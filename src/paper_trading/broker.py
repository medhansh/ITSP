"""Paper trading: broker adapters, Indian transaction-cost model, and a
persistent portfolio ledger.

The broker is behind an interface so that swapping the simulator for a live
Zerodha Kite Connect account changes one class and one config line, and
nothing in the strategy, scoring or reporting layers.

``SimulatedBroker`` fills at the next day's open (or close, if no open is
available) rather than at the price used to make the decision. That
one-day lag matters: filling at the decision price is the same look-ahead
bug that was found and fixed in the backtest engine, and it is just as
easy to reintroduce here.

**What this simulator does and does not validate.** It does test whether
the strategy works on genuinely forward data, whether point-in-time
fundamentals are actually published when the backtest assumes, and whether
realised turnover matches the backtest's estimate. It does NOT test real
fill prices, market impact, or liquidity limits on smaller NIFTY500 names
-- filling at the open with a modelled slippage is optimistic. Those need
a real broker.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


# --- Indian delivery-equity friction, all rates as fractions of turnover ---
BROKERAGE = 0.0            # Zerodha delivery equity is free
STT_RATE = 0.001           # both buy and sell
EXCHANGE_RATE = 0.0000297
SEBI_RATE = 0.000001
STAMP_DUTY_RATE = 0.00015  # buy side only
GST_RATE = 0.18            # on brokerage + exchange + SEBI


@dataclass
class Fill:
    date: str
    symbol: str
    side: str           # "BUY" or "SELL"
    quantity: int
    price: float
    costs: float

    @property
    def value(self) -> float:
        return self.quantity * self.price


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    entry_date: str     # when this holding was LAST opened from flat -- see Ledger.apply_fill
    last_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.last_price - self.avg_price)

    @property
    def return_since_entry(self) -> float:
        return (self.last_price / self.avg_price - 1.0) if self.avg_price else 0.0


def transaction_costs(value: float, side: str, slippage_bps: float = 5.0) -> float:
    """Total friction on one leg. ``value`` is absolute turnover in rupees.

    Slippage is charged as a modelled spread cost because the simulator
    fills at a single observed price; without it, paper results would be
    systematically better than anything achievable.
    """
    value = abs(value)
    slippage = value * slippage_bps / 10_000.0
    stt = value * STT_RATE
    exchange = value * EXCHANGE_RATE
    sebi = value * SEBI_RATE
    stamp = value * STAMP_DUTY_RATE if side == "BUY" else 0.0
    gst = (BROKERAGE + exchange + sebi) * GST_RATE
    return slippage + stt + exchange + sebi + stamp + gst + BROKERAGE


class BrokerAdapter:
    """Interface a live broker must satisfy. See ``SimulatedBroker``."""

    def execute(self, target_weights: pd.Series, prices: pd.Series,
                nav: float, positions: dict[str, Position],
                trade_date: str) -> list[Fill]:
        raise NotImplementedError


class SimulatedBroker(BrokerAdapter):
    """Fills every order at the supplied price, applying the cost model.

    Whole shares only, so small target weights can round to zero -- which
    is realistic and is why the ledger reports the resulting cash drag
    rather than hiding it.
    """

    def __init__(self, slippage_bps: float = 5.0):
        self.slippage_bps = slippage_bps

    def execute(self, target_weights, prices, nav, positions, trade_date) -> list[Fill]:
        fills: list[Fill] = []
        target_qty: dict[str, int] = {}
        for sym, w in target_weights.items():
            px = prices.get(sym)
            if px is None or not pd.notna(px) or px <= 0:
                continue
            target_qty[sym] = int((nav * float(w)) // px)

        # Sells first, so their proceeds are available for the buys.
        for sym, pos in list(positions.items()):
            tgt = target_qty.get(sym, 0)
            if tgt < pos.quantity:
                px = prices.get(sym)
                if px is None or not pd.notna(px) or px <= 0:
                    continue
                qty = pos.quantity - tgt
                fills.append(Fill(trade_date, sym, "SELL", qty, float(px),
                                  transaction_costs(qty * px, "SELL", self.slippage_bps)))
        for sym, tgt in target_qty.items():
            held = positions[sym].quantity if sym in positions else 0
            if tgt > held:
                px = float(prices[sym])
                qty = tgt - held
                fills.append(Fill(trade_date, sym, "BUY", qty, px,
                                  transaction_costs(qty * px, "BUY", self.slippage_bps)))
        return fills


class KiteBroker(BrokerAdapter):
    """Zerodha Kite Connect adapter -- NOT IMPLEMENTED.

    Deliberately left as a stub rather than a half-working client. Kite
    Connect costs Rs 2,000/month, needs an active Zerodha account and app
    registration, and has no free sandbox, so it cannot be written or
    tested without a paid subscription. Implementing this class and setting
    ``paper_trading.broker: kite`` is the only change required -- the
    strategy, scoring and reporting layers do not know which broker is in
    use.

    A live implementation additionally needs: an order state machine
    (submitted / open / partially filled / filled / rejected), reconciliation
    between the WebSocket stream and the REST order book, and handling for
    circuit freezes. None of that is required by the simulator because it
    fills synchronously.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "KiteBroker is a stub. Implement it against kiteconnect, or keep "
            "paper_trading.broker: simulated."
        )


@dataclass
class Ledger:
    """Persistent portfolio state. Serialises to a single JSON file so a
    paper-trading run can be stopped and resumed without losing history."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    nav_history: list[dict] = field(default_factory=list)
    rebalance_dates: list[str] = field(default_factory=list)
    starting_capital: float = 0.0

    @classmethod
    def new(cls, starting_capital: float) -> "Ledger":
        return cls(cash=starting_capital, starting_capital=starting_capital)

    def apply_fill(self, fill: Fill) -> None:
        if fill.side == "BUY":
            self.cash -= fill.value + fill.costs
            if fill.symbol in self.positions:
                p = self.positions[fill.symbol]
                total = p.quantity + fill.quantity
                p.avg_price = (p.avg_price * p.quantity + fill.price * fill.quantity) / total
                p.quantity = total
            else:
                # entry_date is set only when a position is opened FROM FLAT,
                # so "how has this done since it entered the portfolio" measures
                # the current holding period and is not reset by a top-up.
                self.positions[fill.symbol] = Position(
                    fill.symbol, fill.quantity, fill.price, fill.date, fill.price)
        else:
            self.cash += fill.value - fill.costs
            p = self.positions.get(fill.symbol)
            if p:
                p.quantity -= fill.quantity
                if p.quantity <= 0:
                    del self.positions[fill.symbol]
        self.fills.append(fill)

    def mark_to_market(self, prices: pd.Series, on: str) -> float:
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is not None and pd.notna(px) and px > 0:
                pos.last_price = float(px)
        nav = self.cash + sum(p.market_value for p in self.positions.values())
        self.nav_history.append({
            "date": on, "nav": nav, "cash": self.cash,
            "n_positions": len(self.positions),
        })
        return nav

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "cash": self.cash,
            "starting_capital": self.starting_capital,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "fills": [asdict(f) for f in self.fills],
            "nav_history": self.nav_history,
            "rebalance_dates": self.rebalance_dates,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        d = json.loads(Path(path).read_text())
        return cls(
            cash=d["cash"], starting_capital=d.get("starting_capital", 0.0),
            positions={k: Position(**v) for k, v in d["positions"].items()},
            fills=[Fill(**f) for f in d["fills"]],
            nav_history=d["nav_history"], rebalance_dates=d.get("rebalance_dates", []),
        )
