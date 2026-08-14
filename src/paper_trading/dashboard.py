"""Renders the paper-trading ledger as a single self-contained HTML file.

All data is embedded as JSON, so the report is one file that opens in any
browser with no server, no build step, and no dependency on the Python
environment that produced it. Charts are drawn with inline SVG rather than
a charting library so the file works offline.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.logging_utils import get_logger
from src.paper_trading.engine import (
    compute_statistics, holdings_detail, interval_returns,
)

logger = get_logger(__name__)

TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ITSP Paper Trading</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#10151f; --ink-2:#3d4757; --ink-3:#6b7688;
  --paper:#eef1f6; --card:#fbfcfe; --line:#d7dde7;
  --up:#0f8a6a; --down:#c8483c; --accent:#2f4f8f; --accent-soft:#e3e9f6;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.num{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1220px;margin:0 auto;padding:28px 20px 72px}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;flex-wrap:wrap;
  padding-bottom:18px;border-bottom:2px solid var(--ink)}
.eyebrow{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-3);font-weight:600}
h1{margin:2px 0 0;font-size:20px;font-weight:600;letter-spacing:-.01em}
.nav-block{text-align:right}
.nav-val{font-size:38px;font-weight:600;letter-spacing:-.02em;line-height:1.1}
.nav-sub{font-size:13px;color:var(--ink-2);margin-top:2px}
.up{color:var(--up)} .down{color:var(--down)}
section{margin-top:32px}
h2{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink-3);
  font-weight:600;margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px}
.pad{padding:18px 20px}
.intervals{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.chip{border:1px solid var(--line);background:var(--card);border-radius:999px;
  padding:7px 14px;cursor:pointer;font:inherit;font-size:13px;color:var(--ink-2);
  display:flex;gap:8px;align-items:center;transition:.14s}
.chip:hover{border-color:var(--accent)}
.chip[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.chip .v{font-family:"IBM Plex Mono",monospace;font-size:12px;font-weight:500}
.chip[aria-pressed="true"] .v{color:#fff}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(158px,1fr))}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.stat .k{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);font-weight:600}
.stat .v{font-family:"IBM Plex Mono",monospace;font-size:21px;font-weight:500;margin-top:5px;letter-spacing:-.01em}
.stat .b{font-size:11px;color:var(--ink-3);margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
  font-weight:600;text-align:right;padding:0 12px 9px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{padding:10px 12px;border-bottom:1px solid #eef1f5;text-align:right;white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--accent-soft)}
.sym{font-weight:600;letter-spacing:-.01em}
.meta{font-size:11px;color:var(--ink-3);font-weight:400;margin-top:1px}
.spark{display:block}
.empty{padding:40px 20px;text-align:center;color:var(--ink-3);font-size:14px}
footer{margin-top:36px;padding-top:16px;border-top:1px solid var(--line);
  font-size:12px;color:var(--ink-3);display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
@media (max-width:720px){.nav-val{font-size:30px}.wrap{padding:20px 14px 56px}
  th:nth-child(4),td:nth-child(4),th:nth-child(5),td:nth-child(5){display:none}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body><div class="wrap">

<header>
  <div><div class="eyebrow">ITSP &middot; Paper Trading</div><h1 id="period"></h1></div>
  <div class="nav-block">
    <div class="nav-val num" id="nav"></div>
    <div class="nav-sub num" id="navsub"></div>
  </div>
</header>

<section>
  <h2>Portfolio return by interval</h2>
  <div class="intervals" id="chips"></div>
  <div class="card pad"><div id="chart"></div></div>
</section>

<section><h2>Statistics</h2><div class="grid" id="stats"></div></section>

<section>
  <h2>Holdings &middot; performance since entry</h2>
  <div class="card"><div class="pad" style="padding-bottom:4px"><div id="holdings"></div></div></div>
</section>

<section><h2>Recent activity</h2><div class="card"><div class="pad" style="padding-bottom:4px"><div id="fills"></div></div></div></section>

<footer><span id="gen"></span><span>Simulated fills &middot; costs modelled &middot; not live execution</span></footer>
</div>
<script>const DATA = __DATA__;</script>
<script>
const $=s=>document.querySelector(s);
const pct=(v,d=2)=>v==null||isNaN(v)?"&mdash;":(v>=0?"+":"")+(v*100).toFixed(d)+"%";
const cur=v=>v==null||isNaN(v)?"—":"₹"+Math.round(v).toLocaleString("en-IN");
const cls=v=>v==null||isNaN(v)?"":v>=0?"up":"down";
const fx=(v,d=2)=>v==null||isNaN(v)?"&mdash;":v.toFixed(d);

$("#period").textContent=DATA.stats.start_date?`${DATA.stats.start_date} to ${DATA.stats.end_date} · ${DATA.stats.trading_days} trading days`:"No history yet";
$("#nav").textContent=cur(DATA.stats.current_nav);
const d1=DATA.intervals["1D"];
$("#navsub").innerHTML=`<span class="${cls(d1)}">${pct(d1)}</span> today &nbsp;·&nbsp; <span class="${cls(DATA.stats.total_return)}">${pct(DATA.stats.total_return)}</span> since start`;
$("#gen").textContent="Generated "+DATA.generated_at;

// interval chips
const order=["1D","3D","5D","10D","21D","63D","126D","252D","ALL"];
const chips=order.filter(k=>k in DATA.intervals);
$("#chips").innerHTML=chips.map((k,i)=>
  `<button class="chip" role="button" aria-pressed="${i===chips.length-1}" data-k="${k}">${k}<span class="v ${cls(DATA.intervals[k])}">${pct(DATA.intervals[k],1)}</span></button>`).join("");

// equity curve
const nav=DATA.nav_history;
function draw(k){
  const n=k==="ALL"?nav.length:Math.min(nav.length,(parseInt(k)||1)+1);
  const s=nav.slice(-n); const W=1100,H=250,P={t:14,r:14,b:24,l:56};
  if(s.length<2){$("#chart").innerHTML='<div class="empty">Not enough history for this interval.</div>';return;}
  const vs=s.map(r=>r.nav),lo=Math.min(...vs),hi=Math.max(...vs),pad=(hi-lo)*0.10||1;
  const y0=lo-pad,y1=hi+pad;
  const X=i=>P.l+i*(W-P.l-P.r)/(s.length-1);
  const Y=v=>P.t+(1-(v-y0)/(y1-y0))*(H-P.t-P.b);
  const line=s.map((r,i)=>`${i?"L":"M"}${X(i).toFixed(1)},${Y(r.nav).toFixed(1)}`).join("");
  const area=`${line}L${X(s.length-1).toFixed(1)},${Y(y0)}L${X(0)},${Y(y0)}Z`;
  const rising=vs[vs.length-1]>=vs[0], col=rising?"var(--up)":"var(--down)";
  let ticks="";
  for(let i=0;i<=3;i++){const v=y0+(y1-y0)*i/3,yy=Y(v);
    ticks+=`<line x1="${P.l}" y1="${yy}" x2="${W-P.r}" y2="${yy}" stroke="#e6eaf1"/>
    <text x="${P.l-9}" y="${yy+4}" text-anchor="end" font-size="10.5" fill="#6b7688" font-family="IBM Plex Mono">${Math.round(v/1000)}k</text>`;}
  const marks=DATA.rebalance_dates.map(d=>s.findIndex(r=>r.date===d)).filter(i=>i>0)
    .map(i=>`<line x1="${X(i).toFixed(1)}" y1="${P.t}" x2="${X(i).toFixed(1)}" y2="${H-P.b}" stroke="var(--accent)" stroke-width="1" stroke-dasharray="2 4" opacity=".38"/>`).join("");
  $("#chart").innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" role="img" aria-label="Portfolio value">
    <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="${col}" stop-opacity=".16"/><stop offset="100%" stop-color="${col}" stop-opacity="0"/>
    </linearGradient></defs>${ticks}${marks}
    <path d="${area}" fill="url(#g)"/><path d="${line}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round"/>
    <text x="${P.l}" y="${H-6}" font-size="10.5" fill="#6b7688" font-family="IBM Plex Mono">${s[0].date}</text>
    <text x="${W-P.r}" y="${H-6}" text-anchor="end" font-size="10.5" fill="#6b7688" font-family="IBM Plex Mono">${s[s.length-1].date}</text>
  </svg>`;
}
document.querySelectorAll(".chip").forEach(c=>c.addEventListener("click",()=>{
  document.querySelectorAll(".chip").forEach(x=>x.setAttribute("aria-pressed","false"));
  c.setAttribute("aria-pressed","true"); draw(c.dataset.k);}));
draw(chips[chips.length-1]||"ALL");

// stats
const S=DATA.stats;
// Annualised measures are withheld below ~a quarter of data: extrapolating a
// few days to a yearly figure is arithmetically valid and totally misleading.
const warm=S.annualised===false?`needs ${63-S.trading_days} more days`:null;
const cards=[
  ["CAGR",S.cagr==null?"&mdash;":pct(S.cagr),warm||(S.benchmark_cagr!=null?"index "+pct(S.benchmark_cagr):null),cls(S.cagr)],
  ["Sharpe",fx(S.sharpe),warm||(S.benchmark_sharpe!=null?"index "+fx(S.benchmark_sharpe):null),""],
  ["Max drawdown",pct(S.max_drawdown),S.benchmark_max_drawdown!=null?"index "+pct(S.benchmark_max_drawdown):null,"down"],
  ["Volatility",S.volatility==null?"&mdash;":pct(S.volatility),warm,""],
  ["Sortino",fx(S.sortino),warm,""],
  ["Hit rate",pct(S.hit_rate,1),"of trading days",""],
  ["Current drawdown",pct(S.current_drawdown),null,S.current_drawdown<0?"down":""],
  ["Positions",String(S.n_positions??0),cur(S.cash)+" cash",""],
  ["Rebalances",String(S.n_rebalances??0),(S.n_fills??0)+" fills",""],
  ["Costs paid",cur(S.total_costs),"since inception",""],
];
if(S.excess_return!=null)cards.splice(3,0,["vs index",pct(S.excess_return),"total return",cls(S.excess_return)]);
$("#stats").innerHTML=cards.map(([k,v,b,c])=>
  `<div class="stat"><div class="k">${k}</div><div class="v ${c}">${v}</div>${b?`<div class="b">${b}</div>`:""}</div>`).join("");

// holdings, each with a price path since it entered the portfolio
function spark(p){
  if(!p||p.length<2)return"";
  const W=104,H=26,lo=Math.min(...p),hi=Math.max(...p),r=(hi-lo)||1;
  const pts=p.map((v,i)=>`${(i*W/(p.length-1)).toFixed(1)},${(H-2-((v-lo)/r)*(H-4)).toFixed(1)}`).join(" ");
  const c=p[p.length-1]>=p[0]?"var(--up)":"var(--down)";
  return `<svg class="spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" aria-hidden="true">
    <polyline points="${pts}" fill="none" stroke="${c}" stroke-width="1.5"/>
    <circle cx="0" cy="${(H-2-((p[0]-lo)/r)*(H-4)).toFixed(1)}" r="2" fill="var(--ink-3)"/></svg>`;
}
const H=DATA.holdings;
$("#holdings").innerHTML=!H.length?'<div class="empty">No open positions.</div>':
`<table><thead><tr><th>Stock</th><th>Since entry</th><th>Path since entry</th><th>Yesterday</th>
<th>Qty</th><th>Avg</th><th>Last</th><th>Value</th><th>Weight</th></tr></thead><tbody>
${H.sort((a,b)=>b.market_value-a.market_value).map(h=>`<tr>
<td><div class="sym">${h.symbol}</div><div class="meta">held ${h.held_trading_days}d · from ${h.entry_date}</div></td>
<td class="num ${cls(h.return_since_entry)}">${pct(h.return_since_entry)}</td>
<td>${spark(h.path)}</td>
<td class="num ${cls(h.day_change)}">${pct(h.day_change)}</td>
<td class="num">${h.quantity}</td><td class="num">${h.avg_price.toFixed(1)}</td>
<td class="num">${h.last_price.toFixed(1)}</td><td class="num">${cur(h.market_value)}</td>
<td class="num">${(h.weight*100).toFixed(1)}%</td></tr>`).join("")}</tbody></table>`;

const F=DATA.recent_fills;
$("#fills").innerHTML=!F.length?'<div class="empty">No trades yet.</div>':
`<table><thead><tr><th>Date</th><th>Stock</th><th>Side</th><th>Qty</th><th>Price</th><th>Value</th><th>Costs</th></tr></thead><tbody>
${F.map(f=>`<tr><td class="num">${f.date}</td><td class="sym">${f.symbol}</td>
<td class="${f.side==="BUY"?"up":"down"}" style="font-weight:600">${f.side}</td>
<td class="num">${f.quantity}</td><td class="num">${f.price.toFixed(1)}</td>
<td class="num">${cur(f.quantity*f.price)}</td><td class="num">${cur(f.costs)}</td></tr>`).join("")}</tbody></table>`;
</script></body></html>"""


def render_dashboard(ledger, stock_prices: pd.DataFrame,
                     benchmark: pd.Series | None = None,
                     out_path: str = "reports/paper_trading.html") -> str:
    stats = compute_statistics(ledger, benchmark)
    holdings = holdings_detail(ledger, stock_prices)
    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "stats": stats,
        "intervals": interval_returns(ledger),
        "nav_history": ledger.nav_history,
        "rebalance_dates": ledger.rebalance_dates,
        "holdings": holdings,
        "recent_fills": [
            {"date": f.date, "symbol": f.symbol, "side": f.side,
             "quantity": f.quantity, "price": f.price, "costs": f.costs}
            for f in ledger.fills[-40:][::-1]
        ],
    }
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, default=str))
    p = Path(out_path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p)
