"""Consistent logging setup across scripts and modules."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        # Deliberately propagate=True (the default) here: pytest's `caplog`
        # fixture captures log records via a handler on the ROOT logger, so
        # if this logger doesn't propagate, caplog silently sees nothing
        # even though the message printed fine via the StreamHandler above.
        # That was the root cause of test_data_fetchers.py's
        # test_fetch_multiple_quarterly_history_warns_on_near_total_failure
        # failing: the warning genuinely fired (visible in captured stdout)
        # but caplog.records stayed empty. Nothing else in this project
        # configures the root logger, so there's no duplicate-console-output
        # risk from leaving propagation on.
    return logger
