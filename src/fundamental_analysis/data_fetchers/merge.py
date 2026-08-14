"""Merge fundamentals from multiple sources into one SNAPSHOT_SCHEMA row per
symbol, field by field, with source-priority fallback and full provenance.

Why field-level rather than row-level merging: no single free source covers
everything (see docs/data_sourcing_spec.md's coverage table) — Screener has
the best financial-statement data but no analyst estimates; yfinance has
analyst estimates but weaker/inconsistent statement data for NSE names;
Trendlyne has neither reliably (most of its good data is paywalled) but
offers a couple of free supplementary fields. Falling back source-by-source
per field (rather than picking one source's whole row) uses as much of the
available free data as actually exists.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def merge_field_records(
    records_by_source: dict[str, dict[str, Any]], source_priority: list[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge one symbol's per-source field dicts into a single dict, plus a
    parallel provenance dict mapping field -> source name it was taken from
    (or "missing" if no source had it). ``source_priority`` lists source
    names in fallback order, e.g. ["screener", "yfinance", "trendlyne"].
    """
    all_fields: set[str] = set()
    for record in records_by_source.values():
        all_fields.update(record.keys())
    all_fields.discard("symbol")

    merged: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for field in all_fields:
        chosen_value = None
        chosen_source = "missing"
        for source in source_priority:
            record = records_by_source.get(source, {})
            value = record.get(field)
            if not _is_missing(value):
                chosen_value = value
                chosen_source = source
                break
        merged[field] = chosen_value if chosen_value is not None else np.nan
        provenance[field] = chosen_source
    return merged, provenance


def merge_sources(
    source_dataframes: dict[str, pd.DataFrame], source_priority: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DataFrame-level wrapper around merge_field_records: merges snapshot
    DataFrames (indexed by symbol) from multiple sources into one merged
    DataFrame plus a same-shaped provenance DataFrame.

    Any source missing from ``source_dataframes`` (e.g. Trendlyne wasn't run
    this time) is simply skipped — merging degrades gracefully to whatever
    sources are actually present.
    """
    all_symbols: set[str] = set()
    for df in source_dataframes.values():
        all_symbols.update(df.index)

    merged_rows: dict[str, dict[str, Any]] = {}
    provenance_rows: dict[str, dict[str, str]] = {}
    for symbol in sorted(all_symbols):
        records_by_source = {}
        for source, df in source_dataframes.items():
            if symbol in df.index:
                records_by_source[source] = df.loc[symbol].to_dict()
        merged, provenance = merge_field_records(records_by_source, source_priority)
        merged["symbol"] = symbol
        merged_rows[symbol] = merged
        provenance_rows[symbol] = provenance

    merged_df = pd.DataFrame.from_dict(merged_rows, orient="index")
    provenance_df = pd.DataFrame.from_dict(provenance_rows, orient="index")

    coverage = merged_df.notna().mean().sort_values(ascending=False)
    logger.info("Field coverage after merge (fraction of symbols with a non-NaN value):\n%s", coverage)

    return merged_df, provenance_df
