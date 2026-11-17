from contextlib import contextmanager
from datetime import UTC, datetime

import polars as pl

AVAILABLE_AT = "available_at"
TS = pl.Datetime("us", "UTC")

_audits: list[dict] = []


def utc(value) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return datetime.fromisoformat(str(value)).replace(tzinfo=UTC)


def asof(table: pl.DataFrame | pl.LazyFrame, t: datetime) -> pl.DataFrame | pl.LazyFrame:
    t = utc(t)
    out = table.filter(pl.col(AVAILABLE_AT) <= t)
    if _audits and isinstance(out, pl.DataFrame) and out.height:
        latest = out.get_column(AVAILABLE_AT).max()
        for audit in _audits:
            audit["rows"] += out.height
            if audit["max_available_at"] is None or latest > audit["max_available_at"]:
                audit["max_available_at"] = latest
    return out


def asof_latest(table: pl.DataFrame, t: datetime, keys: list[str], order: str) -> pl.DataFrame:
    visible = asof(table, t)
    return visible.sort(order).group_by(keys, maintain_order=True).last()


@contextmanager
def audit():
    record = {"rows": 0, "max_available_at": None}
    _audits.append(record)
    try:
        yield record
    finally:
        _audits.remove(record)


def check_schema(table: pl.DataFrame) -> pl.DataFrame:
    if AVAILABLE_AT not in table.columns:
        raise ValueError("table has no available_at column")
    if table.schema[AVAILABLE_AT] != TS:
        raise TypeError(f"available_at must be {TS}, got {table.schema[AVAILABLE_AT]}")
    return table
