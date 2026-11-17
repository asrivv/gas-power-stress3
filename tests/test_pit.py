from datetime import UTC, datetime, timedelta

import polars as pl

from coldsnap.pit import TS, asof, asof_latest, audit


def table(times):
    return pl.DataFrame({"key": list(range(len(times))), "available_at": times}).with_columns(
        pl.col("available_at").cast(TS)
    )


def test_asof_excludes_rows_one_second_late():
    t = datetime(2021, 2, 14, 12, tzinfo=UTC)
    df = table([t - timedelta(days=1), t, t + timedelta(seconds=1)])
    assert asof(df, t)["key"].to_list() == [0, 1]


def test_asof_latest_picks_newest_visible_vintage():
    t = datetime(2021, 2, 14, 12, tzinfo=UTC)
    df = pl.DataFrame(
        {
            "k": ["a", "a", "a"],
            "v": [1, 2, 3],
            "available_at": [t - timedelta(days=2), t, t + timedelta(hours=1)],
        }
    ).with_columns(pl.col("available_at").cast(TS))
    assert asof_latest(df, t, ["k"], "available_at")["v"].to_list() == [2]


def test_audit_records_latest_input():
    t = datetime(2021, 2, 14, 12, tzinfo=UTC)
    df = table([t - timedelta(days=3), t - timedelta(hours=1), t + timedelta(hours=1)])
    with audit() as record:
        asof(df, t)
    assert record["max_available_at"] == t - timedelta(hours=1)
    assert record["rows"] == 2
