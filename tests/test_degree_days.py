from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from coldsnap import degree_days


def test_hdd_cdd_at_base_and_either_side():
    df = pl.DataFrame({"t": [65.0, 40.0, 80.0, -10.0]}).select(
        degree_days.hdd(pl.col("t")).alias("h"), degree_days.cdd(pl.col("t")).alias("c")
    )
    assert df["h"].to_list() == [0.0, 25.0, 0.0, 75.0]
    assert df["c"].to_list() == [0.0, 0.0, 15.0, 0.0]


def test_kelvin_to_fahrenheit():
    out = pl.DataFrame({"k": [273.15, 233.15]}).select(degree_days.k_to_f(pl.col("k")))
    assert out.to_series().to_list() == pytest.approx([32.0, -40.0])


def test_regional_hdd_is_weighted_mean_of_station_hdd_not_hdd_of_mean():
    temps = pl.DataFrame({"date": [date(2021, 2, 15)] * 2, "station": ["A", "B"], "tmean_f": [45.0, 75.0]})
    weights = pl.DataFrame({"region": ["r", "r"], "station": ["A", "B"], "weight": [0.5, 0.5]})
    out = degree_days.regional(temps, weights, ["date"])
    assert out["hdd"].item() == pytest.approx(10.0)
    assert out["cdd"].item() == pytest.approx(5.0)
    assert out["temp_f"].item() == pytest.approx(60.0)


def test_regional_renormalizes_missing_station():
    temps = pl.DataFrame({"date": [date(2021, 2, 15)], "station": ["A"], "tmean_f": [45.0]})
    weights = pl.DataFrame({"region": ["r", "r"], "station": ["A", "B"], "weight": [0.25, 0.75]})
    assert degree_days.regional(temps, weights, ["date"])["hdd"].item() == pytest.approx(20.0)


def test_local_day_takes_four_samples_in_station_zone():
    run = datetime(2021, 2, 14, tzinfo=UTC)
    steps = range(6, 199, 6)
    rows = [
        {
            "source": "gefs",
            "run_time": run,
            "member": "c00",
            "station": s,
            "valid_time": run + timedelta(hours=h),
            "t2m_k": 273.15 + h,
            "available_at": run + timedelta(hours=6),
        }
        for s in ("EAST", "WEST")
        for h in steps
    ]
    fc = pl.DataFrame(rows).with_columns(
        pl.col("run_time", "valid_time", "available_at").cast(pl.Datetime("us", "UTC"))
    )
    stations = pl.DataFrame({"station": ["EAST", "WEST"], "tz": ["America/New_York", "America/Los_Angeles"]})
    out = degree_days.local_daily_mean(fc, stations)
    east = out.filter((pl.col("station") == "EAST") & (pl.col("date") == date(2021, 2, 15)))
    west = out.filter((pl.col("station") == "WEST") & (pl.col("date") == date(2021, 2, 15)))
    assert east["tmean_f"].item() == pytest.approx(32 + (30 + 36 + 42 + 48) / 4 * 9 / 5)
    assert west["tmean_f"].item() == pytest.approx(32 + (36 + 42 + 48 + 54) / 4 * 9 / 5)
    assert out.filter(pl.col("station") == "EAST")["date"].max() == date(2021, 2, 21)
