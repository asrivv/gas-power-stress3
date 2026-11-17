from datetime import date, timedelta

import polars as pl

from coldsnap import calendar


def window(y, m, d):
    return calendar.gas_delivery_window(date(y, m, d))


def days(*pairs):
    return [date(*p) for p in pairs]


def test_midweek_trade_flows_next_day():
    assert window(2021, 2, 9) == days((2021, 2, 10))


def test_friday_package_covers_weekend_and_monday():
    assert window(2023, 1, 6) == days((2023, 1, 7), (2023, 1, 8), (2023, 1, 9))


def test_long_weekend_presidents_day_during_uri():
    assert window(2021, 2, 12) == days((2021, 2, 13), (2021, 2, 14), (2021, 2, 15), (2021, 2, 16))


def test_long_weekend_mlk():
    assert window(2024, 1, 12) == days((2024, 1, 13), (2024, 1, 14), (2024, 1, 15), (2024, 1, 16))


def test_thanksgiving_package_runs_wednesday_to_monday():
    assert window(2022, 11, 23) == days(
        (2022, 11, 24), (2022, 11, 25), (2022, 11, 26), (2022, 11, 27), (2022, 11, 28)
    )


def test_christmas_on_sunday_observed_monday():
    assert window(2022, 12, 23) == days((2022, 12, 24), (2022, 12, 25), (2022, 12, 26), (2022, 12, 27))


def test_christmas_on_saturday_observed_friday():
    assert window(2021, 12, 23) == days((2021, 12, 24), (2021, 12, 25), (2021, 12, 26), (2021, 12, 27))


def test_new_year_on_saturday_observed_previous_friday():
    assert window(2021, 12, 30) == days((2021, 12, 31), (2022, 1, 1), (2022, 1, 2), (2022, 1, 3))


def test_new_year_midweek():
    assert window(2019, 12, 31) == days((2020, 1, 1), (2020, 1, 2))


def test_every_calendar_day_has_exactly_one_trade():
    trades = [
        d
        for d in (date(2020, 11, 1) + timedelta(days=i) for i in range(500))
        if calendar.is_gas_trading_day(d)
    ]
    flows = [f for t in trades for f in calendar.gas_delivery_window(t)]
    assert len(flows) == len(set(flows))
    assert set(flows) >= {date(2020, 11, 10) + timedelta(days=i) for i in range(400)}


def test_expand_assigns_package_price_to_each_day():
    trades = pl.DataFrame({"hub": ["Chicago"], "trade_date": [date(2021, 2, 12)], "price": [129.52]})
    out = calendar.expand_gas_trades(trades).sort("flow_date")
    assert out["flow_date"].to_list() == days((2021, 2, 13), (2021, 2, 14), (2021, 2, 15), (2021, 2, 16))
    assert out["price"].to_list() == [129.52] * 4


def test_trade_available_at_is_2359_eastern_in_utc():
    out = pl.DataFrame({"d": [date(2021, 2, 12), date(2021, 7, 12)]}).select(
        calendar.trade_available_at(pl.col("d")).alias("t")
    )
    assert [t.isoformat() for t in out["t"]] == ["2021-02-13T04:59:00+00:00", "2021-07-13T03:59:00+00:00"]


def test_storage_release_regular_and_holiday_weeks():
    assert calendar.weekly_storage_release(date(2021, 2, 5)).isoformat() == "2021-02-11T15:30:00+00:00"
    assert calendar.weekly_storage_release(date(2021, 7, 9)).isoformat() == "2021-07-15T14:30:00+00:00"
    assert calendar.weekly_storage_release(date(2021, 2, 12)).isoformat() == "2021-02-19T17:00:00+00:00"
    assert calendar.weekly_storage_release(date(2022, 11, 18)).isoformat() == "2022-11-25T17:00:00+00:00"


def test_winter_membership():
    assert calendar.winter_of(date(2021, 2, 15)) == 2020
    assert calendar.winter_of(date(2022, 12, 24)) == 2022
    assert calendar.winter_of(date(2022, 7, 1)) is None
    assert len(calendar.winter_days(2020)) == 121
