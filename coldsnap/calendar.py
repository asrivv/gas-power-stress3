from datetime import UTC, date, datetime, time, timedelta
from functools import cache
from zoneinfo import ZoneInfo

import polars as pl

from coldsnap.config import ISSUE_HOUR_UTC, SEASON_END, SEASON_START

EASTERN = ZoneInfo("America/New_York")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _easter(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return date(year, month, day)


@cache
def nerc_holidays(year: int) -> frozenset[date]:
    days = {
        _observed(date(year, 1, 1)),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    if date(year + 1, 1, 1).weekday() == 5:
        days.add(date(year, 12, 31))
    return frozenset(d for d in days if d.year == year)


@cache
def gas_holidays(year: int) -> frozenset[date]:
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    days = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _observed(date(year, 11, 11)),
        thanksgiving,
        thanksgiving + timedelta(days=1),
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))
    if date(year + 1, 1, 1).weekday() == 5:
        days.add(date(year, 12, 31))
    return frozenset(d for d in days if d.year == year)


def is_gas_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in gas_holidays(day.year)


def next_gas_trading_day(day: date) -> date:
    day += timedelta(days=1)
    while not is_gas_trading_day(day):
        day += timedelta(days=1)
    return day


def gas_delivery_window(trade_date: date) -> list[date]:
    # Next-day gas traded on T flows on every calendar day after T up to and including the next trading day.
    end = next_gas_trading_day(trade_date)
    return [trade_date + timedelta(days=i) for i in range(1, (end - trade_date).days + 1)]


def expand_gas_trades(trades: pl.DataFrame) -> pl.DataFrame:
    trade_dates = trades.get_column("trade_date").unique().to_list()
    windows = pl.DataFrame(
        [(t, d) for t in trade_dates for d in gas_delivery_window(t)],
        schema={"trade_date": pl.Date, "flow_date": pl.Date},
        orient="row",
    )
    out = trades.join(windows, on="trade_date", how="inner")
    out = out.sort("trade_date").group_by(["hub", "flow_date"], maintain_order=True).last()
    return out.with_columns(trade_available_at(pl.col("trade_date")).alias("available_at"))


def trade_available_at(trade_date: pl.Expr) -> pl.Expr:
    # ICE next-day trades settle by T 23:59 ET; the UTC instant shifts with daylight saving.
    return (
        trade_date.cast(pl.Datetime("us"))
        .dt.offset_by("23h59m")
        .dt.replace_time_zone("America/New_York")
        .dt.convert_time_zone("UTC")
    )


def is_power_onpeak_day(day: date) -> bool:
    return day.weekday() < 5 and day not in nerc_holidays(day.year)


def winter_of(day: date) -> int | None:
    start_m, start_d = SEASON_START
    end_m, end_d = SEASON_END
    if (day.month, day.day) >= (start_m, start_d):
        return day.year
    if (day.month, day.day) <= (end_m, end_d):
        return day.year - 1
    return None


def winter_days(winter: int) -> list[date]:
    start = date(winter, *SEASON_START)
    end = date(winter + 1, *SEASON_END)
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def issue_time(day: date) -> datetime:
    return datetime.combine(day, time(ISSUE_HOUR_UTC), tzinfo=UTC)


def end_of_day_utc_plus(day: date, days: int) -> datetime:
    return datetime.combine(day + timedelta(days=days), time(0), tzinfo=UTC)


def is_weekend(day: date) -> bool:
    return day.weekday() >= 5


def is_holiday(day: date) -> bool:
    return day in nerc_holidays(day.year)


def weekly_storage_release(week_ending: date) -> datetime:
    thursday = week_ending + timedelta(days=6)
    monday = thursday - timedelta(days=3)
    shifted = any(
        monday + timedelta(days=i) in gas_holidays((monday + timedelta(days=i)).year) for i in range(4)
    )
    # Holiday weeks move the report by a day either way; I take the later bound, Friday noon ET.
    release = (
        datetime.combine(thursday + timedelta(days=1), time(12, 0))
        if shifted
        else datetime.combine(thursday, time(10, 30))
    )
    return release.replace(tzinfo=EASTERN).astimezone(UTC)
