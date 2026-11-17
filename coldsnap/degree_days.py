import polars as pl

from coldsnap.config import BASE_TEMP_F


def k_to_f(kelvin: pl.Expr) -> pl.Expr:
    return (kelvin - 273.15) * 9.0 / 5.0 + 32.0


def hdd(temp_f: pl.Expr) -> pl.Expr:
    return (BASE_TEMP_F - temp_f).clip(lower_bound=0.0)


def cdd(temp_f: pl.Expr) -> pl.Expr:
    return (temp_f - BASE_TEMP_F).clip(lower_bound=0.0)


def local_daily_mean(forecasts: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    zones = stations.select("station", "tz")
    frames = []
    for tz in zones["tz"].unique().sort():
        ids = zones.filter(pl.col("tz") == tz)["station"]
        part = forecasts.filter(pl.col("station").is_in(ids.implode()))
        frames.append(
            part.with_columns(pl.col("valid_time").dt.convert_time_zone(tz).dt.date().alias("date"))
        )
    local = pl.concat(frames)
    keys = [c for c in ("source", "run_time", "member", "station", "date") if c in local.columns]
    return (
        local.group_by(keys)
        .agg(
            k_to_f(pl.col("t2m_k").mean()).alias("tmean_f"),
            pl.len().alias("samples"),
            pl.col("available_at").max().alias("available_at"),
        )
        .filter(pl.col("samples") == 4)
        .drop("samples")
    )


def regional(temps: pl.DataFrame, weights: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    joined = temps.join(weights.select("region", "station", "weight"), on="station", how="inner")
    out = joined.group_by(keys + ["region"]).agg(
        (hdd(pl.col("tmean_f")) * pl.col("weight")).sum().alias("hdd"),
        (cdd(pl.col("tmean_f")) * pl.col("weight")).sum().alias("cdd"),
        (pl.col("tmean_f") * pl.col("weight")).sum().alias("temp_f"),
        pl.col("weight").sum().alias("coverage"),
    )
    return out.with_columns(pl.col(c) / pl.col("coverage") for c in ("hdd", "cdd", "temp_f"))
