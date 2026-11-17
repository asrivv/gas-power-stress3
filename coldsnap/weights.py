from datetime import datetime

import numpy as np
import polars as pl

from coldsnap.config import REGIONS, TABLES
from coldsnap.pit import asof

EARTH_RADIUS_KM = 6371.0
NATIONAL = "national"


def haversine_nearest(lat: np.ndarray, lon: np.ndarray, slat: np.ndarray, slon: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat)[:, None], np.radians(slat)[None, :]
    dphi = p2 - p1
    dlmb = np.radians(slon)[None, :] - np.radians(lon)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return np.argmin(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)), axis=1)


def assign(counties: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    nearest = haversine_nearest(
        counties["lat"].to_numpy(),
        counties["lon"].to_numpy(),
        stations["lat"].to_numpy(),
        stations["lon"].to_numpy(),
    )
    return counties.with_columns(pl.Series("station", stations["station"].to_numpy()[nearest]))


def region_of_state() -> dict[str, str]:
    return {s: r.name for r in REGIONS for s in r.states}


def station_weights(assigned: pl.DataFrame, value: str) -> pl.DataFrame:
    states = region_of_state()
    regional = (
        assigned.with_columns(pl.col("state").replace_strict(states, default=None).alias("region"))
        .filter(pl.col("region").is_not_null())
        .group_by(["region", "station"])
        .agg(pl.col(value).sum().alias("w"))
    )
    national = (
        assigned.group_by("station")
        .agg(pl.col(value).sum().alias("w"))
        .with_columns(pl.lit(NATIONAL).alias("region"))
    )
    out = pl.concat([regional, national.select(regional.columns)])
    return out.with_columns((pl.col("w") / pl.col("w").sum().over("region")).alias("weight")).drop("w")


def build(centers: pl.DataFrame, households: pl.DataFrame, stations: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for vintage in centers["vintage"].unique().sort():
        c = centers.filter(pl.col("vintage") == vintage)
        assigned = assign(c, stations)
        w = station_weights(assigned, "population").with_columns(
            pl.lit("pop").alias("kind"),
            pl.lit(int(vintage)).alias("vintage"),
            pl.lit(c["available_at"][0]).alias("available_at"),
        )
        frames.append(w)
    for vintage in households["vintage"].unique().sort():
        h = households.filter(pl.col("vintage") == vintage)
        center_vintage = 2010 if vintage < 2020 else 2020
        c = centers.filter(pl.col("vintage") == center_vintage).select("fips", "lat", "lon", "available_at")
        joined = h.join(c, on="fips", how="inner", suffix="_center")
        assigned = assign(joined, stations)
        available = max(h["available_at"][0], c["available_at"][0])
        w = station_weights(assigned, "gas_households").with_columns(
            pl.lit("gas").alias("kind"),
            pl.lit(int(vintage)).alias("vintage"),
            pl.lit(available).alias("available_at"),
        )
        frames.append(w)
    return pl.concat(frames).select("kind", "vintage", "region", "station", "weight", "available_at")


def asof_weights(table: pl.DataFrame, t: datetime, kind: str) -> pl.DataFrame:
    visible = asof(table.filter(pl.col("kind") == kind), t)
    if visible.height == 0:
        return visible
    latest = visible["vintage"].max()
    return visible.filter(pl.col("vintage") == latest)


def save(table: pl.DataFrame) -> None:
    table.write_parquet(TABLES / "weights.parquet")


def load() -> pl.DataFrame:
    return pl.read_parquet(TABLES / "weights.parquet")
