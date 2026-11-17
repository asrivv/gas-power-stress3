import asyncio
import gzip
import io
from datetime import date

import polars as pl

from coldsnap.config import RAW
from coldsnap.ingest import fetch
from coldsnap.ingest.census import CONUS

BUCKET = "https://noaa-ghcn-pds.s3.amazonaws.com"
NCEI = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
FIRST_NORMAL_YEAR = 1991
COMPLETE_FROM = date(2009, 1, 1)


def _dir():
    return RAW / "ghcn"


def fetch_metadata() -> None:
    for name in ("ghcnd-stations.txt", "ghcnd-inventory.txt"):
        fetch.download(f"{BUCKET}/{name}", _dir() / name)


def stations_metadata() -> pl.DataFrame:
    rows = []
    for line in (_dir() / "ghcnd-stations.txt").read_text().splitlines():
        rows.append(
            {
                "station": line[0:11],
                "lat": float(line[12:20]),
                "lon": float(line[21:30]),
                "elevation": float(line[31:37]),
                "state": line[38:40].strip(),
                "name": line[41:71].strip(),
                "gsn": line[72:75].strip(),
                "hcn": line[76:79].strip(),
                "wmo": line[80:85].strip(),
            }
        )
    return pl.DataFrame(rows)


def inventory() -> pl.DataFrame:
    rows = []
    for line in (_dir() / "ghcnd-inventory.txt").read_text().splitlines():
        if line[31:35] in ("TMAX", "TMIN"):
            rows.append(
                {
                    "station": line[0:11],
                    "element": line[31:35],
                    "first": int(line[36:40]),
                    "last": int(line[41:45]),
                }
            )
    return pl.DataFrame(rows)


def candidates() -> pl.DataFrame:
    meta = stations_metadata()
    inv = inventory()
    span = (
        inv.group_by("station")
        .agg(pl.col("first").max().alias("first"), pl.col("last").min().alias("last"), pl.len().alias("n"))
        .filter((pl.col("n") == 2) & (pl.col("first") <= FIRST_NORMAL_YEAR) & (pl.col("last") >= 2026))
    )
    return (
        meta.join(span, on="station")
        .filter(
            pl.col("station").str.starts_with("USW") & (pl.col("wmo") != "") & pl.col("state").is_in(CONUS)
        )
        .sort("station")
    )


async def _fetch_station(http, station: str, stats) -> None:
    path = _dir() / "by_station" / f"{station}.csv.gz"
    if path.exists():
        return
    response = await fetch.get(http, f"{NCEI}/by_station/{station}.csv.gz", stats)
    if response is None or response.status_code != 200:
        raise RuntimeError(f"GHCN download failed for {station}")
    path.write_bytes(response.content)


def fetch_stations(stations: list[str]) -> dict:
    (_dir() / "by_station").mkdir(parents=True, exist_ok=True)
    stats = fetch.Stats()

    async def run():
        async with fetch.client(http2=False) as http:
            await fetch.gather_limited(
                [lambda s=s: _fetch_station(http, s, stats) for s in stations], limit=16
            )

    asyncio.run(run())
    return stats.as_dict()


def read_station(station: str) -> pl.DataFrame:
    raw = gzip.decompress((_dir() / "by_station" / f"{station}.csv.gz").read_bytes())
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=False,
        new_columns=["ID", "DATE", "ELEMENT", "DATA_VALUE", "M_FLAG", "Q_FLAG", "S_FLAG", "OBS_TIME"],
        schema_overrides={
            "DATE": pl.String,
            "Q_FLAG": pl.String,
            "DATA_VALUE": pl.Float64,
            "M_FLAG": pl.String,
            "S_FLAG": pl.String,
            "OBS_TIME": pl.String,
        },
    )
    df = df.filter(pl.col("ELEMENT").is_in(["TMAX", "TMIN"]) & pl.col("Q_FLAG").is_null())
    df = df.filter(pl.col("DATE") >= f"{FIRST_NORMAL_YEAR}0101")
    wide = df.pivot(on="ELEMENT", index="DATE", values="DATA_VALUE").drop_nulls()
    for col in ("TMAX", "TMIN"):
        if col not in wide.columns:
            return pl.DataFrame()
    return wide.select(
        pl.lit(station).alias("station"),
        pl.col("DATE").str.strptime(pl.Date, "%Y%m%d").alias("date"),
        (pl.col("TMAX") / 10 * 9 / 5 + 32).alias("tmax_f"),
        (pl.col("TMIN") / 10 * 9 / 5 + 32).alias("tmin_f"),
    ).filter(pl.col("tmax_f") >= pl.col("tmin_f"))


def observations(stations: list[str]) -> pl.DataFrame:
    frames = [read_station(s) for s in stations]
    df = pl.concat([f for f in frames if f.height])
    return df.with_columns(
        ((pl.col("tmax_f") + pl.col("tmin_f")) / 2).alias("tmean_f"),
        # GHCN-Daily for day d is in the public files by d + 2 days 00Z.
        (pl.col("date").cast(pl.Datetime("us")).dt.replace_time_zone("UTC") + pl.duration(days=2)).alias(
            "available_at"
        ),
    )


def completeness(obs: pl.DataFrame, end: date) -> pl.DataFrame:
    days = (end - COMPLETE_FROM).days + 1
    return (
        obs.filter((pl.col("date") >= COMPLETE_FROM) & (pl.col("date") <= end))
        .group_by("station")
        .agg((pl.len() / days).alias("complete"))
    )
