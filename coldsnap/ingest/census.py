import json
import time
from datetime import UTC, datetime

import polars as pl

from coldsnap.config import RAW
from coldsnap.ingest import fetch

ACS_URL = "https://data.census.gov/api/access/data/table?id=ACSDT5Y{year}.B25040&g=010XX00US$0500000"
ACS_YEARS = tuple(range(2010, 2025))
CENPOP_URL = "https://www2.census.gov/geo/docs/reference/cenpop{year}/county/CenPop{year}_Mean_CO.txt"
CENPOP_YEARS = (2010, 2020)
STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_20m.zip"
FIPS_STATE = {
    "01": "AL",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}
CONUS = frozenset(FIPS_STATE.values())


def release_available_at(year: int) -> datetime:
    return datetime(year, 12, 31, tzinfo=UTC)


def fetch_all() -> dict:
    stats = {"bytes": 0, "files": 0}
    start = time.perf_counter()
    for year in ACS_YEARS:
        path = RAW / "census" / f"acs5_b25040_{year}.json"
        if not path.exists():
            response = fetch.get_sync(ACS_URL.format(year=year))
            payload = response.json()
            if "response" not in payload:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            time.sleep(1.0)
        stats["bytes"] += path.stat().st_size
        stats["files"] += 1
    for year in CENPOP_YEARS:
        path = fetch.download(CENPOP_URL.format(year=year), RAW / "census" / f"cenpop{year}.txt")
        stats["bytes"] += path.stat().st_size
        stats["files"] += 1
    path = fetch.download(STATES_URL, RAW / "census" / "cb_2020_us_state_20m.zip")
    stats["bytes"] += path.stat().st_size
    stats["files"] += 1
    stats["wall_seconds"] = round(time.perf_counter() - start, 1)
    return stats


def county_centers() -> pl.DataFrame:
    frames = []
    for year in CENPOP_YEARS:
        df = pl.read_csv(
            RAW / "census" / f"cenpop{year}.txt",
            encoding="utf8-lossy",
            schema_overrides={"STATEFP": pl.String, "COUNTYFP": pl.String},
        )
        df = df.rename({c: c.lstrip("﻿") for c in df.columns})
        frames.append(
            df.select(
                (pl.col("STATEFP") + pl.col("COUNTYFP")).alias("fips"),
                pl.col("STATEFP").replace_strict(FIPS_STATE, default=None).alias("state"),
                pl.col("POPULATION").cast(pl.Float64).alias("population"),
                pl.col("LATITUDE").cast(pl.Float64).alias("lat"),
                pl.col("LONGITUDE").cast(pl.Float64).alias("lon"),
                pl.lit(year).alias("vintage"),
                pl.lit(release_available_at(year + 1)).alias("available_at"),
            ).filter(pl.col("state").is_not_null())
        )
    return pl.concat(frames)


def gas_households() -> pl.DataFrame:
    frames = []
    for year in ACS_YEARS:
        path = RAW / "census" / f"acs5_b25040_{year}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())["response"]["data"]
        header, rows = data[0], data[1:]
        geo, total, gas = header.index("GEO_ID"), header.index("B25040_001E"), header.index("B25040_002E")
        frames.append(
            pl.DataFrame(
                {
                    "fips": [r[geo][-5:] for r in rows],
                    "households": [float(r[total]) for r in rows],
                    "gas_households": [float(r[gas]) for r in rows],
                }
            )
            .with_columns(
                pl.col("fips").str.slice(0, 2).replace_strict(FIPS_STATE, default=None).alias("state"),
                pl.lit(year).alias("vintage"),
                pl.lit(release_available_at(year + 1)).alias("available_at"),
            )
            .filter(pl.col("state").is_not_null())
        )
    return pl.concat(frames)
