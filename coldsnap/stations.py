from datetime import date

import polars as pl

from coldsnap.config import TABLES
from coldsnap.ingest import ghcn

COMPLETENESS_MIN = 0.95
STATE_ZONE = {
    **dict.fromkeys(
        [
            "CT",
            "DE",
            "DC",
            "GA",
            "ME",
            "MD",
            "MA",
            "NH",
            "NJ",
            "NY",
            "NC",
            "OH",
            "PA",
            "RI",
            "SC",
            "VT",
            "VA",
            "WV",
            "MI",
            "IN",
            "KY",
            "FL",
        ],
        "America/New_York",
    ),
    **dict.fromkeys(
        ["AL", "AR", "IL", "IA", "LA", "MN", "MS", "MO", "OK", "WI", "TX", "TN", "KS", "NE", "SD", "ND"],
        "America/Chicago",
    ),
    **dict.fromkeys(["CO", "MT", "NM", "UT", "WY", "ID"], "America/Denver"),
    "AZ": "America/Phoenix",
    **dict.fromkeys(["CA", "NV", "OR", "WA"], "America/Los_Angeles"),
}


def zone_for(state: str, lon: float, lat: float) -> str:
    # Split states: the boundary longitudes below are approximate and only decide which 6-hourly samples fall
    # into a local day.
    if state == "TX" and lon < -104.9:
        return "America/Denver"
    if state == "FL" and lon < -85.0:
        return "America/Chicago"
    if state == "IN" and (lon < -87.0 and (lat > 41.0 or lat < 38.5)):
        return "America/Chicago"
    if state == "KY" and lon < -85.9:
        return "America/Chicago"
    if state == "TN" and lon > -85.0:
        return "America/New_York"
    if state == "MI" and lon < -87.6:
        return "America/Chicago"
    if state in ("KS", "NE", "SD", "ND") and lon < -101.5:
        return "America/Denver"
    if state == "ID" and lat > 45.5:
        return "America/Los_Angeles"
    if state == "OR" and lon > -117.5 and lat < 44.5:
        return "America/Boise"
    return STATE_ZONE[state]


def select(end: date) -> pl.DataFrame:
    cands = ghcn.candidates()
    obs = ghcn.observations(cands["station"].to_list())
    comp = ghcn.completeness(obs, end)
    keep = cands.join(comp, on="station").filter(pl.col("complete") >= COMPLETENESS_MIN)
    keep = keep.with_columns(
        pl.struct(["state", "lon", "lat"])
        .map_elements(lambda r: zone_for(r["state"], r["lon"], r["lat"]), return_dtype=pl.String)
        .alias("tz")
    )
    return keep.select("station", "name", "state", "lat", "lon", "elevation", "wmo", "tz", "complete").sort(
        "station"
    )


def build(end: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    TABLES.mkdir(parents=True, exist_ok=True)
    stations = select(end)
    stations.write_parquet(TABLES / "stations.parquet")
    obs = ghcn.observations(stations["station"].to_list())
    obs.write_parquet(TABLES / "observations.parquet")
    return stations, obs


def load() -> pl.DataFrame:
    return pl.read_parquet(TABLES / "stations.parquet")
