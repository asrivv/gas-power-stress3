from datetime import UTC, datetime

import polars as pl

from coldsnap import weights
from coldsnap.config import REGIONS


def synthetic():
    stations = pl.DataFrame(
        {
            "station": ["BOS", "NYC", "CHI", "HOU"],
            "lat": [42.36, 40.78, 41.98, 29.98],
            "lon": [-71.0, -73.97, -87.9, -95.36],
        }
    )
    counties = pl.DataFrame(
        {
            "fips": ["25025", "25017", "36061", "36047", "17031", "18089", "48201", "06037"],
            "state": ["MA", "MA", "NY", "NY", "IL", "IN", "TX", "CA"],
            "population": [800.0, 1600.0, 1600.0, 2600.0, 5200.0, 500.0, 4700.0, 10000.0],
            "gas_households": [100.0, 300.0, 200.0, 0.0, 1500.0, 120.0, 900.0, 2500.0],
            "lat": [42.33, 42.48, 40.78, 40.64, 41.84, 41.47, 29.86, 34.32],
            "lon": [-71.07, -71.39, -73.97, -73.95, -87.82, -87.37, -95.39, -118.22],
        }
    )
    return stations, counties


def test_every_county_assigned_exactly_once():
    stations, counties = synthetic()
    assigned = weights.assign(counties, stations)
    assert assigned.height == counties.height
    assert assigned["fips"].n_unique() == counties.height
    assert assigned.filter(pl.col("fips") == "25025")["station"].item() == "BOS"
    assert assigned.filter(pl.col("fips") == "18089")["station"].item() == "CHI"


def test_region_weights_nonnegative_and_sum_to_one():
    stations, counties = synthetic()
    out = weights.station_weights(weights.assign(counties, stations), "gas_households")
    assert (out["weight"] >= 0).all()
    sums = out.group_by("region").agg(pl.col("weight").sum())
    assert all(abs(s - 1.0) < 1e-12 for s in sums["weight"])
    assert set(sums["region"]) >= {"new_england", "new_york", "midwest", "texas", "california", "national"}


def test_california_county_counts_in_its_region_even_with_distant_station():
    stations, counties = synthetic()
    out = weights.station_weights(weights.assign(counties, stations), "population")
    assert out.filter(pl.col("region") == "california")["weight"].sum() == 1.0


def test_asof_weights_uses_latest_visible_vintage():
    t0, t1 = datetime(2011, 12, 31, tzinfo=UTC), datetime(2012, 12, 31, tzinfo=UTC)
    table = pl.DataFrame(
        {
            "kind": ["gas", "gas"],
            "vintage": [2010, 2011],
            "region": ["midwest", "midwest"],
            "station": ["CHI", "CHI"],
            "weight": [1.0, 1.0],
            "available_at": [t0, t1],
        }
    ).with_columns(pl.col("available_at").cast(pl.Datetime("us", "UTC")))
    assert weights.asof_weights(table, datetime(2012, 6, 1, tzinfo=UTC), "gas")["vintage"].to_list() == [2010]
    assert weights.asof_weights(table, datetime(2011, 6, 1, tzinfo=UTC), "gas").height == 0


def test_every_region_state_is_conus():
    from coldsnap.ingest.census import CONUS

    assert all(s in CONUS for r in REGIONS for s in r.states)
