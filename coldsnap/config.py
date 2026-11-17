from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
TABLES = DATA / "tables"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
FIXTURE = ROOT / "tests" / "fixture"

SEED = 20201
ISSUE_HOUR_UTC = 12
LEADS = (1, 2, 3, 4, 5, 6, 7)
SEASON_START = (11, 15)
SEASON_END = (3, 15)
BASE_TEMP_F = 65.0


@dataclass(frozen=True)
class Region:
    name: str
    states: tuple[str, ...]
    gas_hub: str | None
    power_hub: str | None
    storage_regions: tuple[str, ...]


REGIONS = (
    Region("new_england", ("CT", "MA", "RI", "NH", "VT", "ME"), None, "Nepool MH DA LMP Peak", ("East",)),
    Region("new_york", ("NY",), "New York", None, ("East",)),
    Region("mid_atlantic", ("NJ", "PA", "DE", "MD", "DC", "VA"), None, "PJM WH Real Time Peak", ("East",)),
    Region(
        "midwest",
        ("IL", "IN", "MI", "OH", "WI", "MN", "IA", "MO"),
        "Chicago",
        "Indiana Hub RT Peak",
        ("Midwest", "East"),
    ),
    Region("texas", ("TX",), None, None, ("South Central",)),
    Region("california", ("CA",), "Cal. Comp. Avg.", "SP15 EZ Gen DA LMP Peak", ("Pacific",)),
    Region("pacific_nw", ("WA", "OR", "ID"), None, "Mid C Peak", ("Pacific", "Mountain")),
)
REGION_NAMES = tuple(r.name for r in REGIONS)
GAS_REGIONS = tuple(r.name for r in REGIONS if r.gas_hub)
POWER_REGIONS = tuple(r.name for r in REGIONS if r.power_hub)
HENRY_HUB = "Henry Hub"

STORAGE_REGION_STATES = {
    "East": (
        "CT",
        "DE",
        "DC",
        "FL",
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
    ),
    "Midwest": ("IL", "IN", "IA", "KY", "MI", "MN", "MO", "TN", "WI"),
    "Mountain": ("AZ", "CO", "ID", "MT", "NE", "NV", "NM", "ND", "SD", "UT", "WY"),
    "Pacific": ("CA", "OR", "WA"),
    "South Central": ("AL", "AR", "KS", "LA", "MS", "OK", "TX"),
}

TRAIN_WINTERS = tuple(range(2011, 2020))
TEST_WINTERS = tuple(range(2020, 2026))

GAS_EVENT_FLOOR = 5.00
GAS_EVENT_MULT = 3.0
POWER_EVENT_FLOOR = 150.0
POWER_EVENT_MULT = 3.0
EVENT_LOOKBACK_DAYS = 60
EVENT_PROB_THRESHOLDS = (0.5, 0.3)

BIAS_HALF_LIFE_DAYS = 14.0
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
SAMPLES_PER_MEMBER = 200
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_RESAMPLES = 1000
RELIABILITY_BINS = 10

GEFS_MEMBERS = ("c00",) + tuple(f"p{i:02d}" for i in range(1, 31))
GEFS_STEPS = tuple(range(6, 199, 6))
ECMWF_MEMBERS = (0,) + tuple(range(1, 51))
ECMWF_STEPS = tuple(range(6, 199, 6))
DOWNLOAD_CONCURRENCY = 32
