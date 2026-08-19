"""Offline magnetic-declination lookup on the vns-sdk/ArduPilot grid."""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path


def default_table_path() -> Path:
    return Path(resources.files("vio_px4_bridge").joinpath(
        "data", "declination_table.json"
    ))


def load_table(path: str | Path | None = None) -> dict:
    table_path = Path(path) if path else default_table_path()
    with table_path.open("r", encoding="utf-8") as stream:
        table = json.load(stream)
    grid = table["declination_deg"]
    if len(grid) < 2 or len(grid[0]) < 2 or any(
        len(row) != len(grid[0]) for row in grid
    ):
        raise ValueError("invalid magnetic declination grid shape")
    for key in ("lat_min_deg", "lat_step_deg", "lon_min_deg", "lon_step_deg"):
        if not math.isfinite(float(table[key])):
            raise ValueError(f"invalid declination table field: {key}")
    if float(table["lat_step_deg"]) <= 0 or float(table["lon_step_deg"]) <= 0:
        raise ValueError("declination table steps must be positive")
    return table


def interpolate_declination_deg(table: dict, lat_deg: float, lon_deg: float) -> float:
    """Bilinear interpolation matching vns-sdk/AP_Declination semantics."""
    if not math.isfinite(lat_deg) or not math.isfinite(lon_deg):
        raise ValueError("latitude and longitude must be finite")
    while lon_deg > 180.0:
        lon_deg -= 360.0
    while lon_deg < -180.0:
        lon_deg += 360.0
    grid = table["declination_deg"]
    rows, cols = len(grid), len(grid[0])
    i = min(max((lat_deg-float(table["lat_min_deg"])) /
                float(table["lat_step_deg"]), 0.0), rows-1.0)
    j = min(max((lon_deg-float(table["lon_min_deg"])) /
                float(table["lon_step_deg"]), 0.0), cols-1.0)
    i0, j0 = min(int(i), rows-2), min(int(j), cols-2)
    fi, fj = i-i0, j-j0
    return (
        grid[i0][j0]*(1-fi)*(1-fj)
        + grid[i0+1][j0]*fi*(1-fj)
        + grid[i0][j0+1]*(1-fi)*fj
        + grid[i0+1][j0+1]*fi*fj
    )


def lookup_declination_deg(lat_deg: float, lon_deg: float,
                           path: str | Path | None = None) -> float:
    return interpolate_declination_deg(load_table(path), lat_deg, lon_deg)
