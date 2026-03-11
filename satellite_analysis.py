"""
Module 6 — Satellite Imagery Analysis
Fetches free Esri World Imagery tiles for waypoints (no API key needed)
and runs a lightweight water-detection classifier (NDWI proxy).
"""

import hashlib
import io
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import requests
from PIL import Image

from config import (
    SAT_CACHE_DIR,
    SAT_TILE_ZOOM,
    SAT_WATER_COVERAGE_THRESHOLD,
)


@dataclass
class SatelliteFlag:
    """Result of water detection on a single waypoint satellite image."""
    waypoint_index: int
    lat: float
    lon: float
    water_fraction: float  # 0.0–1.0
    flagged: bool
    image_path: str = ""
    message: str = ""


def _cache_key(lat: float, lon: float, zoom: int) -> str:
    raw = f"{lat:.6f},{lon:.6f},{zoom}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to slippy-map tile coordinates (x, y)."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def fetch_satellite_image(
    lat: float,
    lon: float,
    zoom: int = SAT_TILE_ZOOM,
    use_cache: bool = True,
) -> Image.Image | None:
    """
    Fetch a satellite tile from Esri World Imagery (free, no API key).
    Returns a PIL Image or None on failure.
    """
    cache_path = os.path.join(SAT_CACHE_DIR, f"{_cache_key(lat, lon, zoom)}.png")

    # Check cache (1 hour TTL)
    if use_cache and os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < 3600:
            return Image.open(cache_path)

    tx, ty = _latlon_to_tile(lat, lon, zoom)

    # Esri World Imagery — free, no key required
    url = (
        f"https://server.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        if use_cache:
            img.save(cache_path)
        return img
    except (requests.RequestException, Exception):
        return None


def detect_water_ndwi(img: Image.Image) -> float:
    """
    Estimate water coverage using a simplified NDWI-like index
    on an RGB satellite image.

    NDWI proxy = (Green - NIR) / (Green + NIR)
    Since we only have RGB, we approximate:
        water_index = (Green - Red) / (Green + Red + 1)

    Pixels with water_index > 0.1 and Blue > 100 are classified as water.

    Returns fraction of water pixels (0.0–1.0).
    """
    arr = np.array(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0

    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]

    # Water index = (G - R) / (G + R + 1) to avoid division by zero
    water_index = (green - red) / (green + red + 1.0)

    # Water mask: high green-red ratio AND high blue channel
    water_mask = (water_index > 0.1) & (blue > 100)

    total_pixels = water_mask.size
    water_pixels = water_mask.sum()

    return float(water_pixels) / total_pixels if total_pixels > 0 else 0.0


def analyse_waypoints(
    waypoints: list[tuple[float, float]],
    threshold: float = SAT_WATER_COVERAGE_THRESHOLD,
) -> list[SatelliteFlag]:
    """
    For each waypoint (lat, lon), fetch satellite imagery and
    run water detection. Returns a list of SatelliteFlag results.
    """
    flags = []

    for i, (lat, lon) in enumerate(waypoints):
        img = fetch_satellite_image(lat, lon)

        if img is None:
            flags.append(SatelliteFlag(
                waypoint_index=i,
                lat=lat,
                lon=lon,
                water_fraction=0.0,
                flagged=False,
                message="Could not fetch satellite image",
            ))
            continue

        water_frac = detect_water_ndwi(img)
        is_flagged = water_frac > threshold

        cache_path = os.path.join(
            SAT_CACHE_DIR,
            f"{_cache_key(lat, lon, SAT_TILE_ZOOM)}.png",
        )

        msg = ""
        if is_flagged:
            msg = (
                f"⚠️ Satellite image at waypoint {i} shows possible "
                f"surface water ({water_frac:.0%} coverage). "
                f"Recommend ground confirmation before proceeding."
            )

        flags.append(SatelliteFlag(
            waypoint_index=i,
            lat=lat,
            lon=lon,
            water_fraction=water_frac,
            flagged=is_flagged,
            image_path=cache_path if os.path.exists(cache_path) else "",
            message=msg,
        ))

    return flags
