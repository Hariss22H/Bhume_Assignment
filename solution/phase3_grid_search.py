"""
Phase 3 — Grid Search Candidate Scoring
Strategy:
  - Grid is centred on global shift (gdx, gdy)
  - Shift score rewards staying NEAR the global shift centre
  - Edge + boundary scores pull toward real field boundaries
  - Best candidate = highest weighted total
"""

from __future__ import annotations

import math
import numpy as np
import rasterio
from shapely.affinity import translate
from shapely.geometry.base import BaseGeometry
from bhume.geo import patch_for_plot, geom_to_imagery_crs

# ── Grid constants ────────────────────────────────────────────────────────────
GRID_RANGE_M  = 15.0   # search ±15m around global shift centre
GRID_STEP_M   = 2.5    # step size in metres
MAX_SHIFT_M   = 60.0   # absolute sanity cap

# ── Scoring weights ───────────────────────────────────────────────────────────
W_EDGE        = 0.40
W_BOUNDARY    = 0.30
W_AREA        = 0.15
W_SHIFT       = 0.15   # shift score now relative to global centre


def metres_to_degrees(dx_m: float, dy_m: float,
                       ref_lat: float) -> tuple[float, float]:
    dlat = dy_m / 111320.0
    dlon = dx_m / (111320.0 * math.cos(math.radians(ref_lat)))
    return dlon, dlat


def translate_polygon_metres(polygon: BaseGeometry,
                              dx_m: float, dy_m: float) -> BaseGeometry:
    ref_lat = polygon.centroid.y
    dlon, dlat = metres_to_degrees(dx_m, dy_m, ref_lat)
    return translate(polygon, xoff=dlon, yoff=dlat)


def generate_candidates(gdx: float, gdy: float) -> list[tuple[float, float]]:
    steps = int(GRID_RANGE_M / GRID_STEP_M)
    candidates = []
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            candidates.append((
                gdx + i * GRID_STEP_M,
                gdy + j * GRID_STEP_M
            ))
    return candidates


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[:, :, 0] +
            0.587 * rgb[:, :, 1] +
            0.114 * rgb[:, :, 2]).astype(np.float64)


def _detect_edges(gray: np.ndarray) -> np.ndarray:
    from scipy.ndimage import sobel
    sx = sobel(gray, axis=1)
    sy = sobel(gray, axis=0)
    edges = np.hypot(sx, sy)
    if edges.max() > 0:
        edges = edges / edges.max()
    return edges


def _rasterise_boundary(polygon_img_crs: BaseGeometry,
                         patch_transform,
                         shape: tuple[int, int]) -> np.ndarray:
    from rasterio.features import rasterize
    from shapely.geometry import mapping, LineString
    from scipy.ndimage import binary_dilation
    H, W = shape

    try:
        # Handle both Polygon and MultiPolygon
        if polygon_img_crs.geom_type == 'MultiPolygon':
            geoms = list(polygon_img_crs.geoms)
        else:
            geoms = [polygon_img_crs]

        # Rasterise all exterior rings as lines
        shapes = [
            (mapping(LineString(g.exterior.coords)), 1)
            for g in geoms
        ]

        burned = rasterize(
            shapes,
            out_shape=(H, W),
            transform=patch_transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )

        if burned.sum() == 0:
            # Fallback: filled polygon edge via erosion
            filled_shapes = [(mapping(g), 1) for g in geoms]
            filled = rasterize(
                filled_shapes,
                out_shape=(H, W),
                transform=patch_transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8
            )
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(filled)
            burned = (filled.astype(bool) & ~eroded).astype(np.uint8)

        # Dilate 2px for sub-pixel tolerance
        dilated = binary_dilation(burned, iterations=2).astype(np.float32)
        return dilated

    except Exception:
        return np.zeros((H, W), dtype=np.float32)


def _boundary_patch_under_polygon(boundary_src,
                                   geom_imagery_crs: BaseGeometry,
                                   pad_m: float = 30.0):
    """
    Read a single-band patch from boundaries.tif under a polygon
    (already in imagery CRS), with its own transform.

    Returns: (data_normalised, transform) tuple
        data_normalised: (H,W) float32 array in [0,1]
        transform: rasterio Affine transform for this patch
    """
    from rasterio.windows import from_bounds
    try:
        minx, miny, maxx, maxy = geom_imagery_crs.bounds
        window = from_bounds(
            minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m,
            transform=boundary_src.transform
        )
        data = boundary_src.read(1, window=window).astype(np.float32)
        transform = boundary_src.window_transform(window)

        if data.max() > 0:
            data = data / 255.0

        return data, transform
    except Exception:
        return np.zeros((10, 10), dtype=np.float32), None


def score_candidate(
    polygon: BaseGeometry,
    dx_m: float,
    dy_m: float,
    imagery_patch,
    boundary_values: np.ndarray,
    area_ratio: float,
    edge_map: np.ndarray,
    img_src,
    gdx: float = 0.0,
    gdy: float = 0.0,
) -> dict:

    H, W = imagery_patch.image.shape[:2]
    shifted = translate_polygon_metres(polygon, dx_m, dy_m)

    # ── Signal 1: Edge alignment ──────────────────────────────────────────
    edge_score = 0.0
    try:
        shifted_img = geom_to_imagery_crs(img_src, shifted)
        poly_mask   = _rasterise_boundary(shifted_img,
                                          imagery_patch.transform, (H, W))
        denom = poly_mask.sum()
        if denom > 0:
            edge_score = float((edge_map * poly_mask).sum() / denom)
    except Exception:
        pass

    # ── Signal 2: Boundary hint ───────────────────────────────────────────
    boundary_score = 0.0
    try:
        shifted_img = geom_to_imagery_crs(img_src, shifted)
        bdata, btransform = boundary_values  # now a tuple (data, transform)
        if btransform is not None and bdata.size > 0:
            bH, bW = bdata.shape
            poly_mask_b = _rasterise_boundary(shifted_img, btransform, (bH, bW))
            denom = poly_mask_b.sum()
            if denom > 0:
                boundary_score = float((bdata * poly_mask_b).sum() / denom)
    except Exception:
        pass

    # ── Signal 3: Area ratio prior ────────────────────────────────────────
    deviation  = abs(area_ratio - 1.0)
    area_score = max(0.0, 1.0 - deviation * 3.0)

    # ── Signal 4: Proximity to global shift centre ────────────────────────
    dev_from_global = math.sqrt((dx_m - gdx) ** 2 + (dy_m - gdy) ** 2)
    shift_score     = max(0.0, 1.0 - dev_from_global / GRID_RANGE_M)

    shift_m = math.sqrt(dx_m ** 2 + dy_m ** 2)

    total = (W_EDGE     * edge_score +
             W_BOUNDARY * boundary_score +
             W_AREA     * area_score +
             W_SHIFT    * shift_score)

    return {
        "dx_m":              dx_m,
        "dy_m":              dy_m,
        "total_score":       round(total, 4),
        "edge_score":        round(edge_score, 4),
        "boundary_score":    round(boundary_score, 4),
        "area_score":        round(area_score, 4),
        "shift_score":       round(shift_score, 4),
        "shift_magnitude_m": round(shift_m, 2),
    }


def run_grid_search(
    polygon: BaseGeometry,
    gdx: float,
    gdy: float,
    imagery_patch,
    boundary_values: np.ndarray,
    area_ratio: float,
    img_src,
) -> dict:

    gray     = _to_grayscale(imagery_patch.image.astype(np.float64))
    edge_map = _detect_edges(gray)

    candidates = generate_candidates(gdx, gdy)
    scored = [
        score_candidate(
            polygon, dx_m, dy_m,
            imagery_patch, boundary_values,
            area_ratio, edge_map, img_src,
            gdx=gdx, gdy=gdy
        )
        for dx_m, dy_m in candidates
    ]

    # Score at true origin to compute margin
    origin = score_candidate(
        polygon, 0.0, 0.0,
        imagery_patch, boundary_values,
        area_ratio, edge_map, img_src,
        gdx=gdx, gdy=gdy
    )

    best = max(scored, key=lambda x: x["total_score"])

    # Sanity cap
    if best["shift_magnitude_m"] > MAX_SHIFT_M:
        centre = min(scored,
                     key=lambda x: math.sqrt((x["dx_m"]-gdx)**2 +
                                             (x["dy_m"]-gdy)**2))
        best = centre

    best["score_vs_origin_margin"] = round(
        best["total_score"] - origin["total_score"], 4
    )
    best["origin_score"] = round(origin["total_score"], 4)

    return best