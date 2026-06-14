"""
Phase 2 — Global Median Shift
Extract the village-wide drift vector from example_truths.
This becomes the CENTRE of the grid search in Phase 3.
We do NOT apply it directly as the final answer.
"""

import statistics
import geopandas as gpd


def compute_global_shift(village) -> tuple[float, float]:
    """
    Compute median dx, dy shift in metres using example_truths.

    Uses the same UTM projection trick as the starter kit baseline
    to get true metric distances.

    Returns:
        (global_dx_m, global_dy_m) — the village-wide drift vector in metres
    """
    if village.example_truths is None:
        raise ValueError(f"{village.slug}: no example_truths.geojson found")

    # Project to UTM for true metre distances
    lon = village.example_truths.geometry.iloc[0].centroid.x
    utm = f"EPSG:{32600 + int((lon + 180) // 6) + 1}"

    official_utm = village.plots.to_crs(utm)
    truths_utm   = village.example_truths.to_crs(utm)

    dxs, dys = [], []
    for pn in village.example_truths.index:
        if pn in official_utm.index:
            o = official_utm.loc[pn, "geometry"].centroid
            t = truths_utm.loc[pn, "geometry"].centroid
            dxs.append(t.x - o.x)
            dys.append(t.y - o.y)

    if not dxs:
        raise ValueError("No overlapping plots between example_truths and cadastre")

    gdx = statistics.median(dxs)
    gdy = statistics.median(dys)

    print(f"\n── Phase 2 Global Shift ────────────────────────")
    print(f"  Computed from    : {len(dxs)} example truth plots")
    print(f"  Global dx        : {gdx:+.2f} m  (east/west)")
    print(f"  Global dy        : {gdy:+.2f} m  (north/south)")
    print(f"────────────────────────────────────────────────\n")

    return gdx, gdy