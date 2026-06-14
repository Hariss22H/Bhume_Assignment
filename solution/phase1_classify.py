"""
Phase 1 — Pre-classification
Decide before touching any imagery: is this plot fixable (placement error)
or unfixable (area error / null record)?
Decision is based purely on area_ratio = map_area_sqm / recorded_area_sqm.
"""

import geopandas as gpd

# ── Constants ────────────────────────────────────────────────────────────────
AREA_RATIO_MIN = 0.75   # below → map is too small vs record → flag
AREA_RATIO_MAX = 1.35   # above → map is too large vs record → flag


def classify_plot(row) -> dict:
    """
    Classify a single plot row as 'candidate' or 'flagged'.

    Returns a dict with:
        status      : 'candidate' | 'flagged'
        area_ratio  : float | None
        flag_reason : str | None
    """
    recorded = row.get("recorded_area_sqm")

    # Rule 1 — null or zero recorded area
    if recorded is None or recorded != recorded or recorded == 0:
        return {
            "status": "flagged",
            "area_ratio": None,
            "flag_reason": "null_recorded_area: no 7/12 reference available"
        }

    ratio = row["map_area_sqm"] / recorded

    # Rule 2 — map is too small vs record
    if ratio < AREA_RATIO_MIN:
        return {
            "status": "flagged",
            "area_ratio": round(ratio, 3),
            "flag_reason": f"area_ratio={ratio:.2f}: map too small vs record (area error)"
        }

    # Rule 3 — map is too large vs record
    if ratio > AREA_RATIO_MAX:
        return {
            "status": "flagged",
            "area_ratio": round(ratio, 3),
            "flag_reason": f"area_ratio={ratio:.2f}: map too large vs record (area error)"
        }

    # Passes all checks — send to grid search
    return {
        "status": "candidate",
        "area_ratio": round(ratio, 3),
        "flag_reason": None
    }


def classify_all(plots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Apply classify_plot to every row in the plots GeoDataFrame.
    Adds columns: phase1_status, area_ratio, flag_reason.
    Returns the modified GeoDataFrame.
    """
    results = plots.apply(classify_plot, axis=1, result_type="expand")
    plots = plots.copy()
    plots["phase1_status"] = results["status"]
    plots["area_ratio"]    = results["area_ratio"]
    plots["flag_reason"]   = results["flag_reason"]
    return plots


def summarise(plots: gpd.GeoDataFrame) -> None:
    """Print a quick summary of classification results."""
    total      = len(plots)
    candidates = (plots["phase1_status"] == "candidate").sum()
    flagged    = (plots["phase1_status"] == "flagged").sum()
    null_area  = plots["flag_reason"].str.startswith("null_recorded", na=False).sum()
    ratio_low  = plots["flag_reason"].str.startswith("area_ratio", na=False) & \
                 plots["flag_reason"].str.contains("too small", na=False)
    ratio_high = plots["flag_reason"].str.startswith("area_ratio", na=False) & \
                 plots["flag_reason"].str.contains("too large", na=False)

    print(f"\n── Phase 1 Classification ──────────────────────")
    print(f"  Total plots      : {total}")
    print(f"  Candidates       : {candidates}  ({100*candidates/total:.1f}%)")
    print(f"  Flagged total    : {flagged}  ({100*flagged/total:.1f}%)")
    print(f"    └ null area    : {null_area}")
    print(f"    └ ratio low    : {ratio_low.sum()}")
    print(f"    └ ratio high   : {ratio_high.sum()}")
    print(f"────────────────────────────────────────────────\n")