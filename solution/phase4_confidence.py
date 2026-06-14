"""
Phase 4 — Confidence Scoring
Confidence must vary meaningfully across plots for AUC to be computed.
Strategy:
  - Area ratio is our strongest prior signal
  - Edge/boundary signals create genuine variation in confidence
  - Plots with very weak imagery signal get flagged
  - Restraint: if best candidate is at global centre with no image boost → moderate confidence
"""

CONFIDENCE_THRESHOLD  = 0.38
RESTRAINT_THRESHOLD_M = 1.5


def compute_confidence(best: dict, area_ratio: float) -> float:
    """
    Confidence built from multiple independent signals.
    Each signal contributes independently so values genuinely spread 0.3 - 0.9.
    """
    # ── Component 1: Area ratio (0.0 - 0.50) ──────────────────────────────
    # How close is drawn area to recorded area? Near 1.0 = strong prior
    deviation = abs(area_ratio - 1.0)
    if deviation < 0.05:
        area_component = 0.50      # very close → strong confidence
    elif deviation < 0.10:
        area_component = 0.42
    elif deviation < 0.20:
        area_component = 0.32
    else:
        area_component = 0.20      # far off → weak prior

    # ── Component 2: Edge signal (0.0 - 0.25) ─────────────────────────────
    edge_component = best["edge_score"] * 0.25

    # ── Component 3: Boundary hint (0.0 - 0.15) ───────────────────────────
    boundary_component = best["boundary_score"] * 0.15

    # ── Component 4: Margin over origin (0.0 - 0.10) ─────────────────────
    margin = best.get("score_vs_origin_margin", 0.0)
    if margin > 0.10:
        margin_component = 0.10
    elif margin > 0.05:
        margin_component = 0.06
    elif margin > 0.01:
        margin_component = 0.03
    else:
        margin_component = 0.0

    # ── Penalty: signals disagree strongly ────────────────────────────────
    disagreement = abs(best["edge_score"] - best["boundary_score"])
    penalty = disagreement * 0.08

    confidence = (area_component + edge_component +
                  boundary_component + margin_component - penalty)

    return round(max(0.0, min(1.0, confidence)), 4)


def make_decision(best: dict, area_ratio: float,
                  polygon, gdx: float, gdy: float):
    """
    Final decision for each plot.
    """
    from solution.phase3_grid_search import translate_polygon_metres

    shift_m    = best["shift_magnitude_m"]
    confidence = compute_confidence(best, area_ratio)

    # ── Rule 1: Restraint — already in correct position ───────────────────
    if shift_m < RESTRAINT_THRESHOLD_M:
        return {
            "status":      "corrected",
            "confidence":  min(0.92, confidence + 0.08),
            "geometry":    polygon,
            "method_note": (
                f"already_correct: shift={shift_m:.1f}m conf={confidence:.3f}"
            )
        }

    # ── Rule 2: Flag weak evidence ────────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "status":      "flagged",
            "confidence":  None,
            "geometry":    polygon,
            "method_note": (
                f"weak_evidence: conf={confidence:.3f} "
                f"edge={best['edge_score']:.2f} "
                f"boundary={best['boundary_score']:.2f} "
                f"area_ratio={area_ratio:.2f}"
            )
        }

    # ── Rule 3: Apply correction ──────────────────────────────────────────
    shifted_geom = translate_polygon_metres(polygon, best["dx_m"], best["dy_m"])
    return {
        "status":      "corrected",
        "confidence":  confidence,
        "geometry":    shifted_geom,
        "method_note": (
            f"grid_search: dx={best['dx_m']:+.1f}m dy={best['dy_m']:+.1f}m "
            f"conf={confidence:.3f} edge={best['edge_score']:.2f} "
            f"boundary={best['boundary_score']:.2f} "
            f"area_ratio={area_ratio:.2f} shift={shift_m:.1f}m"
        )
    }