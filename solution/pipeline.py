"""
BhuMe Pipeline — Main Entry Point
Runs all 5 phases end-to-end for a village bundle.

Usage:
    uv run solution/pipeline.py data/34855_vadnerbhairav_chandavad_nashik
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import rasterio

from bhume import load, write_predictions, score
from bhume.geo import open_imagery, patch_for_plot, geom_to_imagery_crs
import numpy as np

from solution.phase1_classify import classify_all, summarise
from solution.phase2_baseline import compute_global_shift
from solution.phase3_grid_search import run_grid_search, _boundary_patch_under_polygon
from solution.phase4_confidence import make_decision


# ── Constants ─────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")


def run_pipeline(village_dir: str) -> None:
    start_time = time.time()
    village_dir = Path(village_dir)

    print("\n" + "=" * 55)
    print("  BhuMe Boundary Correction Pipeline")
    print("=" * 55)

    # ── Phase 0 — Load ────────────────────────────────────────────────────
    print("\n[Phase 0] Loading village bundle...")
    village = load(village_dir)
    n_truth = 0 if village.example_truths is None else len(village.example_truths)
    print(f"  Loaded  : {village.slug}")
    print(f"  Plots   : {len(village.plots)}")
    print(f"  Truths  : {n_truth}")
    print(f"  Bounds  : {'yes' if village.boundaries_path else 'no'}")

    # ── Phase 1 — Classify ────────────────────────────────────────────────
    print("\n[Phase 1] Pre-classifying plots by area ratio...")
    plots = classify_all(village.plots)
    summarise(plots)

    candidates = plots[plots["phase1_status"] == "candidate"]
    flagged_p1 = plots[plots["phase1_status"] == "flagged"]

    # ── Phase 2 — Global shift ────────────────────────────────────────────
    print("\n[Phase 2] Computing global median shift...")
    gdx, gdy = compute_global_shift(village)

    # ── Phase 3 + 4 — Grid search + confidence ────────────────────────────
    print(f"\n[Phase 3+4] Grid search + confidence scoring...")
    print(f"  Processing {len(candidates)} candidate plots...")
    print(f"  (flagging {len(flagged_p1)} plots from Phase 1 directly)\n")

    results = []
    n_corrected = 0
    n_flagged_p4 = 0
    n_already_correct = 0

    with open_imagery(village.imagery_path) as img_src:
        # Open boundaries raster if available
        bounds_src = None
        if village.boundaries_path:
            bounds_src = rasterio.open(village.boundaries_path)

        total = len(candidates)
        for i, (plot_number, row) in enumerate(candidates.iterrows()):

            # Progress indicator every 100 plots
            if i % 100 == 0:
                elapsed = time.time() - start_time
                pct = 100 * i / total
                print(f"  [{i:4d}/{total}] {pct:.0f}% done — {elapsed:.0f}s elapsed")

            polygon     = row["geometry"]
            area_ratio  = row["area_ratio"] if row["area_ratio"] else 1.0

            # ── Get imagery patch ─────────────────────────────────────────
            try:
                imagery_patch = patch_for_plot(img_src, polygon, pad_m=30)
            except Exception as e:
                results.append({
                    "plot_number": plot_number,
                    "status":      "flagged",
                    "geometry":    polygon,
                    "method_note": f"no_imagery: {e}"
                })
                n_flagged_p4 += 1
                continue

            # ── Get boundary hint patch (in imagery CRS) ───────────────────
            if bounds_src:
                polygon_img_crs = geom_to_imagery_crs(img_src, polygon)
                boundary_values = _boundary_patch_under_polygon(
                    bounds_src, polygon_img_crs, pad_m=30
                )
            else:
                boundary_values = (np.zeros((10, 10), dtype=np.float32), None)

            # ── Phase 3: Grid search ──────────────────────────────────────
            try:
                best = run_grid_search(
                    polygon, gdx, gdy,
                    imagery_patch, boundary_values,
                    area_ratio, img_src
                )
            except Exception as e:
                results.append({
                    "plot_number": plot_number,
                    "status":      "flagged",
                    "geometry":    polygon,
                    "method_note": f"grid_search_error: {e}"
                })
                n_flagged_p4 += 1
                continue

            # ── Phase 4: Confidence + decision ───────────────────────────
            decision = make_decision(best, area_ratio, polygon, gdx, gdy)

            results.append({
                "plot_number": plot_number,
                "status":      decision["status"],
                "confidence":  decision.get("confidence"),
                "geometry":    decision["geometry"],
                "method_note": decision["method_note"],
            })

            if decision["status"] == "corrected":
                if "already_correct" in decision["method_note"]:
                    n_already_correct += 1
                else:
                    n_corrected += 1
            else:
                n_flagged_p4 += 1

        if bounds_src:
            bounds_src.close()

    # ── Phase 1 flagged plots — add to results ────────────────────────────
    for plot_number, row in flagged_p1.iterrows():
        results.append({
            "plot_number": plot_number,
            "status":      "flagged",
            "geometry":    row["geometry"],
            "method_note": row["flag_reason"],
        })

    # ── Phase 5 — Build GeoDataFrame and write ────────────────────────────
    print("\n[Phase 5] Writing predictions.geojson...")

    rows_clean = []
    for r in results:
        row = {
            "plot_number": r["plot_number"],
            "status":      r["status"],
            "geometry":    r["geometry"],
            "method_note": r.get("method_note", ""),
        }
        if r["status"] == "corrected" and r.get("confidence") is not None:
            row["confidence"] = r["confidence"]
        rows_clean.append(row)

    preds_gdf = gpd.GeoDataFrame(rows_clean, crs="EPSG:4326")

    out_path = village_dir / "predictions.geojson"
    write_predictions(out_path, preds_gdf)

    # ── Self-score ────────────────────────────────────────────────────────
    print("\n[Score] Self-scoring against example truths...")
    print()
    scorecard = score(preds_gdf, village)
    print(scorecard)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    total_corrected = n_corrected + n_already_correct
    total_flagged   = len(flagged_p1) + n_flagged_p4

    print(f"\n── Final Summary ───────────────────────────────")
    print(f"  Total plots      : {len(plots)}")
    print(f"  Corrected        : {total_corrected}")
    print(f"    └ shifted      : {n_corrected}")
    print(f"    └ already ok   : {n_already_correct}")
    print(f"  Flagged          : {total_flagged}")
    print(f"    └ phase1 ratio : {len(flagged_p1)}")
    print(f"    └ phase4 weak  : {n_flagged_p4}")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print(f"  Output           : {out_path}")
    print(f"────────────────────────────────────────────────\n")

    # ── Save run log ──────────────────────────────────────────────────────
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {
        "village":       village.slug,
        "timestamp":     ts,
        "global_shift":  {"dx_m": round(gdx, 2), "dy_m": round(gdy, 2)},
        "classification": {
            "total":      len(plots),
            "candidates": len(candidates),
            "flagged_p1": len(flagged_p1),
        },
        "results": {
            "corrected_shifted":   n_corrected,
            "already_correct":     n_already_correct,
            "flagged_weak":        n_flagged_p4,
        },
        "elapsed_s": round(elapsed, 1),
        "scorecard": str(scorecard),
    }
    log_path = LOG_DIR / f"run_{ts}.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  Run log saved → {log_path}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run solution/pipeline.py data/<village_folder>")
        sys.exit(1)
    run_pipeline(sys.argv[1])