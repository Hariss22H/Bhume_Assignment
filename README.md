# Bhume_Assignment
BhuMe first round assignment
# BhuMe Take-Home — Land Plot Boundary Correction

## Approach

A 5-phase pipeline that corrects cadastral plot boundaries to align with
satellite imagery, using grid search + multi-signal scoring (no ML training
required).

```
Phase 1 — Pre-classify by area_ratio (map_area / recorded_area)
           → flags plots with area errors (ratio outside 0.75–1.35)

Phase 2 — Compute village-wide global median shift from example_truths

Phase 3 — Grid search ±15m around the global shift centre (2.5m steps)
           → score each candidate on: edge alignment, boundary.tif hint,
             area ratio, proximity to global shift

Phase 4 — Confidence scoring from weighted signal components
           → flag plots below confidence threshold (0.38)
           → restraint check for plots needing <1.5m correction

Phase 5 — Write predictions.geojson, self-score against example truths
```

## Results (self-scored on example truths)

| Village | Official IoU | Our IoU | Improvement | Plots Improved |
|---|---|---|---|---|
| Vadnerbhairav | 0.612 | 0.714 | +0.111 | 100% (6/6) |
| Malatavadi | 0.510 | 0.588 | +0.089 | 67% (2/3) |

Same code, same parameters, **no per-village tuning**.

## How to Run

```bash
uv sync
PYTHONPATH=. uv run solution/pipeline.py data/<village_folder>
```

Outputs `predictions.geojson` in the village folder and a run log in `logs/`.

## Project Structure

```
solution/
├── phase1_classify.py    — area ratio classification
├── phase2_baseline.py     — global median shift computation
├── phase3_grid_search.py  — candidate generation + multi-signal scoring
├── phase4_confidence.py   — confidence calculation + decision rules
└── pipeline.py             — orchestrates all phases end-to-end
```

## Key Decisions

- **Translation only** — no rotation/reshape, since BhuMe's drift is primarily
  a georeferencing offset.
- **Area ratio as a hard pre-filter** — plots with map_area/recorded_area
  outside [0.75, 1.35] are area-mismatch problems, not placement problems,
  so we flag them immediately without spending compute on image analysis.
- **Grid search centred on global shift** — rather than searching from (0,0),
  we centre the search on the village-wide drift vector, since most plots
  share a common drift component.
- **Confidence built from 4 independent signals** (area ratio prior, edge
  alignment, boundary.tif agreement, margin over no-correction) so confidence
  genuinely varies across plots rather than being flat.
- **boundaries.tif handled with its own raster transform** — it's a
  single-band raster at a different resolution than imagery.tif, so it
  needs its own windowed read rather than reusing the RGB patch helper.

## Honest Limitations

- Calibration metrics (AUC/Spearman) are not meaningful on 3–6 example
  truths — both villages either have no "miss" case (Vadnerbhairav, AUC
  undefined) or only one (Malatavadi, AUC=0.5). The confidence logic is
  designed to generalise to the larger hidden test set.
- boundaries.tif contributes a real but modest signal (~0.20 boundary_score
  on a tested plot) — it helped marginally but was not the dominant signal;
  imagery edge detection (Sobel) carried more weight.
- No rotation/local-stretch correction — only translation. Plots that
  drifted with rotation are likely under-corrected.

## AI Usage

Built iteratively with Claude (Anthropic) — see `/transcripts` for full
chat logs covering problem understanding, architecture design, debugging
(MultiPolygon rasterisation bug, boundaries.tif single-band read bug), and
scoring/confidence calibration iterations.