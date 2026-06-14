# AI Transcripts

This folder contains/links to the AI conversations used while building this
submission.

## Claude.ai (Web) — Primary Development Session

Used for: understanding the assignment, designing the 5-phase architecture,
writing all pipeline code, debugging the MultiPolygon rasterisation issue and
the boundaries.tif single-band read issue, and iterating on confidence
calibration.

Share link: **https://claude.ai/share/d384959d-e3ae-430b-88fe-e0b05f828950**



## Notes

- All code in `solution/` was generated through this conversation and run
  locally to validate against `example_truths.geojson` for both villages.
- Key debugging moments documented in the transcript:
  - Initial `ModuleNotFoundError` for the `bhume` package — fixed via
    `PYTHONPATH`.
  - All plots scoring "already correct" (shift < 2.5m) — traced to a
    shift-score weighting issue, fixed by centring the grid search on the
    global shift vector.
  - `edge_score=0.00` and `boundary_score=0.00` for every plot — traced to
    polygon rasterisation returning empty masks for `MultiPolygon`
    geometries, and `boundaries.tif` being a single-band raster incompatible
    with the RGB patch helper. Both fixed with targeted debugging scripts
    before re-running the full pipeline.