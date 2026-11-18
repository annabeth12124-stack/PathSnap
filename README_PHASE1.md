# PathSnap - Phase 1 ML component (Yeshaswinie)

Candidate classification for map matching, built on **Sakshi's actual candidate-generation pipeline** (commit `441f431`). Phase 1 = baseline only:
feature selection, six-model comparison, student-test evaluation, final-model export. **No Bayesian/Optuna optimisation here (Phase 2).**
The earlier stand-in implementation is archived in `legacy_standin_pipeline/` and its numbers are **preliminary / not final**.

## Pipeline (preserved exactly)
Local equirectangular projection centred on the mean track lat/lon (R=6371000) -> cKDTree over arc **midpoints** -> K=100 nearest midpoints per GPS fix ->
exact Euclidean point-to-segment distance -> keep **5** closest, **no distance cutoff**. 13 columns: 6 identifiers (`trip_id, observation_id, arc_id, from_node_id, to_node_id, timestamp`, never features)
+ 7 numeric features: `distance_m, projection_fraction, segment_length_m, gps_speed_mps, gps_heading, arc_heading, heading_difference`.
`pathsnap_ml/candidates.py` is a vectorised port; `tests/test_candidates_match_notebook.py` executes Sakshi's own `process_trip` and checks equality (distances identical to ~1e-13; arc IDs differ only on exact rank-5 distance ties, where the notebook's unstable pandas sort picks arbitrarily).
`trip_id` is always a zero-padded string.

## Labels (train/dev only, never features)
`.route` is a per-trip arc sequence, so each GPS fix gets a reference arc by monotone DP alignment (as in the legacy package). `label = arc_id == aligned reference arc`.
Fixes whose aligned route arc is >75 m away are `ref_ok=0` (route does not cover that part of the track): excluded from training/CV and from Top-1 denominators.

## Candidate audit (reproduced from the tables; `results/candidate_audit.json`)
| | train (80 trips) | student-test (10) |
|---|---|---|
| candidate rows / observations | 1,042,085 / 208,417 | 95,800 / 19,160 |
| rows with NaN features (first fix of each trip) | 400 | 50 |
| nearest candidate >50 m / >100 m | 8.80% / 0.68% | 0.00% / 0.00% |
| max nearest-candidate distance | 256.9 m | 47.3 m |
| ref-unreliable observations | 10.26% | 0.11% |
| **observation-level candidate miss rate** (ref-ok obs) | **2.83%** | **1.93%** |

* Correction to earlier notes: **1,448.8 m is the maximum over all retained candidate rows (the farthest of the 5), not the nearest-candidate maximum (256.9 m, trip 00000096).** Sakshi's saved notebook output shows both.
* Sakshi's ~98.23% is a *trip-level route arc-set coverage*; it is NOT the observation-level miss rate above. Top-1 can never exceed 1 - miss rate (student-test ceiling ~98.1%).
* Nearest-distance heuristic (no learning): Top-1 0.5470, continuity 0.8277 on student-test.

## Experimental protocol
Seed 42. 5-fold `GroupKFold` **by trip** on the 80 training trips; training rows = ref-ok, every 4th observation (233,935 rows). Median imputation (fit per fold) for all models, StandardScaler for scale-sensitive ones.
Decision threshold chosen on out-of-fold training predictions only. Student-test (10 trips) = development comparison, all observations scored. Evaluator data never read. Top-1 counts candidate misses as errors.
**Primary map-matching metric: observation-level Top-1** because the route stage consumes the best candidate per fix. Model selection also considered probability quality for HMM emissions; LightGBM was retained for competitive Top-1 and the best student-test log loss, not because it led CV Top-1.

## Results (source: `results/model_comparison.csv`)

The comparison uses five-fold `GroupKFold` by trip for cross-validation and the 10-trip student-test set for development-time comparison. The evaluator set is not used to train, tune, or select the model.

| Model | CV Macro F1 | CV Top-1 mean ± SD | Student Macro F1 | Student Top-1 | Student Log Loss | Student continuity |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7860 | 0.7457 ± 0.0344 | 0.7970 | 0.7774 | 0.2901 | 0.8717 |
| Gaussian Naive Bayes | 0.8010 | 0.7413 ± 0.0383 | 0.8126 | 0.7805 | 0.5063 | 0.8782 |
| Random Forest | 0.8503 | 0.7704 ± 0.0547 | 0.8712 | 0.8195 | 0.1961 | 0.8972 |
| HistGradientBoosting | 0.8487 | **0.7723 ± 0.0532** | 0.8686 | 0.8114 | 0.1949 | 0.8932 |
| LightGBM (untuned) | 0.8495 | 0.7709 ± 0.0545 | 0.8704 | 0.8151 | **0.1917** | 0.8974 |
| MLP (64,32) | 0.8434 | 0.7640 ± 0.0560 | 0.8667 | 0.8091 | 0.1974 | 0.8982 |

**Selected Phase 1 baseline: untuned LightGBM.** It remained competitive on Top-1 and had the lowest student-test log loss, which supports its use as a probability-producing input to the HMM. It did not lead every metric: HistGradientBoosting had the highest mean CV Top-1, while Random Forest had the highest student-test Top-1 and Macro F1. The CV differences among the leading nonlinear models are small relative to fold-to-fold variation, so this is a practical selection rather than a statistically decisive win on every measure.

## Reproduce (Windows PowerShell)

Activate the virtual environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt

## Integration contract (for Divyanshu)
`pathsnap_ml.predict.predict_trip(trip_dir)` reads only `.track/.nodes/.arcs`. Per candidate: `trip_id, observation_id, timestamp, arc_id, from_node_id, to_node_id, score, prob_obs (sums to 1 per obs - HMM emission), rank, is_top1`. 5 rows per observation, none missing.
Artifacts: `artifacts/final_model.joblib` (sklearn Pipeline incl. imputer), `final_model_config.json`, plus `model_<key>.joblib` for all six. OOF/test scores in `results/*.parquet`.
