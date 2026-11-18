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
**Primary selection metric: CV observation-level Top-1** - it is the quantity the route stage consumes (best candidate per fix), unlike candidate-level accuracy/F1 which is dominated by the 80% negative class.

## Results (all from actual runs)
| Model | Family | CV macro-F1 | CV Top-1 (mean±std) | CV continuity | Test acc | Test prec (macro) | Test recall (macro) | Test macro-F1 | Test log loss | **Test Top-1** | Test continuity | Train Top-1 | Train s | Infer µs/cand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear / probabilistic | 0.7859 | 0.7457±0.0344 | 0.7346 | 0.8667 | 0.7862 | 0.8100 | 0.7970 | 0.2900 | **0.7774** | 0.8717 | 0.7494 | 0.2 | 0.08 |
| Gaussian Naive Bayes | Probabilistic generative | 0.8010 | 0.7414±0.0382 | 0.7345 | 0.8809 | 0.8099 | 0.8154 | 0.8126 | 0.5062 | **0.7804** | 0.8779 | 0.7439 | 0.2 | 0.17 |
| Random Forest | Bagging ensemble (trees) | 0.8503 | 0.7694±0.0559 | 0.7468 | 0.9165 | 0.8612 | 0.8828 | 0.8714 | 0.1952 | **0.8180** | 0.8964 | 0.8356 | 13.6 | 4.18 |
| HistGradientBoosting | Boosting (sklearn) | 0.8476 | 0.7689±0.0548 | 0.7459 | 0.9160 | 0.8616 | 0.8788 | 0.8698 | 0.1943 | **0.8121** | 0.8942 | 0.8403 | 2.4 | 3.72 |
| LightGBM (untuned) | Boosting (leaf-wise GBDT) | 0.8495 | 0.7710±0.0548 | 0.7486 | 0.9155 | 0.8598 | 0.8813 | 0.8699 | 0.1914 | **0.8148** | 0.8970 | 0.8495 | 3.9 | 12.56 |
| MLP (64,32) | Neural network | 0.8446 | 0.7640±0.0555 | 0.7447 | 0.9128 | 0.8564 | 0.8748 | 0.8651 | 0.1973 | **0.8085** | 0.8967 | 0.8040 | 32.5 | 1.65 |

**Selected Phase-1 baseline: LightGBM (untuned)** (highest CV Top-1 0.771). Honest caveat: the top four models are within one CV std (~0.055 across trip folds) of each other - the ranking among tree/boosting/MLP models is **not statistically decisive**; LightGBM is also the slowest tree model at inference. RF has the best student-test Top-1 (0.8180). Linear/NB models are clearly worse.
Overfitting: LightGBM train Top-1 0.850 vs CV 0.771 vs student-test 0.815 (moderate generalisation gap; student-test trips are easier - 0.1% unreliable refs vs 10.3% in train).
Why lower than the legacy 0.88: only 7 per-row features (no neighbouring-fix context, no topology), top-5 candidates, genuine 1.9-2.8% miss rate. The legacy numbers are not comparable.

## Reproduce (Windows PowerShell)

Activate the virtual environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt

## Integration contract (for Divyanshu)
`pathsnap_ml.predict.predict_trip(trip_dir)` reads only `.track/.nodes/.arcs`. Per candidate: `trip_id, observation_id, timestamp, arc_id, from_node_id, to_node_id, score, prob_obs (sums to 1 per obs - HMM emission), rank, is_top1`. 5 rows per observation, none missing.
Artifacts: `artifacts/final_model.joblib` (sklearn Pipeline incl. imputer), `final_model_config.json`, plus `model_<key>.joblib` for all six. OOF/test scores in `results/*.parquet`.
