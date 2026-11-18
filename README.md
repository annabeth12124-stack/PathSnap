# PathSnap

PathSnap maps noisy GPS observations to candidate road arcs, scores candidates with the selected classifier, and reconstructs a connected route with a graph-based Viterbi decoder.

## Run the two-laptop demo

On the server laptop, from the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt
pip install -r requirements-api.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/` on the server laptop. On the second laptop, open `http://<server-laptop-LAN-IP>:8000/`. Both devices must be on a network that permits connections to port 8000. `/health` checks the API connection.

Extract local datasets into `data/student_test/<trip_id>/` and, for the final evaluation, `data/evaluator_test/<trip_id>/`. Dataset files are intentionally excluded from Git. Keep evaluator reference routes on the API server; the browser receives predictions and aggregate scores, not raw labels.

## Project layout

| Path | Purpose |
|---|---|
| `preprocessing/` | Dataset exploration and candidate-generation notebooks |
| `pathsnap_ml/` | Candidate features, model training, prediction, and evaluation |
| `optimisation/` | Bayesian tuning code, study records, and comparisons |
| `route_reconstruction/` | Road graph, transitions, and segmented Viterbi decoding |
| `backend/main.py` | FastAPI service and server-side scoring |
| `frontend/index.html` | Evaluator route viewer and per-trip metrics |
| `artifacts/` | Selected Phase 1 model and configuration |
| `results/` | Reproducible model results and saved demo evidence |
| `visualisations/` | Dataset and route maps |

## Technical notes

- The retained Phase 1 deployment baseline is untuned LightGBM. The model comparison and rationale are in [`README_PHASE1.md`](README_PHASE1.md).
- The completed tuning run and tuned-versus-untuned comparison are in [`README_TUNING.md`](README_TUNING.md).
- The HMM/API walkthrough and two-laptop evidence are in [`README_DIVYANSHU.md`](README_DIVYANSHU.md).
- The saved evaluator screenshot shows one trip, `00000059`; it is a trip-level result, not an average over all ten evaluator trips.
- The legacy stand-in candidate pipeline is not part of the final implementation. Its results must not be used as final results.
