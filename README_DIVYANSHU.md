# Divyanshu: HMM, route reconstruction, and API

## Run locally

From the PathSnap repository root in PowerShell, with the project virtual environment active:

```powershell
pip install -r requirements-ml.txt
pip install -r requirements-api.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` for the route viewer. On the other laptop, open `http://<server-laptop-LAN-IP>:8000/`. Both laptops must be on a network that permits inbound TCP port 8000; Windows Firewall may need an inbound rule. `/health` is the connection check. The service logs trip ID, dataset split, observation count, valid transitions, and segment breaks to the server terminal.

## API

* `GET /health` - service status and selected model name.
* `GET /trips?split=student_test` - available trip IDs. `train` and `evaluator_test` are also supported when their data folders are present.
* `POST /predict/{trip_id}?split=student_test` - classifier-only top candidate, HMM route, GPS fixes, arc geometry, transition counts, processing time, and aggregate evaluation metrics when a `.route` reference file is available.

The route viewer compares classifier-only output with the HMM and shows per-trip metrics, a trip summary, all-trip results, and a request log. It defaults to `student_test` and can also use `evaluator_test` when that split is extracted under `data/evaluator_test/<trip_id>/`. Its optional street basemap uses online OpenStreetMap tiles and shows the required attribution; without internet, the route overlay still works on the plain map canvas. Keep the `.route` files on the API server: the endpoint aligns reference labels and returns only aggregate scores, never the reference route or per-fix labels.

Top-1 accuracy is the share of reference-reliable GPS fixes whose selected arc matches the aligned reference arc. Reliability uses the existing 75 m threshold. Route F1 is the harmonic mean of precision and recall over the unique predicted and reference arc sets at those reliable fixes. Route continuity is the share of consecutive HMM transitions marked connected. These metrics are descriptive evaluation results; the evaluator split is not used to train or tune the model.

The HMM uses classifier `prob_obs` as its emission likelihood and a Laplace compatibility potential based on the difference between GPS displacement and shortest directed network distance. An impossible transition has probability zero. If all transitions into a fix are impossible, Viterbi starts a new segment and decodes that segment independently.

## Completed two-laptop demo and evidence

The service and route viewer were demonstrated across two laptops on the local network. The saved evidence is in `results/integration/`:

- `api-laptop-log.png` shows successful HTTP 200 requests and the prediction log for evaluator-test trip `00000059`.
- `laptop2-ui.png` shows the route map and per-trip metrics for the same trip.

The API accepts `train`, `student_test`, and `evaluator_test` splits when their folders are present. It reads reference routes on the server and sends aggregate per-trip scores to the browser; raw reference routes and aligned per-fix labels are not returned. The UI also provides an all-trips panel, but the saved screenshot is a single-trip result and must not be presented as a ten-trip average.

The recorded trip has 1,031 reference-reliable fixes, HMM Top-1 accuracy 90.69%, classifier-only Top-1 87.97%, route continuity 100%, route arc-set F1 88.95%, and API processing time 8.36 seconds. These figures describe trip `00000059` only. The model and decoder must not be tuned using evaluator-test results.
