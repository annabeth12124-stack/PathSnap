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

* `GET /health` - service health.
* `GET /trips?split=student_test` - available trip IDs (`train` is also supported).
* `POST /predict/{trip_id}?split=student_test` - classifier-only top candidate and HMM route, GPS fixes, arc geometry, and transition counts.

The route viewer defaults to `student_test`; it does not load evaluator data. The HMM uses classifier `prob_obs` as its emission likelihood and a Laplace likelihood of the difference between GPS displacement and shortest directed network distance as its transition likelihood. An impossible transition has probability zero. If all transitions into a fix are impossible, Viterbi starts a new segment and marks that break in `predicted_route`.

## Current scope and remaining evidence

The service and UI expose a classifier-only vs classifier-plus-HMM comparison and report transition continuity. `README_TUNING.md` documents the model interface. Before presenting this as deployment-ready, run the service on the actual server laptop, open it from the second laptop, and save the requested connection log and screenshots under `results/integration/`. Confirm the actual LAN IP and firewall policy on the deployment machines; they are machine-specific.
