"""Phase-1 six-model comparison on Sakshi's real candidate tables. 5-fold GroupKFold BY TRIP on the 80 training trips;
student_test (10 trips) for development comparison. NO Bayesian/Optuna here (Phase 2). Evaluator data never touched.
Usage: python -m pathsnap_ml.run_experiments [--stride 4] [--models lr,gnb,rf,hgb,lgbm,mlp]"""
import argparse, json, time, os, warnings, logging
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from .config import *
from . import evaluation as ev
warnings.filterwarnings("ignore")
OUT = "results"; ART = "artifacts"
imp = lambda: SimpleImputer(strategy="median")          # fitted inside each training fold; only gps_speed/heading/heading_diff of obs 0 are NaN

def make_models():
    return {
     "lr":  dict(name="Logistic Regression", family="Linear / probabilistic", cfg=dict(C=1.0, penalty="l2", solver="lbfgs", max_iter=300, preprocessing="median-impute + StandardScaler"),
                 build=lambda: make_pipeline(imp(), StandardScaler(), LogisticRegression(C=1.0, max_iter=300, random_state=SEED))),
     "gnb": dict(name="Gaussian Naive Bayes", family="Probabilistic generative", cfg=dict(var_smoothing=1e-9, preprocessing="median-impute + StandardScaler"),
                 build=lambda: make_pipeline(imp(), StandardScaler(), GaussianNB())),
     "rf":  dict(name="Random Forest", family="Bagging ensemble (trees)", cfg=dict(n_estimators=60, max_depth=14, min_samples_leaf=20, max_features="sqrt", max_samples=0.5, preprocessing="median-impute"),
                 build=lambda: make_pipeline(imp(), RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=20, max_features="sqrt", max_samples=0.5, n_jobs=-1, random_state=SEED))),
     "hgb": dict(name="HistGradientBoosting", family="Boosting (sklearn)", cfg=dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1, preprocessing="median-impute"),
                 build=lambda: make_pipeline(imp(), HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1, random_state=SEED))),
     "lgbm":dict(name="LightGBM (untuned)", family="Boosting (leaf-wise GBDT)", cfg=dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, preprocessing="median-impute"),
                 build=lambda: make_pipeline(imp(), lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1))),
     "mlp": dict(name="MLP (64,32)", family="Neural network", cfg=dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4, early_stopping=True, max_iter=40, preprocessing="median-impute + StandardScaler"),
                 build=lambda: make_pipeline(imp(), StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, early_stopping=True, max_iter=40, random_state=SEED))),
    }

def curves(model):
    m = model.steps[-1][1]
    if isinstance(m, MLPClassifier): return dict(train_loss=list(map(float, m.loss_curve_)), val_accuracy=list(map(float, m.validation_scores_)))
    if isinstance(m, HistGradientBoostingClassifier): return dict(train_loss=list(map(float, -m.train_score_)), val_loss=list(map(float, -m.validation_score_)))
    return {}

def run_model(key, spec, trs, tr, te, arcs_tr, arcs_te, stride, log):
    X, y, g = trs[FEATURES].values.astype(np.float64), trs.label.values, trs.trip_id.values
    gkf = GroupKFold(n_splits=5); oof = np.zeros(len(trs)); splits = list(gkf.split(X, y, g))
    t_cv = time.time()
    for k, (a, b) in enumerate(splits):
        m = spec["build"](); m.fit(X[a], y[a]); oof[b] = m.predict_proba(X[b])[:, 1]; log(f"  [{key}] fold {k} done")
    cv_time = time.time() - t_cv
    thr = ev.best_threshold(y, oof)                       # threshold from OOF of TRAINING trips only
    folds = []
    for k, (a, b) in enumerate(splits):
        cm = ev.cand_metrics(y[b], oof[b], thr); t1, _ = ev.top1_accuracy(trs.iloc[b], oof[b])
        cont = ev.route_continuity(ev.top1_table(trs.iloc[b], oof[b]), arcs_tr)
        folds.append(dict(fold=k, macro_f1=cm["macro_f1"], top1=t1, log_loss=cm["log_loss"], continuity=cont))
    cvdf = pd.DataFrame(folds)
    t0 = time.time(); m = spec["build"](); m.fit(X, y); train_time = time.time() - t0
    Xt = te[FEATURES].values.astype(np.float64)
    t0 = time.time(); pt = m.predict_proba(Xt)[:, 1]; inf_time = time.time() - t0
    cm = ev.cand_metrics(te.label.values, pt, thr); t1, n_ref = ev.top1_accuracy(te, pt)
    top = ev.top1_table(te, pt); cont = ev.route_continuity(top, arcs_te)
    p_tr = m.predict_proba(X)[:, 1]; tr_t1, _ = ev.top1_accuracy(trs, p_tr)
    res = dict(key=key, model=spec["name"], family=spec["family"], config=spec["cfg"], stride=stride, seed=SEED, features=FEATURES,
               cv_macro_f1_mean=cvdf.macro_f1.mean(), cv_macro_f1_std=cvdf.macro_f1.std(), cv_top1_mean=cvdf.top1.mean(), cv_top1_std=cvdf.top1.std(),
               cv_logloss_mean=cvdf.log_loss.mean(), cv_continuity_mean=cvdf.continuity.mean(), cv_wall_s=cv_time,
               test_accuracy=cm["accuracy"], test_precision_macro=cm["precision_macro"], test_recall_macro=cm["recall_macro"], test_macro_f1=cm["macro_f1"],
               test_precision_pos=cm["precision_pos"], test_recall_pos=cm["recall_pos"], test_f1_pos=cm["f1_pos"], test_log_loss=cm["log_loss"],
               test_top1=t1, test_top1_given_ref_in_candidates=t1 / (1 - ev.candidate_miss_rate(te)), test_continuity=cont, threshold=thr, test_confusion=cm["confusion"],
               train_time_s=train_time, inference_time_s=inf_time, inference_us_per_candidate=1e6 * inf_time / len(Xt),
               n_train_rows=len(X), n_test_rows=len(Xt), n_test_obs_ref_ok=n_ref, train_top1=tr_t1, train_macro_f1=ev.cand_metrics(y, p_tr, thr)["macro_f1"])
    cvdf.to_csv(f"{OUT}/cv_folds_{key}.csv", index=False); json.dump(curves(m), open(f"{OUT}/curves_{key}.json", "w"))
    pd.DataFrame(dict(trip_id=trs.trip_id.values, observation_id=trs.observation_id.values, arc_id=trs.arc_id.values, label=y, score=oof)).to_parquet(f"{OUT}/oof_scores_{key}.parquet")
    pd.DataFrame(dict(trip_id=te.trip_id.values, observation_id=te.observation_id.values, arc_id=te.arc_id.values, label=te.label.values, score=pt)).to_parquet(f"{OUT}/test_scores_{key}.parquet")
    joblib.dump(m, f"{ART}/model_{key}.joblib")
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stride", type=int, default=STRIDE); ap.add_argument("--models", default="lr,gnb,rf,hgb,lgbm,mlp")
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True); os.makedirs(ART, exist_ok=True)
    logging.basicConfig(filename=f"{OUT}/experiment_log.txt", level=logging.INFO, format="%(asctime)s %(message)s")
    def log(s): print(s, flush=True); logging.info(s)
    tr = pd.read_parquet(f"{CACHE}/train.parquet"); te = pd.read_parquet(f"{CACHE}/student_test.parquet")
    trs = tr[(tr.ref_ok == 1) & (tr.observation_id % args.stride == 0)].reset_index(drop=True)
    arcs_tr = ev.load_arcs(DATA_ROOT, "train", tr.trip_id.unique()); arcs_te = ev.load_arcs(DATA_ROOT, "student_test", te.trip_id.unique())
    log(f"train rows used={len(trs)} (ref_ok, stride {args.stride}); student rows={len(te)}; seed={SEED}; features={FEATURES}")
    base = {}
    for nm, d, a in [("train_cv_subset", trs, arcs_tr), ("student_test", te, arcs_te)]:
        t1, n = ev.top1_accuracy(d, -d.distance_m.values); c = ev.route_continuity(ev.top1_table(d, -d.distance_m.values), a)
        base[nm] = dict(top1=t1, continuity=c, n_obs=n, miss_rate=ev.candidate_miss_rate(d)); log(f"nearest-distance heuristic [{nm}] {base[nm]}")
    json.dump(base, open(f"{OUT}/baseline_nearest_distance.json", "w"), indent=1)
    allres = json.load(open(f"{OUT}/model_comparison.json")) if os.path.exists(f"{OUT}/model_comparison.json") else {}
    specs = make_models()
    for key in args.models.split(","):
        t = time.time(); log(f"=== {key} start")
        r = run_model(key, specs[key], trs, tr, te, arcs_tr, arcs_te, args.stride, log); allres[key] = r
        log(f"=== {key} done {time.time()-t:.0f}s CV macroF1={r['cv_macro_f1_mean']:.4f} CV top1={r['cv_top1_mean']:.4f} test top1={r['test_top1']:.4f} testF1={r['test_macro_f1']:.4f} LL={r['test_log_loss']:.4f}")
        json.dump(allres, open(f"{OUT}/model_comparison.json", "w"), indent=1, default=float)
        pd.DataFrame(allres.values()).drop(columns=["config", "features", "test_confusion"]).to_csv(f"{OUT}/model_comparison.csv", index=False)
