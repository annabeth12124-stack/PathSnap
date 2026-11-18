"""Baseline-vs-tuned comparison, final-model selection, optimisation plots, final tuned artifact export.

Selection is rule-based and DECLARED BEFORE seeing results (constants below), and is NOT "highest single score":
  1. Regression guard: a tuned model whose CV Top-1 is worse than its own untuned baseline by > GUARD['top1'], or whose
     CV log loss is worse by > GUARD['log_loss'], is ineligible (unless all three regress).
  2. Each eligible model earns one point per metric on which it is within TOL[metric] of the best eligible model:
     Macro F1, Top-1 (higher is better); log loss, observation-level NLL of prob_obs, fold std of Macro F1 (lower is better).
  3. Most points wins; ties -> higher Macro F1, then lower log loss.
  4. The report flags whether the winner's Macro F1 lead is smaller than the fold std (i.e. not statistically decisive)
     and whether each tuned model's gain over its untuned counterpart exceeds fold noise.
student_test is NOT used here.

Usage: python -m pathsnap_ml.tuning.report
"""
import json
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import FEATURES, ID_COLS  # noqa: E402
from . import common as C  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HIGHER = ["macro_f1", "top1"]
LOWER = ["log_loss", "obs_nll", "macro_f1_std"]
TOL = dict(macro_f1=0.003, top1=0.005, log_loss=0.003, obs_nll=0.01, macro_f1_std=0.005)
GUARD = dict(top1=0.005, log_loss=0.003)


def metric_row(df):
    return dict(macro_f1=float(df.macro_f1.mean()), macro_f1_std=float(df.macro_f1.std()),
                top1=float(df.top1.mean()), top1_std=float(df.top1.std()),
                log_loss=float(df.log_loss.mean()), obs_nll=float(df.obs_nll.mean()),
                continuity=float(df.continuity.mean()), macro_f1_thr050=float(df.macro_f1_thr050.mean()))


def select(tuned, baseline):
    """tuned/baseline: {key: {macro_f1, macro_f1_std, top1, log_loss, obs_nll}} -> selection dict (pure function)."""
    regressed = {k: bool(tuned[k]["top1"] < baseline[k]["top1"] - GUARD["top1"]
                         or tuned[k]["log_loss"] > baseline[k]["log_loss"] + GUARD["log_loss"]) for k in tuned}
    eligible = [k for k in tuned if not regressed[k]] or list(tuned)
    points = {k: 0 for k in eligible}
    within = {}
    for m in HIGHER + LOWER:
        vals = {k: tuned[k][m] for k in eligible}
        best = max(vals.values()) if m in HIGHER else min(vals.values())
        for k, v in vals.items():
            ok = (best - v <= TOL[m]) if m in HIGHER else (v - best <= TOL[m])
            points[k] += int(ok)
            within[f"{k}|{m}"] = bool(ok)
    order = sorted(eligible, key=lambda k: (-points[k], -tuned[k]["macro_f1"], tuned[k]["log_loss"]))
    return dict(selected=order[0], ranking=order, points=points, regressed=regressed, within_tolerance=within,
                tolerances=TOL, regression_guard=GUARD)


def plots(key, tuned_df, base_df, summary_row):
    trials = pd.read_csv(C.RESULTS / f"trials_{key}.csv")
    ok = trials[trials.state == "COMPLETE"].copy()
    # 1) optimisation progress
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(ok.trial, ok.value, s=18, alpha=0.7, label="trial")
    ax.plot(ok.trial, ok.value.cummax(), color="C3", lw=2, label="best so far")
    ax.axhline(base_df.macro_f1.mean(), color="k", ls="--", lw=1, label="untuned baseline (CV mean)")
    ax.set(xlabel="trial", ylabel="CV Macro F1", title=f"{C.MODEL_NAMES[key]} - Optuna progress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.PLOTS / f"progress_{key}.png", dpi=140)
    plt.close(fig)
    # 2) top-1 / log loss of every trial (secondary metrics, to see the trade-off)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(ok.macro_f1_mean, ok.top1_mean, s=18)
    axs[0].set(xlabel="CV Macro F1", ylabel="CV Top-1", title="Macro F1 vs Top-1 per trial")
    axs[1].scatter(ok.macro_f1_mean, ok.log_loss_mean, s=18)
    axs[1].set(xlabel="CV Macro F1", ylabel="CV log loss", title="Macro F1 vs log loss per trial")
    fig.tight_layout()
    fig.savefig(C.PLOTS / f"tradeoffs_{key}.png", dpi=140)
    plt.close(fig)
    # 3) fold-level baseline vs tuned
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, m in zip(axs, ["macro_f1", "top1", "log_loss"]):
        w = 0.38
        ax.bar(np.arange(5) - w / 2, base_df[m], w, label="baseline")
        ax.bar(np.arange(5) + w / 2, tuned_df[m], w, label="tuned")
        ax.set(xlabel="fold", title=m)
        if m != "log_loss":
            lo = min(base_df[m].min(), tuned_df[m].min())
            ax.set_ylim(lo - 0.03, None)
    axs[0].legend()
    fig.suptitle(f"{C.MODEL_NAMES[key]} - fold-level baseline vs tuned")
    fig.tight_layout()
    fig.savefig(C.PLOTS / f"folds_{key}.png", dpi=140)
    plt.close(fig)
    # 4) parameter importance (best effort)
    try:
        st = optuna.load_study(study_name=f"pathsnap_{key}", storage=f"sqlite:///{(C.STUDIES / f'{key}.db').as_posix()}")
        imp = optuna.importance.get_param_importances(st)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1])
        ax.set(title=f"{C.MODEL_NAMES[key]} - hyperparameter importance (indicative)")
        fig.tight_layout()
        fig.savefig(C.PLOTS / f"importance_{key}.png", dpi=140)
        plt.close(fig)
    except Exception as e:  # importance needs >= a few completed trials
        print(f"[{key}] importance plot skipped: {e}")


def summary_plot(rows):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    keys = list(rows.keys())
    for ax, m in zip(axs, ["macro_f1", "top1", "log_loss"]):
        x = np.arange(len(keys))
        b = [rows[k]["baseline"][m] for k in keys]
        t = [rows[k]["tuned"][m] for k in keys]
        sd = [rows[k]["tuned"].get(m + "_std", 0) for k in keys]
        ax.bar(x - 0.19, b, 0.38, label="baseline")
        ax.bar(x + 0.19, t, 0.38, yerr=sd if m != "log_loss" else None, capsize=3, label="tuned (+/- fold std)")
        ax.set_xticks(x)
        ax.set_xticklabels([C.MODEL_NAMES[k] for k in keys], rotation=12)
        ax.set_title(f"CV {m}")
        if m != "log_loss":
            ax.set_ylim(min(b + t) - 0.05, None)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(C.PLOTS / "baseline_vs_tuned_summary.png", dpi=140)
    plt.close(fig)


def main():
    C.make_dirs()
    rows, comp = {}, []
    tuned_dfs, base_dfs = {}, {}
    for key in C.MODEL_KEYS:
        base_dfs[key] = C.baseline_fold_metrics(key)
        tuned_dfs[key] = pd.read_csv(C.RESULTS / f"cv_folds_tuned_{key}.csv")
        b, t = metric_row(base_dfs[key]), metric_row(tuned_dfs[key])
        rows[key] = dict(baseline=b, tuned=t)
        for m in ["macro_f1", "top1", "log_loss", "obs_nll"]:
            d = (tuned_dfs[key][m] - base_dfs[key][m]).values
            se = d.std(ddof=1) / np.sqrt(len(d))
            comp.append(dict(model=C.MODEL_NAMES[key], metric=m, baseline=b[m], tuned=t[m], delta=t[m] - b[m],
                             paired_diff_std=float(d.std(ddof=1)), paired_t=float((d.mean() / se) if se > 0 else np.nan),
                             folds_tuned_better=int((d > 0).sum() if m in HIGHER else (d < 0).sum()),
                             baseline_fold_std=float(base_dfs[key][m].std()), tuned_fold_std=float(tuned_dfs[key][m].std())))
        plots(key, tuned_dfs[key], base_dfs[key], rows[key])
    comp = pd.DataFrame(comp)
    comp.to_csv(C.RESULTS / "comparison_baseline_vs_tuned.csv", index=False)
    json.dump(rows, open(C.RESULTS / "comparison_baseline_vs_tuned.json", "w"), indent=1)
    summary_plot(rows)

    sel = select({k: rows[k]["tuned"] for k in rows}, {k: rows[k]["baseline"] for k in rows})
    w, runner = sel["selected"], (sel["ranking"][1] if len(sel["ranking"]) > 1 else None)
    sel["lead_over_runner_up_macro_f1"] = (rows[w]["tuned"]["macro_f1"] - rows[runner]["tuned"]["macro_f1"]) if runner else None
    sel["lead_smaller_than_fold_std"] = bool(runner and abs(sel["lead_over_runner_up_macro_f1"]) < rows[w]["tuned"]["macro_f1_std"])
    sel["gain_over_own_baseline_exceeds_fold_noise"] = {
        k: bool((rows[k]["tuned"]["macro_f1"] - rows[k]["baseline"]["macro_f1"]) > rows[k]["baseline"]["macro_f1_std"])
        for k in rows}
    json.dump(sel, open(C.RESULTS / "model_selection.json", "w"), indent=1)

    # final tuned artifact in optimisation/artifacts (same file names predict.load_model expects)
    cfg_t = json.load(open(C.ARTIFACTS / f"model_tuned_{w}_config.json"))
    base_cfg = json.load(open(C.BASE_ART / "final_model_config.json"))
    shutil.copy(C.ARTIFACTS / f"model_tuned_{w}.joblib", C.ARTIFACTS / "final_model.joblib")
    final = dict(phase="Phase 2 Bayesian tuning (Optuna TPE)", selected_model=C.MODEL_NAMES[w] + " (tuned)", selected_key=w,
                 selection_method="tolerance-point scorecard over CV Macro F1, Top-1, log loss, obs-level NLL, fold stability "
                                  "(see model_selection.json); NOT highest single score",
                 selection=sel, features=FEATURES, identifier_columns_excluded=ID_COLS,
                 decision_threshold=cfg_t["decision_threshold"], hyperparameters=cfg_t["hyperparameters"], seed=cfg_t["seed"],
                 cv_summary=cfg_t["cv_summary"], baseline_cv_summary=rows[w]["baseline"],
                 candidate_generation=base_cfg.get("candidate_generation"), training=cfg_t["training"],
                 baseline_artifacts_modified=False,
                 prediction_contract="unchanged: trip_id, observation_id, timestamp, arc_id, from_node_id, to_node_id, "
                                     "score, prob_obs (sums to 1 per observation), rank, is_top1")
    json.dump(final, open(C.ARTIFACTS / "final_model_config.json", "w"), indent=1, default=float)

    C.verify_baseline_untouched()
    pd.set_option("display.width", 200)
    print(comp[["model", "metric", "baseline", "tuned", "delta", "paired_t", "folds_tuned_better"]].round(4).to_string(index=False))
    print("\nselected:", w, "| points:", sel["points"], "| regressed:", sel["regressed"])
    print("lead smaller than fold std (not decisive):", sel["lead_smaller_than_fold_std"])
    print("gain > fold noise:", sel["gain_over_own_baseline_exceeds_fold_noise"])


if __name__ == "__main__":
    main()
