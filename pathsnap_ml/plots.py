"""All plots are generated from results/*.json|csv|parquet written by run_experiments (actual results only)."""
import json, os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("results/model_comparison.json")); os.makedirs("results/plots", exist_ok=True)
keys = list(R); names = [R[k]["model"] for k in keys]; x = np.arange(len(keys))
def bar(vals, errs=None, title="", ylab="", fn="", lo=None, hi=None, extra=None):
    f, ax = plt.subplots(figsize=(8, 4)); ax.bar(x, vals, yerr=errs, capsize=4, color="#4C78A8")
    if extra: ax.axhline(extra[1], color="crimson", ls="--", label=extra[0]); ax.legend()
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right"); ax.set_title(title); ax.set_ylabel(ylab)
    if lo is not None: ax.set_ylim(lo, hi)
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    f.tight_layout(); f.savefig(f"results/plots/{fn}.png", dpi=130); plt.close(f)
base = json.load(open("results/baseline_nearest_distance.json"))
bar([R[k]["cv_macro_f1_mean"] for k in keys], [R[k]["cv_macro_f1_std"] for k in keys], "5-fold trip-grouped CV macro-F1 (80 train trips)", "macro F1", "cv_macro_f1")
bar([R[k]["cv_top1_mean"] for k in keys], [R[k]["cv_top1_std"] for k in keys], "CV observation-level Top-1 (selection metric)", "Top-1", "cv_top1", extra=("nearest-distance heuristic (train)", base["train_cv_subset"]["top1"]))
bar([R[k]["test_top1"] for k in keys], None, "Student-test Top-1", "Top-1", "test_top1", extra=("nearest-distance heuristic", base["student_test"]["top1"]))
bar([R[k]["test_macro_f1"] for k in keys], None, "Student-test candidate macro-F1", "macro F1", "test_macro_f1")
bar([R[k]["test_log_loss"] for k in keys], None, "Student-test log loss (lower is better)", "log loss", "test_logloss")
bar([R[k]["test_continuity"] for k in keys], None, "Student-test route continuity of Top-1 sequence", "continuity", "test_continuity", extra=("nearest-distance heuristic", base["student_test"]["continuity"]))
f, ax = plt.subplots(1, 2, figsize=(10, 4)); ax[0].bar(x, [R[k]["train_time_s"] for k in keys]); ax[0].set_title("Training time (s, final fit)"); ax[1].bar(x, [R[k]["inference_us_per_candidate"] for k in keys], color="#F58518"); ax[1].set_title("Inference (µs / candidate)")
for a in ax: a.set_xticks(x); a.set_xticklabels(names, rotation=30, ha="right")
f.tight_layout(); f.savefig("results/plots/timing.png", dpi=130); plt.close(f)
f, ax = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.2))
for a, k in zip(ax, keys):
    cm = np.array(R[k]["test_confusion"]); a.imshow(cm, cmap="Blues"); a.set_title(R[k]["model"], fontsize=8); a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xlabel("pred"); a.set_ylabel("true")
    for i in range(2):
        for j in range(2): a.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8, color="black" if cm[i, j] < cm.max() / 2 else "white")
f.tight_layout(); f.savefig("results/plots/confusion_matrices.png", dpi=130); plt.close(f)
hist = {k: json.load(open(f"results/curves_{k}.json")) for k in keys}; hk = [k for k in keys if hist[k]]
f, ax = plt.subplots(1, len(hk), figsize=(5 * len(hk), 3.6)); ax = np.atleast_1d(ax)
for a, k in zip(ax, hk):
    for n, v in hist[k].items(): a.plot(v, label=n)
    a.set_title(f"{R[k]['model']} (actual training history)"); a.set_xlabel("iteration / epoch"); a.legend()
f.tight_layout(); f.savefig("results/plots/training_histories.png", dpi=130); plt.close(f)
fo = pd.concat([pd.read_csv(f"results/cv_folds_{k}.csv").assign(model=R[k]["model"]) for k in keys])
f, ax = plt.subplots(figsize=(8, 4)); fo.boxplot(column="top1", by="model", ax=ax, rot=25); ax.set_title("Per-fold CV Top-1 (stability)"); plt.suptitle(""); f.tight_layout(); f.savefig("results/plots/cv_fold_stability.png", dpi=130); plt.close(f)
print("plots written")
