#!/usr/bin/env python3
"""
03_cluster_eval.py

Cluster both representations and evaluate against the annotation.
K-means and agglomerative (cosine) for k in {2..6}, 10 seeds each.
ARI, NMI, confusion matrix at k=2, random and single-cluster baselines.

Input:  data/X_lex.npz, data/X_ctx.npy, data/occurrences.tsv
Output: data/eval_results.tsv     (one row per rep x algo x k x seed)
        data/eval_summary.tsv     (mean + sd across seeds)
        data/eval_confusion.tsv   (k=2 confusion matrix, best seed per rep x algo)
"""
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR     = Path("/home/simiyachaq/Documentos/Programming_Stuff/Shan")
DATA_DIR     = BASE_DIR / "data"
OUT_RESULTS  = DATA_DIR / "eval_results.tsv"
OUT_SUMMARY  = DATA_DIR / "eval_summary.tsv"
OUT_CONFUSE  = DATA_DIR / "eval_confusion.tsv"

K_RANGE      = range(2, 7)   # 2..6
N_SEEDS      = 10
BASE_SEED    = 42
N_RANDOM     = 100            # random partitions for the baseline

# ------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------
print("Loading ...")
X_lex = load_npz(DATA_DIR / "X_lex.npz")
X_ctx = np.load(DATA_DIR / "X_ctx.npy")
meta  = pd.read_csv(DATA_DIR / "occurrences.tsv", sep="\t", dtype=str)

assert X_lex.shape[0] == X_ctx.shape[0] == len(meta), \
    f"row count mismatch: {X_lex.shape}, {X_ctx.shape}, {len(meta)}"

# Exclude ambiguous rows from evaluation (if any)
mask = meta["LABEL"].isin({"poss", "nonposs"}).to_numpy()
X_lex = X_lex[mask]
X_ctx = X_ctx[mask]
meta  = meta[mask].reset_index(drop=True)

y_true = (meta["LABEL"] == "poss").astype(int).to_numpy()
print(f"Evaluable rows: {len(meta)}  (poss={y_true.sum()}, nonposs={(1-y_true).sum()})")

reps = {
    "tfidf":      X_lex,
    "contextual": X_ctx,
}

# ------------------------------------------------------------
# 2. Clustering helpers
# ------------------------------------------------------------
def cluster_kmeans(X, k, seed):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
    return km.fit_predict(X.toarray() if hasattr(X, "toarray") else X)

def cluster_agglo(X, k, seed):
    ag = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    return ag.fit_predict(X.toarray() if hasattr(X, "toarray") else X)

ALGOS = {
    "kmeans":     cluster_kmeans,
    "agglomerative": cluster_agglo,
}

# ------------------------------------------------------------
# 3. Baselines
# ------------------------------------------------------------
print("\nBaselines ...")
rng = np.random.default_rng(BASE_SEED)
random_ari = {rep: [] for rep in reps}
random_nmi = {rep: [] for rep in reps}
for rep, X in reps.items():
    n = X.shape[0]
    for _ in range(N_RANDOM):
        pred = rng.integers(0, 2, size=n)
        random_ari[rep].append(adjusted_rand_score(y_true, pred))
        random_nmi[rep].append(normalized_mutual_info_score(y_true, pred))

for rep in reps:
    print(f"  {rep:12s} random ARI = {np.mean(random_ari[rep]):+.4f} "
          f"(sd {np.std(random_ari[rep]):.4f})")
print("  single-cluster: ARI = 0.0000, NMI = 0.0000 (by construction)")

# ------------------------------------------------------------
# 4. Main loop
# ------------------------------------------------------------
print("\nClustering ...")
results = []
confusion_records = []

for rep, X in reps.items():
    for algo_name, algo_fn in ALGOS.items():
        for k in K_RANGE:
            for seed_i in range(N_SEEDS):
                seed = BASE_SEED + seed_i
                try:
                    pred = algo_fn(X, k, seed)
                except Exception as e:
                    print(f"  ERROR {rep}/{algo_name}/k={k}/seed={seed}: {e}")
                    continue
                ari = adjusted_rand_score(y_true, pred)
                nmi = normalized_mutual_info_score(y_true, pred)
                results.append({
                    "representation": rep,
                    "algorithm":      algo_name,
                    "k":              k,
                    "seed":           seed,
                    "ari":            ari,
                    "nmi":            nmi,
                })
                if k == 2 and seed == BASE_SEED:
                    # 2x2 confusion: rows = predicted cluster, cols = true label
                    for cluster_id in range(k):
                        for true_id, true_label in [(0, "nonposs"), (1, "poss")]:
                            n = int(((pred == cluster_id) & (y_true == true_id)).sum())
                            confusion_records.append({
                                "representation": rep,
                                "algorithm":      algo_name,
                                "cluster":        cluster_id,
                                "true_label":     true_label,
                                "count":          n,
                            })
                print(f"  {rep:12s} {algo_name:14s} k={k} seed={seed:4d}  "
                      f"ARI={ari:+.4f}  NMI={nmi:.4f}")

# ------------------------------------------------------------
# 5. Save detailed results
# ------------------------------------------------------------
res_df = pd.DataFrame(results)
res_df.to_csv(OUT_RESULTS, sep="\t", index=False)
print(f"\nSaved {OUT_RESULTS}  ({len(res_df)} rows)")

# ------------------------------------------------------------
# 6. Summarise across seeds
# ------------------------------------------------------------
summary = (res_df.groupby(["representation", "algorithm", "k"])
                 .agg(ari_mean=("ari", "mean"),
                      ari_sd=("ari", "std"),
                      nmi_mean=("nmi", "mean"),
                      nmi_sd=("nmi", "std"),
                      n_seeds=("seed", "count"))
                 .reset_index())
summary.to_csv(OUT_SUMMARY, sep="\t", index=False)
print(f"Saved {OUT_SUMMARY}  ({len(summary)} rows)")

# ------------------------------------------------------------
# 7. Save confusion at k=2
# ------------------------------------------------------------
conf_df = pd.DataFrame(confusion_records)
conf_df.to_csv(OUT_CONFUSE, sep="\t", index=False)
print(f"Saved {OUT_CONFUSE}  ({len(conf_df)} rows)")

# ------------------------------------------------------------
# 8. Report
# ------------------------------------------------------------
print("\n==== SUMMARY (k=2) ====")
print(f"  {'representation':14s} {'algorithm':14s} {'ARI':>10s} {'NMI':>10s}")
print(f"  {'-'*14:14s} {'-'*14:14s} {'-'*10:10s} {'-'*10:10s}")
for _, row in summary[summary["k"] == 2].iterrows():
    print(f"  {row['representation']:14s} {row['algorithm']:14s} "
          f"{row['ari_mean']:+10.4f} {row['nmi_mean']:10.4f}")

print("\n==== SUMMARY (all k, ARI mean) ====")
print(f"  {'representation':14s} {'algorithm':14s} " +
      " ".join(f"k={k}" for k in K_RANGE))
for (rep, algo), sub in summary.groupby(["representation", "algorithm"]):
    vals = {int(r["k"]): r["ari_mean"] for _, r in sub.iterrows()}
    line = f"  {rep:14s} {algo:14s} " + \
           " ".join(f"{vals.get(k, float('nan')):+6.3f}" for k in K_RANGE)
    print(line)

# ------------------------------------------------------------
# 9. Interpretation hints
# ------------------------------------------------------------
ctx_k2 = summary[(summary["representation"] == "contextual") &
                 (summary["algorithm"] == "kmeans") &
                 (summary["k"] == 2)]
lex_k2 = summary[(summary["representation"] == "tfidf") &
                 (summary["algorithm"] == "kmeans") &
                 (summary["k"] == 2)]
if not ctx_k2.empty and not lex_k2.empty:
    gap = ctx_k2.iloc[0]["ari_mean"] - lex_k2.iloc[0]["ari_mean"]
    print(f"\n  Contextual - TF-IDF ARI (k=2, K-means): {gap:+.4f}")
    if gap > 0.10:
        print("  -> contextual beats lexical by a clear margin (H2 supported)")
    elif gap < -0.10:
        print("  -> lexical beats contextual (unexpected)")
    else:
        print("  -> gap is small; category likely lexically identifiable")

for rep in reps:
    ari_rand = np.mean(random_ari[rep])
    ari_best = summary[(summary["representation"] == rep) &
                       (summary["k"] == 2)]["ari_mean"].max()
    if pd.notna(ari_best) and ari_best - ari_rand < 0.05:
        print(f"  WARNING: {rep} k=2 ARI is within 0.05 of the random baseline")

print("\nDone.")