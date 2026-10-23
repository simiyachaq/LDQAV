#!/usr/bin/env python3
"""
04_pca.py

PCA to two dimensions for both representations, scatter plots coloured
by cluster (k=2 K-means, best seed) and by annotation label.

Input:  data/X_lex.npz, data/X_ctx.npy, data/occurrences.tsv
Output: figures/pca_tfidf.png
        figures/pca_contextual.png
        figures/pca_combined.png
        data/pca_coords.tsv
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR   = Path("/home/simiyachaq/Documentos/Programming_Stuff/Shan")
DATA_DIR   = BASE_DIR / "data"
FIG_DIR    = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_COORDS = DATA_DIR / "pca_coords.tsv"
SEED       = 42
K          = 2

plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"]  = 9

# ------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------
print("Loading ...")
X_lex = load_npz(DATA_DIR / "X_lex.npz")
X_ctx = np.load(DATA_DIR / "X_ctx.npy")
meta  = pd.read_csv(DATA_DIR / "occurrences.tsv", sep="\t", dtype=str)

assert X_lex.shape[0] == X_ctx.shape[0] == len(meta), "row count mismatch"

mask = meta["LABEL"].isin({"poss", "nonposs"}).to_numpy()
X_lex = X_lex[mask]
X_ctx = X_ctx[mask]
meta  = meta[mask].reset_index(drop=True)

y_true_bin = (meta["LABEL"] == "poss").astype(int).to_numpy()
y_str      = meta["LABEL"].to_numpy()
print(f"Rows: {len(meta)}   poss={int(y_true_bin.sum())}   nonposs={int((1-y_true_bin).sum())}")

# ------------------------------------------------------------
# 2. Cluster at k=2 (matches 03_cluster_eval)
# ------------------------------------------------------------
def kmeans_labels(X, k, seed):
    arr = X.toarray() if hasattr(X, "toarray") else X
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
    return km.fit_predict(arr)

print("Clustering k=2 (K-means) ...")
cluster_lex = kmeans_labels(X_lex, K, SEED)
cluster_ctx = kmeans_labels(X_ctx, K, SEED)
ari_lex = adjusted_rand_score(y_true_bin, cluster_lex)
ari_ctx = adjusted_rand_score(y_true_bin, cluster_ctx)
print(f"  ARI (tfidf):      {ari_lex:+.4f}")
print(f"  ARI (contextual): {ari_ctx:+.4f}")

# ------------------------------------------------------------
# 3. PCA to 2D
# ------------------------------------------------------------
def pca2(X):
    arr = X.toarray() if hasattr(X, "toarray") else X
    p = PCA(n_components=2, random_state=SEED)
    coords = p.fit_transform(arr)
    return coords, p.explained_variance_ratio_

print("Running PCA ...")
coords_lex, var_lex = pca2(X_lex)
coords_ctx, var_ctx = pca2(X_ctx)
print(f"  PCA tfidf:      PC1={var_lex[0]:.3f}  PC2={var_lex[1]:.3f}")
print(f"  PCA contextual: PC1={var_ctx[0]:.3f}  PC2={var_ctx[1]:.3f}")

# ------------------------------------------------------------
# 4. Save coordinates
# ------------------------------------------------------------
coords_df = pd.DataFrame({
    "hit_id":       meta["hit_id"],
    "LABEL":        meta["LABEL"],
    "source":       meta["source"],
    "year":         meta["year"],
    "cluster_lex":  cluster_lex,
    "cluster_ctx":  cluster_ctx,
    "pc1_lex":      coords_lex[:, 0],
    "pc2_lex":      coords_lex[:, 1],
    "pc1_ctx":      coords_ctx[:, 0],
    "pc2_ctx":      coords_ctx[:, 1],
})
coords_df.to_csv(OUT_COORDS, sep="\t", index=False)
print(f"Saved {OUT_COORDS}")

# ------------------------------------------------------------
# 5. Plot helpers
# ------------------------------------------------------------
LABEL_COLORS  = {"poss": "#d62728", "nonposs": "#1f77b4"}
CLUSTER_COLORS = {0: "#7f7f7f", 1: "#2ca02c"}

def scatter_panel(ax, coords, labels, title, colors, legend_title):
    for key, color in colors.items():
        sel = labels == key
        if sel.sum() == 0:
            continue
        ax.scatter(coords[sel, 0], coords[sel, 1],
                   c=color, s=24, alpha=0.75,
                   edgecolors="white", linewidths=0.4,
                   label=f"{key} (n={int(sel.sum())})")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(title=legend_title, fontsize=8, title_fontsize=8,
              loc="best", framealpha=0.9)

# ------------------------------------------------------------
# 6. Figure 1 — TF-IDF (by label and by cluster)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
scatter_panel(axes[0], coords_lex, y_str, 
              f"TF-IDF — by annotation (ARI={ari_lex:+.3f})",
              LABEL_COLORS, "label")
scatter_panel(axes[1], coords_lex,
              np.array([f"c{c}" for c in cluster_lex]),
              "TF-IDF — by cluster (k=2)",
              {f"c{k}": v for k, v in CLUSTER_COLORS.items()}, "cluster")
fig.suptitle(f"TF-IDF representation  (PC1={var_lex[0]*100:.1f}%, PC2={var_lex[1]*100:.1f}%)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "pca_tfidf.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR/'pca_tfidf.png'}")

# ------------------------------------------------------------
# 7. Figure 2 — contextual (by label and by cluster)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
scatter_panel(axes[0], coords_ctx, y_str,
              f"Contextual — by annotation (ARI={ari_ctx:+.3f})",
              LABEL_COLORS, "label")
scatter_panel(axes[1], coords_ctx,
              np.array([f"c{c}" for c in cluster_ctx]),
              "Contextual — by cluster (k=2)",
              {f"c{k}": v for k, v in CLUSTER_COLORS.items()}, "cluster")
fig.suptitle(f"Contextual representation  (PC1={var_ctx[0]*100:.1f}%, PC2={var_ctx[1]*100:.1f}%)",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "pca_contextual.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR/'pca_contextual.png'}")

# ------------------------------------------------------------
# 8. Figure 3 — combined, coloured by label
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
scatter_panel(axes[0], coords_lex, y_str,
              f"TF-IDF (ARI={ari_lex:+.3f})", LABEL_COLORS, "label")
scatter_panel(axes[1], coords_ctx, y_str,
              f"Contextual (ARI={ari_ctx:+.3f})", LABEL_COLORS, "label")
fig.suptitle("PCA panels coloured by annotation — tfidf vs contextual",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "pca_combined.png", bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR/'pca_combined.png'}")

print("\nDone.")