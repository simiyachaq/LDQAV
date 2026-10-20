import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output"

G = nx.read_graphml(OUT / "graph_whale.graphml")
U = G.to_undirected()               # standard for community detection

print(f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
print(f"density (directed):   {nx.density(G):.4f}")
print(f"density (undirected): {nx.density(U):.4f}")
print(f"avg clustering:       {nx.average_clustering(U):.4f}")

deg = dict(U.degree())
print("\ntop by degree:")
for n, d in sorted(deg.items(), key=lambda x: -x[1])[:10]:
    print(f"  {n:20s} {d}")

btw = nx.betweenness_centrality(U)
print("\ntop by betweenness:")
for n, b in sorted(btw.items(), key=lambda x: -x[1])[:10]:
    print(f"  {n:20s} {b:.4f}")

comms = nx.community.louvain_communities(U, seed=42, resolution=1.0)
comms = sorted(comms, key=len, reverse=True)
mod = nx.community.modularity(U, comms)
print(f"\ncommunities={len(comms)}, modularity={mod:.4f}")
for cid, members in enumerate(comms):
    sample = sorted(members)[:12]
    print(f"  c{cid:02d} n={len(members):>3}: {sample}"
          f"{'...' if len(members) > 12 else ''}")

node_to_comm = {n: cid for cid, members in enumerate(comms) for n in members}
pd.DataFrame(
    [{"node": n, "community": node_to_comm[n], "degree": deg[n],
      "betweenness": round(btw[n], 4)} for n in U.nodes()]
).sort_values(["community", "degree"], ascending=[True, False]).to_csv(
    OUT / "community_whale.csv", index=False)

# community-colored layout
cmap = plt.get_cmap("tab20")
pos = nx.spring_layout(U, k=0.8, seed=42)
plt.figure(figsize=(14, 10))
nx.draw(U, pos,
        node_color=[cmap(cid % 20) for cid in (node_to_comm[n] for n in U.nodes())],
        node_size=[300 + 400 * deg[n] ** 0.7 for n in U.nodes()],
        width=[0.5 + 0.25 * U[u][v].get("weight", 1) for u, v in U.edges()],
        with_labels=True, font_size=8, edgecolors="black", alpha=0.9)
plt.title(f"'whale' dependency graph — {len(comms)} communities, "
          f"modularity={mod:.3f}, density={nx.density(U):.3f}")
plt.tight_layout()
plt.savefig(OUT / "graph_whale_communities.png", dpi=150)
print(f"saved {OUT / 'graph_whale_communities.png'}")