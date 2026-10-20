from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*tight_layout.*")

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output"

CENTER = "whale"
MIN_PRIMARY_W = 3
MIN_SECONDARY_W = 1
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
POS_COLORS = {"NOUN": "#4C72B0", "PROPN": "#DD8452", "VERB": "#55A868",
              "ADJ": "#C44E52", "ADV": "#8172B3", "OTHER": "#937860"}

tokens = pd.read_csv(OUT / "moby_tokens.csv")
pos_of = (tokens[tokens.pos.isin(CONTENT_POS)]
          .groupby("lemma")["pos"].agg(lambda s: s.iloc[0]).to_dict())

prim = pd.read_csv(OUT / "edges_primary.csv")
sec = pd.read_csv(OUT / "edges_secondary.csv")

prim = prim[(prim["head"] != prim["tail"])
            & prim["tail"].str.isalpha()
            & (prim["weight"] >= MIN_PRIMARY_W)]
sec = sec[(sec["head"] != sec["tail"])
          & sec["tail"].str.isalpha()
          & (sec["weight"] >= MIN_SECONDARY_W)]
sec = sec[sec["head"].isin(prim["tail"])]
sec = sec[sec["tail"].isin(pos_of)]          

print(f"primary edges kept:   {len(prim):>3}")
print(f"secondary edges kept: {len(sec):>3}")

G = nx.DiGraph()
G.add_node(CENTER)
for _, r in prim.iterrows():
    G.add_edge(CENTER, r["tail"], dep=r["dep"], weight=int(r["weight"]))
for _, r in sec.iterrows():
    G.add_edge(r["head"], r["tail"], dep=r["dep"], weight=int(r["weight"]))

density = nx.density(G)
print(f"nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}, "
      f"density: {density:.4f}")

pos = nx.spring_layout(G, k=0.8, seed=42)
plt.figure(figsize=(14, 10))
nx.draw(G, pos,
        node_color=[POS_COLORS.get(pos_of.get(n, "OTHER"), "#888") for n in G.nodes()],
        node_size=[300 + 400 * G.degree(n) ** 0.7 for n in G.nodes()],
        width=[0.5 + 0.25 * G[u][v]["weight"] for u, v in G.edges()],
        with_labels=True, font_size=9, arrowsize=10, alpha=0.85,
        edgecolors="black")
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                      markersize=10, label=p)
           for p, c in POS_COLORS.items()
           if any(pos_of.get(n, "OTHER") == p for n in G.nodes())]
plt.legend(handles=handles, title="POS", loc="upper right")
plt.title(f"'{CENTER}' dependency graph — nodes={G.number_of_nodes()}, "
          f"edges={G.number_of_edges()}, density={density:.3f}")
plt.tight_layout()
plt.savefig(OUT / "graph_whale.png", dpi=150)
print(f"saved {OUT / 'graph_whale.png'}")

nx.write_graphml(G, OUT / "graph_whale.graphml")
print(f"saved {OUT / 'graph_whale.graphml'}")