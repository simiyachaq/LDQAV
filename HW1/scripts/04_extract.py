from collections import Counter
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "output/moby_tokens.csv")

CENTER = "whale"
SKIP_DEP = {"punct", "det", "cc", "aux", "auxpass", "mark", "neg", "case"}

kids_df = df[(df.head_i >= 0) & (~df.dep.isin(SKIP_DEP))]
kids_of = {
    (sid, hi): g
    for (sid, hi), g in kids_df.groupby(["sent_id", "head_i"])
}

primary = Counter()     # (center, dep, child) -> count
secondary = Counter()   # (child, dep, grandchild) -> count

for _, w in df[df.lemma == CENTER].iterrows():
    for _, c in kids_of.get((w.sent_id, w.i), pd.DataFrame()).iterrows():
        if c.lemma == CENTER:
            continue
        primary[(CENTER, c.dep, c.lemma)] += 1
        for _, g in kids_of.get((c.sent_id, c.i), pd.DataFrame()).iterrows():
            if g.lemma in (CENTER, c.lemma):
                continue
            secondary[(c.lemma, g.dep, g.lemma)] += 1

out = HERE.parent / "output"

prim_df = pd.DataFrame(
    [(h, d, t, n) for (h, d, t), n in primary.items()],
    columns=["head", "dep", "tail", "weight"],
).sort_values("weight", ascending=False)

sec_df = pd.DataFrame(
    [(h, d, t, n) for (h, d, t), n in secondary.items()],
    columns=["head", "dep", "tail", "weight"],
).sort_values("weight", ascending=False)

prim_df.to_csv(out / "edges_primary.csv", index=False)
sec_df.to_csv(out / "edges_secondary.csv", index=False)

print(f"primary edges:   {len(prim_df):>5} unique, "
      f"{int(prim_df.weight.sum()):>5} total")
print(f"secondary edges: {len(sec_df):>5} unique, "
      f"{int(sec_df.weight.sum()):>5} total")
print("\nTop primary:")
print(prim_df.head(20).to_string(index=False))
print("\nTop secondary:")
print(sec_df.head(20).to_string(index=False))