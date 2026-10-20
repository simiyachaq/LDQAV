"""Probemos 'whale' a ver si se justifica el gráfico entero"""
from pathlib import Path
from collections import Counter
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "output/moby_tokens.csv")

CENTER = "whale"
SKIP_DEP = {"punct", "det", "cc", "aux", "auxpass", "mark", "neg", "case"}

hits = df[df.lemma == CENTER]
children = []
for _, row in hits.iterrows():
    kids = df[(df.sent_id == row.sent_id) & (df.head_i == row.i)]
    for _, k in kids.iterrows():
        if k.dep not in SKIP_DEP:
            children.append((k.lemma, k.dep))

print(f"{len(hits)} occurrences of '{CENTER}'")
print(f"{len(children)} content child edges\n")
for (lemma, dep), n in Counter(children).most_common(30):
    print(f"  {dep:12s} {lemma:20s} {n}")