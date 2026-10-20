from collections import Counter
import spacy
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
IN = HERE.parent / "data/clean/moby_dick.txt"
OUT = HERE.parent / "output/moby_tokens.csv"

text = IN.read_text(encoding="utf-8")
nlp = spacy.load("en_core_web_sm")

rows = []
sent_id = 0
for doc in nlp.pipe(text.split("\n\n"), batch_size=50):
    for sent in doc.sents:
        toks = [t for t in sent if not t.is_space]
        idx = {t.i: k for k, t in enumerate(toks)}
        for i, t in enumerate(toks):
            rows.append({
                "sent_id": sent_id,
                "i": i,
                "token": t.text,
                "lemma": t.lemma_.lower(),
                "pos": t.pos_,
                "dep": t.dep_,
                "head_i": idx.get(t.head.i, -1),
            })
        sent_id += 1

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"wrote {OUT}: {len(df):,} tokens, {sent_id:,} sentences")

content = df[df.pos.isin(["NOUN", "PROPN", "VERB", "ADJ", "ADV"])]
for lemma, n in Counter(content.lemma).most_common(40):
    print(f"  {lemma:20s} {n}")