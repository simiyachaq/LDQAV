import spacy
from pathlib import Path

HERE = Path(__file__).resolve().parent
IN = HERE.parent / "data/clean/moby_dick.txt"
OUT = HERE.parent / "output/moby_lemmatized.txt"

text = IN.read_text(encoding="utf-8")
nlp = spacy.load("en_core_web_sm")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for doc in nlp.pipe(text.split("\n\n"), batch_size=50):
        for sent in doc.sents:
            lemmas = [t.lemma_ for t in sent if not t.is_space and not t.is_punct]
            if lemmas:
                f.write(" ".join(lemmas) + "\n")

print(f"wrote {OUT}")