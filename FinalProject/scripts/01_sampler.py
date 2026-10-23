#!/usr/bin/env python3
"""
Step 1 - Sampler for ၶွင် annotation.

Frame: hit-bearing articles, 2019-2024.
Simple random sample of N articles (no stratification).
1 hit per article if h_i <= 2, 2 hits if h_i >= 3, seeded per article.
Reopens each article via parquet to extract the target sentence and offsets.

Output: annotation/khong_annotation.tsv  (one row per selected hit)
"""
import csv
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR       = Path("/home/simiyachaq/Documentos/Programming_Stuff/Shan")
CENSUS_FILE    = BASE_DIR / "annotation/catalogue/khong_census.tsv"
SHN2ENG_FILE   = BASE_DIR / "shanparsing/dictionary_tsv_clean/shn2eng.tsv"
SHN2SHN_FILE   = BASE_DIR / "shanparsing/dictionary_tsv_clean/shn2shn.tsv"
MASTER_LEX     = BASE_DIR / "master_closed_classes.tsv"
NLP_CORPUS     = BASE_DIR / "shanparsing/ShanNLP-main/shannlp/corpus"
OUT_ANNOT      = BASE_DIR / "annotation/khong_annotation.tsv"

PARQUETS = {
    "shan_news":   BASE_DIR / "corpus/shan_news.parquet",
    "tai_freedom": BASE_DIR / "corpus/tai_freedom.parquet",
}

YEAR_MIN    = 2019
YEAR_MAX    = 2024
N_ARTICLES  = 240
SEED        = 42

TARGET   = "ၶွင်"
DIGIT_RE = re.compile(r"[0-9]+(?:[\/:.\-][0-9]+)*")
LATIN_RE = re.compile(r"[A-Za-z]+")
THAI_RE  = re.compile(r"[\u0e00-\u0e7f]+")
SENT_SPLIT_RE = re.compile(r"။")
PHRASE_RE     = re.compile(r"\S+")

# ------------------------------------------------------------
# 1. Lexicon (used only to build the tokeniser trie)
# ------------------------------------------------------------
word_set: dict[str, set[str]] = {}

def _add(w: str, pos: str) -> None:
    w, pos = w.strip(), pos.strip().lower()
    if not w or not pos:
        return
    word_set.setdefault(w, set()).add(pos)

if SHN2ENG_FILE.exists():
    with open(SHN2ENG_FILE, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        h = next(r); wi, ti = h.index("word"), h.index("type")
        for line in r:
            if len(line) > max(wi, ti):
                _add(line[wi], line[ti])
if SHN2SHN_FILE.exists():
    with open(SHN2SHN_FILE, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        h = next(r); wi, ti = h.index("word"), h.index("type")
        for line in r:
            if len(line) > max(wi, ti):
                _add(line[wi], line[ti])
if MASTER_LEX.exists():
    for _, row in pd.read_csv(MASTER_LEX, sep="\t", dtype=str).iterrows():
        _add(row["word"], row["pos"])

for fn in ["countries_shn.txt", "person_names_female_shn.txt",
           "person_names_male_shn.txt", "shan_state_provinces.txt"]:
    fp = NLP_CORPUS / fn
    if fp.exists():
        for l in fp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                word_set.setdefault(l.strip(), {"propn"})

if TARGET not in word_set:
    word_set[TARGET] = {"particle"}

# ------------------------------------------------------------
# 2. Tokeniser with character offsets
# ------------------------------------------------------------
def _load(name: str) -> set[str]:
    fp = NLP_CORPUS / name
    return {l.strip() for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()} if fp.exists() else set()

CONS = _load("shan_consonents.txt")
VOW  = _load("shan_vowels.txt")
TONE = _load("shan_tone.txt")
MED  = {"ျ", "ြ", "ွ", "ှ"}
ASAT = {"်"}
DEP  = VOW | TONE | MED | ASAT

def is_comb(ch: str) -> bool:
    if unicodedata.category(ch)[0] != "M":
        return False
    cp = ord(ch)
    return 0x1000 <= cp <= 0x109F or 0xAA60 <= cp <= 0xAA7F or 0xA9E0 <= cp <= 0xA9FF

trie: dict = {}
for w in word_set:
    n = trie
    for c in w:
        n = n.setdefault(c, {})
    n[None] = True

def tokenize_phrase_with_offsets(phrase: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    i, L = 0, len(phrase)
    while i < L:
        ch = phrase[i]
        if ch.isdigit():
            m = DIGIT_RE.match(phrase, i)
            if m:
                out.append((m.group(), m.start(), m.end())); i = m.end(); continue
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            m = LATIN_RE.match(phrase, i)
            if m:
                out.append((m.group(), m.start(), m.end())); i = m.end(); continue
        if "\u0e00" <= ch <= "\u0e7f":
            m = THAI_RE.match(phrase, i)
            if m:
                out.append((m.group(), m.start(), m.end())); i = m.end(); continue
        if ch in trie:
            node = trie[ch]; longest = 0; j = i + 1
            while j < L:
                nc = phrase[j]
                if nc not in node:
                    break
                node = node[nc]
                if None in node:
                    longest = j - i + 1
                j += 1
            if longest:
                out.append((phrase[i:i + longest], i, i + longest))
                i += longest; continue
        if ch in CONS:
            s = i; i += 1
            while i < L and (phrase[i] in DEP or is_comb(phrase[i])):
                i += 1
            out.append((phrase[s:i], s, i)); continue
        out.append((ch, i, i + 1)); i += 1

    merged: list[tuple[str, int, int]] = []
    for tok, s, e in out:
        if merged and all(unicodedata.category(c)[0] == "M" for c in tok):
            pt, ps, _ = merged[-1]
            merged[-1] = (pt + tok, ps, e)
        else:
            merged.append((tok, s, e))
    return merged

def tokenize_sentence_with_offsets(sent: str) -> list[tuple[str, int, int]]:
    out = []
    for m in PHRASE_RE.finditer(sent):
        poff = m.start()
        for tok, s, e in tokenize_phrase_with_offsets(m.group()):
            out.append((tok, poff + s, poff + e))
    return out

# ------------------------------------------------------------
# 3. Frame: hit-bearing articles, 2019-2024
# ------------------------------------------------------------
census = pd.read_csv(CENSUS_FILE, sep="\t", dtype={"year": "Int64"})
print(f"Census: {len(census)} hit-bearing articles")

window = census[(census["year"] >= YEAR_MIN) & (census["year"] <= YEAR_MAX)].copy()
print(f"In {YEAR_MIN}-{YEAR_MAX}: {len(window)} hit-bearing articles")
if window.empty:
    raise SystemExit(f"No hit-bearing articles in {YEAR_MIN}-{YEAR_MAX}.")

print("\nWindow composition (descriptive):")
for s, n in window.groupby("source").size().items():
    print(f"  source {s:15s} {n:5d}")
for y, n in window.groupby("year").size().sort_index().items():
    print(f"  year   {int(y):<10d} {n:5d}")

n_sample = min(N_ARTICLES, len(window))
rng = random.Random(SEED)
chosen = window.sample(n=n_sample, random_state=rng.randint(0, 10**9))
print(f"\nSampled {n_sample} articles (uniform over window, seed={SEED}).")

# ------------------------------------------------------------
# 4. Per-article hit extraction
# ------------------------------------------------------------
def extract_hits_from_article(content: str, article_id: str,
                              year: int, article_hit_count: int,
                              source: str) -> list[dict]:
    hits: list[dict] = []
    for sent_id, raw in enumerate(SENT_SPLIT_RE.split(content)):
        raw = raw.strip()
        if not raw or TARGET not in raw:
            continue
        toks = tokenize_sentence_with_offsets(raw)
        for k, (tok, cs, ce) in enumerate(toks):
            if tok != TARGET:
                continue
            hits.append({
                "sentence_id": sent_id,
                "token_index": k,
                "sentence_text": raw,
                "char_start": cs,
                "char_end": ce,
                "left1":  toks[k - 1][0] if k >= 1 else "",
                "right1": toks[k + 1][0] if k + 1 < len(toks) else "",
            })

    if len(hits) != article_hit_count:
        print(f"  WARN: {article_id}: re-tokenised {len(hits)} "
              f"vs census {article_hit_count}")
    if not hits:
        return []

    k_select = 1 if article_hit_count <= 2 else 2
    k_select = min(k_select, len(hits))
    art_rng = random.Random(f"{SEED}_{article_id}")
    selected = art_rng.sample(hits, k_select)

    return [{
        "article_id":        article_id,
        "source":            source,
        "year":              int(year),
        "article_hit_count": article_hit_count,
        "sentence_id":       h["sentence_id"],
        "token_index":       h["token_index"],
        "sentence_text":     h["sentence_text"],
        "char_start":        h["char_start"],
        "char_end":          h["char_end"],
        "left1":             h["left1"],
        "right1":            h["right1"],
        "LABEL":             "",
        "COMPLEMENT_TYPE":   "",
        "NOTES":             "",
    } for h in selected]

# ------------------------------------------------------------
# 5. Load parquets once each, iterate selected articles
# ------------------------------------------------------------
parquet_cache: dict[str, pd.DataFrame] = {}
for source in chosen["source"].unique():
    path = PARQUETS.get(source)
    if not path or not path.exists():
        print(f"  skip {source}: {path} not found"); continue
    parquet_cache[source] = pd.read_parquet(path)
    print(f"  loaded {source}: {len(parquet_cache[source])} articles")

all_rows: list[dict] = []
for _, row in tqdm(chosen.iterrows(), total=len(chosen), desc="extracting", unit="art"):
    source = row["source"]
    if source not in parquet_cache:
        continue
    orig_idx = int(row["original_index"])
    try:
        content = parquet_cache[source].loc[orig_idx, "content"]
    except KeyError:
        print(f"  WARN: {source}[{orig_idx}] missing"); continue
    if not isinstance(content, str) or TARGET not in content:
        continue
    all_rows.extend(extract_hits_from_article(
        content=content,
        article_id=row["article_id"],
        year=int(row["year"]),
        article_hit_count=int(row["article_hit_count"]),
        source=source,
    ))

if not all_rows:
    raise SystemExit("No rows extracted. Check census path and parquet indexing.")

all_rows.sort(key=lambda r: (r["article_id"], r["sentence_id"], r["token_index"]))
for i, r in enumerate(all_rows, 1):
    r["hit_id"] = f"H{i:05d}"

fields = ["hit_id"] + [k for k in all_rows[0] if k != "hit_id"]
with open(OUT_ANNOT, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t",
                       quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    w.writerows(all_rows)

# ------------------------------------------------------------
# 6. Report
# ------------------------------------------------------------
print(f"\n==== SAMPLE ====")
print(f"Rows written: {len(all_rows)}")
print(f"Distinct articles: {len({r['article_id'] for r in all_rows})}")
print("By source:", dict(Counter(r["source"] for r in all_rows)))
print("By year:  ", dict(sorted(Counter(r["year"] for r in all_rows).items())))

long_sents = sum(1 for r in all_rows
                 if len(r["sentence_text"].split()) > 250)
if long_sents:
    print(f"\nNote: {long_sents} hits sit in sentences >250 tokens "
          f"(check truncation in the RoBERTa step).")

print(f"\nAnnotation sheet -> {OUT_ANNOT}")
print("LABEL:          poss | nonposs | ambig")
print("COMPLEMENT_TYPE: pron | full_np | other | NA   (NA if not poss)")
print("NOTES:           free text, single line")