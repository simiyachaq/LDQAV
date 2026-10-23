#!/usr/bin/env python3
"""
Step 0 - Census.
Count occurrences of ၶွင် per article in the news parquets.
One row per hit-bearing article. Short report on stdout, including a
yearly census of all articles in the frame (denominator) alongside the
hit-bearing count and free-token count.

Output: annotation/khong_census.tsv

Trie uses None as the word-terminator key (not "#"), so a literal
'#' in the source text cannot collide with the sentinel.
"""
import csv
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
CATALOGUE_FILE = BASE_DIR / "annotation/catalogue/document_catalogue.tsv"
SHN2ENG_FILE   = BASE_DIR / "shanparsing/dictionary_tsv_clean/shn2eng.tsv"
SHN2SHN_FILE   = BASE_DIR / "shanparsing/dictionary_tsv_clean/shn2shn.tsv"
MASTER_LEX     = BASE_DIR / "master_closed_classes.tsv"
NLP_CORPUS     = BASE_DIR / "shanparsing/ShanNLP-main/shannlp/corpus"
OUT_CENSUS     = BASE_DIR / "annotation/catalogue/khong_census.tsv"

PARQUETS = {
    "shan_news":      BASE_DIR / "corpus/shan_news.parquet",
    "tai_freedom":    BASE_DIR / "corpus/tai_freedom.parquet",
    "shan_blogspots": BASE_DIR / "corpus/shan_blogspots.parquet",
    "shan_novel":     BASE_DIR / "corpus/shan_novel-tainovel.parquet",
}
NEWS_SOURCES = ["shan_news", "tai_freedom"]

TARGET   = "ၶွင်"
DIGIT_RE = re.compile(r"[0-9]+(?:[\/:.\-][0-9]+)*")
LATIN_RE = re.compile(r"[A-Za-z]+")
THAI_RE  = re.compile(r"[\u0e00-\u0e7f]+")

# ------------------------------------------------------------
# 1. Lexicon
# ------------------------------------------------------------
word_set: dict[str, set[str]] = {}
word_list: dict[str, list[str]] = {}

def _add(w: str, pos: str) -> None:
    w, pos = w.strip(), pos.strip().lower()
    if not w or not pos:
        return
    if w not in word_set:
        word_set[w] = set()
        word_list[w] = []
    if pos not in word_set[w]:
        word_list[w].append(pos)
        word_set[w].add(pos)

if SHN2ENG_FILE.exists():
    with open(SHN2ENG_FILE, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        h = next(r)
        wi, ti = h.index("word"), h.index("type")
        for line in r:
            if len(line) > max(wi, ti):
                _add(line[wi], line[ti])
if SHN2SHN_FILE.exists():
    with open(SHN2SHN_FILE, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        h = next(r)
        wi, ti = h.index("word"), h.index("type")
        for line in r:
            if len(line) > max(wi, ti):
                _add(line[wi], line[ti])
if MASTER_LEX.exists():
    for _, row in pd.read_csv(MASTER_LEX, sep="\t", dtype=str).iterrows():
        _add(row["word"], row["pos"])

proper: set[str] = set()
for fn in ["countries_shn.txt", "person_names_female_shn.txt",
           "person_names_male_shn.txt", "shan_state_provinces.txt"]:
    fp = NLP_CORPUS / fn
    if fp.exists():
        proper |= {l.strip() for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()}
for w in proper:
    word_set.setdefault(w, {"propn"})
    word_list.setdefault(w, ["propn"])

if TARGET not in word_set:
    print(f"WARNING: {TARGET!r} not in lexicon; adding so it tokenises as one unit.")
    word_set[TARGET] = {"particle"}
    word_list[TARGET] = ["particle"]
print(f"Lexicon: {len(word_set)} words. {TARGET!r} POS = {word_list.get(TARGET)}")

# ------------------------------------------------------------
# 2. Tokeniser (no offsets; census only needs tokens)
#    Every regex branch is guarded by `if m:`; the trie uses None
#    as the terminator, so '#' in text cannot collide with it.
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
    n[None] = True   # sentinel: None cannot appear in a string

def tokenize(phrase: str) -> list[str]:
    out, i, L = [], 0, len(phrase)
    while i < L:
        ch = phrase[i]

        if ch.isdigit():
            m = DIGIT_RE.match(phrase, i)
            if m:
                out.append(m.group()); i = m.end(); continue

        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            m = LATIN_RE.match(phrase, i)
            if m:
                out.append(m.group()); i = m.end(); continue

        if "\u0e00" <= ch <= "\u0e7f":
            m = THAI_RE.match(phrase, i)
            if m:
                out.append(m.group()); i = m.end(); continue

        if ch in trie:
            node = trie[ch]
            longest = 0
            j = i + 1
            while j < L:
                nc = phrase[j]
                if nc not in node:
                    break
                node = node[nc]
                if None in node:
                    longest = j - i + 1
                j += 1
            if longest:
                out.append(phrase[i:i + longest]); i += longest; continue

        if ch in CONS:
            s = i; i += 1
            while i < L and (phrase[i] in DEP or is_comb(phrase[i])):
                i += 1
            out.append(phrase[s:i]); continue

        out.append(ch); i += 1

    merged = []
    for t in out:
        if merged and all(unicodedata.category(c)[0] == "M" for c in t):
            merged[-1] += t
        else:
            merged.append(t)
    return merged

# ------------------------------------------------------------
# 3. Catalogue metadata
# ------------------------------------------------------------
meta: dict[tuple[str, int], dict] = {}
if CATALOGUE_FILE.exists():
    cat = pd.read_csv(CATALOGUE_FILE, sep="\t", dtype=str)
    for _, row in cat.iterrows():
        try:
            key = (row["source"], int(row["original_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        year = pd.to_numeric(str(row.get("date", ""))[:4], errors="coerce")
        meta[key] = {
            "article_id": row.get("article_id", f"{row['source']}_{row['original_index']}"),
            "year": int(year) if pd.notna(year) else "",
        }
    print(f"Catalogue: {len(meta)} entries")
else:
    print("No catalogue; article_id will be constructed; year will be empty")

# ------------------------------------------------------------
# 4. Iterate parquets; count TARGET per article; census by year
# ------------------------------------------------------------
rows: list[dict] = []
year_total:    Counter = Counter()   # every scanned article, by year
year_hit_arts: Counter = Counter()   # hit-bearing articles, by year
year_hit_toks: Counter = Counter()   # free-token count, by year

scanned = 0
for source in NEWS_SOURCES:
    path = PARQUETS.get(source)
    if not path or not path.exists():
        print(f"skip {source}: {path} not found"); continue
    df = pd.read_parquet(path)
    if "content" not in df.columns:
        print(f"skip {source}: no 'content' column"); continue
    print(f"scanning {source} ({len(df)} articles)...")
    for orig_idx, content in tqdm(df["content"].items(), desc=source, unit="art"):
        scanned += 1

        # Year lookup runs for every article, hit-bearing or not.
        m = meta.get((source, orig_idx), {})
        y = m.get("year", "")
        ykey = y if y != "" else "unknown"
        year_total[ykey] += 1

        if not isinstance(content, str) or TARGET not in content:
            continue
        substr_n = content.count(TARGET)
        tok_n = 0
        for phrase in content.split():
            for tok in tokenize(phrase):
                if tok == TARGET:
                    tok_n += 1
        if tok_n == 0:
            continue

        year_hit_arts[ykey] += 1
        year_hit_toks[ykey] += tok_n

        rows.append({
            "article_id":        m.get("article_id", f"{source}_{orig_idx}"),
            "source":            source,
            "original_index":    orig_idx,
            "year":              y,
            "in_catalogue":      int((source, orig_idx) in meta),
            "article_hit_count": tok_n,
            "substring_count":   substr_n,
        })

if not rows:
    raise SystemExit(f"No articles contain {TARGET!r}. Check lexicon and parquets.")

out = pd.DataFrame(rows).sort_values("article_hit_count", ascending=False)
out.to_csv(OUT_CENSUS, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

# ------------------------------------------------------------
# 5. Report
# ------------------------------------------------------------
total_hits = int(out["article_hit_count"].sum())
total_arts = int(len(out))
print("\n==== CENSUS REPORT ====")
print(f"Articles scanned:              {scanned}")
print(f"Hit-bearing articles:          {total_arts} ({total_arts / scanned * 100:.1f}%)")
print(f"Total ၶွင် tokens:             {total_hits}")
print(f"Substring occurrences:         {int(out['substring_count'].sum())}")
print(f"  (gap = compounds / non-token occurrences)")
print(f"Articles in catalogue:         {int(out['in_catalogue'].sum())} / {total_arts}")

print("\nBy source:")
for s, sub in out.groupby("source"):
    print(f"  {s:15s}  articles={len(sub):5d}  hits={int(sub['article_hit_count'].sum()):6d}"
          f"  max={sub['article_hit_count'].max():3d}")

print("\nHits per article (hit-bearing articles only):")
d = out["article_hit_count"]
print(f"  median={int(d.median())}  mean={d.mean():.2f}  p90={int(d.quantile(0.9))}  max={int(d.max())}")

# ---- yearly census: all articles vs hit-bearing vs free tokens ----
def _year_key(y):
    return (0, int(y)) if str(y).isdigit() else (1, 0)

years_sorted = sorted(year_total.keys(), key=_year_key)
if years_sorted:
    print("\nYearly census (all articles in frame):")
    print(f"  {'year':6s} {'total':>7s} {'hit-art':>8s} {'%':>6s} {'tokens':>8s} {'tok/art':>8s}")
    for y in years_sorted:
        total  = year_total[y]
        n_arts = year_hit_arts.get(y, 0)
        n_toks = year_hit_toks.get(y, 0)
        pct    = (n_arts / total * 100) if total else 0.0
        tokart = (n_toks / total) if total else 0.0
        label  = str(y) if str(y) != "unknown" else "?"
        print(f"  {label:6s} {total:7d} {n_arts:8d} {pct:6.1f} {n_toks:8d} {tokart:8.2f}")

    t_total = sum(year_total.values())
    t_arts  = sum(year_hit_arts.values())
    t_toks  = sum(year_hit_toks.values())
    pct_all = (t_arts / t_total * 100) if t_total else 0.0
    print(f"  {'TOTAL':6s} {t_total:7d} {t_arts:8d} {pct_all:6.1f} {t_toks:8d} {t_toks / t_total:8.2f}")

    # per-source yearly census (confound check: source mix varies by year)
    print("\nYearly census by source:")
    for src, sub in out.groupby("source"):
        src_year = sub.groupby("year") if "year" in sub else None
        # denominator per (source, year) needs the full scan, not just hit-bearing;
        # recompute inline from year_total is impossible for per-source,
        # so derive the hit-bearing / token side from `out` and note that
        # per-source denominators are not tracked in this pass.
        print(f"  [{src}]")
        agg = sub.groupby("year").agg(
            hit_arts=("article_id", "count"),
            tokens=("article_hit_count", "sum"),
        ).sort_index()
        for y, row in agg.iterrows():
            label = str(y) if str(y) != "" else "?"
            print(f"    {label:6s} hit-art={int(row['hit_arts']):5d}  tokens={int(row['tokens']):6d}")
else:
    print("\nYear distribution: unavailable (no catalogue).")

print(f"\nCensus written -> {OUT_CENSUS}")