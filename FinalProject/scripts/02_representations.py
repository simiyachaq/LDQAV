#!/usr/bin/env python3
"""
02_representations.py

Build TF-IDF and RoBERTa contextual representations for each annotated
ၶွင် (khong) hit.

Input:  annotation/khong_annotation.tsv    (LABEL must be filled)
        modelo_entrenado_word/             (epoch 28 checkpoint)
        shan-word-tokenizer/               (word-level tokenizer)

Output: data/X_lex.npz            sparse TF-IDF matrix (N x V), scipy format
        data/X_ctx.npy            dense contextual matrix (N x 384)
        data/occurrences.tsv      metadata aligned by row index
        data/tfidf_vocabulary.tsv vocabulary inspectable
"""
import csv
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import PreTrainedTokenizerFast, RobertaModel, logging as hf_logging
from tqdm import tqdm

hf_logging.set_verbosity_error()

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR       = Path("/home/simiyachaq/Documentos/Programming_Stuff/Shan")
ANNOT_FILE     = BASE_DIR / "annotation/khong_annotation.tsv"
MODEL_DIR      = BASE_DIR / "modelo_entrenado_word"
TOKENIZER_DIR  = BASE_DIR / "shan-word-tokenizer"
SHN2ENG_FILE   = BASE_DIR / "shanparsing/dictionary_tsv_clean/shn2eng.tsv"
SHN2SHN_FILE   = BASE_DIR / "shanparsing/dictionary_tsv_clean/shn2shn.tsv"
MASTER_LEX     = BASE_DIR / "master_closed_classes.tsv"
NLP_CORPUS     = BASE_DIR / "shanparsing/ShanNLP-main/shannlp/corpus"
OUT_DIR        = BASE_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW         = 5
MAX_LEN        = 384
EXPECTED_VOCAB = 12529
TARGET         = "ၶွင်"
VALID_LABELS   = {"poss", "nonposs", "ambig"}

DIGIT_RE  = re.compile(r"[0-9]+(?:[\/:.\-][0-9]+)*")
LATIN_RE  = re.compile(r"[A-Za-z]+")
THAI_RE   = re.compile(r"[\u0e00-\u0e7f]+")
PHRASE_RE = re.compile(r"\S+")

# ------------------------------------------------------------
# 1. Base tokeniser (same as 01_sampler.py)
# ------------------------------------------------------------
word_set: dict[str, set[str]] = {}

def _add(w: str, pos: str) -> None:
    w = w.strip()
    if w:
        word_set.setdefault(w, set()).add(pos.strip().lower())

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
word_set.setdefault(TARGET, {"particle"})

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

def _tok_phrase(phrase: str) -> list[tuple[str, int, int]]:
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
                if nc not in node: break
                node = node[nc]
                if None in node: longest = j - i + 1
                j += 1
            if longest:
                out.append((phrase[i:i+longest], i, i+longest)); i += longest; continue
        if ch in CONS:
            s = i; i += 1
            while i < L and (phrase[i] in DEP or is_comb(phrase[i])): i += 1
            out.append((phrase[s:i], s, i)); continue
        out.append((ch, i, i+1)); i += 1
    merged = []
    for tok, s, e in out:
        if merged and all(unicodedata.category(c)[0] == "M" for c in tok):
            pt, ps, _ = merged[-1]
            merged[-1] = (pt + tok, ps, e)
        else:
            merged.append((tok, s, e))
    return merged

def tokenize_sentence(sent: str) -> list[tuple[str, int, int]]:
    out = []
    for m in PHRASE_RE.finditer(sent):
        poff = m.start()
        for tok, s, e in _tok_phrase(m.group()):
            out.append((tok, poff + s, poff + e))
    return out

# ------------------------------------------------------------
# 2. Load annotation
# ------------------------------------------------------------
if not ANNOT_FILE.exists():
    raise FileNotFoundError(f"Annotation file not found: {ANNOT_FILE}")

df = pd.read_csv(ANNOT_FILE, sep="\t", dtype=str)
df["LABEL"] = df["LABEL"].fillna("").str.strip()
df = df[df["LABEL"].isin(VALID_LABELS)].reset_index(drop=True)
print(f"Coded rows: {len(df)}")
print(f"Label distribution: {df['LABEL'].value_counts().to_dict()}")
if len(df) == 0:
    raise SystemExit("No coded rows. Fill LABEL with poss / nonposs / ambig.")

# ------------------------------------------------------------
# 3. Tokenise, build windows, locate target
# ------------------------------------------------------------
window_docs:      list[str] = []
segmented_sents:  list[str] = []
target_positions: list[int] = []
prefix_lens:      list[int] = []
meta_rows:        list[dict] = []
skipped:          list[str] = []
long_sentences = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="tokenising", unit="hit"):
    sent = row["sentence_text"]
    cs, ce = int(row["char_start"]), int(row["char_end"])
    toks = tokenize_sentence(sent)

    target_idx = None
    for k, (tok, s, e) in enumerate(toks):
        if s <= cs and e >= ce:
            target_idx = k; break
    if target_idx is None:
        skipped.append(f"{row['hit_id']}: span not found")
        continue
    if toks[target_idx][0] != TARGET:
        skipped.append(f"{row['hit_id']}: token '{toks[target_idx][0]}' != {TARGET}")
        continue

    left  = [t for t, _, _ in toks[max(0, target_idx - WINDOW):target_idx]]
    right = [t for t, _, _ in toks[target_idx + 1:target_idx + 1 + WINDOW]]
    window_docs.append(" ".join(left + right))

    words = [t for t, _, _ in toks]
    if len(words) > MAX_LEN - 2:
        long_sentences += 1
    segmented_sents.append(" ".join(words))
    prefix_lens.append(sum(len(w) + 1 for w in words[:target_idx]))
    target_positions.append(target_idx)

    meta_rows.append({
        "hit_id":           row["hit_id"],
        "article_id":       row["article_id"],
        "source":           row["source"],
        "year":             row["year"],
        "LABEL":            row["LABEL"],
        "COMPLEMENT_TYPE":  row.get("COMPLEMENT_TYPE", ""),
        "sentence_id":      row["sentence_id"],
        "target_token_idx": target_idx,
    })

if skipped:
    print(f"\nSkipped {len(skipped)} rows:")
    for s in skipped[:10]:
        print("  ", s)
    print()

print(f"Usable rows: {len(meta_rows)}")
if long_sentences:
    print(f"Sentences longer than MAX_LEN-2 ({MAX_LEN-2} words): {long_sentences}")

if len(meta_rows) == 0:
    raise SystemExit("No usable rows after tokenisation.")

# ------------------------------------------------------------
# 4. TF-IDF
# ------------------------------------------------------------
print("\nFitting TF-IDF ...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    norm="l2",
    token_pattern=r"\S+",
)
X_lex = vectorizer.fit_transform(window_docs)
print(f"  X_lex: {X_lex.shape}   vocabulary {len(vectorizer.vocabulary_)}")
if len(vectorizer.vocabulary_) < 50:
    print("  WARNING: vocabulary very small; consider min_df=1")

# ------------------------------------------------------------
# 5. RoBERTa
# ------------------------------------------------------------
print("\nLoading tokenizer and model ...")
tok = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
if tok.vocab_size != EXPECTED_VOCAB:
    print(f"  WARNING: vocab size {tok.vocab_size}, expected {EXPECTED_VOCAB}")

model = RobertaModel.from_pretrained(str(MODEL_DIR))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"  device: {device}, hidden size: {model.config.hidden_size}")

hidden_dim = model.config.hidden_size
X_ctx = np.zeros((len(segmented_sents), hidden_dim), dtype=np.float32)
kept  = np.ones(len(segmented_sents), dtype=bool)

first_check_done = False

with torch.no_grad():
    for i, (sents, pfx) in enumerate(tqdm(
            zip(segmented_sents, prefix_lens),
            total=len(segmented_sents), desc="roberta", unit="hit")):
        enc = tok(sents, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt", add_special_tokens=True,
                  return_offsets_mapping=True)
        ids = enc["input_ids"].to(device)
        offsets = enc["offset_mapping"][0].tolist()

        # locate the RoBERTa token whose char span contains the target
        tpos = None
        for k, (s, e) in enumerate(offsets):
            if s <= pfx < e:
                tpos = k
                break
        if tpos is None:
            kept[i] = False
            continue

        out = model(input_ids=ids,
                    attention_mask=enc["attention_mask"].to(device))
        vec = out.last_hidden_state[0, tpos, :].cpu().numpy()

        if not first_check_done:
            decoded = tok.convert_ids_to_tokens(ids[0].tolist())
            print(f"\n  target check: token at {tpos} = {decoded[tpos]!r} "
                  f"(expected {TARGET!r})")
            first_check_done = True

        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        X_ctx[i] = vec

if not kept.all():
    print(f"\nDropped {int((~kept).sum())} rows truncated past target")
    X_ctx = X_ctx[kept]
    meta_rows = [m for m, k in zip(meta_rows, kept) if k]
    X_lex = X_lex[kept]

# ------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------
save_npz(OUT_DIR / "X_lex.npz", X_lex)
np.save(OUT_DIR / "X_ctx.npy", X_ctx)

vocab = sorted(vectorizer.vocabulary_.items(), key=lambda kv: kv[1])
pd.DataFrame(vocab, columns=["term", "col"]).to_csv(
    OUT_DIR / "tfidf_vocabulary.tsv", sep="\t", index=False)

meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(OUT_DIR / "occurrences.tsv", sep="\t", index=False)

print(f"\nSaved:")
print(f"  {OUT_DIR/'X_lex.npz'}   {X_lex.shape}")
print(f"  {OUT_DIR/'X_ctx.npy'}   {X_ctx.shape}")
print(f"  {OUT_DIR/'occurrences.tsv'}   {len(meta_df)} rows")
print(f"  {OUT_DIR/'tfidf_vocabulary.tsv'}   {len(vocab)} terms")

print("\nLabel distribution in final matrices:")
print(meta_df["LABEL"].value_counts().to_dict())
norms = np.linalg.norm(X_ctx, axis=1)
print(f"X_ctx row norms: min={norms.min():.4f}  max={norms.max():.4f}  mean={norms.mean():.4f}")