# Possessive in Shan News: Attestation and a Representational Check

A corpus study of the Shan possessive marker `ၶွင်` (khong), with a methodological comparison of TF-IDF and contextual (RoBERTa) representations for unsupervised recovery of the possessive / lexical-noun distinction.

*Daniel Carrasco Bascuñán*

## Summary

The descriptive literature holds that Shan expresses possession by juxtaposition alone: Cushing (1887: §61) states that "the genitive or possessive case has no distinctive sign"; Morev (1983: 95) that "the category of possessivity has not received formal material expression, unlike many Tai languages." Against this, `ၶွင်` (khong) — cognate with Thai ของ and Lao ຂອງ — occurs between full NPs in contemporary Shan news.

This project has two parts:

1. **Attestation.** A census of every free-token `ၶွင်` across 19,766 news articles
   from two Shan news outlets, plus a manually annotated sample of 149 hits
   (2019–2024) coded as possessive / non-possessive / ambiguous.
2. **Representation check.** Two representations of each hit — a TF-IDF vector over
   the ±5-token window, and a contextual vector from a pre-trained Shan RoBERTa —
   clustered with K-means at k=2 and compared against the annotation.

**Main result.** The contextual representation separates the possessive from the lexical-noun use at **ARI = 0.795** (NMI = 0.715), while TF-IDF over the same window reaches **ARI = 0.206** (NMI = 0.273). Both are above the random baseline (ARI ≈ 0).
The possessive share of the annotated hits is **58%** — against Loss (2017)'s 4% for the spoken register — with the remaining 42% being the lexical noun ('thing,  goods, property'), the source meaning from which the grammatical marker evolved.

### Model

The contextual vectors come from **Shan RoBERTa** (word-level, 6 layers, hidden 384, 15.8M parameters, best checkpoint at epoch 28; hold-out loss 4.1968). The model was pre-trained from scratch on 9.1M tokens of segmented Shan text (news, blogs, fiction, conversation) with span masking and strong regularisation.

The model and tokenizer were prepared as prior work for a separate project (functional annotation of Shan news) and are used here as tools, not as this project's contribution.

## Repository contents

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 00_census.py              count ၶွင် per article, yearly census
│   ├── 01_sampler.py             stratified sample + annotation sheet
│   ├── 02_representations.py     TF-IDF + RoBERTa vectors
│   ├── 03_cluster_eval.py        K-means / agglomerative, ARI/NMI
│   └── 04_pca.py                 PCA panels + figures
├── annotation/
│   ├── khong_annotation.tsv      149 coded hits (the ground truth)
│   ├── codebook.md               operational criteria + worked examples
│   └── catalogue/
│       ├── document_catalogue.tsv
│       └── khong_census.tsv      3,498 hit-bearing articles
├── data/
│   ├── X_lex.npz                 sparse TF-IDF matrix (149 × 278)
│   ├── X_ctx.npy                 contextual matrix (149 × 384)
│   ├── occurrences.tsv           metadata aligned by row index
│   ├── tfidf_vocabulary.tsv      278-term vocabulary
│   ├── eval_results.tsv          per-run ARI/NMI (200 rows)
│   ├── eval_summary.tsv          mean + sd across seeds (20 rows)
│   ├── eval_confusion.tsv        k=2 confusion matrices
│   └── pca_coords.tsv            2D PCA coordinates per hit
├── figures/
│   ├── pca_tfidf.png
│   ├── pca_contextual.png
│   └── pca_combined.png
└── report/
    └── report.pdf
```


## Key results

### Attestation

| metric                                           | value         |
| ------------------------------------------------ | ------------- |
| news articles scanned                            | 19,766        |
| articles containing free-token `ၶွင်`            | 3,498 (17.7%) |
| total free-token `ၶွင်` occurrences              | 6,772         |
| substring occurrences (incl. compounds)          | 10,204        |
| annotated hits (2019–2024)                       | 149           |
| of which possessive                              | 86 (58%)      |
| of which lexical noun ('thing, goods, property') | 63 (42%)      |

The share of articles containing `ၶွင်` rises from ~10% in 2017–18 to ~28% in 2023, consistent with an ongoing grammaticalisation in the news register. The full yearly census is in `annotation/catalogue/khong_census.tsv`.

### Representation check (k=2, K-means, mean over 10 seeds)

| representation           | algorithm       | ARI       | NMI       |
| ------------------------ | --------------- | --------- | --------- |
| TF-IDF (±5-token window) | K-means         | 0.206     | 0.273     |
| Contextual (RoBERTa)     | K-means         | **0.795** | **0.715** |
| —                        | random baseline | ≈ 0.000   | ≈ 0.005   |
| —                        | single cluster  | 0.000     | 0.000     |

The contextual representation separates the possessive from the lexical-noun useat ARI = 0.795 while TF-IDF over the same window reaches ARI = 0.206. The gap of **0.59** is the project's core methodological finding: the syntactic frame around `ၶွင်` carries the sense distinction, and the neural model encodes it far more strongly than a bag-of-words baseline.

Agglomerative clustering (average linkage, cosine) fails at k=2 for both representations (ARI ≈ 0) and recovers only the minority class at k≥5 — a known limitation of average linkage on small imbalanced samples, documented in the report.

### PCA

The contextual representation's first two principal components explain 25% of the variance (PC1 = 15.7%, PC2 = 9.3%); TF-IDF's explain 6.7% (PC1 = 3.8%, PC2 = 2.9%).
The PCA plots are shown for visualization only; the ARI/NMI values are computed in the full representations and are the primary evidence.

## Scope and limitations

The frame conditions on `ၶွင်`, so the study describes the marker, not the grammar of possession. Juxtaposed possession is visible in the same sentences but was not annotated; when a writer chooses `ၶွင်` over bare juxtaposition is out of scope.

News is the primary register; blogs, fiction, and conversation were not sampled. Annotation is by a single annotator, and no inter-annotator reliability figure is reported. The 2019–2024 window coincides with COVID, the 2021 coup, and the escalation of conflict in Shan State; the temporal profile is reported descriptively, with no causal claim.

The ARI result measures the pre-trained model's encoding of the construction, not its downstream generalization.

---

## References

Carrasco Bascuñán, D. (2026). *ShanRoBERTa: Word-level pre-training for Shan* [Computer software]. GitHub. https://github.com/simiyachaq/shanroberta

Cushing, J. N. (1887). *Grammar of the Shan Language*. American Baptist Mission Press. §61.

Iwasaki, S. & Ingkaphirom, P. (2005). *A Reference Grammar of Thai*. Cambridge University Press.

Kaikki.org / Wiktionary. (n.d.). Entries for ၶွင် (khong) and ၶွင်မၼ်း (khong man).

Khoe, Mawn (2015). *Tai–English Dictionary*. Mawn Khoe (Keng Tung).

Loss, D. (2017). A Comparison of Grammaticalization in Shan and Thai. *วารสารมหาวิทยาลัยพายัพ*, *26*(2), 203–216. https://doi.org/10.14456/pyuj.2016.15

Morev, L. N. (1983). *Шанский язык*. Nauka. Chapter "Синтагматика", p. 95.
