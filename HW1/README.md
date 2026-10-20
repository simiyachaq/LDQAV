# LDQAV — Homework 1

## Lemmatization and network analysis of a novel

### The novel

*Moby Dick; Or, The Whale*, Herman Melville. Project Gutenberg eBook #2701, <https://www.gutenberg.org/ebooks/2701>. English.

### Layout

```
HW1/
├── data/raw/pg2701.txt
├── data/clean/moby_dick.txt
├── scripts/
│   ├── 00_clean_text.py
│   ├── 01_lemmatize.py
│   ├── 02_dependencies.py
│   ├── 03_peek.py          # diagnostic: prints whale child-edge counts
│   ├── 04_extract.py
│   ├── 05_visualize.py
│   └── 06_analyze.py
└── output/
    ├── moby_lemmatized.txt
    ├── moby_tokens.csv
    ├── edges_primary.csv
    ├── edges_secondary.csv
    ├── graph_whale.graphml
    ├── graph_whale.png
    ├── graph_whale_communities.png
    └── community_whale.csv
```

### Tooling

The course covers `pymystem3`, `pymorphy3`, and NLTK for morphology; spaCy for tokenization, POS tagging, lemmatization, dependency parsing, and NER; `networkx` for graphs, density, centrality, clustering, and community detection; and `pandas` and `matplotlib` for tables and figures.

The pipeline uses these as taught. One tool is not from the course:

- **Louvain** (`networkx.community.louvain_communities`) for community detection. The course notebook demonstrates `greedy_modularity_communities` for the same task; Louvain is used instead because it gives a substantially better partition on this graph. The comparison is reported in Step 5.

## Step 1 — Cleaning

`00_clean_text.py` removes the Gutenberg header and footer, the front matter (`ETYMOLOGY`, `EXTRACTS`, `CONTENTS`), and `[Illustration: …]` markers. It normalizes line endings and collapses runs of blank lines. Unicode punctuation is preserved.

Result: **1,191,640 characters**, **208,458 words**, from `CHAPTER 1. Loomings.` to the Epilogue.

## Step 2 — Lemmatization

`01_lemmatize.py` parses the cleaned text with spaCy (`en_core_web_sm`) and writes one lemmatized sentence per line to `output/moby_lemmatized.txt`, with punctuation and whitespace removed.

Forms collapse as expected: `me, I → I`; `thought → think`; `is → be`. Modals (`would`, `could`) are not mapped by spaCy and remain as they are.

**Implementation note.** Step 3 parses the cleaned text a second time rather than reading `moby_lemmatized.txt`, because the dependency parser needs token boundaries and sentence segmentation that the lemmatized file no longer carries. The same model and the same input text are used, so the lemmas are identical; the re-parse is a technical necessity, not a second lemmatization.

## Step 3 — Center word, children, and secondary dependencies

`02_dependencies.py` parses the cleaned text with the dependency parser enabled and writes `output/moby_tokens.csv`, one row per token: `sent_id`, `i`, `token`, `lemma`, `pos`, `dep`, `head_i`. A token’s head is the token in the same sentence whose `i` equals its `head_i`.

The temptation was too big (it's *Moby Dick*, after all), so the center word I chose is **whale**, the most frequent content lemma in the novel: **1,427 occurrences**.

`04_extract.py` writes two edge lists.

- **`edges_primary.csv`** — every token whose head is an occurrence of *whale*: the direct dependents. **441 unique, 1,037 total.**
- **`edges_secondary.csv`** — every token whose head is one of those direct dependents: the grandchildren. **366 unique, 379 total.**

Self-loops and non-alphabetic characters (punctuation, numerals, symbols) are removed at extraction. Nothing else is removed: function words, hapaxes, and low-frequency edges all remain. **The CSVs are exhaustive.**

This was done in order to fulfill the literal requirements of the assignment: *all* dependencies and *all* their dependencies. The CSVs satisfy that requirement. The graph in Step 4 does not — please see the note there.

## Step 4 — Graph construction

`05_visualize.py` builds a **directed** graph (`networkx.DiGraph`) centered on *whale*: ring 1 holds direct dependents, ring 2 holds dependents of dependents.

**Why directed.** Dependency is a directional relation: `whale → sperm` means “sperm whale,” not “whale sperm.” Direction is preserved in `graph_whale.graphml`.

**Filters, and where I went rogue.** The rubric asks the graph to show all dependencies and all their dependents. The visualized graph does not: it applies two filters, because the unfiltered two-hop neighborhood is too dense to read and contains many one-off edges.

1. Primary edges are kept only if the relation occurs at least **3** times.
2. Secondary edges are kept only if the dependent endpoint is a content lemma: `NOUN`, `PROPN`, `VERB`, `ADJ`, `ADV`.

Result: **111 nodes, 124 edges.**

This is a deliberate departure from the literal instruction. The exhaustive data remain in the CSVs, and the filters are one-line switches in the script; the unfiltered graph can be regenerated at any time (It's not pretty, though). The analyzed object in Step 5 is the filtered graph, and the numbers below describe it, not the full extraction.

**Visual parameters.**

| Parameter  | Encoding                    | Justification                                                                                                                          |
| ---------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Node size  | `300 + 400 · degree^0.7`    | The exponent < 1 compresses the range, so the hub (*whale*, degree 51) is clearly larger than a leaf without making leaves unreadable. |
| Node color | part of speech              | Shows at a glance how adjectives, nouns, verbs, and prepositions are distributed around *whale*.                                       |
| Edge width | co-occurrence weight        | Strong associations (`whale → sperm`, 181) dominate visually; hapaxes stay thin.                                                       |
| Layout     | `spring_layout`, fixed seed | Force-directed placement puts strongly connected nodes near each other; the fixed seed makes the figure reproducible.                  |

Saved as `graph_whale.png` and `graph_whale.graphml`.

## Step 5 — Network analysis

`06_analyze.py` computes density, clustering, degree, betweenness, and Louvain communities.

### Density

- Directed: **0.0102** (124 / (111 · 110))
- Undirected projection: **0.0203** (124 / 6,105)

Both are reported because the saved version is directed while later measures are conventionally undirected. For scale: a spanning tree on 111 nodes has density ≈ 0.018, and this graph has only 14 edges more than a tree. Density here functions mainly as a check that the extraction produced the tree-like object a dependency parse should produce. It is not, by itself, a finding.

### Clustering

Average clustering coefficient (undirected projection): **0.0888**. A pure star gives 0. A value this small but non-zero indicates a star with a few local triangles, consistent with prepositional branches.

### Degree

| Node  | Degree |
| ----- | -----: |
| whale |     51 |
| of    |     17 |
| in    |     15 |
| with  |      7 |
| sperm |      6 |
| head  |      6 |
| on    |      6 |

*Whale*’s dominance is circular — it is the center by construction. The informative part is the ranking below it: the prepositions outrank every content word except *sperm* and *head*.

### Betweenness

Computed on the undirected projection, to match the community detection below and because the pure-star baseline of 1.0 applies only to undirected graphs.

| Node  | Betweenness |
| ----- | ----------: |
| whale |      0.9144 |
| of    |      0.2383 |
| in    |      0.1826 |
| with  |      0.0892 |
| on    |      0.0892 |
| head  |      0.0545 |

In a pure star the center’s betweenness is exactly **1.0**. The value 0.9144 is therefore best read as a **deficit**: roughly 9% of shortest paths bypass *whale*, and the nodes that carry them are the four prepositions. That is the informative result — degree cannot distinguish a hub with many leaves from a hub that bridges separate regions; betweenness can.

### Communities (Louvain)

Louvain is run on the undirected projection, because modularity-based community detection is defined on undirected graphs. The directed graph on disk is unchanged.

Result: **12 communities, modularity 0.1848.**

| Community |     Size | Character                                                                              |
| --------- | -------: | -------------------------------------------------------------------------------------- |
| c00       |       39 | direct modifiers of *whale*: `great`, `dead`, `greenland`, `grinning`, `blasted`, …    |
| c01       |       19 | `of`-frame: `magnitude`, `order`, `course`, `bigness`, `dish`, `day`, …                |
| c02       |       18 | `in`-frame: `ocean`, `mediterranean`, `chapter`, `attitude`, `expression`, `length`, … |
| c03       |       10 | `with`-frame: `head`, `baleen`, `brow`, `hump`, `herd`, …                              |
| c04       |        6 | `on`-frame: `banks`, `coast`, `larboard`, `outskirt`, `side`                           |
| 7 others  | 2–4 each | adverb-of-adjective branches (`large → very`); not analyzed                            |

Two points about this table.

1. **c00 is not prepositional.** The only secondary (ring-2) structure comes from the four prepositions. c00 is a ring-1 community: Louvain groups direct modifiers of *whale* together even though they do not connect to each other, because they sit at distance 2 via the center.
2. **The frame names are my interpretation.** Louvain clustered edges, not meanings. That the clusters align with prepositions is an unsupervised result; that `with` is “anatomy” and `on` is “geography” is a gloss on the member lists, added by me, and should be read as such.

Modularity 0.18 is low. It is reported as measured, not tuned: no adjustment of resolution, thresholds, or weights was made to raise it. But “low” here means low in absolute terms; no null-model baseline is computed, so the number is descriptive rather than inferential.

**Comparison with greedy modularity.** The course notebook demonstrates `networkx.community.greedy_modularity_communities`, which is also a modularity-optimization algorithm but uses a different local search. Run on the same undirected projection, it returns 14 communities at Q=0.0275. The four largest non-trivial communities still align with the prepositions (`of` = c02, `in` = c01, `with` = c03, `on` = c05), but Louvain's largest community (c00, direct modifiers of *whale*, 39 nodes) is split into two fragments and a new grab-bag community appears that mixes *sperm*, *great*, *live*, *portly*, *who*, *by*. Louvain was chosen for the reported partition because it recovers the same four prepositional frames without fragmenting the modifier cluster, and because greedy modularity lands in a weaker local optimum on this graph. The disagreement is itself informative: on a near-star graph, modularity optimization is fragile and the partition depends on the optimizer, so modularity 0.18 is best read as descriptive of the graph's shape, not as evidence of strong community structure.

## Step 6 — Conclusion

The neighborhood of *whale* is a star: it attaches directly to 51 of the graph’s 110 other nodes, and its betweenness of 0.91 — against the 1.0 a pure star would give — locates the only real secondary structure in the ~9% of shortest paths that bypass it, carried by `of`, `in`, `with`, and `on`. Louvain recovers 12 communities at modularity 0.18; the one partition worth reporting is the split between direct adnominal modifiers of *whale* (c00) and prepositional complement frames (c01–c04), a standard distinction inside a noun phrase. Running the same graph through the course-notebook alternative (greedy modularity) yields 14 communities at Q=0.0275 and splits the modifier cluster, which suggests that modularity itself, not the extraction, is the limiting factor on this near-star graph. The semantic labels for those frames are my reading of the member lists, not a result of the clustering, and the low modularity is descriptive of a near-star rather than evidence that the extraction failed.
