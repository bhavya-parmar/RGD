# Notes

## Why does OpenAlex use Inverted-index format for abstract?

Mainly for copyright reasons. Many abstracts are owned by publishers, so OpenAlex can't freely give out the original text. Publishing only which words appear and at which positions is generally considered safer legally, while still being enough for text mining, search and embeddings.

OpenAlex took this format over from Microsoft Academic Graph, which it replaced. Keeping the same format also kept it compatible with tools built for MAG.

Because the positions are kept, the full text can be rebuilt, which is what `reconstruct_abstract` does. Some formatting is lost, such as missing spaces after periods.

## Referenced works vs Related works

| | `referenced_works` | `related_works` |
|---|---|---|
| **What it is** | The paper's reference list (bibliography) | Similar papers picked by OpenAlex's algorithm |
| **Who decides** | The paper's authors | OpenAlex, mainly by shared topics/concepts, leaning towards recent papers |
| **Direction** | Directed: this paper cites that one | No citation link at all |
| **Size** | Varies with the reference list; can be empty if OpenAlex lacks it | Usually about 10 to 20 IDs |
| **Reliability** | Factual | Loose; often off-topic when topic tags are noisy |

## Why are we using MinHash on title + abstract when we already do deduplication on exact ID and title match?

The exact checks only catch records that are identical. MinHash catches the same paper stored as two slightly different records, which is common in OpenAlex.

**What the exact checks miss.** The same paper often shows up under different IDs, for example an arXiv preprint and the published conference version. The titles may also differ slightly:

```bash
W1: "Physics Informed Neural Networks: A Deep Learning Framework"
W2: "Physics-Informed Neural Networks: Deep Learning Framework"
```

- The IDs differ, so the ID check misses it.
- After lowercasing, the titles still differ (a hyphen and the word "A"), so the title check misses it.
- The abstracts may also differ a little between versions.

**What MinHash adds.** It splits the title plus abstract into overlapping 3-word chunks and measures how much two papers overlap (Jaccard similarity). If they are at least 80% similar, it treats them as duplicates and keeps the version with more citations and an abstract.


## MinHash + LSH: an end-to-end example

### Why shingles?

MinHash and LSH work on **sets**: they estimate the Jaccard similarity J = |A ∩ B| / |A ∪ B|. Raw text is not a set, so it has to be turned into one.

| Text as a set of... | Problem |
|---|---|
| Characters | Almost every English text shares most letters, so J is high for everything |
| Single words | Word order is lost: "dog bites man" and "man bites dog" give J = 1 |
| **k-word shingles** | Keeps local word order: two texts share a shingle only if they share a phrase of k words |

Shingles turn "these texts share many phrases" into "these sets overlap a lot", which is what near-duplicate detection needs.

### Setup

Three short texts, with the lecture's sizes: k = 12 hash functions, split into b = 4 bands of r = 3 rows.

```text
A: "Physics-informed neural networks solve partial differential equations."
B: "Physics informed neural networks solve partial differential equations fast."
C: "Graph neural networks classify nodes in citation graphs."
```

### Step 1: Shingling (3-word windows)

```text
A: {physics informed neural, informed neural networks, neural networks solve,
    networks solve partial, solve partial differential, partial differential equations}

B: same 6 shingles + {differential equations fast}

C: {graph neural networks, neural networks classify, networks classify nodes,
    classify nodes in, nodes in citation, in citation graphs}
```

Exact similarities: J(A,B) = 6/7 = **0.857**, J(A,C) = **0**, J(B,C) = **0**.

A and C share the words "neural networks", but no 3-word phrase, so J = 0.

### Step 2: MinHash signatures (k = 12)

Each hash function hᵢ gives every shingle a random-looking number, which stands in for a random ranking. A document's value in row i is the **smallest** hᵢ over its shingles.

```text
sig(A) = [ 66, 132, 181,  34,  27, 59,  156,   4, 160,  189,  81, 207]
sig(B) = [ 60, 132, 181,  34,  27, 59,  156,   4, 160,  189,  81, 207]
sig(C) = [221, 332, 214,  51, 608, 66,  208, 235, 125,    5, 340,  88]
```

- **A and B agree on 11 of 12 rows** (estimate 0.92, true value 0.857). Row 1 differs because B's extra shingle "differential equations fast" happened to get the smallest h₁ value. By the MinHash theorem, this happens with probability 1/7.
- **A and C agree on 0 of 12 rows** (estimate 0, true value 0).

Each document is now 12 numbers, whatever its length.

### Step 3: LSH banding (b = 4 bands × r = 3 rows)

Each band's 3-number tuple is hashed into a bucket table for that band.

```text
          band 1            band 2          band 3           band 4
A:   ( 66, 132, 181)    (34,  27, 59)    (156,   4, 160)    (189,  81, 207)
B:   ( 60, 132, 181)    (34,  27, 59)    (156,   4, 160)    (189,  81, 207)
C:   (221, 332, 214)    (51, 608, 66)    (208, 235, 125)    (  5, 340,  88)
```

- **A and B:** band 1 differs, but bands 2, 3 and 4 match exactly, so they share a bucket and become a **candidate pair**. One matching band is enough.
- **A or B with C:** no band matches, so the pair is **never compared**.

With n papers, this is how LSH avoids checking all n²/2 pairs: only papers that share a bucket are looked at.

### Step 4: Verification (exact Jaccard)

For the candidate A–B, compute the true J on the shingle sets: 6/7 = 0.857 ≥ 0.8 (`lsh_threshold` in `config.yaml`), so it is a **confirmed duplicate**. The code keeps the record that has an abstract and more citations, and removes the other.

If LSH had made A–C a candidate by chance (a false positive), J = 0 would reject it here.

### Side note: hash collisions

A first attempt used hash values modulo 101. Two of B's shingles, "differential equations fast" and "solve partial differential", got the **same** value (98), so the extra shingle could never change B's minimum, and A and B agreed on 12 of 12 rows for every random seed. This is the hash-collision problem from the first half of lecture 6-7. Real implementations, including datasketch, use a much larger range (around 2³²) to make it negligible.

### How this maps to the code

| Step | `CorpusDeduplicator._deduplicate_with_datasketch` |
|---|---|
| 1. Shingling | `create_shingles(title + abstract, k=shingle_size)` |
| 2. Signature | `MinHash(num_perm=128)`, then `m.update(...)` for each shingle |
| 3. Banding | `MinHashLSH(threshold=candidate_threshold)`, `lsh.insert`, `lsh.query` |
| 4. Verify | `_jaccard(shingle_sets[a], shingle_sets[b]) >= threshold` |

The code uses 128 hash functions instead of 12. datasketch chooses b and r itself so that the S-curve threshold (1/b)^(1/r) sits near `candidate_threshold` (0.65, which is `lsh_threshold` − 0.15).

## Validating frontiers: confidence interval and permutation test

Frontier scoring says something like: "communities A and B are similar in topic (0.95) but rarely cite each other (density 0.08), score 0.87". Phase 4 asks two sceptical questions about it.

### 1. Similarity confidence interval: "Is the number stable?"

We only have one sample of papers. A **bootstrap** creates many alternative samples by drawing papers at random from the corpus **with replacement** (some picked twice, some not at all).

```text
repeat 500 times:
    draw 80% of the papers at random, with replacement
    recompute mean similarity between A and B on that sample
95% CI = 2.5th to 97.5th percentile of the 500 values
```

- Narrow interval (e.g. 0.91 to 0.97): the similarity is reliable.
- Wide interval (e.g. -0.2 to 0.6): the similarity depends heavily on which papers happen to be in the corpus.
- `ci_sample_coverage`: fraction of samples that contained both communities; low values mean the interval is based on few samples.

Communities stay fixed, so this measures how stable the **similarity number** is, not whether the frontier would be **found again** (that would need re-clustering on every sample).

### 2. Permutation test: "Is the topic overlap real or chance?"

The communities are **citation silos** (Louvain groups papers that cite each other). A frontier is two silos that work on similar topics. Assume the boring explanation, the **null hypothesis H₀**: "these two silos are no more similar in topic than two random groups of papers of the same sizes". Then simulate that world and see whether the real result looks unusual in it.

```text
observed score for (A, B) = mean_similarity(A, B) x (1 - citation_density(A, B))
repeat 100 times:
    shuffle which embedding belongs to which paper
        (communities and citations stay fixed; topics become random)
    recompute the score for (A, B)
p-value = fraction of shuffled scores >= observed score
```

| p-value | Meaning |
|---|---|
| Small (< 0.05, the `alpha` cutoff) | Random groups rarely look this similar: the two silos **really share a topic** (`statistically_significant = True`) |
| Large (e.g. 0.74) | Random groups look this similar 74% of the time: the overlap is **unremarkable** |

The +1 in `(count + 1) / (n_permutations + 1)` counts the real arrangement as one of the permutations, so p is never exactly 0.

### Design history

- **Version 1 (circular):** citation communities, null = shuffle the citation graph. Communities built from citations always have few citations between them, so the test was always "significant".
- **Version 2 (no signal):** k-means topic communities, null = shuffle the citation graph. k-means separates dissimilar papers by construction (mean similarity within clusters 0.24, between clusters -0.04 on the real corpus), so "similar but different clusters" almost never happened and nothing was significant.
- **Version 3 (current):** citation communities, null = shuffle embeddings. Groups come from citations and the null from the embeddings, so the test is not circular and asks the intended question: do two citation silos share a topic?
- The observed and null scores always use the same `community_similarity()` and `citation_density()` functions.

### Deciding what counts as a frontier

A pair is reported as a frontier (`is_frontier`) only if both hold:

1. **Low citation, relative to the corpus.** `density_ratio` = pair density / whole-graph density. A fixed cutoff like 0.10 is meaningless when the whole graph's density is 0.017 (every pair passes). With `max_density_ratio: 1.0`, the two silos must cite each other **less than an average pair of papers** in the corpus.
2. **Significant after correcting for many tests.** Testing 20 pairs at alpha = 0.05 gives about one "significant" result by luck alone. The **Benjamini-Hochberg** correction controls the *false discovery rate*: among the pairs called significant, at most about 5% are expected to be flukes.

```text
sort the m p-values ascending: p(1) <= p(2) <= ... <= p(m)
adjusted p(i) = min over k >= i of  p(k) * m / k     (capped at 1)
significant if adjusted p < alpha
```

All community pairs are tested and corrected together, and only then is the list cut to `top_k_frontiers`. Testing only the top 10 would hide how many tests were really run.

Significant still means "unlikely to be chance", not "large": with big communities even a small similarity can be significant, so read it together with the similarity and its confidence interval.
