# OpenAlex API Guide

How Phase 1 talks to OpenAlex: what we send, what comes back, and how
[`openalex_fetch.py`](../src/phase1_acquisition/openalex_fetch.py) turns it into the corpus.

Official docs: <https://docs.openalex.org> and <https://help.openalex.org>

---

## 1. Authentication

Since **Feb 13, 2026**, OpenAlex requires an API key.

| Access | Daily budget | Notes |
|---|---|---|
| With API key (free) | 100,000 credits (about $1/day) | Needed for real runs |
| Without key | 100 credits, shared per IP | Then requests fail with **409** |

- Get a key at <https://openalex.org/settings/api>.
- Put it in `Code/RGD/.env`:
  ```ini
  OPENALEX_API_KEY=your_key_here
  ```
- The code sends it as the `api_key` query parameter.
- The old email-based "polite pool" (`mailto=` or email in the User-Agent) is **deprecated and ignored**. No email is needed.

**How to check the key is actually used:** if a response contains `meta.notice` saying
"No API key, so this request shares one free daily budget...", the key was not loaded.
Each response also reports `meta.cost_usd` for that request.

---

## 2. Requests we make

Base URL: `https://api.openalex.org`

### 2.1 Seed search (`fetch_papers_by_query`)

```
GET /works?search=<query>&per_page=25&page=<n>&api_key=<key>
```

- Paginates until `max_results` papers are collected or a page comes back empty.
- Sleeps 0.1 s between pages.

> **Caveat:** `search=` is translated by OpenAlex into `fulltext.search`, which matches the
> **title, abstract, or full text** and ranks by relevance. It does **not** require every word
> to appear. A query like `anisotropy in large language models` therefore returns many
> generic LLM papers. Use `--must-include` (below) for stricter matching.

#### Required terms (`--must-include`)

`--must-include` adds a hard filter on top of the relevance search. `--query` still decides
the ranking, but only papers containing the required terms in their **title or abstract**
are returned.

- Each argument is one **required** group (joined with `AND`).
- Inside a group, `|` separates **alternatives** (joined with `OR`).
- Multi-word terms are matched as exact phrases.
- Commas are not allowed (OpenAlex uses them to separate filters).

```bash
python src/pipeline.py \
  --query "anisotropy in large language models" \
  --must-include anisotropy "language model|transformer"
```

builds:

```
GET /works?search=anisotropy in large language models
          &filter=title_and_abstract.search:anisotropy AND ("language model" OR transformer)
```

The filter is built by `OpenAlexFetcher.build_required_terms_filter`.

### 2.2 Citation expansion (`expand_corpus_via_citations`)

For every seed paper, collect IDs from `referenced_works` (up to `max_references_per_paper`)
and `related_works` (up to half that), capped at `max_total_expanded` overall. Then fetch
them in batches of 25:

```
GET /works?filter=openalex_id:W123|W456|...&per_page=25&api_key=<key>
```

> **Caveat:** `related_works` are chosen by OpenAlex from broad topic overlap, not citations,
> so they are often off-topic. In one run only 3 of 77 expanded papers mentioned the query term.

Add `--filter-expanded` (requires `--must-include`) to apply the same required terms to
expanded papers. The two filters are joined with a comma, which OpenAlex treats as `AND`:

```
GET /works?filter=openalex_id:W123|W456,title_and_abstract.search:anisotropy
```

Leave it off if you want the corpus to include neighbouring work that does not use your
keywords; finding those weakly connected areas is part of gap discovery.

---

## 3. Response structure

```json
{
  "meta": {
    "count": 46304,
    "page": 1,
    "per_page": 25,
    "cost_usd": 0.001,
    "notice": "(only present when something is off, e.g. no API key)"
  },
  "results": [ { "...one work..." }, "..." ],
  "group_by": []
}
```

The code reads only `results`.

### 3.1 A work object

Each work has around 55 fields. The ones that matter here:

| Field | Type | Example / meaning |
|---|---|---|
| `id` | string | `"https://openalex.org/W4404784303"` |
| `doi` | string or null | `"https://doi.org/10.18653/v1/2024.emnlp-main.344"` |
| `title` | string or null | Paper title |
| `publication_year` | int | `2024` |
| `cited_by_count` | int | `6` |
| `authorships` | list of objects | Each has `author.display_name`, `author.id`, `institutions`, `author_position` |
| `concepts` | list of objects | Legacy tags with `display_name`, `level`, `score` |
| `topics`, `primary_topic` | objects | Newer, more accurate topic classification (not used yet) |
| `referenced_works` | list of IDs | Papers this work cites |
| `related_works` | list of IDs | Topic-similar papers picked by OpenAlex |
| `abstract_inverted_index` | dict or null | Abstract stored as `word -> [positions]` |

Other available fields (unused): `publication_date`, `type`, `language`, `open_access`,
`best_oa_location`, `keywords`, `fwci`, `counts_by_year`, `is_retracted`, `institutions`, ...

---

## 4. Parsing (`_parse_work`)

Each work is reduced to a flat record with 10 fields:

| Output field | Source | Transformation |
|---|---|---|
| `id`, `title` | `id`, `title` | If either is missing the work is **dropped** |
| `doi`, `publication_year`, `cited_by_count` | same name | Copied as-is |
| `authors` | `authorships[*].author.display_name` | Flattened to a list of names |
| `concepts` | `concepts[*].display_name` | Flattened to a list of names (score and level discarded) |
| `referenced_works`, `related_works` | same name | Kept as lists of IDs |
| `abstract` | `abstract_inverted_index` | Rebuilt into plain text (see below) |

### 4.1 Abstract reconstruction (`reconstruct_abstract`)

OpenAlex stores abstracts as an inverted index:

```json
{"Language": [0], "models": [1, 19], "strongly": [2], "rely": [3], "on": [4, 134]}
```

The function expands it to `(position, word)` pairs, sorts by position, and joins with spaces:

```
(0,"Language") (1,"models") (2,"strongly") (3,"rely") (4,"on") ...
-> "Language models strongly rely on ..."
```

Returns `""` when the index is missing.

---

## 5. Full example

**Raw work (trimmed):**

```json
{
  "id": "https://openalex.org/W4404784303",
  "doi": "https://doi.org/10.18653/v1/2024.emnlp-main.344",
  "title": "Mitigating Frequency Bias and Anisotropy in Language Model Pre-Training with Syntactic Smoothing",
  "publication_year": 2024,
  "cited_by_count": 6,
  "authorships": [
    {
      "author_position": "first",
      "author": {"id": "https://openalex.org/A5027311833", "display_name": "Richard Diehl Martinez", "orcid": null},
      "institutions": [{"display_name": "University of Cambridge"}]
    }
  ],
  "concepts": [
    {"id": "https://openalex.org/C3770464", "display_name": "Smoothing", "level": 2, "score": 0.78}
  ],
  "referenced_works": ["https://openalex.org/W1821462560", "https://openalex.org/W2933966394"],
  "related_works": ["https://openalex.org/W4388579598"],
  "abstract_inverted_index": {"Language": [0], "models": [1, 19], "strongly": [2]}
}
```

**After `_parse_work`:**

```json
{
  "id": "https://openalex.org/W4404784303",
  "doi": "https://doi.org/10.18653/v1/2024.emnlp-main.344",
  "title": "Mitigating Frequency Bias and Anisotropy in Language Model Pre-Training with Syntactic Smoothing",
  "publication_year": 2024,
  "cited_by_count": 6,
  "abstract": "Language models strongly rely on frequency information because they maximize the likelihood of tokens during pre-training.As a consequence, ...",
  "authors": ["Richard Diehl Martinez", "Zébulon Goriely", "Andrew Caines", "Paula J. Buttery", "Lisa Beinborn"],
  "concepts": ["Smoothing", "Computer science", "Anisotropy", "Training (meteorology)", "Artificial intelligence", "Natural language processing", "Language model", "..."],
  "referenced_works": ["https://openalex.org/W1821462560", "https://openalex.org/W2933966394", "..."],
  "related_works": ["https://openalex.org/W4388579598", "..."]
}
```

`to_dataframe` turns the list of these records into a pandas DataFrame, saved as
`data/raw/corpus_raw.csv`.

---

## 6. Known quirks

- **Missing abstracts:** some works have no `abstract_inverted_index` (about 8% in one run).
  These are embedded from the title only.
- **Missing spaces after periods** (e.g. `pre-training.As`) come from OpenAlex's index, not our code.
- **`concepts` is noisy and legacy.** Tags like "Training (meteorology)" or "Optics" appear on NLP
  papers. Prefer `topics` / `primary_topic` if concepts are ever used downstream.
- **Errors:** a failed seed page stops pagination and logs an error. A failed expansion batch is
  logged as a warning and skipped. If no seeds are returned at all, the pipeline falls back to
  synthetic mock data.
