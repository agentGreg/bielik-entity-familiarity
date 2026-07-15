# Dataset — Polish entity familiarity (v2)

1,440 Polish entities across four domains, decile-stratified by Wikipedia
popularity, with token-length-matched fabricated controls. This is the dataset
behind the paper's gradation (RQ2) and gating (RQ4) experiments.

## Files

- `entities.csv` — all 1,440 rows (combined).
- `entities_athletes.csv`, `entities_cities.csv`, `entities_musicians.csv`,
  `entities_writers.csv` — the same rows split by domain (360 each).

## Composition

| | count |
|---|---|
| real entities | 1,200 (300 per domain, 30 per popularity decile) |
| fabricated entities | 240 (60 per domain) |
| **total** | **1,440** |

## Columns

| column | meaning |
|---|---|
| `qid` | Wikidata QID for real entities; a synthetic id for fabricated ones |
| `entity` | the entity surface string (as presented to the model) |
| `domain` | `athletes`, `cities`, `musicians`, or `writers` |
| `kind` | `real` or `fabricated` |
| `decile` | Wikipedia-pageview popularity decile 0–9 (real only; 0 = long tail, 9 = head) |
| `pageviews_12m` | trailing-12-month Wikipedia pageviews (real only) |
| `sitelinks` | number of Wikidata sitelinks (cross-language presence) |
| `article_len` | Polish Wikipedia article length in bytes |
| `in_v1` | whether the entity was also in the predecessor (v1) dataset |
| `source` | provenance tag for the harvesting step |
| `flags` | JSON: screening flags (e.g. `disambiguation_risk`, `pageview_zero`) |

## Construction (summary; full detail in the paper)

- **Real** entities are harvested from Wikidata by domain, resolved to QIDs, and
  binned into ten equal-count popularity deciles by trailing-12-month pageviews.
- **Fabricated** entities are plausible but non-existent names, screened against
  real-person collisions and **token-length-matched** to the real distribution
  under the model tokenizers so length is not a discriminative shortcut.
- Reproduce with `scripts/build_dataset_v2.py` (see the top-level README).

## Provenance and licence

Entity identifiers and metadata derive from **Wikidata** (CC0) and **Wikimedia
pageview** statistics. Only public information about public entities is used.
Fabricated names are synthetic and do not refer to real individuals. The code in
this repository is MIT-licensed; the derived dataset is released for research use
with attribution to Wikidata/Wikimedia as the upstream sources.
