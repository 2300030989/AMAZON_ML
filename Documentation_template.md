# ML Challenge 2026: Business Entity Resolution

**Team Name:** Not provided  
**Team Members:** Not provided  
**Submission Date:** 2026-09-27

## 1. Executive Summary
This solution uses disk-backed SQLite FTS5 blocking followed by a transparent name/address similarity rule. A global match threshold is chosen on a deterministic holdout from labeled training Source 1 entities using the challenge's macro F0.5 metric. All entity evidence comes from the supplied challenge files.

## 2. Methodology

### 2.1 Problem Analysis
Source 1 is the deduplicated reference; each Source 1 entity can map to zero or multiple Source 2/3 records. Labels are lists rather than one-to-one pairs, so candidates are classified independently. Names and addresses can contain abbreviations, suffix differences, punctuation, word reordering, typos, transliteration and missing components. Country is an open set: test includes France although training includes only US and India. The score is macro-averaged per Source 1 record and includes singleton entities, making unsupported matches costly.

### 2.2 Solution Strategy
**Approach Type:** Multi-pass lexical blocking plus similarity scoring.  
**Core Innovation:** Disk-backed FTS5 indexing avoids all-pairs comparison. The blocking stage first intersects an informative name token and address token, then adds bounded name-only and address-only fallback blocks. This improves recall when one field is absent or corrupted while keeping the candidate set bounded and auditable. Country is treated as an open set and compared as an observed string; no countries are hard-coded or used to filter records.

## 3. Candidate Generation (Blocking)
For each target source (2 and 3), FTS5 uses the longest informative token in each field. It retrieves up to 20 records matching both field tokens, plus up to 10 name-only and 10 address-only records. The union is deduplicated by entity ID and emitted to `candidate_pairs.tsv`; this exact union is scored. The theoretical cap is 40 candidates per target source (80 across Sources 2 and 3) for each Source 1 record. Candidate recall remains bounded by lexical overlap and may be lower for transliterations, severe typos, or records with no usable field tokens. Country mismatches are penalized during scoring rather than blocked.

## 4. Matching Model
**Features:** normalized name character ratio, name-token Jaccard, normalized address character ratio, address-token Jaccard, country agreement/conflict, and overlap of numeric address components. Similarities use Python's `SequenceMatcher` and token set overlap. Name evidence contributes 68% (character/token blend); address evidence contributes 32% (character/token/digit blend), with a small country-agreement bonus and conflict penalty.

**Model type:** interpretable weighted similarity rule (no external pretrained model).  
**Threshold selection:** a deterministic MD5-based holdout is scored against training labels. This run capped the validation sample at 1,000 Source 1 entities. The threshold sweep maximizes mean per-entity F0.5, including singleton rows (1 for correctly empty predictions and 0 for a false match). Observed macro F0.5 was 0.56281 and selected threshold was 0.74. The prior pipeline scored 0.08411 on the same holdout; this is a substantial validation improvement, not a guarantee of leaderboard performance.

## 5. Results & Error Analysis
The 1,000-entity validation sample scored macro F0.5=0.56281 at threshold 0.74, compared with 0.08411 for the earlier pipeline on the same holdout. The largest remaining risks are similar business names at distinct addresses and false negatives caused by weak lexical overlap or incomplete fields. Candidate recall and private leaderboard performance are not yet known.

## 6. Conclusion
The pipeline produces auditable candidate and match TSVs while keeping retrieval disk-backed and constrained to the provided dataset. Its principal limitation is lexical blocking and a hand-weighted similarity score; the emitted candidate file makes the blocking boundary visible for further validation and improvement.

## Appendix

### A. Code Artefacts
`src/pipeline.py` builds indexes and tunes the threshold; `src/generate_parallel.py` generates the complete test output in disjoint shards. Run the commands in `README.md`. The input dataset is under `dataset/`; dependencies are documented in `requirements.txt`.

### B. Additional Results
No charts or full-run metrics are included yet.
