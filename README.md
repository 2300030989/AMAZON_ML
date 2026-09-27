# Business Entity Resolution Challenge

This project solves the provided 3-source business entity resolution challenge with a reproducible, disk-backed pipeline. It uses only the challenge dataset; it does not query external identity, business-registration, address, or geocoding services.

## Requirements

- Python 3.10 or newer
- SQLite compiled with FTS5 (included in standard Windows Python builds)
- No third-party Python packages; see `requirements.txt`
- Several GB of free disk space for temporary FTS indexes. The supplied train and test files are included under `dataset/`.

## Run end-to-end

From this project directory, build the indexes and tune a deterministic 1,000-entity training holdout:

```powershell
python .\src\pipeline.py --data "." --work ".\work" --validation-limit 1000 --prepare-only
```

Then generate the full test set using parallel workers (the threshold is read from `work/threshold.txt`):

```powershell
python .\src\generate_parallel.py --data "." --work ".\work" --output ".\output" --workers 12
```

The run indexes Source 2 and Source 3 for training and test into a disk-backed SQLite FTS5 database, tunes a threshold using only training labels, and generates:

- `output/matching_results.tsv` — leaderboard prediction for every test Source 1 ID
- `output/candidate_pairs.tsv` — the final candidate set scored by the rule model

The test files are never used to tune the threshold. For a fuller validation sweep, omit `--validation-limit 1000`; the pipeline will evaluate the full deterministic holdout. Use `--threshold 0.XX` to override threshold tuning.

After generation, validate from this project directory:

```powershell
python .\utils\validate_submission.py --matching .\output\matching_results.tsv --candidate .\output\candidate_pairs.tsv --test-dir .\dataset\test
```

The validator's optional `--check-ids` can use substantial memory on this dataset. Default format checks are the official lightweight validation.

## Pipeline design

`src/pipeline.py` builds per-source FTS5 indexes. Per target source it retrieves up to 20 records matching one informative name and one address token, plus up to 10 name-only and 10 address-only fallbacks. The deduplicated candidate file contains exactly the records scored during inference (at most 40 per target source, 80 total). Features blend normalized name/address character similarity and token Jaccard, with address digit overlap and country agreement/conflict adjustments. Country is treated as an open string set and is never used to filter candidates.

Observed validation macro F0.5 was 0.56281 on 1,000 held-out Source 1 entities, with threshold 0.74. The earlier pipeline scored 0.08411 on the same holdout. This is a measured validation improvement, not a leaderboard guarantee. Candidate recall was not separately measured. Lexical blocking can still miss transliterations, severe typos, and candidates with no shared lexical tokens.


