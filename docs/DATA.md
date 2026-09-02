# Datasets, licensing, and responsible use

EvalRing itself ships no data. The examples reference several external
datasets, some of them sensitive and most of them carrying redistribution
conditions. This page records where each comes from, what you must do to obtain
it, and what the results do and do not mean.

**The rule:** obtain datasets from their upstream source under their own
licence. Do not commit them to this repository. `.gitignore` excludes
`examples/*/data/` for exactly this reason.

---

## Datasets referenced by the examples

### Humanity's Last Exam (HLE)

- **Source:** [`cais/hle`](https://huggingface.co/datasets/cais/hle) on Hugging Face
- **Obtain with:** `python examples/hle/ingest_hle.py` (requires a Hugging Face
  token in `HF_TOKEN`, and you must accept the dataset's terms first)
- **Licence:** set by the dataset authors on Hugging Face. Check before use.
- **Not redistributed here.**

### ARC-Challenge

- **Source:** [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc)
- **Obtain with:** `python examples/hle/ingest_arc.py`
- **Licence:** CC BY-SA 4.0 at the time of writing; confirm upstream.
- **Not redistributed here.**

### GPQA

- **Source:** [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa)
- **Obtain with:** `python examples/hle/ingest_gpqa.py` (gated; requires an
  accepted licence and a Hugging Face token)
- **Note:** GPQA is deliberately withheld from public indexing to limit
  contamination. Do not republish the questions or your extracted CSV.
- **Not redistributed here.**

### RSD-15k (Reddit suicide-risk posts)

- **Content:** Reddit posts annotated with suicide-risk level (`Ideation`,
  `Indicator`, `Behavior`, `Attempt`).
- **Obtain from:** the original research distribution. It is not downloadable
  from this repository and no download script is provided, because
  redistribution terms for the annotated corpus are restrictive and the
  material is sensitive.
- **Prepare with:** `examples/suicide_detection/data/data_translator.py`, which
  writes the processed CSVs into `examples/suicide_detection/data/`.
- **Not redistributed here.** The committed exploratory statistics and charts
  under `examples/suicide_detection/data/` describe the dataset; they contain
  no post text.

### CounselBench / Counsel Chat

- **Source:** [`nbertagnolli/counsel-chat`](https://huggingface.co/datasets/nbertagnolli/counsel-chat),
  used through the separate [CounselBench](https://github.com/CounselBench)
  project.
- **Note:** CounselBench is an upstream repository, not part of EvalRing. If
  you have a local checkout it is gitignored here.
- **Not redistributed here.**

### Clinical case notes

- **Source:** [`bavehackathon/2026-healthcare-ai`](https://huggingface.co/datasets/bavehackathon/2026-healthcare-ai)
- **Used by:** `examples/sample/med_note/analyze_med_note.ipynb`, which
  downloads `clinical_cases.csv.gz` on first run into a gitignored directory.
- **Not redistributed here.** These are hospital discharge summaries carrying
  MIMIC-style structure (`subject_id`, `hadm_id`, `[**date**]` de-identification
  surrogates). [MIMIC](https://physionet.org/content/mimiciv/) is distributed
  under a PhysioNet credentialed-access data use agreement requiring CITI
  training and a signed DUA, and it **prohibits redistribution** of the notes.
  A mirror on another platform does not transfer those rights. Confirm your own
  authorisation before downloading, and never commit the result.
- See [`examples/sample/med_note/README.md`](../examples/sample/med_note/README.md).

---

## Adding a dataset

Before adding data to an example:

1. Confirm you have the right to redistribute it. If in doubt, you do not —
   write an ingest script instead.
2. Add an entry to this file: source URL, licence, how to obtain it, and any
   sensitivity that affects how results should be read.
3. Keep raw data out of git. Small fixtures for tests belong under `tests/` and
   must be synthetic.
4. If the data contains personal information, do not add it at all.

---

## Responsible use

### Sensitive-domain examples

`examples/suicide_detection/` classifies suicide-risk level in social media
posts, and `examples/sample/med_note/` works with clinical case notes. Both
exist to measure how well models perform on hard, high-stakes classification —
not to be deployed.

**These are not clinical tools.** No output from EvalRing is a screening
instrument, a triage system, a diagnosis, or a risk assessment for any real
person. A model that scores well on RSD-15k has demonstrated agreement with
annotations on a research corpus and nothing more. Deploying such a classifier
against real people would require clinical validation, ethical review,
regulatory clearance, and a human decision-maker in the loop — none of which
this repository provides or substitutes for.

If you are working with this material and need support yourself, contact a
local crisis line; in the US and Canada, 988.

### Your data leaves your machine

Running an evaluation sends dataset text to whichever provider you configured.
Before evaluating restricted, personal, or clinical data, confirm that:

- the data may lawfully be sent to that provider,
- the provider's retention and training policies are acceptable for it,
- any required data-processing agreement is in place.

For data that cannot leave your infrastructure, point `EVALRING_BASE_URL` at a
self-hosted endpoint (vLLM, Ollama, an internal gateway). See
[CONFIGURATION.md](CONFIGURATION.md).

### Run artifacts inherit the sensitivity of the data

Everything under `_EvalRing/` — the SQLite cache, `all_cases.csv`,
`incorrect_cases.txt`, `result.md`, per-model logs — contains dataset text and
model responses. The directory is gitignored. Do not attach these files to
issues, and delete them under the same policy as the source data.

### Reporting results

If you publish comparisons produced with EvalRing, report the model
identifiers and versions, the dataset and split, sample count, seed, and
temperature. Each run's `Meta.json` records all of this. Benchmark scores are
sensitive to prompt wording and answer parsing, so publish the agent's system
prompt alongside the numbers.
