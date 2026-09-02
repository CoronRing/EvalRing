# Dataset: Humanity's Last Exam (HLE)

- Source: [`cais/hle`](https://huggingface.co/datasets/cais/hle)
- Type: extremely hard, expert-authored exam questions
- Split used: `test` (~2,500 entries)
- Grading: **LLM judge** (free-form answers → semantic equivalence), not exact match

## Access (gated)

HLE is gated. Accept the terms on the dataset page, then set a token in `.env`:

```
HF_TOKEN=hf_xxx
```

## Ingest

```bash
python examples/hle/ingest_hle.py
```

Writes:

- `data/hle_full.jsonl` — lossless archive, one row per entry, **including** the
  `image` data-URI (git-ignored; regenerate anytime).
- `data/hle.csv` — text-usable columns only (no image bytes).

## Composition (test split)

| Facet | Counts |
|---|---|
| Total | 2,500 |
| Answer type | exactMatch 1,909 · multipleChoice 591 |
| With image | 342 (13.7%) · text-only 2,158 |
| Top categories | Math 1021 · Biology/Medicine 280 · CS/AI 241 · Physics 230 · … |

## CSV schema

`ID, original_id, question, answer, answer_type, category, raw_subject, author_name, has_image`

Consumed by the runner as `text_field=question`, `label_field=answer`,
`id_field=ID`; the remaining columns land in per-sample metadata.

## Run

Text-only by default (image questions are skipped unless `--include-images`).
See the [HLE example README](../../examples/hle/README.md) and
[providers](../providers/) for model routing.

```bash
python examples/hle/run_hle_suite.py --n-samples 10 --max-workers 5
```

## Gotchas

- Frontier models score in the low double digits — low accuracy is expected.
- Heavy reasoners can exceed the stream/timeout window on hard items and be
  recorded as timeout **errors**; raise `--request-timeout-s` or lower
  `reasoning_effort` if you need them to complete.
