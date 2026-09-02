# Dataset: ARC-Challenge

- Source: [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc), config `ARC-Challenge`
- Type: grade-school science multiple-choice questions (the "Challenge" set)
- Split used: `test` (1,172 entries)
- Access: **public** (no token required)
- Reasoning load: light — useful for a fast pipeline check (vs HLE's heavy reasoning)

## Ingest

```bash
python examples/hle/ingest_arc.py
```

Writes `data/arc_challenge.csv` in the **same schema** the HLE runner consumes,
so it can be fed straight to the runner via `--data-path`.

## CSV schema

`ID, original_id, question, answer, answer_type, category, has_image`

- `question` embeds the answer choices, e.g.:

  ```
  <stem>

  Answer Choices:
  A. ...
  B. ...
  ```
- `answer` is the gold option label (e.g. `A`).
- `answer_type = multipleChoice`, `has_image = 0` for all rows.

## Run

```bash
python examples/hle/run_hle_suite.py `
    --n-samples 20 --max-workers 50 --data-path examples/hle/data/arc_challenge.csv
```

Grading uses the same LLM judge; for a single-letter answer the judge simply
confirms the letter matches the gold key. See [providers](../providers/) for
model routing.
