# Clinical note exploration

[`analyze_med_note.ipynb`](analyze_med_note.ipynb) loads a corpus of hospital
discharge summaries and reports basic structure and statistics. It is an
exploratory notebook, not an evaluation — it exists to show what the data looks
like before you point EvalRing at it.

## The data is not in this repository

No clinical data is committed here, and none should be. The notebook downloads
what it needs on first run into `sample/med_note/`, which is gitignored.

Before running it, satisfy yourself that you are permitted to obtain and use
the corpus. Discharge summaries of the kind this notebook loads are commonly
derived from [MIMIC](https://physionet.org/content/mimiciv/), which is
distributed under a PhysioNet credentialed-access data use agreement: it
requires CITI human-subjects training, a signed DUA, and it **prohibits
redistribution** of the notes, de-identified or otherwise. A mirror existing on
another platform does not transfer those rights to you.

If the corpus you obtain carries such terms:

- do not commit it, and do not commit notebook outputs containing note text,
- do not send it to a third-party model provider without checking that the DUA
  permits it — point `EVALRING_BASE_URL` at a self-hosted endpoint instead,
- do not redistribute derived files that still contain note text.

See [docs/DATA.md](../../../docs/DATA.md) for how this applies across the
examples, and [docs/CONFIGURATION.md](../../../docs/CONFIGURATION.md) for
running against a local endpoint.

## Not a clinical tool

Nothing here is a diagnostic, screening, or triage instrument, and no output
should be used in the care of any real person.
