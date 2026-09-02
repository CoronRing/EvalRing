# EvalRing

[![CI](https://github.com/CoronRing/EvalRing/actions/workflows/ci.yml/badge.svg)](https://github.com/CoronRing/EvalRing/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalring)](https://pypi.org/project/evalring/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**EvalRing helps you answer one question: out of all these options, which one is
actually best at my task?**

You have a job you want a language model to do. Sort support tickets, pull dates
out of contracts, judge whether an answer is any good. There are a lot of ways
you could build it, and no obvious way to tell which one wins.

So you write a quick script. Then a colleague asks whether a cheaper model would
do, and you write another. Then you change the wording of your prompt and you
are not sure whether the score moved because the wording is better or because
you happened to test on different examples. A month later someone asks how you
got the number in the slide, and nobody knows.

EvalRing is for that. You describe your task once, list the options you want to
try, and get back one comparison table plus a saved record of exactly how each
number was produced.

## Who this is for

You will get the most out of EvalRing if you can write basic Python and you have
a task where you know what the right answer looks like, at least for a few
hundred examples. You do not need any background in machine learning, and you do
not need to understand how models work inside.

If you have never run an evaluation before, the [first run](#your-first-run-no-api-key-needed)
below works offline and takes about two minutes.

## What makes it different

Most evaluation tools score **one model** on **a standard benchmark**. That is
useful for reading model announcements, but it is not the question most people
actually have. EvalRing is built the other way around: your task is fixed, and
the thing that varies is which candidate you are testing.

**It compares candidates, not just models.** A candidate can be a model, but it
can just as easily be your prompt with one sentence changed, a two step pipeline
you wrote, a small model running on your own machine, or last month's version of
your code. If you can call it from Python, you can put it in the comparison.

**One bad candidate does not ruin the batch.** When you compare several
candidates, each one runs in its own separate process. If one of them crashes or
runs out of quota halfway through, the others carry on and you still get results
for them.

**Nothing is thrown away when something goes wrong.** An example that fails is
written down as a failure instead of stopping the run. You can rerun just the
failures afterwards, and point a new run at an old one to reuse the answers you
already paid for.

**It is not tied to any one company.** EvalRing talks to anything that speaks the
common OpenAI style API, which in practice is nearly everything: OpenAI,
OpenRouter, your company's internal gateway, or a model running on your own
laptop through Ollama or vLLM. Switching is one environment variable, and no
vendor name is baked into the library.

**Every run leaves a paper trail.** Alongside the scores, each run saves the
per example results and the exact settings that produced them, so the number in
your slide is still explainable in six months.

## Install

```bash
pip install evalring
```

That gives you the core: loading data, running candidates, and scoring. To
actually call a hosted model you also want the `llm` extra:

```bash
pip install "evalring[llm]"    # adds the libraries that talk to model providers
pip install "evalring[all]"    # also adds charts and synthetic data generation
```

Python 3.10 or newer. If you would rather work from the source:

```bash
git clone https://github.com/CoronRing/EvalRing.git
cd EvalRing
pip install -e ".[dev]"
```

## A word on the word "agent"

EvalRing calls the thing being tested an **agent**. That is a big word for a
small idea: an agent is any Python object that takes a piece of text and returns
an answer.

Sending the text to GPT-4o is an agent. A handful of keyword rules is an agent.
A function that searches your database, builds a prompt, calls a model, and
cleans up the reply is also an agent. EvalRing does not care what happens
inside. It hands your agent one example at a time and records what came back.

That is the reason you can compare a model against your own code against a
simple baseline, all in the same table.

## Your first run, no API key needed

This uses a few keyword rules instead of a model, so it costs nothing and needs
no network. It is worth running once to see the shape of things.

Save this as `reviews.json`:

```json
[
  {"id": "1", "text": "this is great, I love it", "label": "positive"},
  {"id": "2", "text": "awful experience, terrible", "label": "negative"},
  {"id": "3", "text": "it arrived on time", "label": "neutral"},
  {"id": "4", "text": "I love the design", "label": "positive"}
]
```

Each example needs three things: an `id` so EvalRing can tell examples apart, the
`text` to judge, and the `label` you believe is correct.

Then run:

```python
from EvalRing import ClassificationEvaluator, JSONDataset, RuleBasedAgent

# 1. Point EvalRing at your examples and say which column is which.
dataset = JSONDataset(name="reviews")
dataset.load_data("reviews.json", text_field="text", label_field="label", id_field="id")

# 2. Describe the thing you want to test. Here it is just keyword matching.
agent = RuleBasedAgent(
    name="keyword-baseline",
    rules={"positive": ["great", "love"], "negative": ["awful", "terrible"]},
    default_output="neutral",
)

# 3. Score it.
evaluator = ClassificationEvaluator(output_dir="results")
result = evaluator.evaluate(agent, dataset, task_name="sentiment")

print(f"accuracy: {result.metrics.get_metric('accuracy'):.3f}")
print(f"f1:       {result.metrics.get_metric('f1_score'):.3f}")

evaluator.save_results(result)
```

You will see:

```
accuracy: 1.000
f1:       1.000
```

**Accuracy** is the share of examples it got exactly right. **F1** is a fairer
summary when some labels are much rarer than others, because plain accuracy can
look great while a model quietly never predicts the rare label at all. EvalRing
also records `precision` and `recall`, which are the two halves F1 combines.

Getting a perfect score here is not a good sign, by the way. It means the toy
examples were chosen to match the keywords. Real data will not be so kind, which
is the whole point of measuring.

Your scores and every individual answer are now saved as a JSON file under
`results/`.

CSV files and pandas data frames work the same way, through `CSVDataset` and
`DataFrameDataset`.

## Now with a real model

First tell EvalRing how to reach a provider. One variable is enough:

```bash
export EVALRING_API_KEY="your-key"
export EVALRING_BASE_URL="https://openrouter.ai/api/v1"   # skip if using OpenAI
export EVALRING_MODEL="anthropic/claude-sonnet-4"         # optional default
```

On Windows PowerShell, use `$env:EVALRING_API_KEY = "your-key"`.

`OPENAI_API_KEY` and `OPENROUTER_API_KEY` are picked up too, so an existing setup
usually just works. To see what EvalRing found, ask it. It never prints your key
itself:

```console
$ evalring check
{
  "evalring_version": "0.2.0",
  "api_key_found": true,
  "api_key_source": "$EVALRING_API_KEY",
  "provider": "evalring",
  "base_url": "https://openrouter.ai/api/v1",
  "model": "anthropic/claude-sonnet-4",
  ...
}
```

The full list of variables and which one wins when several are set is in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

Now swap the keyword rules for a model. Everything else stays the same:

```python
from EvalRing import ClassificationEvaluator, CSVDataset, OpenAIAgent

dataset = CSVDataset(name="reviews")
dataset.load_data("reviews.csv", text_field="review", label_field="sentiment", id_field="row_id")

agent = OpenAIAgent(
    name="sentiment-classifier",
    model_name="gpt-4o",  # or leave it out and let $EVALRING_MODEL decide
    system_prompt="Reply with exactly one word: positive, negative, or neutral.",
    temperature=0.0,  # 0 keeps answers as repeatable as the model allows
)

result = ClassificationEvaluator(output_dir="results").evaluate(
    agent,
    dataset,
    task_name="sentiment",
    max_workers=16,  # how many examples to send at once
    max_retries=3,
)
```

Rate limits and temporary server errors are retried automatically with a growing
wait between attempts. Examples that never succeed are recorded as failures and
the run keeps going, so a wobble in the middle of a thousand examples does not
cost you the other nine hundred.

## Comparing several candidates

This is what EvalRing is really for. Put your evaluation in a script, list the
candidates in a JSON file, and run the whole set:

```bash
evalring models --output model_list.json     # writes a starter list to edit

evalring run-suite \
    --eval-script my_eval.py \
    --models-file model_list.json \
    --n-samples 500 \
    --max-workers 32 \
    --yes
```

Each candidate runs in its own process. Results land in
`_EvalRing/run_suite_<timestamp>/`: one folder per candidate, a combined report
you can read, a JSON summary you can load, and charts if `matplotlib` is
installed.

Two flags matter when a comparison goes wrong partway through:

- `--continue-runs` picks up a suite that was interrupted rather than starting over.
- `--cache <old run folder>` reuses answers already computed in a previous run,
  so you only pay for what is genuinely new.

There is also `retry_failed_cases()` in Python, which rereads a finished run's
metadata and reruns only the examples that failed.

One honest note: reuse works at the level of a whole previous run folder, and is
something you opt into with those flags. A plain `evaluator.evaluate(...)` call
does not silently reuse earlier answers. EvalRing also ships a shared key value
store, `GlobalCache`, if you want your own agent to skip repeat calls itself.

See [docs/USAGE.md](docs/USAGE.md) for the script contract and what each output
file contains.

## Testing your own code

Write a class with two methods and EvalRing can score it. This is how you compare
your pipeline against a plain model, or this month's version against last
month's:

```python
from EvalRing import AgentResponse, BaseAgent


class MyAgent(BaseAgent):
    def initialize(self, **kwargs) -> None:
        # Runs once before evaluation. Load models, open connections here.
        self._model = load_my_model()
        self._is_initialized = True

    def predict(self, input_text: str, **kwargs) -> AgentResponse:
        # Runs once per example.
        label, score = self._model(input_text)
        return AgentResponse(
            input_id="",  # EvalRing fills this in
            input_text=input_text,
            output=label,
            confidence=score,
        )
```

`output` can be a single label, or a dictionary of labels to scores if your code
produces a spread of confidence across options. EvalRing takes the top one for
scoring and keeps the rest in the per example results.

## When there is no single right answer

Sometimes "correct" is not a label. If you are evaluating summaries, or replies
to a customer, there is no key to check against. The usual approach is to have a
second model read each answer and score it against written instructions. EvalRing
supports this, and the written instructions are called a **rubric**, the same way
a teacher's marking guide is:

```python
from EvalRing import LLMJudgeEvaluator

evaluator = LLMJudgeEvaluator.from_rubric(
    rubric="Score 1-5 on whether the response answers the question accurately.",
    criteria="Factual accuracy relative to the reference answer",
    judge_model="gpt-4o",
    threshold=0.6,  # scores at or above this count as a pass
)
result = evaluator.evaluate(agent, dataset, task_name="qa-quality")
```

A rubric can be a plain sentence like this, or a structured `Rubric` object that
spells out what each score level means. You can also score several things at once
and weight them, for example accuracy twice as heavily as tone.

Worth knowing: a model grading another model is a useful signal, not ground
truth. Judges have preferences of their own, including a mild liking for longer
answers. Spot check a sample of the judge's verdicts by hand before you trust the
ranking.

## Command line

```
evalring check                       Show which provider settings were found
evalring info --dataset data.csv     Statistics and sanity checks for a data file
evalring models --output list.json   Write a starter candidate list
evalring run-suite ...               Run one evaluation across many candidates
```

Details in [docs/CLI.md](docs/CLI.md).

## Where results go

Everything is written under `_EvalRing/` in the folder you ran from, or wherever
`EVALRING_WORKSPACE` points. That includes per example results, readable reports,
the settings each run used, and the shared cache database.

These files contain your data and the models' replies to it, so treat them as
carefully as the data itself. The folder is excluded from git already. Do not
attach these files to a public bug report without reading them first.

## Documentation

| Document | What is in it |
| --- | --- |
| [docs/USAGE.md](docs/USAGE.md) | Running comparisons, output files, retries |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every environment variable, and provider setup |
| [docs/CLI.md](docs/CLI.md) | Command line reference |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Every public class and function |
| [docs/DESIGN_SPEC.md](docs/DESIGN_SPEC.md) | How it is built, and the trade-offs behind it |
| [docs/DATA.md](docs/DATA.md) | Where the example datasets come from, and responsible use |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and standards |
| [AGENTS.md](AGENTS.md) | Orientation for coding agents working in this repo |

The documentation ships inside the released package as well as living here, so it
travels with the code.

## Worked examples

Complete, runnable evaluations live in [`examples/`](examples/). They are not
installed with the package, so clone the repository if you want them.

- [`examples/semantic_analysis/`](examples/semantic_analysis/) is the smallest
  one. Start here.
- [`examples/hle/`](examples/hle/) evaluates hard exam questions from Humanity's
  Last Exam, ARC-Challenge, and GPQA, using a judge to check answers.
- [`examples/suicide_detection/`](examples/suicide_detection/) is a multi class
  risk classification task, including a multi role agent setup.

The suicide detection and clinical note examples involve sensitive material, and
their datasets carry conditions on how they may be shared. No data is included in
this repository. Read [docs/DATA.md](docs/DATA.md) before running them.

To be direct about it: EvalRing is a measurement tool. Nothing it produces is a
medical, diagnostic, or screening instrument, and a good score on a research
dataset says nothing about whether something is safe to use on real people.

## Status

EvalRing is before version 1.0. The main building blocks, datasets, agents, and
evaluators, are settled in shape, but smaller details may still change between
minor versions. Pin a version if you are depending on it. Changes are listed in
[CHANGELOG.md](CHANGELOG.md).

Bug reports and pull requests are welcome. If something in the documentation is
wrong or unclear, that counts as a bug worth reporting.

## License

[MIT](LICENSE). Use it for whatever you like.
