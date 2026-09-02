# Security Policy

## Supported versions

EvalRing is pre-1.0. Security fixes are applied to the latest released minor
version only.

| Version | Supported |
| --- | --- |
| 0.2.x | Yes |
| < 0.2 | No |

## Reporting a vulnerability

Do not open a public issue for a security problem.

Report it privately through GitHub Security Advisories:
<https://github.com/CoronRing/EvalRing/security/advisories/new>

Please include what the issue is, how to reproduce it, and what an attacker
could achieve. You should get an acknowledgement within a week. Once a fix is
released we will credit you in the advisory unless you would rather stay
anonymous.

## What is in scope

EvalRing handles API credentials and sends dataset content to third-party model
providers, so we are particularly interested in:

- API keys leaking into logs, run artifacts, error messages, or reports
- Dataset content being written somewhere it should not be
- Cache poisoning: one run reading another's results as if they were its own
- Path traversal through dataset paths, output directories, or model names
  (model names become directory names in suite output)
- Unsafe deserialization of run artifacts or model responses

## What is not in scope

- Prompt injection through dataset content that only affects the model's own
  output quality. EvalRing evaluates models; a model producing a wrong answer
  because the input told it to is a result, not a vulnerability. Prompt
  injection that escapes into *EvalRing's* control flow is in scope.
- Vulnerabilities in provider SDKs, model providers, or datasets. Report those
  upstream.
- Missing hardening in `examples/`, which is illustrative code.

## Operational notes for users

**Credentials.** EvalRing reads API keys from the environment; see
[`docs/CONFIGURATION.md`](docs/CONFIGURATION.md). It never writes a key to a
run artifact, a log line, or the `evalring check` output. Keep keys in your
environment or a gitignored `.env`, never in a committed config file.

**Run artifacts contain your data.** Everything under `_EvalRing/` — the SQLite
cache, `all_cases.csv`, `result.md`, per-model logs — contains prompts, model
responses, and dataset text. The directory is gitignored by default. Treat it
with the same care as the dataset it came from, and do not attach it to a
public issue.

**Third-party transmission.** Running an evaluation sends your dataset text to
whichever provider you configured. Confirm you are permitted to do that before
evaluating restricted or personal data.
