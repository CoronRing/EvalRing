# EvalRing Documentation

These documents ship with the code and are the authoritative description of the
system. A change that alters behaviour is expected to update the relevant
document in the same pull request — see the table in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Start here

| Document | Contents |
| --- | --- |
| [DESIGN_SPEC.md](DESIGN_SPEC.md) | Architecture, object model, runtime flow, concurrency and retry model, persistence contract, extension points |
| [API_REFERENCE.md](API_REFERENCE.md) | Every public class and function in the `EvalRing` namespace |
| [CONFIGURATION.md](CONFIGURATION.md) | Environment variables, provider precedence, worked setups |
| [CLI.md](CLI.md) | The `evalring` command |
| [USAGE.md](USAGE.md) | Running single evaluations and suites, output artifacts, retry workflow |
| [DATA.md](DATA.md) | Dataset provenance, licensing, responsible use |
| [RELEASING.md](RELEASING.md) | Version bump, build, publish |

## Reference material

| Document | Contents |
| --- | --- |
| [providers/](providers/) | Per-provider setup notes: [openai](providers/openai.md), [gemini](providers/gemini.md), [openrouter](providers/openrouter.md), [litellm](providers/litellm.md) |
| [dataset/](dataset/) | Per-dataset ingest notes: [HLE](dataset/hle.md), [ARC-Challenge](dataset/arc_challenge.md) |
| [sample/suicide_detection/](sample/suicide_detection/README.md) | End-to-end walkthrough of the suicide-detection example, including every agent mode |

## Scope

The documents describe the code under:

- [src/EvalRing](../src/EvalRing) — the installable package
- [examples](../examples) — runnable task applications, not packaged

## What is covered

- Unified classification output handling in the core framework: plain string
  labels and structured class-score mappings
- Suicide-detection agent modes: `single-class`, `multi-class-chance`,
  `base-vs-rest-binary` with a configurable `base_class`, `per-class-score`
- In-place retry driven by a previous run's `Meta.json`
- Multi-model suite orchestration with per-model subprocess isolation
- Response caching across runs, and the artifacts each run writes

## Other entry points

- [AGENTS.md](../AGENTS.md) — orientation for coding agents working in this
  repository
- [SECURITY.md](../SECURITY.md) — reporting vulnerabilities, handling
  credentials and run artifacts
- [CHANGELOG.md](../CHANGELOG.md) — what changed in each release
