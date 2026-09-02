# Releasing

EvalRing publishes to PyPI as [`evalring`](https://pypi.org/project/evalring/).
The import name stays `EvalRing`.

## Versioning

The version lives in exactly one place:
[`src/EvalRing/__init__.py`](../src/EvalRing/__init__.py). `pyproject.toml`
reads it through `[tool.setuptools.dynamic]`, so there is nothing to keep in
sync.

Semantic versioning, with pre-1.0 caveats:

| Change | Bump |
| --- | --- |
| Breaking change to a public signature or to environment-variable precedence | minor, while 0.x |
| New capability, backward compatible | minor |
| Bug fix, docs, internals | patch |

## Steps

1. **Confirm the tree is green.**

   ```bash
   ruff check src tests
   ruff format --check src tests
   mypy
   pytest
   ```

2. **Update the changelog.** Move everything under `Unreleased` into a new
   version heading with today's date, and add the comparison links at the
   bottom.

3. **Bump the version.**

   ```python
   # src/EvalRing/__init__.py
   __version__ = "0.3.0"
   ```

   Update the supported-versions table in [`SECURITY.md`](../SECURITY.md) if
   the supported minor changed.

4. **Build and inspect locally.**

   ```bash
   rm -rf dist build
   python -m build
   twine check dist/*
   ```

   The wheel must contain exactly one top-level package:

   ```bash
   python -c "import zipfile,glob; print(sorted({n.split('/')[0] for n in zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist()}))"
   ```

   Expect `['EvalRing', 'evalring-<version>.dist-info']`. Anything else means
   packaging configuration has drifted.

5. **Smoke-test the artifact in a clean environment.**

   ```bash
   python -m venv /tmp/relcheck && /tmp/relcheck/bin/pip install dist/evalring-*.whl
   /tmp/relcheck/bin/python -c "import EvalRing; print(EvalRing.__version__)"
   /tmp/relcheck/bin/evalring --version
   ```

   Install the bare wheel, not `[all]` — this is the check that optional
   dependencies really are optional.

6. **Commit and tag.**

   ```bash
   git commit -am "Release 0.3.0"
   git tag -a v0.3.0 -m "Release 0.3.0"
   git push origin main --follow-tags
   ```

7. **Publishing happens automatically.** The tag triggers
   [`.github/workflows/release.yml`](../.github/workflows/release.yml), which
   rebuilds, verifies the tag matches the packaged version, and publishes
   through PyPI Trusted Publishing.

8. **Write the GitHub release** from the changelog section.

## One-time PyPI setup

Trusted Publishing means no API token is stored in the repository. Configure it
at <https://pypi.org/manage/project/evalring/settings/publishing/>:

| Field | Value |
| --- | --- |
| Owner | `CoronRing` |
| Repository | `EvalRing` |
| Workflow | `release.yml` |
| Environment | `pypi` |

Create a matching `pypi` environment in the repository settings, ideally with a
required reviewer so a publish cannot happen unattended.

For the very first upload, PyPI's "pending publisher" flow lets you register
the project name before it exists.

## Testing the pipeline

Publish to TestPyPI first if you want to rehearse: add a TestPyPI pending
publisher and temporarily point the `pypa/gh-action-pypi-publish` step at
`https://test.pypi.org/legacy/`. Revert before tagging the real release.
