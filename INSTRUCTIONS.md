# Instructions: running `lumi_language_id` with a newer numpy

## The problem

The original package depended on `fasttext` (PyPI `fasttext==0.9.3`, unchanged
since 2020). That package's `predict()` calls `np.array(probs, copy=False)`,
which **numpy 2.x rejects**:

```
ValueError: Unable to avoid copy while creating an array as requested.
```

numpy 2.0 changed `copy=False` to mean "never copy, raise if a copy is needed"
(the old behaviour is now `np.asarray(...)`). So any install with numpy >= 2
crashes on the first detection, even though this project's own code is fine.

## The fix

Swap the dependency from `fasttext` to **`fasttext-numpy2`** — a drop-in fork
that is "fasttext with one line changed for numpy 2". It installs the same
`fasttext` Python module, so no application code changes are needed. The
dependency lives in `pyproject.toml`:

```toml
dependencies = [
    "fasttext-numpy2",
    "numpy",
    "ftfy",
    "langcodes>=2",
]
```

> Note: `fasttext-numpy2` is a single-maintainer community fork. It is the
> pragmatic choice because upstream `fasttext` is unmaintained. The one-line
> `np.asarray` change is backward-compatible, so it also still works on
> numpy 1.x.

## Packaging

The project uses a modern `pyproject.toml` (PEP 621) with the **hatchling**
build backend. Metadata, runtime dependencies, the optional `train` extra, and
a `dev` dependency group all live there; there is no `setup.py`. The pinned
Python (`.python-version`) and `uv.lock` give a reproducible dev environment.

## Setup with `uv`

`fasttext-numpy2` has no prebuilt wheel for macOS arm64, so it builds from
source on install (a C++ toolchain — Xcode command-line tools — is required).
`uv sync` creates the virtual environment (using the pinned Python 3.12),
installs the project plus the `dev` group, and resolves against `uv.lock`:

```sh
uv sync
```

To also install the training extras (scikit-learn, only needed to rebuild the
tuned model):

```sh
uv sync --extra train
```

## Verify it works

```sh
uv run python -c "
from lumi_language_id.tuned import TunedLanguageIdentifier
lid = TunedLanguageIdentifier.load()
print(lid.detect_language('these are words'))
print(lid.detect_language('aquí hay algunas palabras'))
"
```

Run the test suite (pytest comes from the `dev` group):

```sh
uv run pytest
```

## Build distributions

```sh
uv build
```

This produces a wheel and sdist in `dist/`, both containing the bundled model
data files (`lid.176.ftz`, `tuned.npz`).

## Publishing to PyPI

The package is published under the distribution name **`lumi-language-id-2`**
(the original `lumi-language-id` name is taken by the now-archived upstream
project). The import name is unchanged — users still `import lumi_language_id`.

Validate the built artifacts, then upload with an API token:

```sh
uvx twine check dist/*

# Dry run against TestPyPI first (optional, recommended):
uv publish --publish-url https://test.pypi.org/legacy/ --token <testpypi-token>

# Real upload:
uv publish --token <pypi-token>
```

A published version can never be reused, so bump `version` in `pyproject.toml`
before each release.

## Tested configurations

Verified building from source and detecting correctly with:

- **numpy 2.5.0** on Python 3.12, 3.13, and 3.14
- **numpy 1.26.4** on Python 3.12 (backward compatibility)

Python 3.12 is the recommended default. Newer Pythons build fine here but
depend on the source build succeeding on your platform.
