# Contributing

Thanks for helping MATNETS grow.

## Local Setup

```bash
python -m pip install -e ".[dev,docs]"
```

## Checks

```bash
pytest
ruff check .
mypy src
```

Keep changes focused, add tests for behavior changes, and update docs when the
public API changes.
