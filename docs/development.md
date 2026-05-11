# Development Guide

This guide provides instructions for setting up a local development environment, running tests, and contributing to MATNETS.

---

## 1. Local Setup

First, ensure you have Python 3.11 or higher. Clone the repository and install the library in editable mode along with its development and documentation dependencies.

```bash
# Clone the repository
git clone https://github.com/yourusername/matnets.git
cd matnets

# Create a virtual environment
python -m venv .venv

# Activate the environment
# On Windows: .\.venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate

# Install in editable mode with docs and dev dependencies
python -m pip install -e ".[dev,docs]"
```

*(Note: Depending on your `pyproject.toml`, you might just need `python -m pip install -e .` followed by `pip install pytest mkdocs-material mkdocs-material-extensions pymdown-extensions`)*

---

## 2. Running Tests

We use `pytest` for all unit and integration testing. It is critical to ensure the test suite passes before submitting any changes.

```bash
python -m pytest
```

To run a specific test file:

```bash
python -m pytest tests/test_dense.py
```

---

## 3. Running Examples

To verify behavior manually, run the provided example scripts:

```bash
python examples/basic_forward.py
python examples/five_hidden_net.py
python examples/matrix_architectures.py
```

---

## 4. Syntax and Type Checking

Ensure there are no glaring syntax issues across the codebase:

```bash
python -m compileall -q src examples tests
```

*(If the project adopts `mypy` or `ruff` in the future, run those checks here as well).*

---

## 5. Building Documentation

This documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

To build the site locally and strictly verify for broken links:

```bash
python -m mkdocs build --strict
```

To start a live-reloading local server while you edit the Markdown files:

```bash
python -m mkdocs serve
```

The site will be available at `http://127.0.0.1:8000`.

---

## 6. Package Layout Overview

Understanding the structure of the repository will help you navigate the codebase:

```text
src/matnets/
├── _params.py          # MatrixParams PyTree definition and init()
├── _dense.py           # Core tensor contraction primitive
├── activations.py      # Element-wise and determinant-gated activations
├── conv.py             # Pooling and structural pooling operations
├── lax/
│   ├── conv.py         # 1D and 2D matrix convolutions
│   └── attention.py    # Matrix attention primitive
└── nn/
    └── recurrent.py    # RNN, LSTM, and GRU step patterns

tests/                  # Pytest test suite
examples/               # Runnable example scripts
docs/                   # MkDocs Markdown source files
```