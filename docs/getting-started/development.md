# Development

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Run Examples

```powershell
.\.venv\Scripts\python.exe examples\basic_forward.py
.\.venv\Scripts\python.exe examples\five_hidden_net.py
.\.venv\Scripts\python.exe examples\matrix_architectures.py
```

## Syntax Check

```powershell
.\.venv\Scripts\python.exe -m compileall -q src examples tests
```

## Build Documentation

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[docs]"
.\.venv\Scripts\python.exe -m mkdocs build --strict
```

## Package Layout

```text
src/matnets/
  _params.py          MatrixParams and init()
  _dense.py           core dense matrix primitive
  lax/conv.py         convolution-like matrix primitives
  lax/attention.py    matrix attention primitive
  nn/recurrent.py     RNN, LSTM, and GRU step patterns
tests/                test suite
examples/             runnable examples
docs/                 user documentation
```
