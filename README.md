# TraSim
Multi-stage transmission simulation

## Python dependencies

This project uses `numpy`, `pytest`, and the built-in Python `math` module in its unit tests.

### Install in the virtual environment

1. Activate the virtual environment:
   - PowerShell: `e:/Temp/TraSim/.venv/Scripts/Activate.ps1`
2. Install dependencies:
   - `pip install numpy pytest`

### Verify installation

Run:

```powershell
python -c "import numpy; import pytest; print(numpy.__version__)"
```
## Running Tests

Run all tests

```bash
pytest
```

Verbose output

```bash
pytest -v
```

Stop after first failure

```bash
pytest -x
```

Run only one module

```bash
pytest tests/test_vector3d.py
```

Run a single test

```bash
pytest tests/test_vector3d.py::test_cross_product
```