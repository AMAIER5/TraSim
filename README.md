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

## Interactive Optimization App (Streamlit)

End users can run the complete workflow (Excel CSV
input, parameter editing, preview, optimization,
HTML result) without any Python knowledge.

### Install in the virtual environment

```powershell
pip install -r requirements.txt
```

### Start the app

```powershell
streamlit run gui/app.py
```

### Workflow

1. Save `mechanism.csv` and `targetcurve.csv`
   from Excel (German and international CSV
   conventions are detected automatically).
2. Upload both files in the sidebar and review
   the optimization parameters in the table
   (lengths in mm, angles in degrees; equal
   min/max bounds fix a parameter).
3. Optional: open the **Vorschau** tab and click
   **Vorschau aktualisieren** to see the loaded
   mechanism and the target curve in the usual
   HTML representation.
4. Click **Optimierung starten** and follow the
   live progress.
5. Download the optimized mechanism
   (`mechanism_optimized.csv`) and the result
   document (`simulation_result.html`) from the
   **Ergebnis** tab.

### Windows distribution for end users

Build a single executable with PyInstaller so
end users do not need a Python installation.
Refer to `docs/user_guide_gui.md` for details.
