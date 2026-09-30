# TraSim GUI — User Guide

| Property | Value |
|----------|-------|
| Status   | Living Document |
| Purpose  | Usage of the Streamlit app and Windows distribution |

The Streamlit app (`gui/app.py`) wraps the complete
optimization workflow in a browser interface. End
users need no Python knowledge.

## Workflow

1. **Define inputs in Excel**
   - `mechanism.csv`: one row per lever with the
     columns `id, length_min, length_max,
     length_start, angle_min, angle_max,
     angle_start, pivot_x, pivot_y, pivot_z,
     axis_x, axis_y, axis_z, driver, coupled`
     (angles in degrees).
   - `targetcurve.csv`: columns `input_angle,
     output_angle` (degrees), optional third column
     `weight` for per-support-point fitness
     weighting.
   - Both international (comma/dot) and German
     (semicolon/comma) CSV conventions are
     detected automatically from the header line.
2. **Review optimization parameters**
   - After uploading both files, the app derives
     one length parameter (mm) and one angle
     parameter (degrees) per lever from the CSV
     ranges. Values and bounds are editable.
   - Equal minimum and maximum fix a parameter.
   - The **Vorschau** tab simulates the start
     mechanism and shows it together with the
     target curve in the usual HTML representation.
3. **Start the optimization**
   - Population size, generations, children per
     generation, mutation strength and seed are
     configured in the sidebar.
   - Live progress shows the current generation
     and the best fitness; a line chart tracks the
     fitness history.
4. **Download the result**
   - `simulation_result.html`: the usual result
     document (3D lever diagram + Soll/Ist curve).
   - `mechanism_optimized.csv`: the optimized
     mechanism, written in the convention of the
     input files.

## Starting the app (developer machine)

```powershell
pip install -r requirements.txt
streamlit run gui/app.py
```

## Windows distribution (end users)

Distribute a single executable so end users need
neither Python nor a repository checkout.

1. On a Windows machine with Python installed,
   create the executable with PyInstaller:

   ```powershell
   pip install pyinstaller
   pyinstaller --onefile --name TraSim ^
     --collect-all streamlit ^
     --add-data "gui;gui" ^
     run_trasim.py
   ```

2. `run_trasim.py` is a small launcher that starts
   Streamlit programmatically:

   ```python
   from streamlit.web import cli as stcli
   import sys

   sys.argv = [
       "streamlit",
       "run",
       "gui/app.py",
       "--global.developmentMode=false",
   ]
   sys.exit(stcli.main())
   ```

3. The resulting `dist/TraSim.exe` is copied to the
   end user's machine. Double-clicking it starts a
   local browser session; no installation step is
   required. The app only opens a local port on the
   user's own machine.

Note: the executable is roughly 150–250 MB because
it bundles Python, Streamlit and numpy.
