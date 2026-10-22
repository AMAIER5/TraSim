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
   - Optional columns: `ref_x, ref_y, ref_z`
     (explicit reference direction of the lever;
     when all three are empty it is selected
     automatically) and `pivot_on` (mixer lever:
     `lever_id@angle_deg` moves the lever's pivot
     onto the endpoint of the referenced lever
     at the given lever angle; see
     `docs/Architecture.md`, section *pivot_on
     Relationship*, and the example
     `examples/mechanism_optimized.csv`).
   - `targetcurve.csv`: columns `input_angle,
     output_angle` (degrees), optional third column
     `weight` for per-support-point fitness
     weighting.
   - Both international (comma/dot) and German
     (semicolon/comma) CSV conventions are
     detected automatically from the header line.
2. **Review and edit the target curve**
   - After uploading, the target-curve support
     points are shown in an editable table
     (input angle, output angle, weight).
   - Values can be changed; rows can be added and
     deleted. Input angles must be unique and
     are sorted ascending automatically; weights
     must be non-negative (clamped to 100).
   - Two charts next to the table always show the
     current target curve and the per-point
     weights; they update immediately after each
     edit.
   - The edited curve is used for the preview,
     the optimization and the HTML report.
3. **Review and edit the mechanism**
   - After uploading, the mechanism definition is
     shown as an editable table with one row per
     lever (id, lengths, angles in degrees, pivot,
     axis, optional reference direction, driver,
     coupled, pivot_on).
   - Values can be changed; levers can be added
     and deleted on the fly for fast iteration.
     Invalid input (missing values, duplicate ids,
     malformed pivot_on) is rejected with a clear
     message.
   - The optimization parameters below and the
     preview update automatically after each edit.
4. **Review optimization parameters**
   - The app derives
     one length parameter (mm) and one angle
     parameter (degrees) per lever from the CSV
     ranges. Values and bounds are editable.
   - Equal minimum and maximum fix a parameter.
   - The **Vorschau** tab simulates the start
     mechanism and shows it together with the
     target curve in the usual HTML representation.
5. **Adopt optimized values**
   - After an optimization run, the parameter table
     shows an **Optimiert** column with the values
     of the last run next to the original start
     values.
   - A per-parameter **Übernehmen** button (and an
     **Alle optimierten Werte als Startwert
     übernehmen** button for all parameters at once)
     copies the optimized value into the start value
     column, so the next iteration starts from the
     optimized mechanism.
6. **Start the optimization**
   - Population size, generations, children per
     generation, mutation strength and seed are
     configured in the sidebar.
   - Live progress shows the current generation
     and the best fitness; a line chart tracks the
     fitness history.
7. **Download the result**
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
   pyinstaller --onefile --name TraSim --collect-all streamlit --add-data "gui;gui" --add-data "examples;examples" --collect-submodules analysis --collect-submodules core --collect-submodules mechanics --collect-submodules mechanism_io --collect-submodules model --collect-submodules optimization --collect-submodules simulation --collect-submodules validation run_trasim.py
   ```

   Note: use this single-line form or the `TraSim.spec` file
   (`pyinstaller TraSim.spec`); multi-line shell continuations
   differ between cmd.exe (`^`) and PowerShell (`` ` ``).

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
