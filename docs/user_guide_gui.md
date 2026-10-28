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
     `lever_id` moves the lever's pivot
     onto the endpoint of the referenced lever;
     its pivot/axis columns are ignored, see
     `docs/Architecture.md`, section *pivot_on
     Relationship*, and the example
     `examples/mechanism_optimized.csv`).
   - `targetcurve.csv`: columns `input_angle,
     output_angle` (degrees), optional third column
     `weight` for per-support-point fitness
     weighting.
   - **Start angle semantics:** the initial start
     angle (`angle_start`, before optimization)
     of every lever describes the position the
     lever occupies at the **middle support
     position** of the target curve.  The middle
     position is the n-th support position with
     n = (number of support positions) / 2,
     mathematically rounded (round half up); with
     seven support positions at -30, -20, -10, 0,
     10, 20, 30° the support position at 0° is the
     reference point for the start angle.
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
   - This single table is the only place where
     start values, bounds and the optimization
     release of every lever property are
     maintained. Next to `length_start` and
     `angle_start` two columns each are shown:
     an **Optimierung** checkbox (release of the
     parameter for the optimization, enabled by
     default) and a read-only **Optimiert** value
     from the last optimization run.
   - A click on the **Übernehmen** checkbox next
     to an **Optimiert** value adopts that single
     optimized value as the new start value.
   - A cleared checkbox fixes the parameter at
     its start value (minimum = maximum), so it
     is excluded from the optimization.
   - The **Optimiert** cells are cleared as soon
     as the corresponding start value is edited
     or adopted, so every displayed value is
     always the value the diagrams are built
     from.
   - Values can be changed; levers can be added
     and deleted on the fly for fast iteration.
     Invalid input (missing values, duplicate ids,
     malformed pivot_on) is rejected with a clear
     message.
   - The **Vorschau** tab simulates the start
     mechanism and shows it together with the
     target curve in the usual HTML representation;
     press **Vorschau aktualisieren** to rebuild
     it after edits.
4. **Adopt optimized values**
   - After an optimization run, the **Alle
     optimierten Werte als Startwerte übernehmen**
     button writes the optimized length and angle
     values of the last run directly into the
     `length_start` / `angle_start` cells of the
     mechanism table, so the next iteration starts
     from the optimized mechanism.
   - Optimization of pivot positions and axis
     orientations may be added later; today they
     stay fixed.
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
     In the 3D mechanism diagram the button
     **Nullage der Winkeldefinitionen einblenden**
     optionally overlays the zero position of each
     lever's angle definition as a dashed, light-red
     line at every axis.
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
