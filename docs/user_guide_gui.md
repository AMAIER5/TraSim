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

## Mischer-Modus (Bremsklappenbetätigung)

Enthält die geladene `mechanism.csv` einen Hebel mit
einem Eintrag in der Spalte `pivot_on`, erkennt die GUI das
automatisch und schaltet in den Mischer-Modus
(Bremsklappenbetätigung): der Erfolg der
Optimierung wird über **fünf Bremsstellungen**
(0 % / 25 % / 50 % / 75 % / 100 % Bremsweg)
bewertet.

### Bedeutung des pivot_on-Hebels

- `pivot_on = <Verstellhebel-Index>` bezeichnet den
  Hebel, auf dessen Endpunkt der Drehpunkt des
  pivot_on-Hebels (Mischerhebel) sitzt. Der
  Verstellhebel selbst bleibt ein ganz normaler
  Hebel mit festem Drehpunkt — sein Winkelbereich
  (`angle_start` bis `angle_max`) definiert
  zugleich den Bremsweg: `angle_start` ist die
  ungebremste Referenzstellung, `angle_max` die
  voll gebremste Stellung. Die veraltete Syntax
  `lever_id@winkel` wird mit einer Fehlermeldung
  abgelehnt.
- Die Spalten `pivot_x`/`pivot_y`/`pivot_z` und
  `axis_x`/`axis_y`/`axis_z` des Mischerhebels
  werden ignoriert: Drehpunkt, Achse und
  Referenzrichtung werden bei jedem Build
  dynamisch vom Endpunkt des Verstellhebels
  übernommen. Die Winkel-Grenzen des
  Mischerhebels behalten ihre übliche Bedeutung
  (Hebelwinkel relativ zur Referenzrichtung).
- Genau ein Hebel darf `pivot_on` besitzen; der
  Verweis darf nicht auf einen anderen pivot_on-
  Hebel zeigen.

### CSV-Format im Überblick

```csv
id,...,pivot_on
1,...,lever3
2,...,
3,...,
```

### Aufbau der zwei Sollkurven-Bereiche in der GUI

- Wie im Normalmodus wird die **Zielkurve ungebremst**
  (`targetcurve.csv`) hochgeladen und in der
  editierbaren Tabelle gepflegt; sie beschreibt das
  Ausgangsverhalten bei 0 % Bremsweg.
- Zusätzlich erscheint der Ausbereich
  „Mischer-Modus: Zielkurve voll gebremst
  (target_curve_braked.csv)“: hier wird die zweite
  Sollkurve für die voll gebremste Stellung
  (100 % Bremsweg) hochgeladen — eigene Datei
  oder per Knopfdruck **Beispiel-Dateien laden**.
  Ohne diese zweite Kurve startet der Modus nicht.
- Die Zielkurven der Bremsstellungen 25 %, 50 %
  und 75 % werden zwischen den beiden
  Sollkurven interpoliert; die Optimierung
  bewertet den Kandidaten über alle fünf
  Bremsstellungen.

### Bremsstellungs-Schalter

- Im Tab **Vorschau** wählt der Schalter
  **Bremsstellung** (0 % bis 100 %), welche
  Bremsstellung im Mechanismus-Snapshot der
  Vorschau dargestellt wird.
- Im Tab **Ergebnis** wählt derselbe Schalter
  die Bremsstellung des Ergebnisses (optimierte
  Geometrie); die Soll-/Ist-Kurve des Berichts
  ist jeweils mit „Soll Bremse XX %“ / „Bremse
  XX %“ beschriftet.

### Beispiel: kompletter Durchlauf am Mischbrems-Beispiel

1. **CSV vorbereiten:** `examples/mixed_brake_mechanism.csv`
   (Hebel 1 `pivot_on = lever3` = Mischerhebel mit ±12°
   Winkelbereich, Hebel 2 = Abtriebshebel, Hebel 3 =
   Bremshebel mit Winkelbereich 170°–190°,
   `angle_start = 180°`); die Sollkurven sind
   `examples/target_curve_unbraked.csv` und
   `examples/target_curve_braked.csv` (jeweils
   11 Stützpunkte von −10° bis +10°, beide mit
   0° Ausgangswinkel am rechten Rand). Das
   fertige Beispiel-Setup läuft alternativ per
   `python examples/optimize_mixed_brake.py`
   (vom Projektverzeichnis aus, um die Paket-
   Importe aufzulösen).
2. **GUI-Schritte:** Streamlit-App starten, dann
   `mixed_brake_mechanism.csv` als mechanism.csv
   und `target_curve_unbraked.csv` als
   targetcurve.csv hochladen; im aufgehenden
   Ausbereich „Mischer-Modus“ entweder
   `target_curve_braked.csv` hochladen oder
   **Beispiel-Dateien laden** drücken. Beide
   Sollkurven bei Bedarf in den Tabellen
   editieren, Grenzen/Freigaben der Hebel in
   der Parametertabelle prüfen (der Mischerhebel
   hat keine eigenen Drehpunkt-Parameter), in
   der Sidebar Population/Generationen/Seed
   setzen, im Tab **Vorschau** über den
   **Bremsstellung**-Schalter und **Vorschau
   aktualisieren** den Startmechanismus prüfen
   und schließlich **Optimierung starten**
   drücken.
3. **Ergebnis-Interpretation:** Der Kopf zeigt
   Stopgrund, beste Fitness, Generationen und
   den minimalen Übertragungswinkel des besten
   Kandidaten (Soft-Kriterium); der
   Fitness-Verlauf über die Generationen zeigt,
   ob die Konvergenz abflacht. Über den
   **Bremsstellung**-Schalter werden Soll- und
   Istkurve der fünf Bremsstellungen im
   Vergleich durchgeschaltet — am
   0-%-Ende entspricht die Istkurve der
   ungebremsten, am 100-%-Ende der gebremsten
   Sollkurve, und die Zwischenstellungen zeigen
   die interpolierten Verläufe. Große Abweichungen
   bei einzelnen Bremsstellungen deuten auf
   einen Kompromiss des Optimierers hin; dann
   Grenzen oder Sollkurven nachschärfen und
   mit **Alle optimierten Werte als Startwerte
   übernehmen** iterieren. **Ergebnis-HTML
   herunterladen** liefert den Bericht
   (`mixed_brake_result.html`), **Optimierte
   Mechanik herunterladen (CSV)** die optimierte
   Geometrie als `mechanism_optimized.csv` in der
   Konvention der Eingabedateien.

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
