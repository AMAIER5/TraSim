# Performance

Calculation speed of the kinematic simulation, and how it is
improved. Two levers are used:

1. **Refactoring** — allocation-free scalar hot path in pure Python
   (always active, no build step required).
2. **Transpilation** — optional Cython (Python → C) build of the
   hot residual function (native speed, pure-Python fallback).

Both levers are **bit-identical** to the object-based reference
path: every floating-point operation happens in the same order,
so simulation results do not change by a single ulp. This is
enforced by `tests/test_fast_residual.py`.

---

## 1. Hot path

One simulation step solves the output angle of one stage with
Brent's method (`solver/angle_solver.py`, `solver/root_solver.py`).
Each residual evaluation previously went through the object API:

```
Lever.direction()          -> 6 Vector3D allocations (Rodrigues)
Lever.end_position()       -> Vector3D * scalar, Point3D allocation
constraints.rod_length_error() -> attribute access, sqrt
```

With ~14 residual evaluations per solved angle and thousands of
angles per optimization run, these allocations dominated the
runtime (measured: ~85 % of total simulation time).

## 2. Refactoring (pure Python, always active)

- `Lever` caches the constant Rodrigues terms
  `axis x reference` and `axis . reference` once per lever
  (`mechanics/lever.py`).
- `Lever.end_position_components(angle)` computes the endpoint as
  plain floats without allocating `Vector3D`/`Point3D` objects.
- `Lever.endpoint_constants()` exposes the fixed scalar frame of a
  lever (pivot, axis, reference, cross, dot, length) so solvers
  unpack it only once per stage.
- `solver/_rod_residual.py` provides the fused scalar residual
  `stage_residual(...)` and the factory `make_stage_residual(...)`
  used by `solver/objective.py::create_stage_objective`.
- `solver/root_solver.py` hoists `2 * math.ulp(1.0)` into a module
  constant.

Measured effect on the benchmark below: **~4.5x faster** than the
original object-based path, with bit-identical results.

## 3. Transpilation (optional native build)

`solver/_rod_residual.pyx` is the Cython transpilation of
`solver/_rod_residual.py`. Build it in place:

```bash
pip install cython
python setup.py build_ext --inplace
```

The compiled extension (`solver/_rod_residual.*.so`) takes import
precedence over the pure-Python `.py` file in the same package, so
`from solver._rod_residual import ...` automatically uses native
code after the build — no source changes, no runtime flag.

Without a build, the pure-Python module is used. Cython is a build
time dependency only; `requirements.txt` is unchanged.

When packaging with PyInstaller (`TraSim.spec`), build the
extension **before** running PyInstaller so the native module is
bundled; otherwise the pure-Python module is bundled.

Measured effect on the benchmark below: additional **~1.5x** on top
of the refactoring (**~6.8x total** vs. the original path), again
bit-identical.

## 4. Benchmark

```bash
python examples/benchmark_simulation.py
```

Simulates a fixed three-stage mechanism over 481 input angles
(0.5° step) and prints the best wall-clock time of 5 runs plus
which residual module (pure Python / native) is active.

Reference measurement (CPython 3.12, Linux x86-64):

| Variant                    | Time   | Speedup |
|----------------------------|--------|---------|
| original object path       | 0.108 s| 1.0x    |
| refactoring (pure Python)  | 0.024 s| ~4.5x   |
| refactoring + native build | 0.016 s| ~6.8x   |

## 5. Rules for future changes

- Keep the scalar fast path bit-identical to the object path;
  extend `tests/test_fast_residual.py` when adding functions.
- If a function is added to `_rod_residual`, it must exist in both
  `_rod_residual.py` and `_rod_residual.pyx` with identical
  arithmetic.
- Do not add a hard dependency on Cython or any compiler to the
  runtime path.
