# Koppelgetriebe Simulator — TraSim

**Architecture Specification**


| Property | Value                                   |
| -------- | --------------------------------------- |
| Version  | 0.3.0                                   |
| Status   | Updated to match current implementation |
| Language | English (source code)                   |
| Units    | mm / deg (user), mm / rad (internal)    |


---

# 1. Purpose

The project provides a simulation framework for planar and spatial linkage mechanisms.

The primary goals are:

- Reliable kinematic simulation
- Arbitrary number of linkage stages
- Support for spatial mechanisms
- High numerical stability
- Automatic optimization using evolutionary algorithms
- Fully reproducible engineering calculations

The simulator supports:

- 3D lever axes
- Fixed pivot positions
- Rigid coupling rods
- Multi-stage mechanisms
- **CSV**-based mechanism definitions and simulation configurations
- Stage and mechanism motion validation
- Evolutionary optimization
- Adaptive motion sampling
- CSV export of simulation results

---

# 2. Coordinate System

The simulator uses a **right-handed Cartesian coordinate system**.

```text
    +Z
    |
    |
    |
    o──────── +X
   /
  /
 +Y
```

Coordinates are represented by `Point3D(x, y, z)`.

Unit: millimeter \[mm\].

---

# 3. Units

### User Interface


| Quantity | Unit |
| -------- | ---- |
| Length   | mm   |
| Angle    | deg  |


### Internal Calculations


| Quantity | Unit |
| -------- | ---- |
| Length   | mm   |
| Angle    | rad  |


Only input/output layers perform unit conversion.

No internal class stores angles in degrees.

---

# 4. Rotation Convention

Internally all rotations follow the right-hand rule.

Positive rotations are counter-clockwise when looking along the positive rotation axis.

Rotations are represented using **quaternions** (`Quaternion`) and a higher-level `Rotation` utility class.

Advantages:

- No gimbal lock
- Stable interpolation
- Efficient composition

---

# 5. Software Architecture

The project is organized into independent layers.

```text
Application (examples/)
    |
    v
Optimization
    |
    v
Analysis
    |
    v
Validation
    |
    v
Simulation
    |
    v
Solver
    |
    v
Mechanics
    |
    v
Mechanism I/O
    |
    v
Model
    |
    v
Core
```

Dependencies always point downward.

Higher layers shall never be referenced by lower layers.

---

# 6. Geometry Layer (core/)

The geometry layer contains mathematical primitives.

### Point3D

Represents a position in space.

Properties: `x`, `y`, `z`.

Operations:

- Addition with `Vector3D` (translation)
- Subtraction (`Point3D - Point3D` → `Vector3D`, `Point3D - Vector3D` → `Point3D`)
- Distance to another point
- Midpoint
- Translation by vector
- `almost_equal` with configurable tolerance
- NumPy array conversion

### Vector3D

Represents direction or displacement. Immutable (`frozen=True`, `slots=True`).

Operations:

- Addition, subtraction, negation
- Scalar multiplication and division
- Dot product (`dot`, `@` operator)
- Cross product
- Norm (squared and full)
- Normalization
- Distance to another vector
- Angle to another vector
- `is_zero` check
- `almost_equal` with configurable tolerance
- NumPy array conversion

Module-level tolerance: `DEFAULT_TOLERANCE = 1e-12`.

Central tolerances imported from `core/tolerance.py`:


| Constant           | Value | Application             |
| ------------------ | ----- | ----------------------- |
| `LENGTH_TOLERANCE` | 1e-9  | Length comparisons (mm) |
| `ANGLE_TOLERANCE`  | 1e-12 | Angle comparisons (rad) |
| `VECTOR_TOLERANCE` | 1e-12 | Vector comparisons      |


### Quaternion

Immutable quaternion `(w, x, y, z)` for 3D rotations.

Operations:

- Identity
- Construction from axis-angle (`from_axis_angle`)
- Conjugate
- Inverse
- Hamilton product
- Vector rotation (`q * v * q⁻¹`)
- Rotation matrix conversion (`to_rotation_matrix`)
- Normalization

### Rotation

High-level rotation utility class (`core/rotation.py`) providing engineering-oriented static methods:

- `rotate_vector(vector, axis, angle_rad)` — rotate a vector around an axis
- `from_two_vectors(source, target)` — shortest rotation mapping source → target
- `align_z_axis(target)` — rotation aligning global Z with a target vector

---

# 7. Mechanism Definition Model (model/)

Input data is separated from simulation objects.

The external definition model describes a mechanism before construction.

### LeverDefinition

```text
LeverDefinition
├── id
├── pivot : Point3D
├── axis  : Vector3D
├── length_min
├── length_max
├── length_start
├── angle_min
├── angle_max
├── angle_start
├── driver  : int | None
└── coupled : int | None
```

Properties: `is_driver`, `is_coupled`.

### MechanismDefinition

```text
MechanismDefinition
├── LeverDefinition 1
├── LeverDefinition 2
├── LeverDefinition 3
└── ...
```

Contains:

- Fixed geometry
- Optimization ranges
- Linkage relationships
- Unique ID validation
- Access helpers (`get_lever`, `input_lever`, `coupled_levers`, `driven_levers`)

### SimulationConfig

Global optimization and simulation parameters:

```text
SimulationConfig
├── population_size
├── children_per_generation
├── generations
├── target_error
├── mutation_rate
├── elite_size
├── motion_start
├── motion_end
└── motion_step
```

---

# 8. CSV Interface (mechanism\_io/)

**CSV** files are the external user interface.

The **CSV** layer is responsible for:

- Parsing input files
- Writing reproducible definitions
- Unit conversion (deg → rad on read, rad → deg on write)
- Validation of input data

The **CSV** layer does not perform simulation.

### CsvReader

Static methods:

- `read_simulation(path)` → `SimulationConfig`
- `read_mechanism(path)` → `MechanismDefinition`

### CsvWriter

Static methods:

- `write_simulation(config, path)`
- `write_mechanism(mechanism, path)`

### Mechanism CSV Format

Example:

```csv
id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled
1,40,100,60,-40,40,0,0,0,0,0,0,1,,
2,30,90,45,-60,60,0,100,0,0,0,0,1,1,
```

### Driver Relationship

A driver defines a mechanical stage.

Example: `Lever 2 driver = 1` creates:

```text
Stage(
    input  = Lever 1,
    output = Lever 2
)
```

### Coupled Relationship

A coupled lever maintains a constant angular relation.

Example: `Lever 3 coupled = 2` means:

```text
angle(Lever3) - angle(Lever2) = constant
```

A lever shall not be controlled by both `driver` and `coupled`.

If both are specified: **coupled has priority**, driver is ignored.

---

# 9. Mechanical Components (mechanics/)

### Lever

A lever is immutable (`frozen=True`, `slots=True`).

```text
Lever
├── pivot              : Point3D
├── axis               : Vector3D
├── length             : float
├── _normalized_axis   : Vector3D  (cached, init=False)
└── _reference_direction : Vector3D (cached, init=False)
```

The endpoint is calculated from:

```text
tip = pivot + direction(angle) * length
```

The lever uses **Rodrigues' rotation formula** with a reference direction that is **guaranteed to be perpendicular to the rotation axis**. The normalized axis and reference direction are both cached in `__post_init__`.

The lever does not store dynamic angle state.

#### Reference Direction Convention

The initial lever direction (at angle 0) is automatically selected based on the **dominant axis** of the rotation axis vector to ensure it is never parallel to the rotation axis:


| Rotation Axis Dominant Component | Reference Direction at 0° |
| -------------------------------- | ------------------------- |
|                                  | axis\_x                   |
|                                  | axis\_y                   |
|                                  | axis\_z                   |


This ensures the lever can always rotate meaningfully, even when the rotation axis is aligned with a cardinal axis. Without this convention, a lever with X-axis as its rotation axis would have a degenerate initial direction parallel to its axis, resulting in no motion.

Methods:

- `direction(angle_rad)` → `Vector3D` (Rodrigues rotation of the reference direction)
- `end_position(angle_rad)` → `Point3D`

```


### Rod

A rod is an ideal rigid body.

```text
Rod
├── point_a : Point3D
├── point_b : Point3D
└── length  : float
```

Constraint: `distance(point_a, point_b) == length`

Factory: `Rod.from_points(point_a, point_b)` calculates length automatically.

### Stage

A stage consists of two levers connected by an ideal rod. Immutable (`frozen=True`, `slots=True`).

```text
Stage
├── input_lever  : Lever
├── output_lever : Lever
├── rod_length   : float
├── input_angle_offset  : float  (installation offset, rad)
├── output_angle_offset : float  (installation offset, rad)
├── input_angle_min  : float
├── input_angle_max  : float
├── output_angle_min : float
├── output_angle_max : float
├── input_angle      : float  (stored reference)
├── output_angle     : float  (stored reference)
├── input_endpoint   : Point3D
└── output_endpoint  : Point3D
```

Factory: `Stage.from_reference_position(...)` — creates a stage from a valid reference position. The rod length is calculated automatically from the reference endpoints. All working-range limits default to ±∞ if not specified.

Methods:

- `input_position(angle)` → `Point3D` (applies offset automatically)
- `output_position(angle)` → `Point3D` (applies offset automatically)
- `accepts_input_angle(angle)` → `bool` (within working range)
- `accepts_output_angle(angle)` → `bool` (within working range)

### Mechanism

```text
Mechanism
└── stages : tuple[Stage, ...]
```

The order of stages defines the kinematic chain. The number of stages is unlimited. `stages` must be a tuple (validated in `__post_init__`).

---

# 10. Mechanism Builders (mechanics/)

Builders convert definition models or parameter sets into simulation models.

### MechanismBuilder Protocol (optimization/mechanism\_builder.py)

Protocol defining the builder interface:

```text
MechanismBuilder.build(parameters: ParameterSet) -> Mechanism
```

### StandardMechanismBuilder

```text
ParameterSet
    |
    v
Mechanism
```

Builds a single-stage mechanism from optimization parameters. Expected parameters: `input_lever_length`, `output_lever_length`, `input_angle_offset`, `output_angle_offset`.

### CsvMechanismBuilder

```text
MechanismDefinition + ParameterSet
    |
    v
Mechanism
```

Builds a multi-stage mechanism from a `MechanismDefinition`, applying optimization parameters as an overlay. Uses `Stage.from_reference_position` for each driver relationship.

Features:

- Accepts an optional `StageMotionValidator` (creates a default one if none supplied)
- Runs stage validation during `build()` and stores results
- `get_validation_results()` returns the validation results tuple
- Parameter names follow the pattern `lever.<id>.length` and `lever.<id>.angle`

### MechanismFactory (mechanics/mechanism\_factory.py)

```text
MechanismFactory
├── builder : Callable[[ParameterSet], object]
└── create(parameters: ParameterSet) -> object
```

Wraps any callable builder behind a uniform factory interface.

Builders are responsible only for construction. They do not perform simulation.

---

# 11. Solver Layer (solver/)

The solver layer contains numerical root-finding and kinematic solving.

### SolverPrecision (solver\_precision.py)

```text
SolverPrecision
├── tolerance      : float = 1e-10
├── max_iterations : int   = 40
├── bracket_step   : float = 1° (rad)
└── search_window  : float = 30° (rad)
```

Validated in `__post_init__`.

### SolverResult (solver\_result.py)

```text
SolverResult
├── success     : bool
├── angle       : float (rad, NaN if failed)
├── residual    : float
├── iterations  : int
└── reason       : str | None
```

### SolverState (solver\_state.py)

Persistent state tracking one continuous physical motion branch.

```text
SolverState
├── last_input_angle   : float
├── last_output_angle  : float
├── direction          : int  (-1, 0, +1)
└── output_velocity    : float
```

Methods:

- `predict_output(input_angle)` — linear extrapolation using velocity
- `next(input_angle, output_angle)` — create updated state
- `reversed()` — create state for reversed motion
- `initial(input_angle, output_angle)` — class method for initial state

### Constraints (solver/constraints.py)

Pure geometric constraint functions:

- `distance(point_a, point_b)` → float
- `rod_length_error(point_a, point_b, rod_length)` → float (residual)

### Objective (solver/objective.py)

- `stage_error(stage, input_angle, output_angle)` → float
- `create_stage_objective(stage, input_angle)` → `Callable[[float], float]` (residual function with cached input point)

### RootSolver (solver/root\_solver.py)

Generic one-dimensional root finder.

Static methods:

- `find_bracket(function, center, window, step)` — find one bracket around center
- `find_all_brackets(function, minimum, maximum, step)` — find all sign-change intervals
- `find_all_brackets_around(function, center, window, step)` — local bracket search
- `solve_brent(function, left, right, tolerance, max_iterations)` — Brent's method

Brent's method implementation features:

- Machine-epsilon tolerance term (`2 * ulp(1) * |b| + tolerance`)
- Iteration counter starts at 2 (initial bracket evaluations)
- Bisection fallback when inverse quadratic interpolation is not applicable
- Accepts exact roots at bracket boundaries

### AngleSolver (solver/angle\_solver.py)

Solves the output angle for a given input angle, preserving the physical motion branch.

Search strategy (in order):

1. **Adaptive reuse** — reuse previous solution branch with dynamic window
2. **Local search** — search around predicted position
3. **Full-range fallback** — only if no brackets found above

Features:

- `_filter_to_allowed_range()` — filters fast-path brackets to stage output limits
- `_select_branch()` — selects physically continuous branch using prediction distance, output jump, velocity change, and direction penalty
- Performance statistics tracking (`stats` dict with adaptive, local, fallback, bracket, and brent counters)
- Configurable search window, reuse factor, and reuse min/max windows

### StageSolver (solver/stage\_solver.py)

High-level solver wrapper for one stage.

```text
StageSolver
├── stage        : Stage
├── angle_solver : AngleSolver
└── _state       : SolverState | None
```

Manages solver state across simulation steps. Delegates to `AngleSolver` for solving.

---

# 12. Simulation Layer (simulation/)

### MotionRange (simulation/motion\_range.py)

Defines a one-dimensional angular simulation range. Immutable (`frozen=True`, `slots=True`).

```text
MotionRange
├── start_angle : float
├── max_angle   : float  (total travel, always ≥ 0)
├── step        : float
└── direction   : int    (+1 or -1)
```

Iteration uses a travelled-distance accumulator with `ANGLE_TOLERANCE` guard to avoid floating-point drift. `feedback()` is a no-op (fixed step).

### AdaptiveMotionRange (simulation/adaptive\_motion\_range.py)

Adaptive motion provider with feedback-based step adjustment. Mutable (`slots=True`, not frozen).

```text
AdaptiveMotionRange
├── start_angle       : float
├── end_angle         : float
├── initial_step      : float = 5°
├── min_step          : float = 0.25°
├── max_step          : float = 10°
├── max_output_delta  : float = 5°
└── current_step      : float  (init=False)
```

`feedback(output_delta)` adjusts step size: halves on large output jumps, increases on small jumps, clamped to `[min_step, max_step]`.

### MotionProvider Protocol (simulation/motion\_provider.py)

```text
MotionProvider (Protocol)
├── __iter__() -> Iterator[float]
└── feedback(*, output_delta: float) -> None
```

Common interface for all motion angle providers.

### ResultMotionProvider (simulation/result\_motion\_provider.py)

Provides the output angles of a previous simulation stage as input for the next stage. Immutable and re-iterable.

### SimulationResult (simulation/simulation\_result.py)

```text
SimulationResult
├── input_angles  : tuple[float, ...]
├── output_angles : tuple[float, ...]
├── success       : bool
└── blocked_at    : float | None
```

Validates that input and output angle counts match.

### StageSimulator (simulation/stage\_simulator.py)

Stateless simulator for one mechanical stage.

```text
StageSimulator
├── solver_type : type (default: StageSolver)
├── precision   : SolverPrecision | None
└── _solvers    : list  (created per run, accessible via .solvers property)
```

`run(stage, motion)` → `SimulationResult`:

- Creates a fresh solver per run
- Validates stage reference geometry
- Iterates over the motion provider
- Sends feedback to the motion provider after each step
- Returns blocked result on solver failure

### MechanismSimulator (simulation/mechanism\_simulator.py)

Simulates every stage of a mechanism.

```text
MechanismSimulator
├── motion          : MotionRange
├── stage_simulator : StageSimulator
├── stage_limit     : int | None
└── precision       : SolverPrecision | None
```

`simulate(mechanism)` → `tuple[SimulationResult, ...]`:

- Chains stages: each stage's output becomes the next stage's input via `ResultMotionProvider`
- Optionally limits to first N stages

### MechanismMotionSimulator (simulation/mechanism\_motion\_simulator.py)

Runs a complete motion simulation of a multi-stage mechanism and produces a combined result.

```text
MechanismMotionResult
├── input_angles  : tuple[float, ...]
├── stage_outputs : tuple[tuple[float, ...], ...]  (per input angle)
├── success       : bool
└── blocked_at    : float | None
```

### CSVExporter (simulation/csv\_exporter.py)

Writes `MechanismMotionResult` to CSV. Angles exported in radians.

```text
CSVExporter.write(filename, result)
```

Header: `input_angle, stage_0_output, stage_1_output, ...`

---

# 13. Analysis Layer (analysis/)

### TransferCurve (analysis/transfer\_curve.py)

Represents a kinematic transfer function (input/output angle relationship). Immutable.

```text
TransferCurve
├── input_angles  : tuple[float, ...]
├── output_angles : tuple[float, ...]
└── (cached: _ascending, _minimum, _maximum, lookup tables)
```

Supports ascending and descending input sequences. `output_at(input_angle)` performs linear interpolation.

### TargetCurve (analysis/target\_curve.py)

Defines desired kinematic behavior as a callable function. Immutable.

Construction methods:

- `from_points(input_angles, output_angles)` — linear interpolation
- `from_csv(path)` — load from CSV (deg → rad conversion)
- `sample(input_angles)` → `TransferCurve`

### ErrorMetric (analysis/error\_metric.py)

Calculates deviation from a target curve.

```text
ErrorMetric
├── target : TransferCurve
└── calculate(actual: TransferCurve) -> float  (mean absolute error)
```

### CurveFitness (analysis/curve\_fitness.py)

Calculates fitness of a simulated transfer curve. Implements the `FitnessFunction` protocol. Lower values are better.

Features:

- Uses the **last stage's** `input_angles` (not the first stage's) for `TransferCurve` construction, preventing length-mismatch crashes when intermediate stages block
- Caches `ErrorMetric` instances per input-angle grid
- Penalty system for failed/blocked simulations
- `__call__` provides backwards-compatible interface for direct `TransferCurve` input

---

# 14. Validation Layer (validation/)

The validation layer checks whether stages and mechanisms can follow their complete defined motion ranges.

### StageValidationResult (validation/stage\_validation\_result.py)

```text
StageValidationResult
├── valid                  : bool
├── checked_steps          : int
├── failed_at_input_angle  : float | None
├── reason                 : str | None
└── stage_id               : int | None
```

### StageMotionValidator (validation/stage\_motion\_validator.py)

```text
StageMotionValidator
└── steps : int = 50
```

Methods:

- `validate(stage, stage_id)` — validates the stage's complete defined input range
- `validate_motion(stage, input_angles, stage_id)` — validates a supplied sequence of input angles
- `get_validation_results()` on `CsvMechanismBuilder` returns results from the last build

Checks:

1. Solver must find a mathematical solution (else `reason="blocked"`)
2. Solution must be within the stage's output angle range (else `reason="output_angle_limit"`)

### MechanismValidationResult (validation/mechanism\_validation\_result.py)

```text
MechanismValidationResult
├── stages      : tuple[StageValidationResult, ...]
├── valid       : bool  (property: all stages valid)
└── failed_stage : int | None  (property: first failed stage index)
```

### MechanismMotionValidator (validation/mechanism\_motion\_validator.py)

```text
MechanismMotionValidator
└── stage_validator : StageMotionValidator
```

`validate(mechanism, motion)` → `MechanismValidationResult`:

- If no motion supplied: validates each stage's defined input range
- If motion supplied: validates each stage against the explicit input positions

---

# 15. Optimization Layer (optimization/)

### Parameter (optimization/parameter.py)

A bounded optimization parameter. Immutable.

```text
Parameter
├── name    : str
├── minimum : float
├── maximum : float
└── value   : float
```

Validates: `minimum < maximum`, `value` within range, non-empty name.

### ParameterSet (optimization/parameter\_set.py)

Immutable collection of parameters representing one design variant.

```text
ParameterSet
├── parameters : tuple[Parameter, ...]
├── get(name)  -> Parameter
├── values()  -> dict[str, float]
└── __len__()
```

Validates: no duplicate parameter names.

### Population (optimization/population.py)

Immutable collection of candidate designs. One population = one generation.

```text
Population
├── members : tuple[ParameterSet, ...]
├── __len__()
├── __getitem__(index) -> ParameterSet
└── __iter__() -> Iterator[ParameterSet]
```

Validates: non-empty.

### Selection (optimization/selection.py)

Selects candidates with lowest score (lower fitness = better).

```text
Selection.select(population, scores, count) -> Population
```

### ParameterMutation (optimization/parameter\_mutation.py)

Mutation operator using relative change based on parameter range.

```text
ParameterMutation
├── strength         : float = 0.1
├── random_generator : random.Random | None
└── apply(parameter_set) -> ParameterSet
```

Mutation: `delta = uniform(-1,1) * range_size * strength`, clamped to `[minimum, maximum]`.

### Reproduction (optimization/reproduction.py)

Creates children from existing candidates via mutation.

```text
Reproduction
├── mutation : ParameterMutation
└── create(population, count) -> Population
```

Children are created by round-robin parent selection and mutation.

### PopulationFactory (optimization/population\_factory.py)

Creates initial populations from parameter templates.

```text
PopulationFactory
├── random_generator : random.Random | None
├── initial_spread  : float = 0.1
└── create(template, size) -> Population
```

Each candidate is randomized around template values within `initial_spread * range_size`.

### CsvParameterFactory (optimization/csv\_parameter\_factory.py)

Creates optimization parameter templates from a `MechanismDefinition`.

```text
CsvParameterFactory.create(definition) -> ParameterSet
```

Each lever produces two parameters: `lever.<id>.length` and `lever.<id>.angle`.

### FitnessFunction Protocol (optimization/fitness\_function.py)

```text
FitnessFunction (Protocol)
└── evaluate(simulation: tuple[SimulationResult, ...]) -> float
```

### MechanismBuilder Protocol (optimization/mechanism\_builder.py)

```text
MechanismBuilder (Protocol)
└── build(parameters: ParameterSet) -> Mechanism
```

### MechanismOptimizer (optimization/mechanism\_optimizer.py)

Adapter between mechanism simulation and evolutionary optimization.

```text
MechanismOptimizer
├── builder   : MechanismBuilder
├── simulator : MechanismSimulator
├── fitness   : FitnessFunction
└── _cache    : dict[OptimizationCacheKey, float]
```

Features:

- Results cached by `OptimizationCacheKey` (parameters + motion + precision + stage\_limit)
- Cache statistics: `evaluations`, `cache_size`, `cache_hits`, `cache_misses`
- `clear_cache()` resets cache and stats

### EvolutionEngine (optimization/evolution\_engine.py)

Executes evolutionary optimization.

```text
EvolutionEngine
├── population          : Population
├── evaluator           : Callable[[ParameterSet], float]
├── selection_count     : int
├── reproduction        : Reproduction
├── target_fitness      : float | None
├── max_generations     : int = 100
├── stagnation_limit    : int | None
├── stagnation_tolerance: float = 1e-6
├── best_candidate      : ParameterSet | None
├── best_score          : float
├── stop_reason         : str | None
└── scores              : dict[ParameterSet, float]
```

Methods:

- `evaluate_population()` — evaluate all candidates in current population
- `update_best()` — update best known solution (returns `None`)
- `should_stop()` — check stopping criteria (target fitness or stagnation)
- `step(children_count)` — execute one generation (evaluate → select → reproduce)
- `run(children_count)` → `Iterator[int]` — run optimization, yields generation indices

Stop reasons: `"target_fitness_reached"`, `"stagnation_limit_reached"`, `"max_generations_reached"`.

### OptimizationProblem (optimization/optimization\_problem.py)

High-level public entry point wiring together the complete optimization infrastructure.

```text
OptimizationProblem
├── parameter_template : ParameterSet
├── simulator          : MechanismSimulator
├── fitness            : CurveFitness
├── builder            : StandardMechanismBuilder | None
└── random_generator   : Random | None
```

`optimize(population_size, generations, children_per_generation, selection_count)` → `Population`

### OptimizationPipeline (optimization/optimization\_pipeline.py)

End-to-end pipeline wrapper.

```text
OptimizationPipeline
├── problem : OptimizationProblem
└── run(population_size, generations, children_per_generation) -> Population
```

### OptimizerRunner (optimization/optimizer\_runner.py)

High-level execution wrapper running multiple evolutionary steps.

```text
OptimizerRunner
├── engine : EvolutionEngine
└── run(generations, children_count) -> Population
```

---

# 16. Simulation Pipeline

The simulation pipeline is strictly layered.

```text
Mechanism
    |
    v
MechanismSimulator
    |
    v  (per stage)
StageSimulator
    |
    v
StageSolver
    |
    v
AngleSolver
    |
    v
RootSolver (Brent's method)
    |
    v
SolverResult
    |
    v
SimulationResult
```

Each layer has exactly one responsibility. Stage outputs are chained via `ResultMotionProvider` so each stage's output becomes the next stage's input.

---

# 17. Optimization Pipeline

Optimization is independent from solver implementation.

```text
ParameterSet
    |
    v
MechanismBuilder (StandardMechanismBuilder / CsvMechanismBuilder)
    |
    v
Mechanism
    |
    v
MechanismSimulator
    |
    v
SimulationResult
    |
    v
CurveFitness (FitnessFunction)
    |
    v
MechanismOptimizer (cached evaluator)
    |
    v
EvolutionEngine
    |
    v  (per generation)
Selection -> Reproduction -> Population
```

The optimizer never accesses solver internals.

---

# 18. Validation Pipeline

Validation checks whether mechanisms can follow their motion ranges before or during construction.

```text
Mechanism
    |
    v
MechanismMotionValidator
    |
    v  (per stage)
StageMotionValidator
    |
    v
StageSolver
    |
    v
StageValidationResult
    |
    v
MechanismValidationResult
```

`CsvMechanismBuilder` runs `StageMotionValidator` during `build()` and stores results accessible via `get_validation_results()`.

---

# 19. Numerical Rules

Floating-point values shall never be compared with `==` for general equality checks.

Required:

```python
if abs(value) < tolerance:
```

All tolerances are defined centrally in `core/tolerance.py`:


| Constant           | Value | Application             |
| ------------------ | ----- | ----------------------- |
| `LENGTH_TOLERANCE` | 1e-9  | Length comparisons (mm) |
| `ANGLE_TOLERANCE`  | 1e-12 | Angle comparisons (rad) |
| `VECTOR_TOLERANCE` | 1e-12 | Vector comparisons      |


Additionally, `DEFAULT_TOLERANCE = 1e-12` is defined locally in `vector3d.py` and `quaternion.py` for internal zero checks.

Brent's method uses a combined tolerance: `2 * ulp(1) * |b| + tolerance` to handle both machine-epsilon-dominated and tolerance-dominated regimes.

---

# 20. Testing Strategy

Every production module requires:

- Unit tests
- Boundary tests
- Invalid input tests
- Numerical stability tests

**CSV** interfaces require:

- Reader tests
- Writer tests
- Roundtrip tests

Mechanism builders require:

- Construction tests
- Geometry transfer tests
- Relationship tests

Validation requires:

- Valid motion tests
- Blocked motion tests
- Output range violation tests

---

# 21. Immutability

Domain objects are immutable whenever possible.

Required:

- `@dataclass(frozen=True, slots=True)`
- `tuple` instead of `list`
- Validation in `__post_init__()`

Mutable state is restricted to algorithms.

Examples of mutable objects:

- `SolverState` is immutable but `StageSolver` manages it mutably
- `EvolutionEngine` (mutable optimization state)
- `StageSimulator` (tracks solver instances)
- `AdaptiveMotionRange` (mutable step size)
- `ParameterMutation` (holds random generator)
- `CsvMechanismBuilder` (stores validation results)
- `MechanismOptimizer` (cache)

---

# 22. Project Structure

```text
core/
    __init__.py
    point3d.py
    vector3d.py
    quaternion.py
    rotation.py
    tolerance.py

model/
    __init__.py
    lever_definition.py
    mechanism_definition.py
    simulation_config.py

mechanics/
    __init__.py
    lever.py
    rod.py
    stage.py
    mechanism.py
    standard_mechanism_builder.py
    csv_mechanism_builder.py
    mechanism_factory.py

mechanism_io/
    __init__.py
    csv_reader.py
    csv_writer.py

solver/
    angle_solver.py
    root_solver.py
    stage_solver.py
    solver_state.py
    solver_result.py
    solver_precision.py
    objective.py
    constraints.py

simulation/
    motion_range.py
    adaptive_motion_range.py
    motion_provider.py
    result_motion_provider.py
    simulation_result.py
    stage_simulator.py
    mechanism_simulator.py
    mechanism_motion_simulator.py
    csv_exporter.py

analysis/
    transfer_curve.py
    target_curve.py
    error_metric.py
    curve_fitness.py

optimization/
    __init__.py
    parameter.py
    parameter_set.py
    population.py
    population_factory.py
    selection.py
    parameter_mutation.py
    reproduction.py
    fitness_function.py
    mechanism_builder.py
    mechanism_optimizer.py
    evolution_engine.py
    csv_parameter_factory.py
    optimization_problem.py
    optimization_pipeline.py
    optimizer_runner.py

validation/
    stage_motion_validator.py
    stage_validation_result.py
    mechanism_motion_validator.py
    mechanism_validation_result.py

examples/
    __init__.py
    debug.py
    debug_simulation.py
    test_single_simulation.py
    simple_optimization.py
    optimize_csv_mechanism.py
    levers.csv
    mechanism.csv
    simulation.csv
    target_curve.csv
    target_curve2.csv
    target_curve_csv_mechanism.csv

tests/
    conftest.py
    debug_simulation.py
    test_adaptive_motion_range.py
    test_angle_solver.py
    test_blocking_behavior.py
    test_constraints.py
    test_csv_exporter.py
    test_csv_mechanism_builder.py
    test_csv_mechanism_builder_fitness.py
    test_csv_mechanism_builder_parameters.py
    test_csv_mechanism_motion_validation.py
    test_csv_parameter_factory.py
    test_csv_reader.py
    test_csv_roundtrip.py
    test_csv_writer.py
    test_curve_fitness.py
    test_error_metric.py
    test_evaluator.py
    test_evolution_engine.py
    test_full_chain.py
    test_lever.py
    test_mechanism.py
    test_mechanism_definition.py
    test_mechanism_factory.py
    test_mechanism_motion_simulator.py
    test_mechanism_motion_validator.py
    test_mechanism_optimizer.py
    test_mechanism_optimizer_cache.py
    test_mechanism_simulator.py
    test_motion_provider.py
    test_motion_range.py
    test_multistage_curve_fitness.py
    test_multistage_optimization.py
    test_multistage_parameter_influence.py
    test_multistage_simulation.py
    test_objective.py
    test_optimization_pipeline.py
    test_optimization_pipeline2.py
    test_optimization_problem.py
    test_optimizer_runner.py
    test_parameter.py
    test_parameter_mutation.py
    test_parameter_set.py
    test_point3D.py
    test_population.py
    test_population_factory.py
    test_quaternion.py
    test_reproduction.py
    test_rod.py
    test_root_solver.py
    test_rotation.py
    test_selection.py
    test_simulation_result.py
    test_single_stage_optimization.py
    test_stage_motion_validator.py
    test_stage_simulator.py
    test_stage_solver.py
    test_stage_validation_result.py
    test_standard_mechanism_builder.py
    test_vector3d.py

docs/
    Architecture.md
    Ideas.md
    deleoper_guide.md
    roadmap.md
```

---

# 23. Development Principles

The project follows:

- Correctness before performance
- Performance before convenience
- Readability before cleverness

Every public function shall provide:

- Type hints
- Docstring
- Unit tests
- Usage example

Development proceeds in vertical slices.

Each sprint shall:

- Introduce one coherent feature
- Include complete unit tests
- Keep the project executable
- Update documentation

---

# 24. Long-Term Roadmap


| Version | Goal                                                 |
| ------- | ---------------------------------------------------- |
| 0.1     | Geometry library (Point3D, Vector3D, Quaternion)     |
| 0.2     | Spatial lever + Rotation utilities                   |
| 0.3     | Single linkage stage + solver                        |
| 0.4     | Multi-stage mechanism + chaining                     |
| 0.5     | CSV I/O (reader, writer, roundtrip)                  |
| 0.6     | CSV mechanism builder + parameter factory            |
| 0.7     | Validation layer (stage + mechanism)                 |
| 0.8     | Adaptive motion + CSV export                         |
| 1.0     | Stable simulation engine                             |
| 2.0     | Evolutionary optimization (engine, pipeline, runner) |
| 3.0     | Interactive GUI                                      |