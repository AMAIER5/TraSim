# Koppelgetriebe Simulator

**Architecture Specification**

| Property | Value |
|----------|-------|
| Version | 0.1.0 |
| Status | Draft |
| Language | English (source code) |
| Units | mm / deg (user), mm / rad (internal) |

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

The simulator shall support:

- 3D lever axes
- Spherical joints
- Rigid coupling rods
- Multi-stage mechanisms
- Evolutionary optimization

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

Coordinates are represented by

```python
Point3D(x, y, z)
```

Unit:

```
millimeter [mm]
```

---

# 3. Units

## User Interface

| Quantity | Unit |
|----------|------|
| Length | mm |
| Angle | deg |

## Internal Calculations

| Quantity | Unit |
|----------|------|
| Length | mm |
| Angle | rad |

Only the input/output layer performs unit conversion.

No internal class shall store angles in degrees.

## Mathematical notation

All mathematical descriptions shall use ASCII-compatible notation
where possible.

Examples:

cross(a,b)   instead of   a × b
dot(a,b)     instead of   a · b
norm(v)      instead of   |v|

The goal is maximum compatibility between:
- source code
- documentation
- version control
- automated tools

---

# 4. Rotation Convention

Internally all rotations follow the **right-hand rule**.

Positive mathematical rotations are counter-clockwise when looking along the positive rotation axis.

User input/output may use engineering conventions.

All conversions are handled exclusively by the I/O layer.

---

# 5. Basic Geometric Entities

## Point3D

Represents a position in space.

Properties

- x
- y
- z

Example

```python
Point3D(10.0, 20.0, 5.0)
```

---

## Vector3D

Represents a direction or displacement.

Operations

- Addition
- Subtraction
- Scaling
- Dot product
- Cross product
- Normalization
- Angle calculation
- Rotation

---

## Rotation

Rotations are internally represented by **quaternions**.

Advantages

- No gimbal lock
- Stable interpolation
- Efficient composition
- Suitable for animation

---

# 6. Lever Definition

A lever is completely defined by

```text
Lever
│
├── pivot_point : Point3D
├── axis        : Vector3D
├── length      : float
├── reference   : Vector3D
└── angle       : float
```

The tip position is calculated by

```
tip = pivot + rotate(reference * length)
```

---

# 7. Coupler Rod Definition

A rod is an ideal rigid body.

```text
Rod
│
├── point_a
├── point_b
└── length
```

Constraint

```
distance(point_a, point_b) == length
```

Both rod ends are ideal spherical joints.

---

# 8. Stage Definition

A stage consists of

```text
Input Lever
      │
      ●────────────●
      │            │
      │ Coupler    │
      │            │
      ●────────────●
     Output Lever
```

Each stage contains

- Input lever
- Output lever
- Coupler rod

Both pivots are fixed to the global frame.

---

# 9. Mechanism Structure

```text
Mechanism

├── Stage 1
├── Stage 2
├── Stage 3
└── ...
```

The number of stages is unlimited.

---

# 10. Solver Requirements

The solver shall

- accept an arbitrary start angle
- simulate in positive direction
- simulate in negative direction
- stop at configured limits
- stop when no valid solution exists
- return NaN for impossible positions
- preserve continuous angle tracking

---

# 11. Numerical Rules

Floating point values shall never be compared directly.

Forbidden

```python
if value == 0:
```

Required

```python
if abs(value) < tolerance:
```

All tolerances are defined centrally.

---

# 12. Testing Philosophy

Every mathematical module requires

- Unit tests
- Boundary tests
- Invalid input tests
- Numerical stability tests

Mechanism modules may only depend on tested geometry modules.

---

# 13. Future Optimization Interface

The solver shall expose a deterministic interface.

Input

- Mechanism definition
- Simulation configuration

Output

- Joint positions
- Lever angles
- Solver status
- Validity flags

The optimizer shall never access internal solver implementation details.

---

# 14. Development Principles

The project follows these priorities:

1. Correctness before performance
2. Performance before convenience
3. Readability before cleverness

Every public function shall provide

- Type hints
- Docstring
- Unit tests
- Usage example

The project is developed in vertical slices.

Each sprint shall produce a complete, tested and usable increment.

Future ideas are collected separately and never implemented
before the current sprint is complete.

No sprint may introduce unrelated features.

---

# 15. Long-Term Roadmap

| Version | Goal |
|----------|------|
| 0.1 | Geometry library |
| 0.2 | Spatial lever |
| 0.3 | Single linkage stage |
| 0.4 | Multi-stage mechanism |
| 0.5 | CSV I/O |
| 0.6 | Visualization |
| 1.0 | Stable simulation engine |
| 2.0 | Evolutionary optimization |
| 3.0 | Interactive GUI |

# 16. Software Architecture

The project is organized into independent layers.

```Application
    │
    ▼
Optimization
    │
    ▼
Simulation
    │
    ▼
Solver
    │
    ▼
Mechanics
    │
    ▼
Geometry
```
Dependencies always point downward.
Higher layers shall never be referenced by lower layers.

# 17. Project Structure

```core/
    Point3D
    Vector3D
    Quaternion

mechanics/
    Lever
    Rod
    Stage
    Mechanism
    MechanismFactory
    StandardMechanismBuilder

solver/
    AngleSolver
    StageSolver
    SolverState
    SolverResult

simulation/
    MotionRange
    SimulationResult
    StageSimulator
    MechanismSimulator
    MechanismMotionSimulator
    CSVExporter

analysis/
    TransferCurve
    TargetCurve
    ErrorMetric
    CurveFitness

optimization/
    Parameter
    ParameterSet
    Population
    PopulationFactory
    ParameterMutation
    Selection
    Reproduction
    EvolutionEngine
    OptimizerRunner
    FitnessFunction
    MechanismOptimizer
    OptimizationProblem

examples/
tests/
```
# 18. Simulation Pipeline

The simulation pipeline is strictly layered.

```Mechanism
        │
        ▼
StageSimulator
        │
        ▼
StageSolver
        │
        ▼
AngleSolver
        │
        ▼
SolverResult
```
Each layer has exactly one responsibility.

# 19. Optimization Pipeline

The optimization process is independent from the solver implementation.

```ParameterSet
        │
        ▼
MechanismBuilder
        │
        ▼
Mechanism
        │
        ▼
MechanismSimulator
        │
        ▼
SimulationResult
        │
        ▼
TransferCurve
        │
        ▼
CurveFitness
        │
        ▼
EvolutionEngine
```
The optimization layer never accesses solver internals.

# 20. Design Principles

The project follows the Single Responsibility Principle.

Each class owns exactly one responsibility.

Examples
```StageSolver
    computes one solver step

StageSimulator
    simulates one stage

MechanismSimulator
    simulates all stages

CurveFitness
    evaluates one transfer curve

EvolutionEngine
    performs one evolutionary generation
```
# 21. Immutability

Domain objects are immutable whenever possible.

Required

- dataclass(frozen=True)
- slots=True
- tuple instead of list
- explicit validation in post_init()

Mutable state is restricted to algorithms.

Examples
```SolverState
PopulationFactory
EvolutionEngine
```

# 22. Dependency Rules

The following dependencies are forbidden.

```Geometry
    -> Simulation

Mechanics
    -> Optimization

Solver
    -> Analysis

Analysis
    -> Solver

Optimization
    -> Solver internals
```

Allowed dependency graph

```Optimization
        │
        ▼
Analysis
        │
        ▼
Simulation
        │
        ▼
Solver
        │
        ▼
Mechanics
        │
        ▼
Core
```

# 23. Public API

Public interfaces shall be strongly typed.

Forbidden
```Callable
Any
dict
```

for public APIs where dedicated domain types exist.

Preferred

```MechanismSimulator
CurveFitness
Population
ParameterSet
SimulationResult
TransferCurve
```

# 24. Testing Strategy

Every production module shall have a corresponding test module.

Naming convention

```module.py

↓

test_module.py
```

Each public class requires tests for

- construction
- invalid parameters
- nominal behaviour
- edge cases
- regression tests

Whenever an interface changes, all dependent tests shall be updated in the same change set.

# 25. Development Workflow

Development proceeds in vertical slices.

Each sprint shall

- introduce one coherent feature,
-include complete unit tests,
- keep the project fully executable,
- avoid partially migrated APIs.

API refactorings shall update

- production code,
- tests,
- examples,
- documentation

within the same sprint.