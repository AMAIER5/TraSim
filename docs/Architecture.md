# Koppelgetriebe Simulator

**Architecture Specification**

| Property | Value |
|----------|-------|
| Version | 0.2.0 |
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

The simulator supports:

- 3D lever axes
- Fixed pivot positions
- Rigid coupling rods
- Multi-stage mechanisms
- **CSV**-based mechanism definitions
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

Coordinates are represented by

Point3D(x, y, z)

Unit:

millimeter [mm]
3. Units
### User Interface
Quantity	Unit
Length	mm
Angle	deg
### Internal Calculations
Quantity	Unit
Length	mm
Angle	rad

Only input/output layers perform unit conversion.

No internal class stores angles in degrees.

## Rotation Convention

Internally all rotations follow the right-hand rule.

Positive rotations are counter-clockwise when looking along the positive rotation axis.

Rotations are represented using quaternions.

## Software Architecture

The project is organized into independent layers.

Application
    |
    v
Optimization
    |
    v
Analysis
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
Core

Dependencies always point downward.

Higher layers shall never be referenced by lower layers.

## Geometry Layer

The geometry layer contains mathematical primitives.

Point3D

Represents a position in space.

Properties:

x y z Vector3D

Represents direction or displacement.

Operations:

Addition Subtraction Scaling Dot product Cross product Normalization Rotation Quaternion

Used internally for rotations.

Advantages:

No gimbal lock Stable interpolation Efficient composition ## Mechanism Definition Model

Input data is separated from simulation objects.

The external definition model describes a mechanism before construction.

MechanismDefinition

├── LeverDefinition 1 ├── LeverDefinition 2 ├── LeverDefinition 3 └── ...

A lever definition contains:

LeverDefinition

├── id ├── pivot : Point3D ├── axis  : Vector3D │ ├── length_min ├── length_max ├── length_start │ ├── angle_min ├── angle_max ├── angle_start │ ├── driver └── coupled

The definition contains:

fixed geometry optimization ranges linkage relationships ## CSV Interface

**CSV** files are the external user interface.

The **CSV** layer is responsible for:

parsing input files writing reproducible definitions unit conversion validation of input data

The **CSV** layer does not perform simulation.

Mechanism **CSV** Format

Example:

id,length_min,length_max,length_start,angle_min,angle_max,angle_start,pivot_x,pivot_y,pivot_z,axis_x,axis_y,axis_z,driver,coupled 1,40,**100**,60,-40,40,0,0,0,0,0,0,1,, 2,30,90,45,-60,60,0,**100**,0,0,0,0,1,1, ### Driver Relationship

A driver defines a mechanical stage.

Example:

Lever 2 driver = 1

creates:

Stage(
    input = Lever 1,
    output = Lever 2
)
### Coupled Relationship

A coupled lever maintains a constant angular relation.

Example:

Lever 3 coupled = 2

means:

angle(Lever3) - angle(Lever2) = constant

A lever shall not be controlled by both:

driver coupled

If both are specified:

coupled has priority driver is ignored ## Mechanical Components Lever

A lever is immutable.

Lever

├── pivot : Point3D ├── axis  : Vector3D └── length

The endpoint is calculated from:

tip = pivot + rotate(reference * length)

The lever does not store dynamic angle state.

Rod

A rod is an ideal rigid body.

Rod

├── point_a ├── point_b └── length

Constraint:

distance(point_a, point_b) == length ## Stage Definition

A stage consists of two levers connected by an ideal rod.

### Input Lever

    |
    ●────────────●
    |
    |
    ●────────────●
### Output Lever

Each stage contains:

Input lever Output lever Rod length Reference configuration

Rod length is calculated automatically from the reference position.

## Mechanism Structure

The simulation model contains stages.

Mechanism

├── Stage 1 ├── Stage 2 ├── Stage 3 └── ...

The number of stages is unlimited.

## Mechanism Builders

Builders convert definition models into simulation models.

Available builders:

StandardMechanismBuilder

ParameterSet
    |
    v
Mechanism
CsvMechanismBuilder

MechanismDefinition
    |
    v
Mechanism

Builders are responsible only for construction.

They do not perform simulation.

## Simulation Pipeline

The simulation pipeline is strictly layered.

Mechanism
    |
    v
StageSimulator
    |
    v
StageSolver
    |
    v
AngleSolver
    |
    v
SolverResult

Each layer has exactly one responsibility.

## Optimization Pipeline

Optimization is independent from solver implementation.

ParameterSet
    |
    v
MechanismBuilder
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
TransferCurve
    |
    v
CurveFitness
    |
    v
EvolutionEngine

The optimizer never accesses solver internals.

## Numerical Rules

Floating point values shall never be compared directly.

Forbidden:

if value == 0:

Required:

if abs(value) < tolerance:

All tolerances are defined centrally.

## Testing Strategy

Every production module requires:

Unit tests Boundary tests Invalid input tests Numerical stability tests

**CSV** interfaces require:

Reader tests Writer tests Roundtrip tests

Mechanism builders require:

Construction tests Geometry transfer tests Relationship tests ## Immutability

Domain objects are immutable whenever possible.

Required:

dataclass(frozen=True) slots=True tuple instead of list validation in post_init()

Mutable state is restricted to algorithms.

Examples:

SolverState
Population
EvolutionEngine
## Project Structure
core/
    point3d.py
    vector3d.py
    quaternion.py

model/
    lever_definition.py
    mechanism_definition.py
    simulation_config.py

mechanics/
    lever.py
    rod.py
    stage.py
    mechanism.py
    standard_mechanism_builder.py
    csv_mechanism_builder.py

mechanism_io/
    csv_reader.py
    csv_writer.py

solver/
    angle_solver.py
    stage_solver.py
    solver_state.py
    solver_result.py

simulation/
    motion_range.py
    simulation_result.py
    stage_simulator.py
    mechanism_simulator.py

analysis/
    transfer_curve.py
    target_curve.py
    error_metric.py
    curve_fitness.py

optimization/
    parameter.py
    parameter_set.py
    evolution_engine.py
    mechanism_optimizer.py

examples/

tests/ ## Development Principles

The project follows:

Correctness before performance Performance before convenience Readability before cleverness

Every public function shall provide:

Type hints Docstring Unit tests Usage example

Development proceeds in vertical slices.

Each sprint shall:

introduce one coherent feature
include complete unit tests
keep the project executable
update documentation
## Long-Term Roadmap
Version	Goal
0.1	Geometry library
0.2	Spatial lever
0.3	Single linkage stage
0.4	Multi-stage mechanism
0.5	**CSV** I/O
0.6	**CSV** mechanism builder
0.7	Visualization
1.0	Stable simulation engine
2.0	Evolutionary optimization
3.0	Interactive **GUI**