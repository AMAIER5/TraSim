# Koppelgetriebe Simulator

# Roadmap

| Property | Value |
|----------|-------|
| Version | 0.1.0 |
| Status | Living Document |
| Purpose | Development Roadmap |

---

# Vision

TraSim shall become a professional engineering framework for the design,
simulation and optimization of multi-stage linkage mechanisms.

The long-term goal is to evolve from a numerical simulator into a complete
engineering environment capable of automatically synthesizing linkage systems
from desired transfer characteristics.

---

# Current Status

Current development stage

```
Sprint 15
```

Implemented

- Geometry library
- Quaternion-based rotations
- Mechanical component model
- Stage solver
- Motion simulation
- Transfer curve analysis
- Error metrics
- Evolutionary optimization framework
- Mechanism optimization pipeline
- Unit test suite

Current test status

```
All tests passing
```

---

# Release Roadmap

| Version | Goal | Status |
|----------|------|--------|
| 0.1 | Geometry Library | Completed |
| 0.2 | Mechanics | Completed |
| 0.3 | Solver | Completed |
| 0.4 | Simulation | Completed |
| 0.5 | Analysis | Completed |
| 0.6 | Evolutionary Optimization | In Progress |
| 0.7 | Visualization | Planned |
| 0.8 | Project Persistence | Planned |
| 0.9 | GUI Prototype | Planned |
| 1.0 | Stable Engineering Framework | Planned |

---

# Sprint Overview

## Sprint 1–4

Geometry

Completed

- Point3D
- Vector3D
- Quaternion
- Rotation
- Numerical utilities

Status

Completed

---

## Sprint 5–8

Mechanics

Completed

- Lever
- Rod
- Stage
- Mechanism
- Builders
- Factory classes

Status

Completed

---

## Sprint 9–11

Solver

Completed

- AngleSolver
- StageSolver
- SolverState
- SolverResult
- Blocking detection

Status

Completed

---

## Sprint 12–14

Simulation and Analysis

Completed

- MotionRange
- StageSimulator
- TransferCurve
- TargetCurve
- ErrorMetric
- Fitness calculation

Status

Completed

---

## Sprint 15

Evolutionary Optimization

Completed

- Parameter
- ParameterSet
- Mutation
- Population
- PopulationFactory
- Selection
- Reproduction
- EvolutionEngine
- MechanismOptimizer
- OptimizationPipeline

Currently under refactoring

- StageSimulator
- MechanismSimulator
- OptimizationProblem

Goal

Stable optimization API

---

# Upcoming Sprints

## Sprint 16

Optimization API

Goals

- Complete simulator refactoring
- Remove remaining generic Callables
- Strongly typed public API
- Stable optimization workflow
- Complete end-to-end example

Deliverables

- OptimizationProblem
- MechanismSimulator
- Example project

---

## Sprint 17

Visualization

Goals

Generate engineering plots

Features

- Transfer curves
- Fitness history
- Population statistics
- Blocking visualization

Possible libraries

- matplotlib

---

## Sprint 18

Project Persistence

Goals

Save and load complete optimization projects.

Features

- JSON project format
- Parameter sets
- Target curves
- Optimization settings
- Result export

---

## Sprint 19

Engineering Reports

Goals

Automatic report generation.

Possible outputs

- PDF
- CSV
- Markdown

Include

- mechanism geometry
- optimized parameters
- fitness
- transfer curve
- convergence plots

---

## Sprint 20

Interactive GUI

Goals

Desktop application.

Possible technologies

- Qt
- PySide6

Functions

- Build mechanisms
- Start optimization
- Display curves
- Inspect solutions

---

# Future Research Topics

Possible future developments

- Multi-objective optimization
- Genetic crossover operators
- Adaptive mutation rates
- Parallel fitness evaluation
- Constraint optimization
- Automatic topology synthesis
- Multi-stage optimization
- Closed-loop optimization
- Sensitivity analysis
- Robust optimization
- Monte Carlo simulation
- Manufacturing tolerances

---

# Long-Term Vision

The optimizer shall eventually solve the following engineering problem.

Input

- Desired transfer curve
- Packaging constraints
- Mechanical limits
- Manufacturing constraints

Output

- Complete optimized linkage mechanism
- Geometric parameters
- Predicted transfer curve
- Engineering report

without manual tuning.

---

# Technical Debt

Known refactoring tasks

High Priority

- Complete simulator refactoring
- Eliminate remaining generic APIs
- Unify simulation interfaces

Medium Priority

- Improve naming consistency
- Reduce duplicate code
- Extend documentation

Low Priority

- Performance profiling
- Micro-optimizations

---

# Quality Goals

Every release shall satisfy

- 100% passing test suite
- Complete type hints
- Public API documentation
- Executable examples
- No known regressions

---

# Success Criteria

Version 1.0 is reached when

- the complete optimization workflow is stable,
- engineering examples execute without modification,
- optimization results are reproducible,
- documentation is complete,
- the framework is suitable for real-world linkage design.

---

# Guiding Principle

The project evolves incrementally.

Each sprint shall leave the repository in a fully working state.

Correctness, maintainability and reproducibility always take precedence over rapid feature growth.