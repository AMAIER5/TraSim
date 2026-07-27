# Koppelgetriebe Simulator

# Developer Guide

| Property | Value |
|----------|-------|
| Version | 0.1.0 |
| Status | Living Document |
| Purpose | Development Rules and Coding Standards |

---

# 1. Purpose

This document defines the development rules for the project.

It is intended to ensure that

- every module follows the same architecture,
- all source files have a consistent style,
- the public API remains stable,
- refactorings are predictable,
- future contributors can work efficiently.

This document complements **architecture.md**.

Architecture describes **what** the software is.

Developer Guide describes **how** it is developed.

---

# 2. General Principles

The following priorities always apply.

1. Correctness before performance
2. Performance before convenience
3. Readability before cleverness
4. Explicitness before magic
5. Simplicity before abstraction

---

# 3. Language

The complete source code shall use English.

Including

- identifiers
- comments
- docstrings
- exception messages
- documentation

Only user-facing applications may later provide localized output.

---

# 4. Python Version

Current target

```
Python 3.12+
```

The project intentionally uses modern Python features.

Examples

- dataclasses
- slots=True
- frozen=True
- pattern matching (when appropriate)
- modern type hints

---

# 5. Code Formatting

Maximum line length

```
79 characters
```

Imports

```
Standard Library

↓

Third-party packages

↓

Project imports
```

One import per line whenever practical.

No wildcard imports.

---

# 6. Type Hints

Every public function shall have complete type hints.

Required

```python
def solve(
    stage: Stage,
) -> SolverResult:
```

Forbidden

```python
Any
```

unless absolutely unavoidable.

Prefer domain classes over generic types.

Example

Good

```python
MechanismSimulator
CurveFitness
SimulationResult
```

instead of

```python
Callable
dict
tuple
```

in public APIs.

---

# 7. Dataclasses

Immutable objects shall use

```python
@dataclass(
    frozen=True,
    slots=True,
)
```

Examples

- Point3D
- Vector3D
- Lever
- Rod
- Stage
- Mechanism
- Parameter
- ParameterSet

---

# 8. Validation

All constructor validation belongs into

```python
__post_init__()
```

Examples

- invalid length
- invalid axis
- empty population
- negative dimensions

Fail fast.

---

# 9. Collections

Immutable collections

```
tuple
```

Mutable collections

```
list
```

shall only exist inside algorithms.

Never expose mutable lists through the public API.

---

# 10. Public API

Public interfaces shall be strongly typed.

Example

```python
MechanismSimulator
```

instead of

```python
Callable
```

Public interfaces shall evolve towards explicit domain objects.

---

# 11. Architecture Rules

Dependencies are only allowed downwards.

```
Optimization
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

Never introduce cyclic dependencies.

---

# 12. Class Responsibilities

Each class shall own exactly one responsibility.

Examples

```
StageSolver

→ solves one kinematic step
```

```
StageSimulator

→ simulates one stage
```

```
MechanismSimulator

→ simulates a complete mechanism
```

```
CurveFitness

→ evaluates one transfer curve
```

```
EvolutionEngine

→ performs one evolutionary generation
```

---

# 13. Testing

Every production module shall have a matching test module.

```
module.py

↓

test_module.py
```

Every public class requires tests for

- construction
- nominal behaviour
- invalid parameters
- edge cases
- regression

Bug fixes shall include a regression test.

---

# 14. Refactoring

Refactorings shall be complete.

Whenever a public interface changes,
the following shall be updated together.

- production code
- unit tests
- examples
- documentation

No partially migrated APIs.

---

# 15. Examples

Every larger subsystem shall provide at least one executable example.

Examples are part of the public documentation.

They should represent realistic user workflows.

---

# 16. Documentation

Every public class shall contain

- docstring
- parameter description
- return value description

Complex algorithms should additionally explain

- mathematical background
- assumptions
- limitations

---

# 17. Numerical Rules

Floating-point values shall never be compared directly.

Forbidden

```python
if value == 0:
```

Required

```python
if abs(value) < tolerance:
```

All tolerances shall be defined centrally.

---

# 18. Development Workflow

Development proceeds in vertical slices.

Each sprint shall

- implement one coherent feature,
- include complete unit tests,
- keep the project executable,
- avoid unrelated changes.

---

# 19. ChatGPT Workflow

Large changes shall be implemented in coherent refactoring phases.

For every affected module ChatGPT shall output

- the complete source file,
- not only patches.

Whenever interfaces change,
all dependent files shall be updated together.

No code shall rely on guessed interfaces.
If required interfaces are unknown,
they shall be inspected before refactoring.

---

# 20. Sprint Completion Criteria

A sprint is complete when

- all tests pass,
- examples run,
- documentation is updated,
- no TODOs remain inside the implemented feature,
- the public API is internally consistent.

Future ideas shall never be implemented before the current sprint is complete.

---

# 21. Long-Term Goal

The project shall evolve into a reusable framework for

- kinematic simulation,
- optimization,
- visualization,
- engineering design.

Every architectural decision should support long-term maintainability over short-term convenience.