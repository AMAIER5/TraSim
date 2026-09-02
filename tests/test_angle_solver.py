"""
tests/test_angle_solver.py

Integration tests for the numerical AngleSolver.
"""

from __future__ import annotations

import math

from core.point3d import Point3D
from core.vector3d import Vector3D
from mechanics.lever import Lever
from mechanics.stage import Stage
from solver.angle_solver import AngleSolver
from solver.root_solver import RootSolver
from solver.solver_state import SolverState


def create_test_stage(
    *,
    input_angle_min: float = float("-inf"),
    input_angle_max: float = float("inf"),
    output_angle_min: float = float("-inf"),
    output_angle_max: float = float("inf"),
) -> Stage:

    input_lever = Lever(
        pivot=Point3D(0, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=50,
    )

    output_lever = Lever(
        pivot=Point3D(100, 0, 0),
        axis=Vector3D(0, 0, 1),
        length=50,
    )

    return Stage.from_reference_position(
        input_lever=input_lever,
        output_lever=output_lever,
        input_angle=0.0,
        output_angle=0.0,
        input_angle_min=input_angle_min,
        input_angle_max=input_angle_max,
        output_angle_min=output_angle_min,
        output_angle_max=output_angle_max,
        validate_reference=False,
    )


# ---------------------------------------------------------------------------
# Basic solving
# ---------------------------------------------------------------------------


def test_angle_solver_finds_solution():
    """
    Solver returns a valid solution for a solvable geometry.
    """

    stage = create_test_stage()

    solver = AngleSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        input_angle=0.0,
        state=state,
    )

    assert result.success is True

    assert result.residual < 1e-10


# ---------------------------------------------------------------------------
# Branch continuity
# ---------------------------------------------------------------------------


def test_angle_solver_preserves_branch():
    """
    Consecutive solutions remain continuous.
    """

    stage = create_test_stage()

    solver = AngleSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
        direction=1,
    )

    previous = state.last_output_angle

    for input_angle in (
        math.radians(5),
        math.radians(10),
        math.radians(15),
    ):

        result = solver.solve(
            input_angle=input_angle,
            state=state,
        )

        assert result.success is True

        assert abs(
            result.angle - previous
        ) < math.radians(20)

        state = state.next(
            input_angle=input_angle,
            output_angle=result.angle,
        )

        previous = result.angle


# ---------------------------------------------------------------------------
# Kinematic branch selection
# ---------------------------------------------------------------------------


def test_angle_solver_prefers_continuous_velocity_branch():
    """
    Branch selection prefers kinematic continuity over
    a closer prediction if the velocity jump becomes
    unrealistic.
    """

    brackets = [
        (
            math.radians(-23),
            math.radians(-21),
            1,
        ),
        (
            math.radians(-36),
            math.radians(-34),
            1,
        ),
    ]

    state = SolverState(
        last_input_angle=math.radians(10),
        last_output_angle=math.radians(-10),
        direction=-1,
        output_velocity=-5.0,
    )

    selected = AngleSolver._select_branch(
        brackets,
        reference_angle=math.radians(-22),
        state=state,
        input_angle=math.radians(15),
    )

    left, right, _ = selected

    selected_angle = (
        left + right
    ) / 2.0

    assert math.isclose(
        selected_angle,
        math.radians(-35),
        abs_tol=math.radians(1),
    )


def test_angle_solver_without_motion_history_uses_prediction():
    """
    Initial solver state behaves like the previous
    prediction-based branch selection.
    """

    brackets = [
        (
            math.radians(-31),
            math.radians(-29),
            1,
        ),
        (
            math.radians(20),
            math.radians(22),
            1,
        ),
    ]

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
        direction=0,
        output_velocity=0.0,
    )

    selected = AngleSolver._select_branch(
        brackets,
        reference_angle=math.radians(-30),
        state=state,
        input_angle=math.radians(5),
    )

    left, right, _ = selected

    selected_angle = (
        left + right
    ) / 2.0

    assert math.isclose(
        selected_angle,
        math.radians(-30),
        abs_tol=math.radians(1),
    )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_angle_solver_iteration_limit():
    """
    Numerical solver should converge quickly.
    """

    stage = create_test_stage()

    solver = AngleSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    result = solver.solve(
        input_angle=0.0,
        state=state,
    )

    assert result.success is True

    assert solver.get_stats()["solved"] == 1


# ---------------------------------------------------------------------------
# Output angle limits
# ---------------------------------------------------------------------------


def test_angle_solver_global_search_respects_stage_output_limits(
    monkeypatch,
):
    """
    Global bracket search is restricted to the Stage
    output angle limits.

    Fix #3: The full-range search is now a fallback that
    only runs when local/adaptive searches find nothing.
    We monkeypatch both find_all_brackets_around (used by
    the local search) and find_all_brackets (used by the
    fallback and the diagnostic block) so that the local
    search returns empty and the fallback's
    [minimum, maximum] can be captured.

    The fallback call uses allowed_min/allowed_max (the
    stage output limits).  The diagnostic call (which runs
    after the fallback also returns empty) uses
    self.search_min/self.search_max.  We capture only the
    first find_all_brackets call to verify the fallback
    range.
    """

    stage = create_test_stage(
        output_angle_min=math.radians(-20),
        output_angle_max=math.radians(35),
    )

    solver = AngleSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    captured = {}

    def fake_find_all_brackets_around(
        *,
        function,
        center,
        window,
        step,
    ):
        # Simulate local/adaptive search finding nothing.
        return []

    call_count = {"n": 0}

    def fake_find_all_brackets(
        *,
        function,
        minimum,
        maximum,
        step,
    ):
        # Capture only the first call (the fallback).
        # The second call is the diagnostic block with
        # self.search_min/self.search_max.
        if call_count["n"] == 0:
            captured["minimum"] = minimum
            captured["maximum"] = maximum
        call_count["n"] += 1
        return []

    monkeypatch.setattr(
        RootSolver,
        "find_all_brackets",
        fake_find_all_brackets,
    )
    monkeypatch.setattr(
        RootSolver,
        "find_all_brackets_around",
        fake_find_all_brackets_around,
    )

    result = solver.solve(
        input_angle=math.radians(90),
        state=state,
    )

    assert result.success is False

    assert captured["minimum"] == math.radians(-20)

    assert captured["maximum"] == math.radians(35)


# ---------------------------------------------------------------------------
# Adaptive / local search fast-path behaviour (Issue #3)
# ---------------------------------------------------------------------------


def test_adaptive_search_short_circuits_full_range_scan(
    monkeypatch,
):
    """
    Fix #3: When the adaptive search finds brackets, the
    full-range fallback must NOT be called.
    """

    stage = create_test_stage()

    solver = AngleSolver(stage)

    # Prime the adaptive reuse path.
    solver._last_root = 0.0
    solver._last_bracket_width = math.radians(2)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
        direction=1,
        output_velocity=1.0,
    )

    fallback_called = False

    def fake_find_all_brackets_around(
        *,
        function,
        center,
        window,
        step,
    ):
        # Return a valid bracket around the predicted angle.
        return [
            (
                center - step,
                center + step,
                2,
            )
        ]

    def fake_find_all_brackets(
        *,
        function,
        minimum,
        maximum,
        step,
    ):
        nonlocal fallback_called
        fallback_called = True
        return []

    def fake_solve_brent(
        *,
        function,
        left,
        right,
        tolerance,
        max_iterations,
    ):
        # Return a fake successful Brent result.
        return (left + right) / 2.0, 0.0, 3

    monkeypatch.setattr(
        RootSolver,
        "find_all_brackets_around",
        fake_find_all_brackets_around,
    )
    monkeypatch.setattr(
        RootSolver,
        "find_all_brackets",
        fake_find_all_brackets,
    )
    monkeypatch.setattr(
        RootSolver,
        "solve_brent",
        fake_solve_brent,
    )

    result = solver.solve(
        input_angle=math.radians(5),
        state=state,
    )

    assert result.success is True

    assert fallback_called is False

    assert solver.get_stats()["adaptive_success"] == 1

    assert solver.get_stats()["fallback_searches"] == 0


def test_local_search_short_circuits_full_range_scan(
    monkeypatch,
):
    """
    Fix #3: When the local search (no _last_root) finds
    brackets, the full-range fallback must NOT be called.
    """

    stage = create_test_stage()

    solver = AngleSolver(stage)

    state = SolverState(
        last_input_angle=0.0,
        last_output_angle=0.0,
    )

    fallback_called = False

    def fake_find_all_brackets_around(
        *,
        function,
        center,
        window,
        step,
    ):
        return [
            (
                center - step,
                center + step,
                2,
            )
        ]

    def fake_find_all_brackets(
        *,
        function,
        minimum,
        maximum,
        step,
    ):
        nonlocal fallback_called
        fallback_called = True
        return []

    def fake_solve_brent(
        *,
        function,
        left,
        right,
        tolerance,
        max_iterations,
    ):
        # Return a fake successful Brent result.
        return (left + right) / 2.0, 0.0, 3

    monkeypatch.setattr(
        RootSolver,
        "find_all_brackets_around",
        fake_find_all_brackets_around,
    )
    monkeypatch.setattr(
        RootSolver,
        "find_all_brackets",
        fake_find_all_brackets,
    )
    monkeypatch.setattr(
        RootSolver,
        "solve_brent",
        fake_solve_brent,
    )

    result = solver.solve(
        input_angle=0.0,
        state=state,
    )

    assert result.success is True

    assert fallback_called is False

    assert solver.get_stats()["local_searches"] == 1

    assert solver.get_stats()["fallback_searches"] == 0


# ---------------------------------------------------------------------------
# Fast-path output range filtering (Issue #4)
# ---------------------------------------------------------------------------


def test_fast_path_respects_output_limits(monkeypatch):
    """
    Fix #4: Brackets found by the adaptive/local searches
    that lie outside the stage output angle limits must be
    filtered out, causing a fallback to the full-range search.
    """

    stage = create_test_stage(
        output_angle_min=math.radians(-10),
        output_angle_max=math.radians(10),
    )

    solver = AngleSolver(stage)

    # Prime the adaptive path so find_all_brackets_around
    # is called with a bracket outside the allowed range.
    solver._last_root = math.radians(50)
    solver._last_bracket_width = math.radians(5)

    state = SolverState(
        last_input_angle=math.radians(5),
        last_output_angle=math.radians(50),
        direction=1,
        output_velocity=1.0,
    )

    fallback_called = False

    def fake_find_all_brackets_around(
        *,
        function,
        center,
        window,
        step,
    ):
        # Return a bracket centred well outside the allowed
        # output range of [-10°, 10°].
        return [
            (
                math.radians(40),
                math.radians(42),
                2,
            )
        ]

    def fake_find_all_brackets(
        *,
        function,
        minimum,
        maximum,
        step,
    ):
        nonlocal fallback_called
        fallback_called = True
        # Return a bracket inside the allowed range.
        return [
            (
                math.radians(-1),
                math.radians(1),
                2,
            )
        ]

    def fake_solve_brent(
        *,
        function,
        left,
        right,
        tolerance,
        max_iterations,
    ):
        # Return a fake successful Brent result.
        return (left + right) / 2.0, 0.0, 3

    monkeypatch.setattr(
        RootSolver,
        "find_all_brackets_around",
        fake_find_all_brackets_around,
    )
    monkeypatch.setattr(
        RootSolver,
        "find_all_brackets",
        fake_find_all_brackets,
    )
    monkeypatch.setattr(
        RootSolver,
        "solve_brent",
        fake_solve_brent,
    )

    result = solver.solve(
        input_angle=math.radians(10),
        state=state,
    )

    # The out-of-range adaptive bracket must be filtered,
    # and the fallback must be invoked.
    assert fallback_called is True

    assert result.success is True

    assert (
        math.radians(-10)
        <= result.angle
        <= math.radians(10)
    )