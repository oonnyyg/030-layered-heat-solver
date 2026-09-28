"""Backward-Euler finite-volume step with linear tridiagonal solve."""
from __future__ import annotations

from .model import AREA, value_at


class StepFailure(RuntimeError):
    """Raised when a step fails (overflow or energy-balance tolerance)."""


def _interface_conductance(dx_i, k_i, dx_j, k_j):
    # R = half-cell thickness / own k on each side, summed. Area = 1 m^2.
    r = (dx_i * 0.5) / k_i + (dx_j * 0.5) / k_j
    return 1.0 / r


def _half_cell_conductance(dx, k):
    return 1.0 / ((dx * 0.5) / k)


def _thomas(lower, diag, upper, rhs):
    """Solve tridiagonal system in O(n). lower[i] multiplies x[i-1] (i>=1)."""
    n = len(diag)
    c = [0.0] * n
    d = [0.0] * n
    beta = diag[0]
    if beta == 0.0:
        raise StepFailure("singular matrix")
    c[0] = upper[0] / beta if n > 1 else 0.0
    d[0] = rhs[0] / beta
    for i in range(1, n):
        m = diag[i] - lower[i] * c[i - 1]
        if m == 0.0:
            raise StepFailure("singular matrix")
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / m
    x = [0.0] * n
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def assemble(grid, case, t0, dt):
    """Build tridiagonal system for one backward-Euler step starting at t0."""
    n = grid["n"]
    dx, k, cv = grid["dx"], grid["k"], grid["cv"]
    lower = [0.0] * n
    diag = [0.0] * n
    upper = [0.0] * n
    rhs_add = [0.0] * n  # everything except cv*V/dt*T_old
    cap = [cv[i] * dx[i] * AREA / dt for i in range(n)]

    for i in range(n - 1):
        g = _interface_conductance(dx[i], k[i], dx[i + 1], k[i + 1])
        lower[i + 1] = -g
        upper[i] = -g
        diag[i] += g
        diag[i + 1] += g

    left = case["left"]
    if left["type"] == "temperature":
        tval = value_at(left["points"], t0)
        g = _half_cell_conductance(dx[0], k[0])
        diag[0] += g
        rhs_add[0] += g * tval
    else:
        rhs_add[0] += value_at(left["points"], t0) * AREA  # inward-positive flux

    right = case["right"]
    if right["type"] == "temperature":
        tval = value_at(right["points"], t0)
        g = _half_cell_conductance(dx[-1], k[-1])
        diag[-1] += g
        rhs_add[-1] += g * tval
    else:
        rhs_add[-1] += value_at(right["points"], t0) * AREA

    for i in range(n):
        diag[i] += cap[i]
        li = grid["layer_of"][i]
        q = value_at(case["sources"][li], t0)
        rhs_add[i] += q * dx[i] * AREA

    return lower, diag, upper, rhs_add, cap


def step(grid, case, temps, t0, t1):
    """Advance one step. Returns (new_temps, record) with energy terms in joules."""
    dt = t1 - t0
    if dt <= 0.0:
        raise StepFailure("non-positive step size")
    n = grid["n"]
    lower, diag, upper, rhs_add, cap = assemble(grid, case, t0, dt)
    rhs = [rhs_add[i] + cap[i] * temps[i] for i in range(n)]
    try:
        new = _thomas(lower, diag, upper, rhs)
    except OverflowError as exc:
        raise StepFailure(f"numeric overflow: {exc}") from exc
    for v in new:
        if v != v or v in (float("inf"), float("-inf")):
            raise StepFailure("non-finite temperature produced")

    # Energy accounting for the step (area = 1 m^2).
    left = case["left"]
    if left["type"] == "temperature":
        g = _half_cell_conductance(grid["dx"][0], grid["k"][0])
        q_left = g * (value_at(left["points"], t0) - new[0])
    else:
        q_left = value_at(left["points"], t0)
    right = case["right"]
    if right["type"] == "temperature":
        g = _half_cell_conductance(grid["dx"][-1], grid["k"][-1])
        q_right = g * (value_at(right["points"], t0) - new[-1])
    else:
        q_right = value_at(right["points"], t0)

    boundary_in = (q_left + q_right) * AREA * dt
    source_in = 0.0
    for li, vol in enumerate(grid["layer_vol"]):
        source_in += value_at(case["sources"][li], t0) * vol * dt
    storage = sum(
        grid["cv"][i] * grid["dx"][i] * AREA * (new[i] - temps[i]) for i in range(n))
    residual = storage - (boundary_in + source_in)

    record = {
        "t0": t0, "t1": t1, "dt": dt,
        "q_left": q_left, "q_right": q_right,
        "boundary_in": boundary_in,
        "source_in": source_in,
        "storage": storage,
        "residual": residual,
        "denom_add": (abs(q_left * AREA * dt) + abs(q_right * AREA * dt)
                      + sum(abs(value_at(case["sources"][li], t0) * vol * dt)
                            for li, vol in enumerate(grid["layer_vol"]))),
    }
    return new, record


def initial_enthalpy_abs_sum(grid, temps):
    return sum(abs(grid["cv"][i] * grid["dx"][i] * AREA * temps[i])
               for i in range(grid["n"]))


TOL = 1e-9


def check_balance(cum_residual, denom):
    return abs(cum_residual) <= TOL * max(1.0, denom)
