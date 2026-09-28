"""Conservative finite-volume solver: backward Euler, tridiagonal Thomas solve."""

from __future__ import annotations

import math

from .errors import HeatError
from .model import AREA, change_points, series_value

RESIDUAL_TOL = 1e-9


def time_grid(case):
    """Union of uniform dt grid from 0, all change points, and t_end."""
    pts = {0.0, case.t_end}
    n = int(case.t_end / case.dt) + 1
    for i in range(n + 1):
        t = i * case.dt
        if t <= case.t_end:
            pts.add(t)
    pts |= change_points(case)
    return sorted(pts)


def _interface_resistance(dx_i, k_i, dx_j, k_j):
    return 0.5 * dx_i / k_i + 0.5 * dx_j / k_j


def solve_step(case, layout, t_old, t_new, temps):
    """One backward-Euler step. Returns (new_temps, record) with energy terms."""
    dx, k, cv, owner = layout
    n = len(temps)
    dt = t_new - t_old
    if dt <= 0.0 or not math.isfinite(dt):
        raise HeatError("non-positive step size")

    left = case.left
    right = case.right
    tb_left = series_value(left.series, t_new) if left.kind == "temperature" else None
    tb_right = series_value(right.series, t_new) if right.kind == "temperature" else None
    q_left = series_value(left.series, t_new) if left.kind == "flux" else None
    q_right = series_value(right.series, t_new) if right.kind == "flux" else None
    qv = [series_value(case.sources[owner[i]], t_new) for i in range(n)]

    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    d = [0.0] * n

    for i in range(n):
        cap = cv[i] * AREA * dx[i] / dt
        b[i] = cap
        d[i] = cap * temps[i] + qv[i] * AREA * dx[i]
        if i > 0:
            g = AREA / _interface_resistance(dx[i - 1], k[i - 1], dx[i], k[i])
            a[i] = -g
            b[i] += g
        if i < n - 1:
            g = AREA / _interface_resistance(dx[i], k[i], dx[i + 1], k[i + 1])
            c[i] = -g
            b[i] += g

    if left.kind == "temperature":
        g = AREA / (0.5 * dx[0] / k[0])
        b[0] += g
        d[0] += g * tb_left
    else:
        d[0] += q_left * AREA
    if right.kind == "temperature":
        g = AREA / (0.5 * dx[-1] / k[-1])
        b[-1] += g
        d[-1] += g * tb_right
    else:
        d[-1] += q_right * AREA

    new_temps = _thomas(a, b, c, d)
    for t in new_temps:
        if not math.isfinite(t):
            raise HeatError("non-finite temperature (overflow)")

    if left.kind == "temperature":
        g = AREA / (0.5 * dx[0] / k[0])
        in_left = g * (tb_left - new_temps[0]) * dt
    else:
        in_left = q_left * AREA * dt
    if right.kind == "temperature":
        g = AREA / (0.5 * dx[-1] / k[-1])
        in_right = g * (tb_right - new_temps[-1]) * dt
    else:
        in_right = q_right * AREA * dt
    boundary_in = in_left + in_right
    layer_source = {}
    for i in range(n):
        layer_source[owner[i]] = layer_source.get(owner[i], 0.0) + qv[i] * AREA * dx[i]
    source_in = sum(layer_source.values()) * dt
    scale_step = (
        abs(in_left)
        + abs(in_right)
        + sum(abs(v) for v in layer_source.values()) * dt
    )
    storage = sum(cv[i] * AREA * dx[i] * (new_temps[i] - temps[i]) for i in range(n))

    record = {
        "t0": t_old,
        "t1": t_new,
        "dt": dt,
        "boundary_in": boundary_in,
        "boundary_left": in_left,
        "boundary_right": in_right,
        "source_in": source_in,
        "storage": storage,
        "residual": boundary_in + source_in - storage,
        "scale_step": scale_step,
    }
    return new_temps, record


def _thomas(a, b, c, d):
    n = len(d)
    cp = [0.0] * n
    dp = [0.0] * n
    beta = b[0]
    if beta == 0.0:
        raise HeatError("singular system")
    cp[0] = c[0] / beta
    dp[0] = d[0] / beta
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        if m == 0.0:
            raise HeatError("singular system")
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def enthalpy(layout, temps):
    """Heat content relative to 0 C."""
    dx, _, cv, _ = layout
    return sum(cv[i] * AREA * dx[i] * temps[i] for i in range(len(temps)))


def check_residual(cum_residual, scale):
    return abs(cum_residual) <= RESIDUAL_TOL * max(1.0, scale)
