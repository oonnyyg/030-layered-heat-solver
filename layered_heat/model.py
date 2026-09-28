"""Component/case validation and grid construction.

All units SI except temperatures in degC. Cross-sectional area is 1 m^2.
"""
from __future__ import annotations

import math

MAX_CELLS = 5000
AREA = 1.0  # m^2


class ValidationError(ValueError):
    pass


def _finite(x, name):
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        raise ValidationError(f"{name} must be a finite number, got {x!r}")
    return float(x)


def _positive(x, name):
    v = _finite(x, name)
    if v <= 0.0:
        raise ValidationError(f"{name} must be positive, got {x!r}")
    return v


def validate_component(spec):
    """Validate a component (layered wall) spec. Returns normalized dict."""
    if not isinstance(spec, dict):
        raise ValidationError("component must be an object")
    layers_in = spec.get("layers")
    if not isinstance(layers_in, list) or not layers_in:
        raise ValidationError("component.layers must be a non-empty list")
    layers = []
    total_cells = 0
    for i, lay in enumerate(layers_in):
        if not isinstance(lay, dict):
            raise ValidationError(f"layer {i} must be an object")
        thickness = _positive(lay.get("thickness"), f"layer {i} thickness")
        k = _positive(lay.get("k"), f"layer {i} k")
        cv = _positive(lay.get("cv"), f"layer {i} cv")
        n = lay.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValidationError(f"layer {i} n must be a positive integer")
        total_cells += n
        layers.append({"thickness": thickness, "k": k, "cv": cv, "n": n})
    if total_cells > MAX_CELLS:
        raise ValidationError(f"total cells {total_cells} exceeds {MAX_CELLS}")
    return {"layers": layers}


def validate_points(points, name):
    """Piecewise-constant time points: [[t, value], ...], t from 0, strictly increasing."""
    if not isinstance(points, list) or not points:
        raise ValidationError(f"{name} must be a non-empty list of [t, value]")
    out = []
    prev_t = None
    for i, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValidationError(f"{name}[{i}] must be [t, value]")
        t = _finite(p[0], f"{name}[{i}] time")
        v = _finite(p[1], f"{name}[{i}] value")
        if t < 0.0:
            raise ValidationError(f"{name}[{i}] time must be >= 0")
        if prev_t is not None and t <= prev_t:
            raise ValidationError(f"{name} times must be strictly increasing")
        prev_t = t
        out.append([t, v])
    if out[0][0] != 0.0:
        raise ValidationError(f"{name} must start at t=0")
    return out


def validate_boundary(spec, name):
    if not isinstance(spec, dict):
        raise ValidationError(f"{name} must be an object")
    btype = spec.get("type")
    if btype not in ("temperature", "flux"):
        raise ValidationError(f"{name}.type must be 'temperature' or 'flux'")
    points = validate_points(spec.get("points"), f"{name}.points")
    return {"type": btype, "points": points}


def validate_case(spec, component):
    """Validate a case spec against a component. Returns normalized config."""
    if not isinstance(spec, dict):
        raise ValidationError("case must be an object")
    n_layers = len(component["layers"])
    total_cells = sum(l["n"] for l in component["layers"])

    t_end = _positive(spec.get("t_end"), "t_end")
    dt = _positive(spec.get("dt"), "dt")

    initial = spec.get("initial_temp")
    if isinstance(initial, (int, float)) and not isinstance(initial, bool):
        initial_temps = [_finite(initial, "initial_temp")] * total_cells
    elif isinstance(initial, list):
        if len(initial) != total_cells:
            raise ValidationError(
                f"initial_temp length {len(initial)} != total cells {total_cells}")
        initial_temps = [_finite(v, "initial_temp") for v in initial]
    else:
        raise ValidationError("initial_temp must be a number or a list of per-cell values")

    bnd = spec.get("boundaries")
    if not isinstance(bnd, dict):
        raise ValidationError("boundaries must be an object with left/right")
    left = validate_boundary(bnd.get("left"), "boundaries.left")
    right = validate_boundary(bnd.get("right"), "boundaries.right")

    sources_in = spec.get("sources")
    if sources_in is None:
        sources = [[[0.0, 0.0]] for _ in range(n_layers)]
    else:
        if not isinstance(sources_in, list) or len(sources_in) != n_layers:
            raise ValidationError(f"sources must be a list of {n_layers} point lists")
        sources = [validate_points(s, f"sources[{i}]") for i, s in enumerate(sources_in)]

    # Change points must not exceed t_end.
    for pts, name in [(left["points"], "left"), (right["points"], "right")] + [
            (s, f"sources[{i}]") for i, s in enumerate(sources)]:
        for t, _ in pts:
            if t > t_end:
                raise ValidationError(f"{name} has a change point past t_end")

    return {
        "t_end": t_end,
        "dt": dt,
        "initial_temps": initial_temps,
        "left": left,
        "right": right,
        "sources": sources,
    }


def build_grid(component):
    """Return per-cell arrays: dx, k, cv, layer index; and per-layer volumes."""
    dx, k, cv, layer_of = [], [], [], []
    layer_vol = []
    for li, lay in enumerate(component["layers"]):
        n = lay["n"]
        d = lay["thickness"] / n
        dx.extend([d] * n)
        k.extend([lay["k"]] * n)
        cv.extend([lay["cv"]] * n)
        layer_of.extend([li] * n)
        layer_vol.append(lay["thickness"] * AREA)
    return {"dx": dx, "k": k, "cv": cv, "layer_of": layer_of,
            "layer_vol": layer_vol, "n": len(dx)}


def value_at(points, t):
    """Piecewise-constant value in effect for a step starting at time t."""
    v = points[0][1]
    for pt, pv in points:
        if pt <= t:
            v = pv
        else:
            break
    return v


def build_time_grid(case):
    """Union of 0, k*dt <= t_end, all change points, and t_end (sorted, unique)."""
    t_end, dt = case["t_end"], case["dt"]
    ts = {0.0, t_end}
    k = 1
    while k * dt < t_end:
        ts.add(k * dt)
        k += 1
    for pts in [case["left"]["points"], case["right"]["points"], *case["sources"]]:
        for t, _ in pts:
            if 0.0 < t < t_end:
                ts.add(t)
    return sorted(ts)
