"""Data model and validation for components (layer stacks) and cases."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .errors import HeatError

AREA = 1.0  # m^2, fixed cross-section
MAX_CELLS = 5000


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HeatError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise HeatError(f"{name} must be finite")
    return value


def _positive(value, name):
    value = _finite(value, name)
    if value <= 0.0:
        raise HeatError(f"{name} must be positive")
    return value


@dataclass
class Layer:
    thickness: float  # m
    k: float  # W/(m*K)
    cv: float  # volumetric heat capacity J/(m^3*K)
    n: int  # number of equal cells

    @property
    def dx(self):
        return self.thickness / self.n


@dataclass
class Component:
    name: str
    layers: list  # list[Layer]

    @property
    def ncells(self):
        return sum(layer.n for layer in self.layers)

    def cell_layout(self):
        """Return (dx, k, cv, layer_index) per cell."""
        dx, k, cv, owner = [], [], [], []
        for li, layer in enumerate(self.layers):
            for _ in range(layer.n):
                dx.append(layer.dx)
                k.append(layer.k)
                cv.append(layer.cv)
                owner.append(li)
        return dx, k, cv, owner


def parse_component(name, layers_spec):
    if not isinstance(name, str) or not name:
        raise HeatError("component name must be a non-empty string")
    if not isinstance(layers_spec, list) or not layers_spec:
        raise HeatError("layers must be a non-empty list")
    layers = []
    for i, spec in enumerate(layers_spec):
        if not isinstance(spec, dict):
            raise HeatError(f"layer {i} must be an object")
        try:
            thickness = _positive(spec["thickness"], f"layer {i} thickness")
            k = _positive(spec["k"], f"layer {i} k")
            cv = _positive(spec["cv"], f"layer {i} cv")
            n = spec["n"]
        except KeyError as exc:
            raise HeatError(f"layer {i} missing field {exc}") from exc
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise HeatError(f"layer {i} n must be a positive integer")
        layers.append(Layer(thickness, k, cv, n))
    comp = Component(name, layers)
    if comp.ncells > MAX_CELLS:
        raise HeatError(f"total cells {comp.ncells} exceed limit {MAX_CELLS}")
    return comp


def parse_series(value, name):
    """Piecewise-constant series: [[t, v], ...], times strictly increasing from 0."""
    if not isinstance(value, list) or not value:
        raise HeatError(f"{name} must be a non-empty list of [time, value]")
    points = []
    prev_t = None
    for i, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise HeatError(f"{name}[{i}] must be [time, value]")
        t = _finite(item[0], f"{name}[{i}] time")
        v = _finite(item[1], f"{name}[{i}] value")
        if t < 0.0:
            raise HeatError(f"{name}[{i}] time must be >= 0")
        if prev_t is not None and t <= prev_t:
            raise HeatError(f"{name} times must be strictly increasing")
        points.append((t, v))
        prev_t = t
    if points[0][0] != 0.0:
        raise HeatError(f"{name} must start at time 0")
    return points


def series_value(points, t):
    """Value of a piecewise-constant series at time t (change point uses new value)."""
    v = points[0][1]
    for tp, vp in points:
        if tp <= t:
            v = vp
        else:
            break
    return v


@dataclass
class Boundary:
    kind: str  # "temperature" or "flux"
    series: list  # list[(t, value)]; temperature in C, flux W/m^2 inward positive


@dataclass
class Case:
    name: str
    component: str
    dt: float
    t_end: float
    initial: list  # per-cell initial temperature, C
    left: Boundary
    right: Boundary
    sources: list  # per-layer series list[(t, W/m^3)]
    status: str = "created"  # created | running | paused | done | failed
    error: str = ""


def _parse_boundary(spec, name):
    if not isinstance(spec, dict):
        raise HeatError(f"{name} boundary must be an object")
    kind = spec.get("type")
    if kind not in ("temperature", "flux"):
        raise HeatError(f"{name} boundary type must be 'temperature' or 'flux'")
    if "series" not in spec:
        raise HeatError(f"{name} boundary missing series")
    return Boundary(kind, parse_series(spec["series"], f"{name} boundary series"))


def parse_case(name, component, spec):
    if not isinstance(name, str) or not name:
        raise HeatError("case name must be a non-empty string")
    if not isinstance(spec, dict):
        raise HeatError("case spec must be an object")
    dt = _positive(spec.get("dt"), "dt")
    t_end = _positive(spec.get("t_end"), "t_end")
    initial = spec.get("initial")
    if isinstance(initial, (int, float)) and not isinstance(initial, bool):
        initial = [_finite(initial, "initial")] * component.ncells
    elif isinstance(initial, list):
        if len(initial) != component.ncells:
            raise HeatError(
                f"initial temperature size {len(initial)} != cell count {component.ncells}"
            )
        initial = [_finite(v, "initial temperature") for v in initial]
    else:
        raise HeatError("initial must be a number or a list of per-cell temperatures")
    left = _parse_boundary(spec.get("left"), "left")
    right = _parse_boundary(spec.get("right"), "right")
    src_spec = spec.get("sources")
    nlayers = len(component.layers)
    if src_spec is None:
        sources = [[(0.0, 0.0)] for _ in range(nlayers)]
    else:
        if isinstance(src_spec, dict):
            src_list = [src_spec.get(str(i), src_spec.get(i)) for i in range(nlayers)]
        elif isinstance(src_spec, list):
            if len(src_spec) != nlayers:
                raise HeatError(f"sources must have {nlayers} entries")
            src_list = src_spec
        else:
            raise HeatError("sources must be a list or object keyed by layer index")
        sources = []
        for i, s in enumerate(src_list):
            if s is None:
                sources.append([(0.0, 0.0)])
            elif isinstance(s, (int, float)) and not isinstance(s, bool):
                sources.append([(0.0, _finite(s, f"source layer {i}"))])
            else:
                sources.append(parse_series(s, f"source layer {i}"))
    return Case(
        name=name,
        component=component.name,
        dt=dt,
        t_end=t_end,
        initial=initial,
        left=left,
        right=right,
        sources=sources,
    )


def change_points(case):
    """All series change points within (0, t_end]."""
    pts = set()
    for series in [case.left.series, case.right.series, *case.sources]:
        for t, _ in series:
            if 0.0 < t <= case.t_end:
                pts.add(t)
    return pts
