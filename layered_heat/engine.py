"""Run / resume engine: commits steps transactionally, enforces energy closure."""

from __future__ import annotations

from .errors import HeatError
from .model import parse_case, parse_component
from .solver import check_residual, enthalpy, solve_step, time_grid


def _load(store, case_name):
    row = store.get_case_row(case_name)
    comp_spec = store.get_component_spec(row["component"])
    comp = parse_component(row["component"], comp_spec["layers"])
    case = parse_case(case_name, comp, __import__("json").loads(row["spec"]))
    return row, comp, case


def run_case(store, case_name, max_steps=None):
    """Start or resume a case. Returns progress dict."""
    row, comp, case = _load(store, case_name)
    if row["status"] == "done":
        return progress(store, case_name)
    if row["status"] == "failed":
        raise HeatError(f"case '{case_name}' failed: {row['error']}")

    layout = comp.cell_layout()
    grid = time_grid(case)
    total_steps = len(grid) - 1

    ckpt = store.get_checkpoint(case_name)
    if ckpt is None:
        temps = list(case.initial)
        scale0 = sum(
            abs(layout[2][i] * layout[0][i] * temps[i]) for i in range(len(temps))
        )
        store.init_checkpoint(case_name, temps, scale0)
        ckpt = store.get_checkpoint(case_name)
        store.set_status(case_name, "running")

    temps = ckpt["temps"]
    cum = ckpt["cum"]
    step = ckpt["step"]
    if abs(grid[step] - ckpt["time"]) > 1e-12 * max(1.0, abs(ckpt["time"])):
        raise HeatError("checkpoint time inconsistent with time grid")

    done = 0
    try:
        while step < total_steps:
            if max_steps is not None and done >= max_steps:
                store.set_status(case_name, "paused")
                return progress(store, case_name)
            t0, t1 = grid[step], grid[step + 1]
            temps, record = solve_step(case, layout, t0, t1, temps)
            cum["boundary"] += record["boundary_in"]
            cum["source"] += record["source_in"]
            cum["storage"] += record["storage"]
            cum["residual"] = cum["boundary"] + cum["source"] - cum["storage"]
            cum["scale"] += record["scale_step"]
            if not check_residual(cum["residual"], cum["scale"]):
                raise HeatError(
                    f"energy residual {cum['residual']:.3e} exceeds tolerance"
                    f" at t={t1}"
                )
            step += 1
            store.commit_step(case_name, step, t1, temps, record, cum)
            done += 1
    except HeatError as exc:
        store.set_status(case_name, "failed", str(exc))
        raise

    store.set_status(case_name, "done")
    return progress(store, case_name)


def progress(store, case_name):
    row, comp, case = _load(store, case_name)
    total_steps = len(time_grid(case)) - 1
    ckpt = store.get_checkpoint(case_name)
    done_steps = ckpt["step"] if ckpt else 0
    return {
        "case": case_name,
        "status": row["status"],
        "error": row["error"],
        "steps_done": done_steps,
        "steps_total": total_steps,
        "time": ckpt["time"] if ckpt else 0.0,
        "t_end": case.t_end,
        "complete": row["status"] == "done",
    }


def export_temperature(store, case_name):
    row, comp, case = _load(store, case_name)
    ckpt = store.get_checkpoint(case_name)
    if ckpt is None:
        raise HeatError(f"case '{case_name}' has not been started")
    dx, _, _, owner = comp.cell_layout()
    x = []
    pos = 0.0
    for i in range(len(dx)):
        x.append(pos + 0.5 * dx[i])
        pos += dx[i]
    return {
        "case": case_name,
        "status": row["status"],
        "complete": row["status"] == "done",
        "time": ckpt["time"],
        "x": x,
        "layer": owner,
        "temperature": ckpt["temps"],
    }


def export_energy(store, case_name):
    row, comp, case = _load(store, case_name)
    ckpt = store.get_checkpoint(case_name)
    if ckpt is None:
        raise HeatError(f"case '{case_name}' has not been started")
    layout = comp.cell_layout()
    return {
        "case": case_name,
        "status": row["status"],
        "complete": row["status"] == "done",
        "time": ckpt["time"],
        "cumulative": {
            "boundary_in": ckpt["cum"]["boundary"],
            "source_in": ckpt["cum"]["source"],
            "storage_change": ckpt["cum"]["storage"],
            "residual": ckpt["cum"]["residual"],
            "scale": ckpt["cum"]["scale"],
        },
        "enthalpy": enthalpy(layout, ckpt["temps"]),
        "steps": store.get_steps(case_name),
    }
