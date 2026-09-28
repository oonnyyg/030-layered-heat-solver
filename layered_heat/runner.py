"""Run/resume engine: commits steps with checkpoints, verifies energy balance."""
from __future__ import annotations

from .model import build_grid, build_time_grid
from .solver import (StepFailure, check_balance, initial_enthalpy_abs_sum,
                     step)


def run_case(store, case_id, max_steps=None):
    """Advance a case from its checkpoint. Returns a progress dict."""
    case = store.get_case(case_id)
    if case["status"] == "completed":
        return progress(store, case_id)
    if case["status"] == "failed":
        raise StepFailure(f"case {case_id} is failed; create a new case")
    component = store.get_component(case["component_id"])
    config = case["config"]
    grid = build_grid(component)
    times = build_time_grid(config)

    ckpt = store.get_checkpoint(case_id)
    if ckpt is None:
        temps = list(config["initial_temps"])
        step_no = 0
        totals = {"cum_boundary": 0.0, "cum_source": 0.0, "cum_storage": 0.0,
                  "cum_residual": 0.0,
                  "denom": initial_enthalpy_abs_sum(grid, temps)}
    else:
        temps = list(ckpt["temps"])
        step_no = ckpt["step"]
        totals = {"cum_boundary": ckpt["cum_boundary"],
                  "cum_source": ckpt["cum_source"],
                  "cum_storage": ckpt["cum_storage"],
                  "cum_residual": ckpt["cum_residual"],
                  "denom": ckpt["denom"]}
        # Integrity: checkpoint must sit on the time grid.
        ref = max(1.0, abs(times[step_no]) if step_no < len(times) else 0.0)
        if step_no >= len(times) or abs(ckpt["time"] - times[step_no]) > 1e-9 * ref:
            store.set_status(case_id, "failed")
            raise StepFailure(
                f"case {case_id}: checkpoint inconsistent with time grid")

    if case["status"] == "new":
        store.set_status(case_id, "running")

    done = 0
    while step_no < len(times) - 1:
        if max_steps is not None and done >= max_steps:
            break
        t0, t1 = times[step_no], times[step_no + 1]
        try:
            temps, record = step(grid, config, temps, t0, t1)
        except StepFailure:
            store.set_status(case_id, "failed")
            raise
        totals["cum_boundary"] += record["boundary_in"]
        totals["cum_source"] += record["source_in"]
        totals["cum_storage"] += record["storage"]
        totals["cum_residual"] += record["residual"]
        totals["denom"] += record["denom_add"]
        if not check_balance(totals["cum_residual"], totals["denom"]):
            store.set_status(case_id, "failed")
            raise StepFailure(
                f"case {case_id}: energy balance residual "
                f"{totals['cum_residual']} exceeds tolerance at step "
                f"{step_no + 1}; step not committed")
        step_no += 1
        store.commit_step(case_id, step_no, t1, temps, record, totals)
        done += 1

    if step_no == len(times) - 1:
        store.set_status(case_id, "completed")
    return progress(store, case_id)


def progress(store, case_id):
    case = store.get_case(case_id)
    config = case["config"]
    ckpt = store.get_checkpoint(case_id)
    n_steps = len(build_time_grid(config)) - 1
    done = ckpt["step"] if ckpt else 0
    return {
        "case_id": case_id,
        "status": case["status"],
        "time": ckpt["time"] if ckpt else 0.0,
        "t_end": config["t_end"],
        "steps_done": done,
        "steps_total": n_steps,
        "cum_boundary": ckpt["cum_boundary"] if ckpt else 0.0,
        "cum_source": ckpt["cum_source"] if ckpt else 0.0,
        "cum_storage": ckpt["cum_storage"] if ckpt else 0.0,
        "cum_residual": ckpt["cum_residual"] if ckpt else 0.0,
    }
