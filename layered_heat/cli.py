"""JSONL command interface: one JSON object per line on stdin/stdout."""
from __future__ import annotations

import json
import sys

from .model import ValidationError, validate_case, validate_component
from .runner import progress, run_case
from .solver import StepFailure
from .store import Store


def _export_temperatures(store, case_id):
    case = store.get_case(case_id)
    steps = store.get_steps(case_id)
    ckpt = store.get_checkpoint(case_id)
    completed = case["status"] == "completed"
    times = [0.0] + [s["t1"] for s in steps]
    temps = [case["config"]["initial_temps"]] + [s["temps"] for s in steps]
    return {
        "case_id": case_id,
        "status": case["status"],
        "completed": completed,
        "exported_through": ckpt["time"] if ckpt else 0.0,
        "times": times,
        "temperatures": temps,
    }


def _export_energy(store, case_id):
    case = store.get_case(case_id)
    steps = store.get_steps(case_id)
    ckpt = store.get_checkpoint(case_id)
    return {
        "case_id": case_id,
        "status": case["status"],
        "completed": case["status"] == "completed",
        "exported_through": ckpt["time"] if ckpt else 0.0,
        "steps": [
            {"step": s["step"], "t0": s["t0"], "t1": s["t1"], "dt": s["dt"],
             "q_left": s["q_left"], "q_right": s["q_right"],
             "boundary_in": s["boundary_in"], "source_in": s["source_in"],
             "storage": s["storage"], "residual": s["residual"]}
            for s in steps
        ],
        "cumulative": {
            "boundary_in": ckpt["cum_boundary"] if ckpt else 0.0,
            "source_in": ckpt["cum_source"] if ckpt else 0.0,
            "storage": ckpt["cum_storage"] if ckpt else 0.0,
            "residual": ckpt["cum_residual"] if ckpt else 0.0,
        },
    }


def handle(store, req):
    cmd = req.get("cmd")
    if cmd == "import_component":
        spec = validate_component(req.get("component"))
        cid = store.add_component(spec)
        return {"ok": True, "component_id": cid}
    if cmd == "create_case":
        cid = req.get("component_id")
        component = store.get_component(cid)
        config = validate_case(req.get("case"), component)
        case_id = store.add_case(cid, config)
        return {"ok": True, "case_id": case_id}
    if cmd == "start":
        return {"ok": True, "progress": run_case(
            store, req.get("case_id"), req.get("max_steps"))}
    if cmd == "resume":
        case_id = req.get("case_id")
        case = store.get_case(case_id)
        if case["status"] == "new":
            raise ValidationError("case has not been started; use 'start'")
        return {"ok": True, "progress": run_case(
            store, case_id, req.get("max_steps"))}
    if cmd == "progress":
        return {"ok": True, "progress": progress(store, req.get("case_id"))}
    if cmd == "export_temperatures":
        return {"ok": True, **_export_temperatures(store, req.get("case_id"))}
    if cmd == "export_energy":
        return {"ok": True, **_export_energy(store, req.get("case_id"))}
    raise ValidationError(f"unknown cmd {cmd!r}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    state_dir = "."
    if "--state" in argv:
        i = argv.index("--state")
        state_dir = argv[i + 1]
        del argv[i:i + 2]
    if argv:
        print(json.dumps({"ok": False, "error": f"unexpected args: {argv}"}),
              file=sys.stderr)
        return 2
    store = Store(state_dir)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = handle(store, req)
            except (ValidationError, StepFailure, KeyError) as exc:
                resp = {"ok": False, "error": str(exc)}
            except Exception as exc:  # defensive: never crash the protocol loop
                resp = {"ok": False, "error": f"internal: {exc}"}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    finally:
        store.close()
    return 0
