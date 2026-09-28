"""JSONL request/response server over stdin/stdout."""

from __future__ import annotations

import json
import sys

from . import engine
from .errors import HeatError
from .model import parse_case, parse_component
from .store import Store


def _cmd_import_component(store, req):
    name = req.get("name")
    layers = req.get("layers")
    comp = parse_component(name, layers)
    store.put_component(comp.name, {"layers": layers})
    return {"component": comp.name, "cells": comp.ncells}


def _cmd_create_case(store, req):
    name = req.get("name")
    if store.case_exists(name):
        raise HeatError(f"case '{name}' already exists; create a new case to change configuration")
    comp_name = req.get("component")
    comp_spec = store.get_component_spec(comp_name)
    comp = parse_component(comp_name, comp_spec["layers"])
    spec = {
        "dt": req.get("dt"),
        "t_end": req.get("t_end"),
        "initial": req.get("initial"),
        "left": req.get("left"),
        "right": req.get("right"),
        "sources": req.get("sources"),
    }
    case = parse_case(name, comp, spec)
    store.put_case(case.name, comp_name, spec)
    return {"case": case.name, "cells": comp.ncells}


def _cmd_start(store, req):
    return engine.run_case(store, req.get("case"), req.get("max_steps"))


def _cmd_resume(store, req):
    return engine.run_case(store, req.get("case"), req.get("max_steps"))


def _cmd_progress(store, req):
    return engine.progress(store, req.get("case"))


def _cmd_export_temperature(store, req):
    return engine.export_temperature(store, req.get("case"))


def _cmd_export_energy(store, req):
    return engine.export_energy(store, req.get("case"))


def _cmd_list_components(store, req):
    return {"components": store.list_components()}


def _cmd_list_cases(store, req):
    return {"cases": store.list_cases()}


COMMANDS = {
    "import_component": _cmd_import_component,
    "create_case": _cmd_create_case,
    "start": _cmd_start,
    "resume": _cmd_resume,
    "progress": _cmd_progress,
    "export_temperature": _cmd_export_temperature,
    "export_energy": _cmd_export_energy,
    "list_components": _cmd_list_components,
    "list_cases": _cmd_list_cases,
}


def handle(store, req):
    cmd = req.get("cmd")
    handler = COMMANDS.get(cmd)
    if handler is None:
        raise HeatError(f"unknown cmd '{cmd}'")
    return handler(store, req)


def serve(state_dir, instream=None, outstream=None):
    instream = instream or sys.stdin
    outstream = outstream or sys.stdout
    store = Store(state_dir)
    try:
        for line in instream:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise HeatError("request must be a JSON object")
                result = handle(store, req)
                resp = {"id": req.get("id"), "ok": True, "result": result}
            except HeatError as exc:
                resp = {"id": None, "ok": False, "error": str(exc)}
                try:
                    resp["id"] = json.loads(line).get("id")
                except Exception:
                    pass
            except Exception as exc:  # malformed JSON etc.
                resp = {"id": None, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            outstream.write(json.dumps(resp) + "\n")
            outstream.flush()
    finally:
        store.close()
