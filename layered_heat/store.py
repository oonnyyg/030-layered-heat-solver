"""SQLite-backed state: components, cases, committed steps, checkpoints."""

from __future__ import annotations

import json
import os
import sqlite3

from .errors import HeatError

SCHEMA = """
CREATE TABLE IF NOT EXISTS components (
    name TEXT PRIMARY KEY,
    spec TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
    name TEXT PRIMARY KEY,
    component TEXT NOT NULL,
    spec TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS checkpoints (
    case_name TEXT PRIMARY KEY,
    step INTEGER NOT NULL,
    time REAL NOT NULL,
    temps TEXT NOT NULL,
    cum_boundary REAL NOT NULL,
    cum_source REAL NOT NULL,
    cum_storage REAL NOT NULL,
    cum_residual REAL NOT NULL,
    scale REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    case_name TEXT NOT NULL,
    idx INTEGER NOT NULL,
    t0 REAL NOT NULL,
    t1 REAL NOT NULL,
    dt REAL NOT NULL,
    boundary_in REAL NOT NULL,
    boundary_left REAL NOT NULL,
    boundary_right REAL NOT NULL,
    source_in REAL NOT NULL,
    storage REAL NOT NULL,
    residual REAL NOT NULL,
    cum_boundary REAL NOT NULL,
    cum_source REAL NOT NULL,
    cum_storage REAL NOT NULL,
    cum_residual REAL NOT NULL,
    PRIMARY KEY (case_name, idx)
);
"""


class Store:
    def __init__(self, state_dir):
        os.makedirs(state_dir, exist_ok=True)
        self.db = sqlite3.connect(os.path.join(state_dir, "layered_heat.db"))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    # components
    def put_component(self, name, spec):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO components(name, spec) VALUES (?, ?)",
                (name, json.dumps(spec)),
            )

    def get_component_spec(self, name):
        row = self.db.execute(
            "SELECT spec FROM components WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise HeatError(f"unknown component '{name}'")
        return json.loads(row["spec"])

    def list_components(self):
        return [
            r["name"]
            for r in self.db.execute("SELECT name FROM components ORDER BY name")
        ]

    # cases
    def put_case(self, name, component, spec, status="created", error=""):
        with self.db:
            self.db.execute(
                "INSERT INTO cases(name, component, spec, status, error)"
                " VALUES (?, ?, ?, ?, ?)",
                (name, component, json.dumps(spec), status, error),
            )

    def get_case_row(self, name):
        row = self.db.execute(
            "SELECT * FROM cases WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise HeatError(f"unknown case '{name}'")
        return row

    def case_exists(self, name):
        return (
            self.db.execute(
                "SELECT 1 FROM cases WHERE name = ?", (name,)
            ).fetchone()
            is not None
        )

    def set_status(self, name, status, error=""):
        with self.db:
            self.db.execute(
                "UPDATE cases SET status = ?, error = ? WHERE name = ?",
                (status, error, name),
            )

    def list_cases(self):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT name, component, status, error FROM cases ORDER BY name"
            )
        ]

    # checkpoint + steps (committed atomically)
    def commit_step(self, case_name, idx, time, temps, record, cum):
        with self.db:
            self.db.execute(
                "INSERT INTO steps(case_name, idx, t0, t1, dt, boundary_in,"
                " boundary_left, boundary_right, source_in, storage, residual,"
                " cum_boundary, cum_source, cum_storage, cum_residual)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    case_name,
                    idx,
                    record["t0"],
                    record["t1"],
                    record["dt"],
                    record["boundary_in"],
                    record["boundary_left"],
                    record["boundary_right"],
                    record["source_in"],
                    record["storage"],
                    record["residual"],
                    cum["boundary"],
                    cum["source"],
                    cum["storage"],
                    cum["residual"],
                ),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO checkpoints(case_name, step, time, temps,"
                " cum_boundary, cum_source, cum_storage, cum_residual, scale)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    case_name,
                    idx,
                    time,
                    json.dumps(temps),
                    cum["boundary"],
                    cum["source"],
                    cum["storage"],
                    cum["residual"],
                    cum["scale"],
                ),
            )

    def init_checkpoint(self, case_name, temps, scale0):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO checkpoints(case_name, step, time, temps,"
                " cum_boundary, cum_source, cum_storage, cum_residual, scale)"
                " VALUES (?, 0, 0.0, ?, 0.0, 0.0, 0.0, 0.0, ?)",
                (case_name, json.dumps(temps), scale0),
            )

    def get_checkpoint(self, case_name):
        row = self.db.execute(
            "SELECT * FROM checkpoints WHERE case_name = ?", (case_name,)
        ).fetchone()
        if row is None:
            return None
        return {
            "step": row["step"],
            "time": row["time"],
            "temps": json.loads(row["temps"]),
            "cum": {
                "boundary": row["cum_boundary"],
                "source": row["cum_source"],
                "storage": row["cum_storage"],
                "residual": row["cum_residual"],
                "scale": row["scale"],
            },
        }

    def get_steps(self, case_name):
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM steps WHERE case_name = ? ORDER BY idx", (case_name,)
            )
        ]

    def reset_run(self, case_name):
        with self.db:
            self.db.execute("DELETE FROM steps WHERE case_name = ?", (case_name,))
            self.db.execute("DELETE FROM checkpoints WHERE case_name = ?", (case_name,))
