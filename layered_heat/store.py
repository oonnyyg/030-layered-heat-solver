"""SQLite persistence: components, cases, checkpoints, committed steps."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS components (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component_id INTEGER NOT NULL REFERENCES components(id),
    config TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'new'
);
CREATE TABLE IF NOT EXISTS checkpoints (
    case_id INTEGER PRIMARY KEY REFERENCES cases(id),
    step INTEGER NOT NULL,
    time REAL NOT NULL,
    temps TEXT NOT NULL,
    cum_boundary REAL NOT NULL,
    cum_source REAL NOT NULL,
    cum_storage REAL NOT NULL,
    cum_residual REAL NOT NULL,
    denom REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    case_id INTEGER NOT NULL REFERENCES cases(id),
    step INTEGER NOT NULL,
    t0 REAL NOT NULL,
    t1 REAL NOT NULL,
    dt REAL NOT NULL,
    q_left REAL NOT NULL,
    q_right REAL NOT NULL,
    boundary_in REAL NOT NULL,
    source_in REAL NOT NULL,
    storage REAL NOT NULL,
    residual REAL NOT NULL,
    temps TEXT NOT NULL,
    PRIMARY KEY (case_id, step)
);
"""


class Store:
    def __init__(self, state_dir):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.dir / "layered_heat.db")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    # --- components / cases -------------------------------------------------
    def add_component(self, spec):
        cur = self.db.execute("INSERT INTO components(spec) VALUES (?)",
                              (json.dumps(spec),))
        self.db.commit()
        return cur.lastrowid

    def get_component(self, cid):
        row = self.db.execute("SELECT spec FROM components WHERE id=?",
                              (cid,)).fetchone()
        if row is None:
            raise KeyError(f"component {cid} not found")
        return json.loads(row[0])

    def add_case(self, component_id, config):
        cur = self.db.execute(
            "INSERT INTO cases(component_id, config) VALUES (?, ?)",
            (component_id, json.dumps(config)))
        self.db.commit()
        return cur.lastrowid

    def get_case(self, case_id):
        row = self.db.execute(
            "SELECT component_id, config, status FROM cases WHERE id=?",
            (case_id,)).fetchone()
        if row is None:
            raise KeyError(f"case {case_id} not found")
        return {"component_id": row[0], "config": json.loads(row[1]),
                "status": row[2]}

    def set_status(self, case_id, status):
        self.db.execute("UPDATE cases SET status=? WHERE id=?",
                        (status, case_id))
        self.db.commit()

    # --- checkpoint / steps ---------------------------------------------------
    def get_checkpoint(self, case_id):
        row = self.db.execute(
            "SELECT step, time, temps, cum_boundary, cum_source, cum_storage,"
            " cum_residual, denom FROM checkpoints WHERE case_id=?",
            (case_id,)).fetchone()
        if row is None:
            return None
        return {"step": row[0], "time": row[1], "temps": json.loads(row[2]),
                "cum_boundary": row[3], "cum_source": row[4],
                "cum_storage": row[5], "cum_residual": row[6], "denom": row[7]}

    def commit_step(self, case_id, step_no, time, temps, record, totals):
        """Atomically record a committed step and its checkpoint."""
        with self.db:
            self.db.execute(
                "INSERT INTO steps(case_id, step, t0, t1, dt, q_left, q_right,"
                " boundary_in, source_in, storage, residual, temps)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (case_id, step_no, record["t0"], record["t1"], record["dt"],
                 record["q_left"], record["q_right"], record["boundary_in"],
                 record["source_in"], record["storage"], record["residual"],
                 json.dumps(temps)))
            self.db.execute(
                "INSERT INTO checkpoints(case_id, step, time, temps,"
                " cum_boundary, cum_source, cum_storage, cum_residual, denom)"
                " VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(case_id) DO UPDATE SET"
                " step=excluded.step, time=excluded.time, temps=excluded.temps,"
                " cum_boundary=excluded.cum_boundary,"
                " cum_source=excluded.cum_source,"
                " cum_storage=excluded.cum_storage,"
                " cum_residual=excluded.cum_residual, denom=excluded.denom",
                (case_id, step_no, time, json.dumps(temps),
                 totals["cum_boundary"], totals["cum_source"],
                 totals["cum_storage"], totals["cum_residual"],
                 totals["denom"]))

    def get_steps(self, case_id):
        rows = self.db.execute(
            "SELECT step, t0, t1, dt, q_left, q_right, boundary_in, source_in,"
            " storage, residual, temps FROM steps WHERE case_id=? ORDER BY step",
            (case_id,)).fetchall()
        return [{"step": r[0], "t0": r[1], "t1": r[2], "dt": r[3],
                 "q_left": r[4], "q_right": r[5], "boundary_in": r[6],
                 "source_in": r[7], "storage": r[8], "residual": r[9],
                 "temps": json.loads(r[10])} for r in rows]
