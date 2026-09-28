import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layered_heat.errors import HeatError
from layered_heat.model import parse_case, parse_component
from layered_heat.server import serve
from layered_heat.solver import solve_step, time_grid
from layered_heat.store import Store
from layered_heat import engine


LAYERS = [
    {"thickness": 0.1, "k": 1.0, "cv": 1.0e6, "n": 4},
    {"thickness": 0.2, "k": 0.5, "cv": 2.0e6, "n": 4},
]


def make_case_spec(**over):
    spec = {
        "dt": 10.0,
        "t_end": 100.0,
        "initial": 20.0,
        "left": {"type": "temperature", "series": [[0, 100.0]]},
        "right": {"type": "flux", "series": [[0, 0.0]]},
        "sources": None,
    }
    spec.update(over)
    return spec


class Session:
    def __init__(self, tmp):
        self.dir = tempfile.TemporaryDirectory()
        self.store = Store(self.dir.name)
        self.tmp = tmp

    def close(self):
        self.store.close()
        self.dir.cleanup()

    def setup_case(self, name="c1", layers=None, **over):
        from layered_heat.model import parse_component

        layers = layers or LAYERS
        comp = parse_component("wall", layers)
        self.store.put_component("wall", {"layers": layers})
        spec = make_case_spec(**over)
        comp2 = parse_component("wall", layers)
        from layered_heat.model import parse_case as pc

        pc(name, comp2, spec)
        self.store.put_case(name, "wall", spec)
        return name


class TestValidation(unittest.TestCase):
    def test_bad_materials(self):
        for bad in [
            {"thickness": 0, "k": 1, "cv": 1, "n": 1},
            {"thickness": 0.1, "k": -1, "cv": 1, "n": 1},
            {"thickness": 0.1, "k": 1, "cv": float("nan"), "n": 1},
            {"thickness": 0.1, "k": 1, "cv": 1, "n": 0},
            {"thickness": 0.1, "k": 1, "cv": 1, "n": 1.5},
        ]:
            with self.assertRaises(HeatError):
                parse_component("x", [bad])

    def test_cell_limit(self):
        with self.assertRaises(HeatError):
            parse_component("x", [{"thickness": 1, "k": 1, "cv": 1, "n": 5001}])

    def test_initial_size(self):
        comp = parse_component("x", LAYERS)
        with self.assertRaises(HeatError):
            parse_case("c", comp, make_case_spec(initial=[1.0, 2.0]))

    def test_series_rules(self):
        comp = parse_component("x", LAYERS)
        with self.assertRaises(HeatError):  # not starting at 0
            parse_case("c", comp, make_case_spec(
                left={"type": "flux", "series": [[5, 1.0]]}))
        with self.assertRaises(HeatError):  # not increasing
            parse_case("c", comp, make_case_spec(
                left={"type": "flux", "series": [[0, 1.0], [0, 2.0]]}))
        with self.assertRaises(HeatError):  # non-finite
            parse_case("c", comp, make_case_spec(
                left={"type": "flux", "series": [[0, float("inf")]]}))


class TestSolver(unittest.TestCase):
    def setUp(self):
        self.comp = parse_component("w", LAYERS)
        self.layout = self.comp.cell_layout()

    def _case(self, **over):
        return parse_case("c", self.comp, make_case_spec(**over))

    def test_adiabatic_conservation(self):
        # both ends zero flux, no source: temperature and enthalpy stay constant
        case = self._case(
            left={"type": "flux", "series": [[0, 0.0]]},
            right={"type": "flux", "series": [[0, 0.0]]},
            initial=[10, 20, 30, 40, 50, 60, 70, 80],
        )
        temps = list(case.initial)
        e0 = sum(self.layout[2][i] * self.layout[0][i] * temps[i] for i in range(8))
        for t0, t1 in zip(time_grid(case)[:-1], time_grid(case)[1:]):
            temps, rec = solve_step(case, self.layout, t0, t1, temps)
            self.assertAlmostEqual(rec["boundary_in"], 0.0)
            self.assertAlmostEqual(rec["source_in"], 0.0)
        e1 = sum(self.layout[2][i] * self.layout[0][i] * temps[i] for i in range(8))
        self.assertAlmostEqual(e0, e1)

    def test_uniform_stays_uniform_adiabatic(self):
        case = self._case(
            left={"type": "flux", "series": [[0, 0.0]]},
            right={"type": "flux", "series": [[0, 0.0]]},
        )
        temps = [20.0] * 8
        temps, rec = solve_step(case, self.layout, 0.0, 10.0, temps)
        for t in temps:
            self.assertAlmostEqual(t, 20.0)

    def test_interface_resistance(self):
        # steady 1-cell-per-layer, fixed end temps: flux = dT / sum(R)
        comp = parse_component("w", [
            {"thickness": 0.1, "k": 1.0, "cv": 1.0, "n": 1},
            {"thickness": 0.2, "k": 0.5, "cv": 1.0, "n": 1},
        ])
        layout = comp.cell_layout()
        case = parse_case("c", comp, make_case_spec(
            dt=1e9, t_end=1e9,
            left={"type": "temperature", "series": [[0, 100.0]]},
            right={"type": "temperature", "series": [[0, 0.0]]},
            initial=0.0,
        ))
        temps, rec = solve_step(case, layout, 0.0, 1e9, [0.0, 0.0])
        r_total = 0.05 / 1.0 + 0.05 / 1.0 + 0.1 / 0.5 + 0.1 / 0.5
        q = 100.0 / r_total
        self.assertAlmostEqual(temps[0], 100.0 - q * (0.05 / 1.0), places=6)
        self.assertAlmostEqual(temps[1], q * (0.1 / 0.5), places=6)

    def test_energy_closure_per_step(self):
        case = self._case(sources=[1000.0, -500.0])
        temps = [20.0] * 8
        cum_r = 0.0
        scale = 0.0
        for t0, t1 in zip(time_grid(case)[:-1], time_grid(case)[1:]):
            temps, rec = solve_step(case, self.layout, t0, t1, temps)
            cum_r += rec["residual"]
            scale += rec["scale_step"]
        self.assertLessEqual(abs(cum_r), 1e-9 * max(1.0, scale))

    def test_change_point_split(self):
        case = self._case(
            dt=10.0, t_end=25.0,
            left={"type": "flux", "series": [[0, 5.0], [7.5, 50.0]]},
        )
        grid = time_grid(case)
        self.assertIn(7.5, grid)
        self.assertEqual(grid[-1], 25.0)
        # step starting at 7.5 uses the new value
        temps = [0.0] * 8
        i = grid.index(7.5)
        for j in range(i):
            temps, _ = solve_step(case, self.layout, grid[j], grid[j + 1], temps)
        before = list(temps)
        temps, rec = solve_step(case, self.layout, 7.5, grid[i + 1], temps)
        self.assertAlmostEqual(rec["boundary_left"], 50.0 * (grid[i + 1] - 7.5))

    def test_final_step_truncation(self):
        case = self._case(dt=10.0, t_end=25.0)
        grid = time_grid(case)
        self.assertEqual(grid[-1], 25.0)
        self.assertAlmostEqual(grid[-1] - grid[-2], 5.0)


class TestEngine(unittest.TestCase):
    def test_run_pause_resume_export(self):
        sess = Session(self)
        self.addCleanup(sess.close)
        name = sess.setup_case(sources=[100.0, 0.0])
        p = engine.run_case(sess.store, name, max_steps=3)
        self.assertEqual(p["status"], "paused")
        self.assertEqual(p["steps_done"], 3)
        t_paused = engine.export_temperature(sess.store, name)
        self.assertFalse(t_paused["complete"])
        p = engine.run_case(sess.store, name)
        self.assertEqual(p["status"], "done")
        self.assertEqual(p["steps_done"], p["steps_total"])
        # resume after done is a no-op, committed steps not recomputed
        p2 = engine.run_case(sess.store, name)
        self.assertEqual(p2["steps_done"], p["steps_done"])
        e = engine.export_energy(sess.store, name)
        self.assertTrue(e["complete"])
        cum = e["cumulative"]
        self.assertAlmostEqual(
            cum["boundary_in"] + cum["source_in"] - cum["storage_change"],
            cum["residual"],
        )
        self.assertLessEqual(abs(cum["residual"]), 1e-9 * max(1.0, cum["scale"]))
        self.assertEqual(len(e["steps"]), p["steps_total"])

    def test_resume_matches_single_run(self):
        s1 = Session(self)
        s2 = Session(self)
        self.addCleanup(s1.close)
        self.addCleanup(s2.close)
        s1.setup_case()
        s2.setup_case()
        engine.run_case(s1.store, "c1")
        engine.run_case(s2.store, "c1", max_steps=4)
        engine.run_case(s2.store, "c1")
        t1 = engine.export_temperature(s1.store, "c1")["temperature"]
        t2 = engine.export_temperature(s2.store, "c1")["temperature"]
        self.assertEqual(t1, t2)

    def test_failed_step_not_committed(self):
        sess = Session(self)
        self.addCleanup(sess.close)
        sess.setup_case(left={"type": "temperature", "series": [[0, 1e308]]},
                        dt=1e308, t_end=1e308)
        with self.assertRaises(HeatError):
            engine.run_case(sess.store, "c1")
        row = sess.store.get_case_row("c1")
        self.assertEqual(row["status"], "failed")
        self.assertEqual(sess.store.get_steps("c1"), [])

    def test_incomplete_not_exported_as_complete(self):
        sess = Session(self)
        self.addCleanup(sess.close)
        sess.setup_case()
        engine.run_case(sess.store, "c1", max_steps=2)
        out = engine.export_energy(sess.store, "c1")
        self.assertFalse(out["complete"])
        self.assertEqual(out["status"], "paused")


class TestJsonlCli(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            reqs = [
                {"id": 1, "cmd": "import_component", "name": "wall", "layers": LAYERS},
                {"id": 2, "cmd": "create_case", "name": "c1", "component": "wall",
                 **make_case_spec(sources=[10.0, 0.0])},
                {"id": 3, "cmd": "start", "case": "c1", "max_steps": 2},
                {"id": 4, "cmd": "progress", "case": "c1"},
                {"id": 5, "cmd": "resume", "case": "c1"},
                {"id": 6, "cmd": "export_temperature", "case": "c1"},
                {"id": 7, "cmd": "export_energy", "case": "c1"},
                {"id": 8, "cmd": "create_case", "name": "c1", "component": "wall",
                 **make_case_spec()},
            ]
            proc = subprocess.run(
                [sys.executable, "-m", "layered_heat", "--state", d],
                input="\n".join(json.dumps(r) for r in reqs),
                capture_output=True, text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            resps = [json.loads(line) for line in proc.stdout.strip().splitlines()]
            self.assertEqual(len(resps), 8)
            by_id = {r["id"]: r for r in resps}
            self.assertTrue(by_id[1]["ok"])
            self.assertEqual(by_id[1]["result"]["cells"], 8)
            self.assertTrue(by_id[3]["result"]["status"] == "paused")
            self.assertEqual(by_id[4]["result"]["steps_done"], 2)
            self.assertEqual(by_id[5]["result"]["status"], "done")
            self.assertTrue(by_id[6]["result"]["complete"])
            self.assertEqual(len(by_id[6]["result"]["temperature"]), 8)
            self.assertTrue(by_id[7]["result"]["complete"])
            self.assertFalse(by_id[8]["ok"])  # duplicate case name rejected


if __name__ == "__main__":
    unittest.main()
