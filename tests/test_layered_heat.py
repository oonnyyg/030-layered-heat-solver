import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layered_heat.model import (ValidationError, build_grid, build_time_grid,
                                validate_case, validate_component)
from layered_heat.runner import run_case
from layered_heat.store import Store


def make_component(layers=None):
    return validate_component({"layers": layers or [
        {"thickness": 0.1, "k": 1.0, "cv": 1e6, "n": 4},
        {"thickness": 0.2, "k": 0.5, "cv": 2e6, "n": 4},
    ]})


def base_case(**kw):
    case = {
        "t_end": 100.0,
        "dt": 10.0,
        "initial_temp": 20.0,
        "boundaries": {
            "left": {"type": "temperature", "points": [[0.0, 100.0]]},
            "right": {"type": "flux", "points": [[0.0, 0.0]]},
        },
    }
    case.update(kw)
    return case


class TestValidation(unittest.TestCase):
    def test_rejects_bad_materials(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), "x"):
            with self.assertRaises(ValidationError):
                validate_component({"layers": [
                    {"thickness": 0.1, "k": bad, "cv": 1.0, "n": 2}]})
            with self.assertRaises(ValidationError):
                validate_component({"layers": [
                    {"thickness": bad, "k": 1.0, "cv": 1.0, "n": 2}]})
            with self.assertRaises(ValidationError):
                validate_component({"layers": [
                    {"thickness": 0.1, "k": 1.0, "cv": bad, "n": 2}]})

    def test_rejects_bad_cell_count(self):
        with self.assertRaises(ValidationError):
            validate_component({"layers": [
                {"thickness": 0.1, "k": 1.0, "cv": 1.0, "n": 0}]})
        with self.assertRaises(ValidationError):
            validate_component({"layers": [
                {"thickness": 1.0, "k": 1.0, "cv": 1.0, "n": 5001}]})

    def test_rejects_initial_temp_size(self):
        comp = make_component()
        with self.assertRaises(ValidationError):
            validate_case(base_case(initial_temp=[1.0, 2.0]), comp)
        with self.assertRaises(ValidationError):
            validate_case(base_case(initial_temp=float("nan")), comp)

    def test_rejects_non_monotonic_points(self):
        comp = make_component()
        with self.assertRaises(ValidationError):
            validate_case(base_case(boundaries={
                "left": {"type": "flux", "points": [[0.0, 1.0], [0.0, 2.0]]},
                "right": {"type": "flux", "points": [[0.0, 0.0]]},
            }), comp)


class TestPhysics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _setup(self, component, case):
        comp = validate_component({"layers": component})
        cid = self.store.add_component(comp)
        cfg = validate_case(case, comp)
        case_id = self.store.add_case(cid, cfg)
        return comp, cfg, case_id

    def test_adiabatic_source_uniform_heating(self):
        # Zero-flux both ends + uniform source: T rises exactly q*t/cv.
        comp, cfg, case_id = self._setup(
            [{"thickness": 0.1, "k": 1.0, "cv": 2e6, "n": 5}],
            base_case(t_end=50.0, dt=10.0, initial_temp=10.0,
                      boundaries={
                          "left": {"type": "flux", "points": [[0.0, 0.0]]},
                          "right": {"type": "flux", "points": [[0.0, 0.0]]}},
                      sources=[[[0.0, 1000.0]]]))
        prog = run_case(self.store, case_id)
        self.assertEqual(prog["status"], "completed")
        temps = self.store.get_steps(case_id)[-1]["temps"]
        expect = 10.0 + 1000.0 * 50.0 / 2e6
        for t in temps:
            self.assertAlmostEqual(t, expect, places=9)
        # Energy: source_in == storage, boundary zero.
        self.assertAlmostEqual(prog["cum_boundary"], 0.0, places=6)
        self.assertAlmostEqual(prog["cum_source"], prog["cum_storage"], places=6)
        self.assertLessEqual(abs(prog["cum_residual"]), 1e-9)

    def test_steady_state_series_resistance(self):
        # Two layers, fixed temps both ends, long time -> linear steady state.
        layers = [{"thickness": 0.1, "k": 1.0, "cv": 1.0, "n": 2},
                  {"thickness": 0.2, "k": 0.5, "cv": 1.0, "n": 2}]
        comp, cfg, case_id = self._setup(
            layers,
            base_case(t_end=1e7, dt=1e6, initial_temp=0.0,
                      boundaries={
                          "left": {"type": "temperature", "points": [[0.0, 100.0]]},
                          "right": {"type": "temperature", "points": [[0.0, 0.0]]}}))
        run_case(self.store, case_id)
        temps = self.store.get_steps(case_id)[-1]["temps"]
        r_total = 0.1 / 1.0 + 0.2 / 0.5  # R = L/k per layer, area 1
        q = 100.0 / r_total
        # Cell centers: left layer cells at x=0.025, 0.075; right at 0.15, 0.25.
        expect = [100.0 - q * x for x in (0.025, 0.075)]
        expect += [100.0 - q * (0.1 + (x - 0.1) * 1.0 / 0.5) for x in (0.15, 0.25)]
        for got, want in zip(temps, expect):
            self.assertAlmostEqual(got, want, places=6)
        # Steady state: boundary in == boundary out, storage ~ 0 on last step.
        last = self.store.get_steps(case_id)[-1]
        self.assertAlmostEqual(last["q_left"], -last["q_right"], places=6)

    def test_change_point_splits_steps_and_applies_new_value(self):
        comp, cfg, case_id = self._setup(
            [{"thickness": 0.1, "k": 1.0, "cv": 1e6, "n": 2}],
            base_case(t_end=30.0, dt=10.0,
                      boundaries={
                          "left": {"type": "flux",
                                   "points": [[0.0, 0.0], [15.0, 500.0]]},
                          "right": {"type": "flux", "points": [[0.0, 0.0]]}}))
        times = build_time_grid(cfg)
        self.assertIn(15.0, times)
        prog = run_case(self.store, case_id)
        steps = self.store.get_steps(case_id)
        # Step starting at 15 uses the new flux; step ending at 15 uses old.
        s_end = next(s for s in steps if s["t1"] == 15.0)
        s_start = next(s for s in steps if s["t0"] == 15.0)
        self.assertEqual(s_end["q_left"], 0.0)
        self.assertEqual(s_start["q_left"], 500.0)
        # Total stored energy equals 500 W/m^2 applied for 15 s over 1 m^2.
        cap = 1e6 * 0.1  # cv * volume, area = 1
        mean_t = sum(steps[-1]["temps"]) / len(steps[-1]["temps"])
        self.assertAlmostEqual(mean_t, 20.0 + 500.0 * 15.0 / cap, places=9)
        self.assertAlmostEqual(prog["cum_boundary"], 500.0 * 15.0, places=6)

    def test_final_step_truncated(self):
        comp, cfg, case_id = self._setup(
            [{"thickness": 0.1, "k": 1.0, "cv": 1e6, "n": 2}],
            base_case(t_end=25.0, dt=10.0))
        prog = run_case(self.store, case_id)
        self.assertEqual(prog["status"], "completed")
        steps = self.store.get_steps(case_id)
        self.assertEqual(steps[-1]["t1"], 25.0)
        self.assertAlmostEqual(steps[-1]["dt"], 5.0)
        self.assertAlmostEqual(prog["time"], 25.0)

    def test_resume_matches_uninterrupted(self):
        comp = make_component()
        cfg = validate_case (base_case(
            boundaries={
                "left": {"type": "temperature",
                         "points": [[0.0, 100.0], [35.0, 50.0]]},
                "right": {"type": "flux", "points": [[0.0, 20.0]]}},
            sources=[[[0.0, 10.0], [12.0, 0.0]], [[0.0, 5.0]]]), comp)
        cid = self.store.add_component(comp)
        full = self.store.add_case(cid, cfg)
        run_case(self.store, full)
        part = self.store.add_case(cid, cfg)
        run_case(self.store, part, max_steps=3)
        mid = run_case(self.store, part, max_steps=2)
        self.assertEqual(mid["status"], "running")
        done = run_case(self.store, part)
        self.assertEqual(done["status"], "completed")
        full_steps = self.store.get_steps(full)
        part_steps = self.store.get_steps(part)
        self.assertEqual(len(full_steps), len(part_steps))
        for a, b in zip(full_steps, part_steps):
            self.assertEqual(a["t1"], b["t1"])
            self.assertEqual(a["temps"], b["temps"])
            self.assertEqual(a["boundary_in"], b["boundary_in"])
        # Committed steps not recomputed: step count equals total, not more.
        self.assertEqual(len(part_steps), done["steps_total"])

    def test_export_incomplete_not_marked_complete(self):
        from layered_heat.cli import _export_energy, _export_temperatures
        comp = make_component()
        cid = self.store.add_component(comp)
        cfg = validate_case(base_case(), comp)
        case_id = self.store.add_case(cid, cfg)
        run_case(self.store, case_id, max_steps=2)
        out = _export_temperatures(self.store, case_id)
        self.assertFalse(out["completed"])
        self.assertEqual(len(out["times"]), 3)  # t=0 plus 2 committed steps
        self.assertLess(out["exported_through"], 100.0)
        en = _export_energy(self.store, case_id)
        self.assertFalse(en["completed"])
        self.assertEqual(len(en["steps"]), 2)

    def test_energy_balance_tolerance(self):
        comp = make_component()
        cid = self.store.add_component(comp)
        cfg = validate_case(base_case(
            t_end=1000.0, dt=7.0,
            boundaries={
                "left": {"type": "temperature",
                         "points": [[0.0, 100.0], [300.0, 20.0]]},
                "right": {"type": "flux", "points": [[0.0, 5.0]]}},
            sources=[[[0.0, 3.0]], [[0.0, 0.0], [500.0, 8.0]]]), comp)
        case_id = self.store.add_case(cid, cfg)
        prog = run_case(self.store, case_id)
        self.assertEqual(prog["status"], "completed")
        denom = max(1.0, abs(prog["cum_boundary"]) + abs(prog["cum_source"]))
        self.assertLessEqual(abs(prog["cum_residual"]), 1e-9 * denom)
        self.assertAlmostEqual(
            prog["cum_storage"],
            prog["cum_boundary"] + prog["cum_source"] - prog["cum_residual"],
            places=6)


class TestCli(unittest.TestCase):
    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            reqs = [
                {"cmd": "import_component", "component": {"layers": [
                    {"thickness": 0.1, "k": 1.0, "cv": 1e6, "n": 3}]}},
                {"cmd": "create_case", "component_id": 1, "case": {
                    "t_end": 20.0, "dt": 10.0, "initial_temp": 0.0,
                    "boundaries": {
                        "left": {"type": "temperature", "points": [[0.0, 50.0]]},
                        "right": {"type": "flux", "points": [[0.0, 0.0]]}}}},
                {"cmd": "start", "case_id": 1},
                {"cmd": "progress", "case_id": 1},
                {"cmd": "export_temperatures", "case_id": 1},
                {"cmd": "export_energy", "case_id": 1},
            ]
            proc = subprocess.run(
                [sys.executable, "-m", "layered_heat", "--state", tmp],
                input="\n".join(json.dumps(r) for r in reqs),
                capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[1])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outs = [json.loads(l) for l in proc.stdout.strip().splitlines()]
            self.assertTrue(all(o["ok"] for o in outs), outs)
            self.assertEqual(outs[3]["progress"]["status"], "completed")
            self.assertTrue(outs[4]["completed"])
            self.assertEqual(len(outs[4]["times"]), 3)
            self.assertEqual(len(outs[5]["steps"]), 2)
            # Resume a completed case is a no-op.
            proc2 = subprocess.run(
                [sys.executable, "-m", "layered_heat", "--state", tmp],
                input=json.dumps({"cmd": "resume", "case_id": 1}),
                capture_output=True, text=True,
                cwd=Path(__file__).resolve().parents[1])
            out2 = json.loads(proc2.stdout.strip())
            self.assertEqual(out2["progress"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
