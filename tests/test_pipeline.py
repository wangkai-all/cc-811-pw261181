"""端到端管线测试：场景、锁定、极端情形、验证、确定性。"""

import unittest

from countyrate.data import generate_dataset, default_locked_ids
from countyrate.pipeline import compute_estimates, compute_validation


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset()
        cls.locked = default_locked_ids(cls.dataset)

    def test_dataset_has_required_edge_cases(self):
        counties = self.dataset["counties"]
        zero = [c for c in counties if c["n_sample"] == 0]
        tiny = [c for c in counties if 0 < c["n_sample"] <= 3]
        confirmed = [c for c in counties if c["confirmed_count"] is not None]
        self.assertGreaterEqual(len(zero), 2, "需要零样本县")
        self.assertGreaterEqual(len(tiny), 1, "需要小样本县")
        self.assertGreaterEqual(len(confirmed), 4, "需要已确认县")
        # 人口量级差异
        pops = [c["population"] for c in counties]
        self.assertGreater(max(pops) / min(pops), 20)

    def test_baseline_feasible_and_total_matched(self):
        res = compute_estimates(self.dataset, "baseline", self.locked)
        self.assertTrue(res["calibration"]["feasible"])
        T = res["scenario"]["city_total"]
        self.assertAlmostEqual(res["calibration"]["achieved_total"], T, places=4)

    def test_conflict_scenario_reports_infeasible(self):
        res = compute_estimates(self.dataset, "conflict", self.locked)
        cal = res["calibration"]
        self.assertFalse(cal["feasible"])
        self.assertEqual(cal["reason"], "total_below_locked")
        self.assertIn("不可改动", cal["message"])
        # 不可行时不得输出校准后率（避免误用）
        for c in res["counties"]:
            self.assertIsNone(c["cal_rate"])

    def test_conflict_resolved_by_unlocking(self):
        # 解锁全部后，冲突场景在数学上变得可行（0 ≤ T ≤ 总容量）
        res = compute_estimates(self.dataset, "conflict", [])
        self.assertTrue(res["calibration"]["feasible"])

    def test_locked_county_equals_confirmed_rate(self):
        res = compute_estimates(self.dataset, "baseline", self.locked)
        by_id = {c["id"]: c for c in self.dataset["counties"]}
        for row in res["counties"]:
            if row["id"] in self.locked:
                expect = by_id[row["id"]]["confirmed_count"] / row["population"]
                self.assertAlmostEqual(row["cal_rate"], expect, places=12)

    def test_zero_sample_county_is_extrapolated(self):
        res = compute_estimates(self.dataset, "baseline", self.locked)
        zero_rows = [r for r in res["counties"] if r["n_sample"] == 0]
        self.assertTrue(zero_rows)
        for r in zero_rows:
            self.assertTrue(r["extrapolated"])
            self.assertEqual(r["shrink_weight_B"], 1.0)
            self.assertIsNone(r["raw_rate"])

    def test_tiny_sample_county_shrinks_hard(self):
        res = compute_estimates(self.dataset, "baseline", self.locked)
        tiny = [r for r in res["counties"] if 0 < r["n_sample"] <= 3]
        self.assertTrue(tiny)
        for r in tiny:
            self.assertGreater(r["shrink_weight_B"], 0.5,
                               "小样本县应大幅收缩向模型")

    def test_shrunk_intervals_within_unit_interval(self):
        res = compute_estimates(self.dataset, "baseline", self.locked)
        for r in res["counties"]:
            self.assertGreater(r["shrunk_lo"], 0.0)
            self.assertLess(r["shrunk_hi"], 1.0)
            self.assertGreaterEqual(r["cal_hi"], r["cal_lo"])

    def test_low_scenario_triggers_bounds(self):
        res = compute_estimates(self.dataset, "low", self.locked)
        self.assertTrue(res["calibration"]["feasible"])
        clamped = [r for r in res["counties"] if r["clamped"] is not None]
        self.assertTrue(clamped, "偏低总量应触发至少一个边界约束")

    def test_locking_unconfirmed_raises(self):
        unconfirmed = next(c for c in self.dataset["counties"]
                           if c["confirmed_count"] is None)
        with self.assertRaises(ValueError):
            compute_estimates(self.dataset, "baseline", [unconfirmed["id"]])

    def test_determinism(self):
        a = compute_estimates(self.dataset, "baseline", self.locked)
        b = compute_estimates(self.dataset, "baseline", self.locked)
        self.assertEqual(a, b)

    def test_validation_runs_and_sane(self):
        val = compute_validation(self.dataset)
        self.assertEqual(len(val["folds"]), 5)
        ov = val["overall"]
        for k in ("coverage_extrapolated", "coverage_shrunk"):
            self.assertGreaterEqual(ov[k], 0.0)
            self.assertLessEqual(ov[k], 1.0)
        # 收缩估计的加权误差应优于或接近原始率
        self.assertLess(ov["wmae_shrunk"], ov["wmae_raw"] * 1.5)


if __name__ == "__main__":
    unittest.main()
