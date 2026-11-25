"""校准层单元测试：约束满足、边界、不可行报告、不确定性联动。"""

import unittest

from countyrate.calibrate import calibrate


def cs(specs):
    """specs: list of (id, population, mean, var)"""
    return [{"id": s[0], "population": s[1], "mean": s[2], "var": s[3]} for s in specs]


class TestFeasible(unittest.TestCase):
    def setUp(self):
        self.counties = cs([
            ("A", 1000, 0.10, 0.001),
            ("B", 2000, 0.20, 0.002),
            ("C", 3000, 0.30, 0.003),
        ])
        self.model_total = 1000 * 0.10 + 2000 * 0.20 + 3000 * 0.30  # 1400

    def test_constraint_satisfied_exactly(self):
        res = calibrate(self.counties, {}, 1200.0)
        self.assertTrue(res["feasible"])
        self.assertAlmostEqual(res["achieved_total"], 1200.0, places=6)

    def test_locked_values_untouched(self):
        res = calibrate(self.counties, {"A": 0.15}, 1300.0)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["counties"]["A"]["rate"], 0.15)
        self.assertTrue(res["counties"]["A"]["locked"])
        self.assertAlmostEqual(res["achieved_total"], 1300.0, places=6)

    def test_bounds_respected_under_extreme_total(self):
        # 极低总量 -> 部分县贴 0 界，但总量仍精确满足
        res = calibrate(self.counties, {}, 50.0)
        self.assertTrue(res["feasible"])
        for rc in res["counties"].values():
            self.assertGreaterEqual(rc["rate"], 0.0)
            self.assertLessEqual(rc["rate"], 1.0)
        self.assertAlmostEqual(res["achieved_total"], 50.0, places=6)
        self.assertTrue(any(rc["clamped"] == "lower" for rc in res["counties"].values()))

    def test_upper_bound_respected(self):
        res = calibrate(self.counties, {}, 5900.0)  # 接近容量 6000
        self.assertTrue(res["feasible"])
        for rc in res["counties"].values():
            self.assertLessEqual(rc["rate"], 1.0)
        self.assertAlmostEqual(res["achieved_total"], 5900.0, places=6)

    def test_uncertainty_shrinks_with_estimates(self):
        res = calibrate(self.counties, {}, 1200.0)
        for c in self.counties:
            rc = res["counties"][c["id"]]
            if rc["clamped"] is None:
                self.assertLess(rc["var"], c["var"],
                                "自由未贴边县的条件方差应小于模型方差")

    def test_locked_and_clamped_have_zero_variance(self):
        # 锁定 A=0.15（贡献 150），总量 200 -> 自由县只剩 50，被迫贴 0 界
        res = calibrate(self.counties, {"A": 0.15}, 200.0)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["counties"]["A"]["var"], 0.0)
        for rc in res["counties"].values():
            if rc["clamped"] is not None:
                self.assertEqual(rc["var"], 0.0)

    def test_zero_variance_free_county_handled(self):
        counties = cs([("A", 1000, 0.1, 0.0), ("B", 1000, 0.2, 0.01)])
        res = calibrate(counties, {}, 250.0)
        self.assertTrue(res["feasible"])
        self.assertAlmostEqual(res["achieved_total"], 250.0, places=6)


class TestInfeasible(unittest.TestCase):
    def setUp(self):
        self.counties = cs([
            ("A", 1000, 0.10, 0.001),
            ("B", 2000, 0.20, 0.002),
        ])

    def test_total_below_locked_sum(self):
        # 锁定 A=0.5（贡献 500），总量 400 < 500 -> 不可行
        res = calibrate(self.counties, {"A": 0.5}, 400.0)
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "total_below_locked")
        self.assertAlmostEqual(res["detail"]["deficit"], 100.0, places=9)
        self.assertIn("不可改动", res["message"])

    def test_total_above_capacity(self):
        # 锁定 A=0.1（贡献 100），自由容量 2000，总量 2500 -> 需求 2400 > 2000
        res = calibrate(self.counties, {"A": 0.1}, 2500.0)
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "total_above_capacity")

    def test_infeasible_does_not_modify_locked(self):
        # 不可行时结果中不含任何县估计，确认数据无从被改
        res = calibrate(self.counties, {"A": 0.5}, 400.0)
        self.assertNotIn("counties", res)

    def test_all_locked_exact_total(self):
        res = calibrate(self.counties, {"A": 0.1, "B": 0.2}, 500.0)
        self.assertTrue(res["feasible"])
        self.assertAlmostEqual(res["achieved_total"], 500.0, places=9)


if __name__ == "__main__":
    unittest.main()
