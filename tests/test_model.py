"""模型层单元测试：logit 变换、有效样本量、分层拟合、后验收缩。"""

import math
import unittest

from countyrate import model
from countyrate.linalg import solve, weighted_least_squares


def make_county(n, W, y_w, n_eff, urban=0.5, aging=0.2, income=1.0):
    """构造测试用县记录。"""
    return {
        "id": "T", "name": "测试县", "region": "测试区",
        "population": 10000, "urban": urban, "aging": aging, "income": income,
        "n_sample": n, "sum_weights": W, "n_eff": n_eff,
        "deff": (n / n_eff) if n_eff else None,
        "weighted_successes": y_w,
        "raw_rate": (y_w / W) if W else None,
        "confirmed_count": None, "true_rate": 0.1,
    }


class TestTransforms(unittest.TestCase):
    def test_logit_logistic_inverse(self):
        for p in (0.01, 0.1, 0.5, 0.9, 0.99):
            self.assertAlmostEqual(model.logistic(model.logit(p)), p, places=12)

    def test_interval_within_unit_interval(self):
        lo, hi = model.interval_logit(0.0, 100.0)
        self.assertGreater(lo, 0.0)
        self.assertLess(hi, 1.0)


class TestLinalg(unittest.TestCase):
    def test_solve_identity(self):
        x = solve([[1, 0], [0, 1]], [3, 4])
        self.assertAlmostEqual(x[0], 3.0)
        self.assertAlmostEqual(x[1], 4.0)

    def test_solve_2x2(self):
        # 2x + y = 5 ; x + 3y = 5  ->  x=2, y=1
        x = solve([[2, 1], [1, 3]], [5, 5])
        self.assertAlmostEqual(x[0], 2.0, places=10)
        self.assertAlmostEqual(x[1], 1.0, places=10)

    def test_wls_equal_weights_matches_ols_closed_form(self):
        # y = 1 + 2x，精确无噪声 -> 应精确还原
        X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        y = [1.0, 3.0, 5.0, 7.0]
        beta = weighted_least_squares(X, y, [1.0] * 4)
        self.assertAlmostEqual(beta[0], 1.0, places=8)
        self.assertAlmostEqual(beta[1], 2.0, places=8)


class TestLikelihoodStats(unittest.TestCase):
    def test_zero_sample_returns_none(self):
        self.assertIsNone(model.likelihood_stats(make_county(0, 0.0, 0.0, 0.0)))

    def test_hand_computed(self):
        # n=4, 等权重 w=1 -> W=4, n_eff=4, ỹ=1
        # p̃ = (1+0.5)/(4+1) = 0.3 ; z = logit(0.3)
        # s² = logit((X+0.5)/5) 在 X~Bin(4, 0.3) 下的精确方差（独立核算值）
        st = model.likelihood_stats(make_county(4, 4.0, 1.0, 4.0))
        self.assertAlmostEqual(st["p_tilde"], 0.3, places=12)
        self.assertAlmostEqual(st["z"], math.log(0.3 / 0.7), places=12)
        self.assertAlmostEqual(st["s2"], 0.916878549978113, places=12)

    def test_extreme_rate_smoothed(self):
        # 1/1 = 100% 的极端率必须被连续性修正拉回 (0,1)
        st = model.likelihood_stats(make_county(1, 1.0, 1.0, 1.0))
        self.assertLess(st["p_tilde"], 1.0)
        self.assertTrue(math.isfinite(st["z"]))
        self.assertTrue(math.isfinite(st["s2"]))


class TestFitAndPosterior(unittest.TestCase):
    def _synthetic_counties(self, n=30, true_beta=(-2.0, 1.0), tau=0.3, seed=7):
        import random
        rng = random.Random(seed)
        counties = []
        for i in range(n):
            urban = rng.uniform(0.2, 0.9)
            mu = true_beta[0] + true_beta[1] * urban + rng.gauss(0, tau)
            theta = model.logistic(mu)
            n_eff = 200.0  # 较大有效样本量，减小 logit 变换的 Jensen 偏差
            # 直接按二项抽样构造加权计数（等权重 -> n_eff = n）
            succ = sum(1 for _ in range(int(n_eff)) if rng.random() < theta)
            counties.append(make_county(
                int(n_eff), n_eff, float(succ), n_eff,
                urban=urban,
                aging=rng.uniform(0.1, 0.35),
                income=rng.uniform(0.6, 1.5)))
        return counties

    def test_fit_recovers_beta_and_tau(self):
        counties = self._synthetic_counties()
        fit = model.fit_model(counties)
        self.assertAlmostEqual(fit["beta"][0], -2.0, delta=0.25)
        self.assertAlmostEqual(fit["beta"][1], 1.0, delta=0.35)
        self.assertGreaterEqual(fit["tau2"], 0.0)
        self.assertLess(fit["tau2"], 0.5)  # 真值 0.09，不应爆涨

    def test_zero_sample_posterior_equals_prior(self):
        counties = self._synthetic_counties()
        fit = model.fit_model(counties)
        c0 = make_county(0, 0.0, 0.0, 0.0, urban=0.6)
        post = model.posterior_for(c0, fit)
        pri = model.prior_for(c0, fit)
        self.assertTrue(post["extrapolated"])
        self.assertEqual(post["B"], 1.0)
        self.assertAlmostEqual(post["mu"], pri["mean"], places=15)
        self.assertAlmostEqual(post["v"], pri["var"], places=15)

    def test_shrinkage_weight_monotonic_in_precision(self):
        counties = self._synthetic_counties()
        fit = model.fit_model(counties)
        # 同一观察率、不同有效样本量：n_eff 越大，收缩权重 B 越小
        Bs = []
        for n_eff in (2.0, 10.0, 50.0, 200.0):
            y = 0.2 * n_eff
            c = make_county(int(n_eff), n_eff, y, n_eff)
            Bs.append(model.posterior_for(c, fit)["B"])
        self.assertTrue(all(Bs[i] > Bs[i + 1] for i in range(len(Bs) - 1)),
                        f"B 应随 n_eff 递减: {Bs}")

    def test_posterior_between_prior_and_data(self):
        counties = self._synthetic_counties()
        fit = model.fit_model(counties)
        c = make_county(20, 20.0, 6.0, 20.0)  # p̃ 偏高
        post = model.posterior_for(c, fit)
        lo = min(post["z"], post["prior_mean"])
        hi = max(post["z"], post["prior_mean"])
        self.assertLessEqual(lo - 1e-12, post["mu"])
        self.assertLessEqual(post["mu"], hi + 1e-12)


if __name__ == "__main__":
    unittest.main()
