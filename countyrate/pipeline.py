"""端到端编排：CLI 与 Web 服务共用同一套计算，保证结果可核查一致。

compute_estimates: 拟合分层模型 -> 县级收缩估计 -> 全市总量校准
compute_validation: 整地区留出验证
"""

from . import model
from .calibrate import calibrate
from .validate import leave_one_region_out


def compute_estimates(dataset, scenario_key, locked_ids):
    """对给定场景与锁定集合，返回完整估计结果（JSON 友好的 dict）。"""
    counties = dataset["counties"]
    scenario = dataset["scenarios"][scenario_key]
    city_total = scenario["city_total"]
    locked_ids = set(locked_ids or [])

    fit = model.fit_model(counties)

    # ---- 收缩估计（校准前）----
    per_county = {}
    for c in counties:
        post = model.posterior_for(c, fit)
        m_rate, v_rate = model.rate_moments(post["mu"], post["v"])
        lo, hi = model.interval_logit(post["mu"], post["v"])
        x = model.features(c)
        per_county[c["id"]] = {
            "posterior": post,
            "mean": m_rate,
            "var": v_rate,
            "lo": lo,
            "hi": hi,
            "prior_rate": model.logistic(post["prior_mean"]),
            "covariate_contribs": [
                {"name": name, "x": xv, "beta": b, "contrib": xv * b}
                for name, xv, b in zip(model.FEATURE_NAMES, x, fit["beta"])
            ],
        }

    # ---- 总量校准 ----
    locked_rates = {}
    for c in counties:
        if c["id"] in locked_ids:
            if c["confirmed_count"] is None:
                raise ValueError(f"县 {c['id']} 没有已确认计数，无法锁定")
            locked_rates[c["id"]] = c["confirmed_count"] / c["population"]

    cal_input = [
        {"id": c["id"], "population": c["population"],
         "mean": per_county[c["id"]]["mean"], "var": per_county[c["id"]]["var"]}
        for c in counties
    ]
    cal = calibrate(cal_input, locked_rates, city_total)

    # ---- 汇总输出 ----
    out_counties = []
    for c in counties:
        pc = per_county[c["id"]]
        post = pc["posterior"]
        row = {
            "id": c["id"], "name": c["name"], "region": c["region"],
            "population": c["population"],
            "urban": c["urban"], "aging": c["aging"], "income": c["income"],
            "n_sample": c["n_sample"], "n_eff": c["n_eff"], "deff": c["deff"],
            "sum_weights": c["sum_weights"],
            "weighted_successes": c["weighted_successes"],
            "raw_rate": c["raw_rate"],
            "confirmed_count": c["confirmed_count"],
            "true_rate": c["true_rate"],
            # 收缩估计
            "shrunk_rate": pc["mean"],
            "shrunk_lo": pc["lo"], "shrunk_hi": pc["hi"],
            "shrink_weight_B": post["B"],          # 1=完全靠模型, 0=完全靠本县数据
            "own_weight": 1.0 - post["B"],
            "prior_rate": pc["prior_rate"],
            "extrapolated": post["extrapolated"],
            "covariate_contribs": pc["covariate_contribs"],
        }
        if cal["feasible"]:
            rc = cal["counties"][c["id"]]
            sd = rc["var"] ** 0.5
            row.update({
                "cal_rate": rc["rate"],
                "cal_lo": max(0.0, rc["rate"] - model.Z90 * sd),
                "cal_hi": min(1.0, rc["rate"] + model.Z90 * sd),
                "cal_var": rc["var"],
                "clamped": rc["clamped"],
                "locked": rc["locked"],
                "cal_shift": rc["rate"] - pc["mean"],
            })
        else:
            row.update({
                "cal_rate": None, "cal_lo": None, "cal_hi": None,
                "cal_var": None, "clamped": None,
                "locked": c["id"] in locked_ids,
                "cal_shift": None,
            })
        out_counties.append(row)

    model_total = sum(c["population"] * per_county[c["id"]]["mean"] for c in counties)

    return {
        "scenario": {"key": scenario_key, "label": scenario["label"],
                     "city_total": city_total},
        "model": {
            "beta": fit["beta"], "tau2": fit["tau2"],
            "feature_names": model.FEATURE_NAMES,
            "n_train": fit["n_train"],
        },
        "calibration": cal,
        "model_total_before_calibration": model_total,
        "locked_ids": sorted(locked_ids),
        "counties": out_counties,
    }


def compute_validation(dataset):
    """整地区留出验证（耗时很短，纯标准库）。"""
    return leave_one_region_out(dataset)
