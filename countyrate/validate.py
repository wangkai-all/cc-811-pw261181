"""整地区留出验证（leave-one-region-out）。

每一折：把一个地区的所有县整体留出，仅用其余地区训练模型，
再预测留出县 —— 检验模型对"整个未见地区"的外推能力。

两种预测模式：
- 外推：仅用地区特征 x'β̂（视同该县无样本）
- 收缩：训练得到的 β̂、τ̂² 不变，再结合留出县自己的抽样数据

指标（设计权重进入验证）：
- 90% 区间覆盖率（对模拟真值）
- MAE 与 n_eff 加权 MAE（wMAE，权重即 Kish 有效样本量）
- 与两个基线对比：原始率（仅有样本县）、全市均值
"""

from .model import fit_model, posterior_for, prior_for, rate_moments, interval_logit, logistic


def _metrics(rows, key):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    mae = sum(abs(r[key] - r["true_rate"]) for r in rows) / n
    return {"n": n, "mae": mae}


def _weighted_mae(rows, key):
    """n_eff 加权 MAE；零样本县权重为 0，不计入。"""
    sw = sum(r["n_eff"] for r in rows)
    if sw <= 0:
        return None
    return sum(r["n_eff"] * abs(r[key] - r["true_rate"]) for r in rows) / sw


def leave_one_region_out(dataset):
    counties = dataset["counties"]
    regions = []
    for c in counties:
        if c["region"] not in regions:
            regions.append(c["region"])

    folds = []
    for region in regions:
        train = [c for c in counties if c["region"] != region]
        test = [c for c in counties if c["region"] == region]
        fit = fit_model(train)

        rows = []
        for c in test:
            # 外推：只用先验
            pri = prior_for(c, fit)
            m_ext, _ = rate_moments(pri["mean"], pri["var"])
            lo_ext, hi_ext = interval_logit(pri["mean"], pri["var"])
            # 收缩：结合本县样本
            post = posterior_for(c, fit)
            m_shr, _ = rate_moments(post["mu"], post["v"])
            lo_shr, hi_shr = interval_logit(post["mu"], post["v"])
            truth = c["true_rate"]
            rows.append({
                "id": c["id"], "name": c["name"],
                "n_eff": c["n_eff"],
                "true_rate": truth,
                "raw_rate": c["raw_rate"],
                "extrapolated_rate": m_ext,
                "shrunk_rate": m_shr,
                "covered_extrapolated": lo_ext <= truth <= hi_ext,
                "covered_shrunk": lo_shr <= truth <= hi_shr,
                "width_extrapolated": hi_ext - lo_ext,
                "width_shrunk": hi_shr - lo_shr,
            })

        sampled = [r for r in rows if r["raw_rate"] is not None]
        folds.append({
            "region": region,
            "n_train": fit["n_train"],
            "tau2": fit["tau2"],
            "rows": rows,
            "coverage_extrapolated": sum(r["covered_extrapolated"] for r in rows) / len(rows),
            "coverage_shrunk": sum(r["covered_shrunk"] for r in rows) / len(rows),
            "mae_extrapolated": _metrics(rows, "extrapolated_rate")["mae"],
            "mae_shrunk": _metrics(rows, "shrunk_rate")["mae"],
            "wmae_shrunk": _weighted_mae(rows, "shrunk_rate"),
            "mae_raw": _metrics(sampled, "raw_rate")["mae"] if sampled else None,
            "wmae_raw": _weighted_mae(sampled, "raw_rate"),
        })

    all_rows = [r for f in folds for r in f["rows"]]
    all_sampled = [r for r in all_rows if r["raw_rate"] is not None]
    overall = {
        "coverage_extrapolated": sum(r["covered_extrapolated"] for r in all_rows) / len(all_rows),
        "coverage_shrunk": sum(r["covered_shrunk"] for r in all_rows) / len(all_rows),
        "mae_extrapolated": _metrics(all_rows, "extrapolated_rate")["mae"],
        "mae_shrunk": _metrics(all_rows, "shrunk_rate")["mae"],
        "wmae_shrunk": _weighted_mae(all_rows, "shrunk_rate"),
        "mae_raw": _metrics(all_sampled, "raw_rate")["mae"],
        "wmae_raw": _weighted_mae(all_sampled, "raw_rate"),
        "mean_width_extrapolated": sum(r["width_extrapolated"] for r in all_rows) / len(all_rows),
        "mean_width_shrunk": sum(r["width_shrunk"] for r in all_rows) / len(all_rows),
    }
    return {"folds": folds, "overall": overall}
