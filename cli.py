#!/usr/bin/env python3
"""县域比率分层估计与总量校准 —— 命令行入口（可核查计算）。

用法：
    python3 cli.py                      # 基准场景，结果写入 out/
    python3 cli.py --scenario conflict  # 冲突总量场景（报告不可行）
    python3 cli.py --unlock-all         # 不锁定任何已确认项

输出（out/ 目录）：
    estimates_<场景>.csv   各县原始率/收缩率/校准后率与区间明细
    validation.csv         整地区留出验证明细
    units.csv              抽样单元级明细（权重、结果）
    summary.json           全部结果（模型系数、校准诊断、验证汇总）
"""

import argparse
import csv
import json
import os
import sys

from countyrate.data import generate_dataset, default_locked_ids
from countyrate.pipeline import compute_estimates, compute_validation

PCT = lambda x: "" if x is None else f"{x:.9f}"


def write_estimates_csv(path, result):
    cols = ["id", "name", "region", "population", "n_sample", "n_eff", "deff",
            "sum_weights", "weighted_successes", "raw_rate", "confirmed_count",
            "shrunk_rate", "shrunk_lo", "shrunk_hi", "shrink_weight_B",
            "extrapolated", "cal_rate", "cal_lo", "cal_hi",
            "locked", "clamped", "true_rate"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in result["counties"]:
            w.writerow([
                c["id"], c["name"], c["region"], c["population"],
                c["n_sample"], c["n_eff"], c["deff"] or "",
                c["sum_weights"], c["weighted_successes"],
                PCT(c["raw_rate"]), c["confirmed_count"] or "",
                PCT(c["shrunk_rate"]), PCT(c["shrunk_lo"]), PCT(c["shrunk_hi"]),
                f"{c['shrink_weight_B']:.4f}", int(c["extrapolated"]),
                PCT(c["cal_rate"]), PCT(c["cal_lo"]), PCT(c["cal_hi"]),
                int(c["locked"]), c["clamped"] or "", PCT(c["true_rate"]),
            ])


def write_validation_csv(path, val):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["held_out_region", "county_id", "name", "n_eff", "true_rate",
                    "raw_rate", "extrapolated_rate", "shrunk_rate",
                    "covered_extrapolated", "covered_shrunk"])
        for fold in val["folds"]:
            for r in fold["rows"]:
                w.writerow([fold["region"], r["id"], r["name"], r["n_eff"],
                            PCT(r["true_rate"]), PCT(r["raw_rate"]),
                            PCT(r["extrapolated_rate"]), PCT(r["shrunk_rate"]),
                            int(r["covered_extrapolated"]), int(r["covered_shrunk"])])


def write_units_csv(path, units):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["county_id", "unit", "stratum", "weight", "outcome"])
        w.writeheader()
        w.writerows(units)


def main(argv=None):
    ap = argparse.ArgumentParser(description="县域比率分层估计与总量校准")
    ap.add_argument("--scenario", default="baseline",
                    choices=["baseline", "low", "conflict"], help="总量场景")
    ap.add_argument("--out", default="out", help="输出目录")
    ap.add_argument("--unlock-all", action="store_true", help="不锁定任何已确认项")
    args = ap.parse_args(argv)

    dataset = generate_dataset()
    locked = [] if args.unlock_all else default_locked_ids(dataset)

    result = compute_estimates(dataset, args.scenario, locked)
    val = compute_validation(dataset)

    os.makedirs(args.out, exist_ok=True)
    est_path = os.path.join(args.out, f"estimates_{args.scenario}.csv")
    write_estimates_csv(est_path, result)
    write_validation_csv(os.path.join(args.out, "validation.csv"), val)
    write_units_csv(os.path.join(args.out, "units.csv"), dataset["units"])
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"estimates": result, "validation": val}, f,
                  ensure_ascii=False, indent=2)

    m = result["model"]
    cal = result["calibration"]
    print(f"场景: {result['scenario']['label']}  全市确认总量 T = {result['scenario']['city_total']}")
    print(f"模型: τ̂² = {m['tau2']:.4f}（收缩强度，由 {m['n_train']} 个有样本县估计，未使用人口量）")
    for name, b in zip(m["feature_names"], m["beta"]):
        print(f"  β[{name}] = {b:+.4f}")
    print(f"校准前模型隐含总量: {result['model_total_before_calibration']:.1f}")
    print(f"校准: {cal['message']}")
    if cal["feasible"]:
        print(f"{'县':<6}{'原始率':>9}{'收缩率':>9}{'校准后率':>9}  90%区间")
        for c in result["counties"]:
            raw = "—" if c["raw_rate"] is None else f"{c['raw_rate']:.3f}"
            print(f"{c['name']:<6}{raw:>9}{c['shrunk_rate']:>9.3f}{c['cal_rate']:>9.3f}"
                  f"  [{c['cal_lo']:.3f}, {c['cal_hi']:.3f}]"
                  + ("  [锁定]" if c["locked"] else "")
                  + ("  [外推]" if c["extrapolated"] else ""))
    ov = val["overall"]
    print(f"\n整地区留出验证: 外推覆盖率 {ov['coverage_extrapolated']:.0%}, "
          f"收缩覆盖率 {ov['coverage_shrunk']:.0%}, "
          f"加权MAE 收缩 {ov['wmae_shrunk']:.4f} vs 原始 {ov['wmae_raw']:.4f}")
    print(f"\n已写出: {est_path} 及 validation.csv / units.csv / summary.json")
    return 0 if cal["feasible"] else 2


if __name__ == "__main__":
    sys.exit(main())
