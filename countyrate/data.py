"""虚构抽样数据生成（固定随机种子，完全可复现）。

构造一个虚构市，下辖 5 个地区、24 个县。对每个县生成：
- 人口量 population 与地区特征（城镇化率 urban、老龄化指数 aging、收入指数 income）
- 模拟真值率 true_rate（仅用于验证，现实场景中不可见）
- 分层不等概率抽样：样本量 n、设计权重、加权成功数、Kish 有效样本量
- 部分县有行政确认计数 confirmed_count（已确认项，锁定后不可改动）

内置极端情形（需求要求）：
- 零样本县（n=0）：估计完全依赖模型外推
- 小样本极端率县（n=1、n=3）：原始率 0% 或 100%，极不稳定
- 冲突总量场景：全市确认总量与县级已确认项矛盾，必须报告不可行

种子固定为 20261125；所有随机抽取按固定顺序进行，任何机器上结果一致。
"""

import math
import random

SEED = 20261125

# 真实数据生成机制（仅用于造数；估计模型看不到 region_effect 与噪声）
_TRUE_BETA = {"intercept": -2.3, "urban": 1.2, "aging": 1.5, "income": -0.6}
_REGION_EFFECT = {"北山区": 0.25, "东川区": -0.15, "南湖区": 0.05, "西林区": -0.30, "中城区": 0.15}
_COUNTY_SD = 0.30  # 县级真值噪声（模型外的异质性）

_COUNTY_NAMES = {
    "北山区": ["青云县", "白岩县", "松岭县", "石门县", "寒泉县"],
    "东川区": ["临江县", "金滩县", "白沙县", "青川县"],
    "南湖区": ["镜湖县", "荷风县", "芦花县", "清波县", "涟水县"],
    "西林区": ["梧桐县", "樟木县", "枫叶县", "竹溪县", "松林县"],
    "中城区": ["朝晖县", "望江县", "文华县", "安澜县", "同安县"],
}

# 极端情形指定（按县名）
_ZERO_SAMPLE = {"石门县", "涟水县"}          # 无样本：纯模型外推
_TINY_SAMPLE = {"白岩县": 1, "荷风县": 3}     # 小样本极端率
_CONFIRMED = {"青云县", "金滩县", "镜湖县", "梧桐县", "朝晖县", "安澜县"}


def _logit(p):
    return math.log(p / (1.0 - p))


def _logistic(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def generate_dataset(seed=SEED):
    """生成整套虚构数据。返回 dict：counties / units / scenarios / meta。"""
    rng = random.Random(seed)
    counties = []
    units = []  # 单元级抽样明细（县id, 单元号, 权重, 结果），供核查与导出

    cid = 0
    for region, names in _COUNTY_NAMES.items():
        for name in names:
            cid += 1
            # ---- 人口：刻意拉开量级（小县几千，大县几十万）----
            size_class = rng.random()
            if size_class < 0.35:
                population = rng.randint(2_000, 8_000)       # 小县
            elif size_class < 0.8:
                population = rng.randint(15_000, 60_000)     # 中等县
            else:
                population = rng.randint(80_000, 250_000)    # 大县

            # ---- 地区特征 ----
            urban = round(rng.uniform(0.25, 0.90), 3)
            aging = round(rng.uniform(0.08, 0.35), 3)
            income = round(rng.uniform(0.60, 1.50), 3)

            # ---- 模拟真值率 ----
            z = (_TRUE_BETA["intercept"] + _TRUE_BETA["urban"] * urban
                 + _TRUE_BETA["aging"] * aging + _TRUE_BETA["income"] * income
                 + _REGION_EFFECT[region] + rng.gauss(0.0, _COUNTY_SD))
            true_rate = _logistic(z)

            # ---- 抽样设计 ----
            if name in _ZERO_SAMPLE:
                n_sample = 0
            elif name in _TINY_SAMPLE:
                n_sample = _TINY_SAMPLE[name]
            else:
                base = 6 + population / 4000.0
                n_sample = max(5, min(80, int(round(base * rng.uniform(0.5, 1.5)))))

            # 每个县 3 个层，层内等概率、层间不等概率 -> 设计权重有实际差异
            sum_w = 0.0
            sum_w2 = 0.0
            weighted_succ = 0.0
            raw_succ = 0
            for j in range(n_sample):
                stratum = j % 3
                inclusion_prob = [0.020, 0.008, 0.004][stratum]
                weight = 1.0 / inclusion_prob * rng.uniform(0.85, 1.15)
                x = 1 if rng.random() < true_rate else 0
                raw_succ += x
                sum_w += weight
                sum_w2 += weight * weight
                weighted_succ += weight * x
                units.append({"county_id": f"C{cid:02d}", "unit": j + 1,
                              "stratum": stratum + 1,
                              "weight": round(weight, 4), "outcome": x})

            if n_sample > 0:
                n_eff = sum_w * sum_w / sum_w2          # Kish 有效样本量
                deff = n_sample / n_eff                  # 设计效应
                raw_rate = weighted_succ / sum_w         # 设计加权原始率
            else:
                n_eff = 0.0
                deff = None
                raw_rate = None

            # ---- 行政确认计数（仅部分县；贴近真值但不等于真值）----
            confirmed = None
            if name in _CONFIRMED:
                confirmed = max(0, round(population * true_rate * rng.uniform(0.96, 1.04)))

            counties.append({
                "id": f"C{cid:02d}", "name": name, "region": region,
                "population": population,
                "urban": urban, "aging": aging, "income": income,
                "n_sample": n_sample,
                "sum_weights": round(sum_w, 4),
                "n_eff": round(n_eff, 2),
                "deff": round(deff, 3) if deff else None,
                "weighted_successes": round(weighted_succ, 4),
                "raw_rate": raw_rate,
                "confirmed_count": confirmed,
                "true_rate": true_rate,
            })

    true_total = sum(c["population"] * c["true_rate"] for c in counties)
    confirmed_sum = sum(c["confirmed_count"] for c in counties if c["confirmed_count"] is not None)

    scenarios = {
        # 可行：全市确认总量 = 模拟真值总量（取整）
        "baseline": {"label": "基准（可行总量）", "city_total": round(true_total)},
        # 可行但偏低：校准会触发部分县的下限（0）约束
        "low": {"label": "偏低总量（触发边界）", "city_total": round(0.60 * true_total)},
        # 不可行：总量低于已确认项之和 —— 必须报告冲突，不得暗改确认数据
        "conflict": {"label": "冲突总量（不可行）", "city_total": round(0.40 * confirmed_sum)},
    }

    return {
        "meta": {
            "seed": seed,
            "true_total": true_total,
            "confirmed_sum": confirmed_sum,
            "note": "虚构数据；true_rate 为模拟真值，仅用于验证，估计流程不使用。",
        },
        "counties": counties,
        "units": units,
        "scenarios": scenarios,
    }


def default_locked_ids(dataset):
    """默认锁定集合：所有带行政确认计数的县。"""
    return [c["id"] for c in dataset["counties"] if c["confirmed_count"] is not None]
