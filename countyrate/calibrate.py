"""全市总量校准（benchmarking），估计与不确定性一起调整。

问题：给定模型后验（率尺度均值 m_i、方差 V_i）、人口量 N_i、
已确认锁定项（locked，不得改动）与全市确认总量 T，求

    min  Σ_{自由县} (θ_i - m_i)² / V_i
    s.t. Σ_全部县 N_i θ_i = T,   0 ≤ θ_i ≤ 1,
         锁定县 θ_i = 确认率（固定）

KKT 条件给出 θ_i(λ) = clip(m_i + λ·V_i·N_i, 0, 1)，
λ 由总量方程唯一确定（左端关于 λ 单调不减）——用二分法求解，
自动满足非负与上限约束（clip 即激活集）。

不确定性联动调整：把 (θ_自由) 看作联合高斯（对角协方差 diag(V)），
在等式约束 N'θ = T' 下取条件协方差：
    Ṽ_i = V_i - (N_i V_i)² / Σ_{自由未贴边} N_j² V_j
锁定县与贴边县方差归零。于是区间随点估计一起被总量信息收紧。

不可行判定（绝不暗改已确认数据）：
    T - Σ_锁定 N_i·确认率 < 0            -> 总量低于已确认项之和
    T - Σ_锁定 N_i·确认率 > Σ_自由 N_i   -> 总量超出自由县承载上限
"""

_VAR_FLOOR = 1e-12


def calibrate(counties, locked_rates, city_total):
    """执行总量校准。

    counties: list[dict]，每项含 id / population / mean / var（率尺度后验矩）。
    locked_rates: dict {county_id: 锁定率}（确认项，不可改动）。
    city_total: 全市确认总量（计数）。
    返回校准结果 dict（含 feasible 标志与诊断）。
    """
    locked_rates = locked_rates or {}
    locked_sum = 0.0
    free = []
    for c in counties:
        cid = c["id"]
        if cid in locked_rates:
            locked_sum += c["population"] * locked_rates[cid]
        else:
            free.append(c)

    target = city_total - locked_sum
    capacity = sum(c["population"] for c in free)

    diag = {
        "city_total": city_total,
        "locked_sum": locked_sum,
        "target_remaining": target,
        "free_capacity": capacity,
        "n_locked": len(locked_rates),
        "n_free": len(free),
    }

    if target < -1e-9:
        return {"feasible": False, "reason": "total_below_locked",
                "detail": {**diag, "deficit": -target},
                "message": (f"全市总量 {city_total:.0f} 低于已确认锁定项之和 "
                            f"{locked_sum:.0f}，缺口 {-target:.0f}。"
                            f"已确认数据不可改动，校准不可行。")}
    if target > capacity + 1e-9:
        return {"feasible": False, "reason": "total_above_capacity",
                "detail": {**diag, "excess": target - capacity},
                "message": (f"全市总量 {city_total:.0f} 超出自由县最大承载 "
                            f"{capacity:.0f}（剩余需求 {target:.0f}）。"
                            f"已确认数据不可改动，校准不可行。")}

    # ---- 二分求 λ：f(λ) = Σ N·clip(m + λ·V·N, 0, 1) - target = 0 ----
    items = [
        (c["id"], c["population"], c["mean"], max(c["var"], _VAR_FLOOR))
        for c in free
    ]

    def total_at(lam):
        t = 0.0
        for _, N, m, V in items:
            theta = m + lam * V * N
            if theta < 0.0:
                theta = 0.0
            elif theta > 1.0:
                theta = 1.0
            t += N * theta
        return t

    lo, hi = -1.0, 1.0
    while total_at(lo) > 0.0:
        lo *= 2.0
    while total_at(hi) < target:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total_at(mid) < target:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)

    # ---- 点估计与贴边标记 ----
    rates = {}
    clamped_free = set()
    for cid, N, m, V in items:
        theta = m + lam * V * N
        mark = None
        if theta <= 0.0:
            theta, mark = 0.0, "lower"
        elif theta >= 1.0:
            theta, mark = 1.0, "upper"
        if mark:
            clamped_free.add(cid)
        rates[cid] = {"rate": theta, "clamped": mark, "locked": False}

    # ---- 不确定性联动调整（条件协方差的对角元）----
    active = [it for it in items if it[0] not in clamped_free]
    D = sum(N * N * V for _, N, _, V in active)
    for cid, N, m, V in items:
        if cid in clamped_free or D <= 0.0:
            rates[cid]["var"] = 0.0
        else:
            rates[cid]["var"] = max(V - (N * V) ** 2 / D, 0.0)

    for cid, r in locked_rates.items():
        rates[cid] = {"rate": r, "var": 0.0, "clamped": None, "locked": True}

    achieved = sum(
        next(c for c in counties if c["id"] == cid)["population"] * rc["rate"]
        for cid, rc in rates.items()
    )

    return {
        "feasible": True,
        "reason": None,
        "lambda": lam,
        "detail": diag,
        "counties": rates,
        "achieved_total": achieved,
        "message": (f"校准可行：λ={lam:.4g}，校准后总量 "
                    f"{achieved:.1f} = 全市确认总量 {city_total:.0f}。"),
    }
