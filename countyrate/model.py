"""logit 尺度分层经验贝叶斯模型。

模型（抽样噪声用二项精确代入方差，见 likelihood_stats）：
    z_i = logit(p̃_i) = x_i'β + u_i + e_i
    u_i ~ N(0, τ²)        县间异质性（收缩强度）
    e_i ~ N(0, s_i²)      抽样噪声（由设计权重与有效样本量决定）

关键点（对应需求）：
- 收缩强度 τ² 用 DerSimonian–Laird 矩估计从训练数据估计，
  全程不使用人口量 —— 人口只在校准阶段作为总量权重出现。
- 设计权重通过加权成功数 ỹ 与 Kish 有效样本量 n_eff 进入 s_i²。
- 零样本县没有 (z, s²)，直接取先验 N(x_i'β, τ² + 系数不确定性)，即模型外推。
"""

import math
from math import comb

from .linalg import weighted_least_squares, solve

FEATURE_NAMES = ["截距", "城镇化率", "老龄化指数", "收入指数"]
Z90 = 1.6448536269514722  # 90% 正态区间半宽


def logistic(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logit(p):
    return math.log(p / (1.0 - p))


def features(county):
    """设计矩阵行：截距 + 三个地区特征。"""
    return [1.0, county["urban"], county["aging"], county["income"]]


def _binomial_logit_variance(m, p):
    """统计量 z = logit((X+0.5)/(m+1)) 在 X ~ Binomial(m, p) 下的精确方差。

    对 x = 0..m 求和（代入法：p 取观测到的平滑率）。
    小样本下远优于 delta 近似 —— 例如 m=1 时 delta 法给出 5.33，
    而真实方差不足 1，虚大的方差会淹没县间真实异质性 τ²。
    """
    zs = [logit((x + 0.5) / (m + 1)) for x in range(m + 1)]
    ps = [comb(m, x) * p ** x * (1.0 - p) ** (m - x) for x in range(m + 1)]
    ez = sum(pi * zi for pi, zi in zip(ps, zs))
    return sum(pi * (zi - ez) ** 2 for pi, zi in zip(ps, zs))


def likelihood_stats(county):
    """由抽样数据计算 (z, s²)。零样本县返回 None。

    z = logit(p̃)，连续性修正 p̃ = (ỹ + 0.5·w̄) / (W + w̄)，w̄ 为平均权重，
    使 0%/100% 的极端原始率也能取 logit，且修正幅度随样本量缩小。
    s² = z 在 Binomial(round(n_eff), p̃) 下的精确代入方差，
    其中 n_eff 为 Kish 有效样本量（设计权重由此进入估计）。
    """
    n = county["n_sample"]
    if n == 0:
        return None
    W = county["sum_weights"]
    w_bar = W / n
    p_tilde = (county["weighted_successes"] + 0.5 * w_bar) / (W + w_bar)
    p_tilde = min(max(p_tilde, 1e-9), 1 - 1e-9)
    z = logit(p_tilde)
    m = max(1, int(round(county["n_eff"])))
    s2 = _binomial_logit_variance(m, p_tilde)
    return {"z": z, "s2": s2, "p_tilde": p_tilde}


def _invert(A):
    """用线性方程组求逆（按单位列逐列求解）。仅用于 k×k 小矩阵。"""
    n = len(A)
    cols = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        cols.append(solve(A, e))
    # cols[j] 是逆的第 j 列 -> 转行
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def fit_model(counties):
    """在训练县集合上估计 β 与 τ²。

    流程（确定性，固定 2 轮迭代）：
      1. w0 = 1/s² 的 WLS 得 β̂₀
      2. DL 矩估计 τ̂² = max(0, (Q - (n-k)) / C)
      3. 用 w = 1/(τ̂²+s²) 重估 β̂，再更新一次 τ̂²
    返回 dict：beta、tau2、系数协方差逆矩阵 (X'WX)、训练县数。
    """
    train = []
    for c in counties:
        st = likelihood_stats(c)
        if st is not None:
            train.append((features(c), st["z"], st["s2"]))
    n = len(train)
    k = len(train[0][0]) if train else 0
    if n <= k:
        raise ValueError(f"训练样本县数({n})不足以估计 {k} 个系数")

    X = [t[0] for t in train]
    z = [t[1] for t in train]
    s2 = [t[2] for t in train]

    tau2 = 0.0
    beta = None
    for _ in range(2):
        w = [1.0 / (tau2 + s) for s in s2]
        beta = weighted_least_squares(X, z, w)
        # DL 矩估计（Q 用当前权重下的残差）
        w0 = [1.0 / s for s in s2]
        beta0 = weighted_least_squares(X, z, w0)
        Q = sum(wi * (zi - sum(b * x for b, x in zip(beta0, xi))) ** 2
                for wi, zi, xi in zip(w0, z, X))
        sum_w0 = sum(w0)
        C = sum_w0 - sum(wi * wi for wi in w0) / sum_w0
        tau2 = max(0.0, (Q - (n - k)) / C) if C > 0 else 0.0

    # 系数不确定性 (X'WX)^{-1}，用于外推县的先验方差
    XtWX = [[0.0] * k for _ in range(k)]
    for xi, wi in zip(X, w):
        for a in range(k):
            for b in range(k):
                XtWX[a][b] += wi * xi[a] * xi[b]
    beta_cov = _invert(XtWX)

    return {"beta": beta, "tau2": tau2, "beta_cov": beta_cov, "n_train": n}


def prior_for(county, fit):
    """先验（模型部分）：均值 x'β̂，方差 τ̂² + x'(X'WX)^{-1}x。"""
    x = features(county)
    mean = sum(b * xi for b, xi in zip(fit["beta"], x))
    # x' beta_cov x
    quad = 0.0
    for a in range(len(x)):
        for b in range(len(x)):
            quad += x[a] * fit["beta_cov"][a][b] * x[b]
    var = fit["tau2"] + max(quad, 0.0)
    return {"mean": mean, "var": var}


def posterior_for(county, fit):
    """县级后验（logit 尺度）。

    有样本：μ = (1-B)·z + B·x'β̂，B = s²/(先验方差+s²) 为收缩权重，
            v = (1/s² + 1/先验方差)^{-1}
    零样本：μ = x'β̂，v = 先验方差（纯模型外推，B=1）。
    """
    prior = prior_for(county, fit)
    st = likelihood_stats(county)
    if st is None:
        return {"mu": prior["mean"], "v": prior["var"], "B": 1.0,
                "prior_mean": prior["mean"], "prior_var": prior["var"],
                "extrapolated": True, "z": None, "s2": None}
    B = st["s2"] / (prior["var"] + st["s2"])
    mu = (1.0 - B) * st["z"] + B * prior["mean"]
    v = 1.0 / (1.0 / st["s2"] + 1.0 / prior["var"])
    return {"mu": mu, "v": v, "B": B,
            "prior_mean": prior["mean"], "prior_var": prior["var"],
            "extrapolated": False, "z": st["z"], "s2": st["s2"]}


def rate_moments(mu, v):
    """logit 后验 -> 率尺度矩（delta 方法）。

    E[θ] ≈ logistic(μ)，Var(θ) ≈ [θ(1-θ)]² · v
    """
    m = logistic(mu)
    g = m * (1.0 - m)
    return m, g * g * v


def interval_logit(mu, v, z=Z90):
    """logit 尺度区间变换回率尺度（保证落在 (0,1) 内）。"""
    sd = math.sqrt(max(v, 0.0))
    return logistic(mu - z * sd), logistic(mu + z * sd)
