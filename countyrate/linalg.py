"""极小线性代数工具（纯标准库）。

只实现本项目需要的运算：矩阵乘法、线性方程组求解（高斯消元 + 部分主元）。
所有矩阵用 list[list[float]] 表示，向量用 list[float]。
维度都很小（特征数 k=4），无需 numpy。
"""


def mat_vec_mul(A, x):
    """A (m×n) 乘向量 x (n) -> 向量 (m)。"""
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in A]


def solve(A, b):
    """解线性方程组 A x = b，高斯消元 + 部分主元。

    A 为 n×n 可逆方阵。返回解向量。奇异时抛 ValueError。
    """
    n = len(A)
    # 增广矩阵，避免改动入参
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        # 部分主元
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14:
            raise ValueError("矩阵奇异，无法求解")
        M[col], M[piv] = M[piv], M[col]
        inv = 1.0 / M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                factor = M[r][col] * inv
                for c in range(col, n + 1):
                    M[r][c] -= factor * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def weighted_least_squares(X, y, w):
    """加权最小二乘：min sum_i w_i (y_i - x_i'b)^2。

    返回 beta = (X'WX)^{-1} X'W y。
    """
    k = len(X[0])
    XtWX = [[0.0] * k for _ in range(k)]
    XtWy = [0.0] * k
    for xi, yi, wi in zip(X, y, w):
        for a in range(k):
            XtWy[a] += wi * xi[a] * yi
            for b in range(k):
                XtWX[a][b] += wi * xi[a] * xi[b]
    return solve(XtWX, XtWy)
