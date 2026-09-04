import numpy as np
import scipy.spatial.distance as sp_dist
import scipy.optimize as opt
from common.classification_utils import sigmoid

def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0, gamma=None, h=None):
    """
    動径基底関数 (RBF / ガウス) カーネル
    k(x, x') = variance * exp(- 0.5 * ||x - x'||^2 / length_scale^2)
    gamma: exp(- gamma * ||x - x'||^2)
    h: バンド幅パラメータ (h = length_scale)
    """
    if h is not None:
        length_scale = h
    elif gamma is not None:
        length_scale = 1.0 / np.sqrt(2.0 * gamma)

    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    dists_sq = sp_dist.cdist(X1, X2, metric='sqeuclidean')
    return variance * np.exp(-0.5 * dists_sq / (length_scale**2))


def ard_kernel(X1, X2, theta0, etas):
    """
    自動適合性決定 (ARD) カーネル (PRML 式 6.79)
    k(x, x') = theta0 * exp( - 0.5 * sum_i eta_i (x_i - x'_i)^2 )
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    # スケーリングされた入力
    scaled_X1 = X1 * np.sqrt(etas)
    scaled_X2 = X2 * np.sqrt(etas)
    dists_sq = sp_dist.cdist(scaled_X1, scaled_X2, metric='sqeuclidean')
    return theta0 * np.exp(-0.5 * dists_sq)

def prml_regression_kernel(X1, X2, theta0=1.0, theta1=4.0, theta2=0.0, theta3=0.0):
    """
    PRML 式 6.63 の汎用カーネル関数
    k(x, x') = theta0 * exp(-0.5 * theta1 * ||x - x'||^2) + theta2 + theta3 * x^T x'
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    dists_sq = sp_dist.cdist(X1, X2, metric='sqeuclidean')
    linear_term = X1 @ X2.T
    return theta0 * np.exp(-0.5 * theta1 * dists_sq) + theta2 + theta3 * linear_term

def linear_kernel(X1, X2):

    """線形カーネル k(x, x') = x^T x'"""
    return np.atleast_2d(X1) @ np.atleast_2d(X2).T

def resolve_kernel(kernel):
    """文字列または関数から適切なカーネル関数を解決"""
    if callable(kernel):
        return kernel
    if isinstance(kernel, str):
        k_str = kernel.lower()
        if k_str in ('rbf', 'gaussian'):
            return rbf_kernel
        elif k_str == 'linear':
            return linear_kernel
        elif k_str == 'prml':
            return prml_regression_kernel
    raise ValueError(f"Unknown kernel: {kernel}")

class KernelRidgeRegression:
    """
    双対表現に基づく正則化最小二乗法 (Kernel Ridge Regression, PRML 6.1節)
    a = (K + lambda * I)^(-1) * t
    y(x) = k(x)^T a
    """
    def __init__(self, kernel='rbf', reg_lambda=1e-3, **kernel_kwargs):
        self.kernel = resolve_kernel(kernel)
        self.reg_lambda = reg_lambda
        self.kernel_kwargs = kernel_kwargs
        self.X_train = None
        self.a = None


    def fit(self, X, t):
        self.X_train = np.atleast_2d(X)
        N = len(self.X_train)
        K = self.kernel(self.X_train, self.X_train, **self.kernel_kwargs)
        self.a = np.linalg.solve(K + self.reg_lambda * np.eye(N), t)
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        k_star = self.kernel(X, self.X_train, **self.kernel_kwargs) # (N_test, N_train)
        return k_star @ self.a

class NadarayaWatsonRegressor:
    """
    Nadaraya-Watson カーネル回帰モデル (PRML 6.3.1節)
    y(x) = sum_n k(x, x_n) * t_n / sum_m k(x, x_m)
    """
    def __init__(self, kernel='rbf', **kernel_kwargs):
        self.kernel = resolve_kernel(kernel)
        self.kernel_kwargs = kernel_kwargs
        self.X_train = None
        self.t_train = None

    def fit(self, X, t):
        self.X_train = np.atleast_2d(X)
        self.t_train = np.asarray(t).ravel()
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        K_cross = self.kernel(X, self.X_train, **self.kernel_kwargs) # (N_test, N_train)
        # 各行の重みの和で規格化
        weights = K_cross / (np.sum(K_cross, axis=1, keepdims=True) + 1e-12)
        return weights @ self.t_train

class GaussianProcessRegressor:
    """
    ガウス過程回帰 (Gaussian Process Regression: GPR, PRML 6.4節)
    """
    def __init__(self, kernel='prml', beta=25.0, **kernel_kwargs):
        self.kernel = resolve_kernel(kernel)

        self.beta = beta # ノイズ精度 (ノイズ分散 sigma_n^2 = 1/beta)
        self.kernel_kwargs = kernel_kwargs
        self.X_train = None
        self.t_train = None
        self.C_N = None
        self.C_N_inv = None

    def fit(self, X, t):
        self.X_train = np.atleast_2d(X)
        self.t_train = np.asarray(t).ravel()
        N = len(self.X_train)
        K = self.kernel(self.X_train, self.X_train, **self.kernel_kwargs)
        self.C_N = K + (1.0 / self.beta) * np.eye(N)
        self.C_N_inv = np.linalg.pinv(self.C_N)
        return self

    def predict(self, X, return_std=True, return_cov=False):
        X = np.atleast_2d(X)
        N_test = len(X)
        # 共分散ベクトル・行列の計算
        k_star = self.kernel(X, self.X_train, **self.kernel_kwargs) # (N_test, N_train)
        c_star = self.kernel(X, X, **self.kernel_kwargs) + (1.0 / self.beta) * np.eye(N_test) # (N_test, N_test)
        
        # 予測分布の平均 (PRML 式 6.66)
        mu = k_star @ self.C_N_inv @ self.t_train
        
        # 予測分布の共分散 (PRML 式 6.67)
        cov = c_star - k_star @ self.C_N_inv @ k_star.T
        # 数値的安定化
        cov = 0.5 * (cov + cov.T)
        var = np.maximum(np.diag(cov), 1e-10)
        
        if return_cov:
            return mu, cov
        if return_std:
            return mu, np.sqrt(var)
        return mu

    def log_marginal_likelihood(self, theta_params=None):
        """
        対数周辺尤度 (PRML 式 6.69)
        ln p(t|theta) = -0.5 * ln|C_N| - 0.5 * t^T C_N^(-1) t - 0.5 * N * ln(2pi)
        """
        if theta_params is not None:
            # パラメータを更新して評価
            kwargs = self.kernel_kwargs.copy()
            # theta_params は辞書
            kwargs.update(theta_params)
            N = len(self.X_train)
            K = self.kernel(self.X_train, self.X_train, **kwargs)
            C_N = K + (1.0 / self.beta) * np.eye(N)
        else:
            C_N = self.C_N
            N = len(self.X_train)
            
        sign, logdet = np.linalg.slogdet(C_N)
        if sign <= 0:
            return -1e10
        inv_CN = np.linalg.pinv(C_N)
        data_fit = -0.5 * self.t_train @ inv_CN @ self.t_train
        complexity = -0.5 * logdet
        const = -0.5 * N * np.log(2.0 * np.pi)
        return data_fit + complexity + const

class GaussianProcessClassifier:
    """
    ガウス過程分類 (Gaussian Process Classification: GPC, PRML 6.4.5-6.4.6節)
    潜在変数 a(x) に対する GP 事前分布 + シグモイド尤度 + ラプラス近似
    """
    def __init__(self, kernel='rbf', **kernel_kwargs):
        self.kernel = resolve_kernel(kernel)
        self.kernel_kwargs = kernel_kwargs
        self.X_train = None
        self.t_train = None
        self.a_map = None
        self.K_N = None
        self.W = None
        self.H_inv = None

    def fit(self, X, t, max_iter=100, tol=1e-5):
        self.X_train = np.atleast_2d(X)
        self.t_train = np.asarray(t).ravel() # t in {0, 1}
        N = len(self.X_train)
        self.K_N = self.kernel(self.X_train, self.X_train, **self.kernel_kwargs) + 1e-6 * np.eye(N)
        K_inv = np.linalg.pinv(self.K_N)
        
        # ラプラス近似のための最頻値 a_map の Newton-Raphson 更新 (PRML 式 6.83)
        a = np.zeros(N)
        for i in range(max_iter):
            sigma_a = sigmoid(a)
            # W は対角行列 W_nn = sigma_a_n * (1 - sigma_a_n)
            W_diag = sigma_a * (1.0 - sigma_a)
            # 勾配: grad = t - sigma(a) - K^(-1) a
            # Newton-Raphson: a_new = K (I + W K)^(-1) [t - sigma(a) + W a]
            W_half = np.sqrt(W_diag)
            B = np.eye(N) + (W_half[:, None] * self.K_N) * W_half[None, :] # B = I + W^(1/2) K W^(1/2)
            
            # 安定的な中間計算
            b = W_diag * a + (self.t_train - sigma_a)
            # a_new = K * (b - W^(1/2) B^(-1) W^(1/2) K b)
            # または直接 (K^(-1) + W)^(-1) (t - sigma + W a)
            inv_K_plus_W = np.linalg.pinv(K_inv + np.diag(W_diag))
            a_new = inv_K_plus_W @ (self.t_train - sigma_a + W_diag * a)
            
            if np.max(np.abs(a_new - a)) < tol:
                a = a_new
                break
            a = a_new
            
        self.a_map = a
        self.W_diag = sigmoid(a) * (1.0 - sigmoid(a))
        # 事後共分散行列 H^(-1) = (K^(-1) + W)^(-1)
        self.H_inv = np.linalg.pinv(K_inv + np.diag(self.W_diag))
        return self

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        N_test = len(X)
        k_star = self.kernel(X, self.X_train, **self.kernel_kwargs) # (N_test, N_train)
        c_star = np.diag(self.kernel(X, X, **self.kernel_kwargs))   # (N_test,)
        
        # 潜在関数 a* の平均と分散 (PRML 式 6.87, 6.88)
        K_inv = np.linalg.pinv(self.K_N)
        mu_a = k_star @ (self.t_train - sigmoid(self.a_map))
        
        # 分散: var_a = c* - k* (K + W^(-1))^(-1) k*^T
        W_inv_diag = 1.0 / (self.W_diag + 1e-12)
        inv_term = np.linalg.pinv(self.K_N + np.diag(W_inv_diag))
        sigma_a2 = c_star - np.sum((k_star @ inv_term) * k_star, axis=1)
        sigma_a2 = np.maximum(sigma_a2, 1e-10)
        
        # プロビット畳み込みによる軟化された予測確率 (PRML 式 6.89, 4.153)
        kappa = 1.0 / np.sqrt(1.0 + np.pi * sigma_a2 / 8.0)
        p_C1 = sigmoid(kappa * mu_a)
        return p_C1

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
