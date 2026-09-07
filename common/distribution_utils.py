import numpy as np
import scipy.special as sp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def simplex_to_xy(p):
    """
    Transforms a 3-dimensional Dirichlet / Multinomial point p = (p1, p2, p3)
    satisfying p1 + p2 + p3 = 1 into 2D Cartesian coordinates on an equilateral triangle.
    Vertices are:
    (0, 0) -> (1, 0, 0)
    (1, 0) -> (0, 1, 0)
    (0.5, sqrt(3)/2) -> (0, 0, 1)
    """
    p = np.asarray(p)
    # x = p2 + 0.5 * p3
    # y = sqrt(3)/2 * p3
    x = p[..., 1] + 0.5 * p[..., 2]
    y = (np.sqrt(3) / 2.0) * p[..., 2]
    return x, y


def plot_dirichlet_contour(alpha, n_grid=200, ax=None, cmap='viridis', levels=20):
    """
    Plots the contour of a Dirichlet distribution with parameter alpha (3-vector)
    on the standard 2-simplex (equilateral triangle).
    """
    if ax is None:
        ax = plt.gca()

    # Generate grid on simplex
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, np.sqrt(3)/2, n_grid)
    X, Y = np.meshgrid(x, y)

    # Invert (x, y) to (p1, p2, p3)
    # p3 = y / (sqrt(3)/2)
    # p2 = x - 0.5 * p3
    # p1 = 1 - p2 - p3
    P3 = Y / (np.sqrt(3) / 2.0)
    P2 = X - 0.5 * P3
    P1 = 1.0 - P2 - P3

    valid = (P1 >= 0) & (P2 >= 0) & (P3 >= 0)
    Z = np.zeros_like(X)

    # Dirichlet pdf = 1/B(alpha) * prod(p_i^{alpha_i - 1})
    log_b = np.sum(sp.gammaln(alpha)) - sp.gammaln(np.sum(alpha))
    
    # Calculate log pdf for valid points
    log_pdf = -log_b + (alpha[0] - 1) * np.log(np.maximum(P1, 1e-12)) \
                     + (alpha[1] - 1) * np.log(np.maximum(P2, 1e-12)) \
                     + (alpha[2] - 1) * np.log(np.maximum(P3, 1e-12))
    
    Z[valid] = np.exp(log_pdf[valid])
    Z[~valid] = np.nan

    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    
    # Draw triangle boundary
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', lw=1.5)
    ax.text(-0.05, -0.05, r'$\mathbf{x}_1$', fontsize=12)
    ax.text(1.02, -0.05, r'$\mathbf{x}_2$', fontsize=12)
    ax.text(0.48, np.sqrt(3)/2 + 0.03, r'$\mathbf{x}_3$', fontsize=12)
    ax.axis('off')
    ax.set_aspect('equal')
    return cs


def plot_gaussian_ellipse(mean, cov, ax=None, n_std=2.0, **kwargs):
    """
    Plots an ellipse representing the covariance of a 2D Gaussian distribution.
    """
    if ax is None:
        ax = plt.gca()

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def student_t_pdf(x, mu, lam, nu):
    """
    Univariate Student's t-distribution pdf:
    St(x | mu, lambda, nu) = Gamma(nu/2 + 1/2)/Gamma(nu/2) * (lambda / (pi * nu))^(1/2) * [1 + lambda(x-mu)^2/nu]^(-(nu+1)/2)
    """
    c = sp.gamma((nu + 1) / 2.0) / (sp.gamma(nu / 2.0) * np.sqrt(np.pi * nu / lam))
    return c * (1.0 + (lam * (x - mu)**2) / nu) ** (-(nu + 1) / 2.0)


def von_mises_pdf(theta, mu, kappa):
    """
    Von Mises distribution pdf for periodic variables:
    p(theta | mu, kappa) = 1 / (2*pi*I_0(kappa)) * exp(kappa * cos(theta - mu))
    """
    i0 = sp.i0(kappa)
    return np.exp(kappa * np.cos(theta - mu)) / (2.0 * np.pi * i0)


# ==============================================================================
# Comprehensive PRML Chapter 2 Probability Distribution Classes
# ==============================================================================

class Gaussian1D:
    """
    1次元ガウス分布 (Univariate Gaussian Distribution) - PRML Section 1.2.4 & 2.3
    
    尤度推定 (MLE)、共役事前分布によるベイズ更新（平均に対するガウス事前分布、精度に対するガンマ事前分布）、
    および事後予測分布の導出をサポート。
    """
    def __init__(self, mu=0.0, var=1.0):
        self.mu = float(mu)
        self.var = float(var)
        self.precision = 1.0 / self.var if self.var > 0 else np.inf

    def fit(self, X):
        """
        1次元データセット X から平均と分散の最尤推定量 (MLE) および不偏分散を算出。
        """
        X = np.asarray(X, dtype=float).ravel()
        N = len(X)
        if N == 0:
            raise ValueError("Data array X cannot be empty.")
        self.mu = float(np.mean(X))
        self.var = float(np.var(X, ddof=0))
        self.unbiased_var = float(np.var(X, ddof=1)) if N > 1 else self.var
        self.precision = 1.0 / self.var if self.var > 0 else np.inf
        return self

    def pdf(self, x):
        """確率密度関数 N(x | mu, var)"""
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * (x - self.mu)**2 / self.var) / np.sqrt(2.0 * np.pi * self.var)

    def log_pdf(self, x):
        """対数確率密度関数 ln N(x | mu, var)"""
        x = np.asarray(x, dtype=float)
        return -0.5 * np.log(2.0 * np.pi * self.var) - 0.5 * (x - self.mu)**2 / self.var

    @staticmethod
    def bayesian_mean_update(X, prior_mean, prior_var, noise_var):
        """
        分散既知における平均 mu の共役ベイズ更新 (PRML 式 2.140 - 2.141)
        事前分布: p(mu) = N(mu | mu_0, sigma_0^2)
        事後分布: p(mu | X) = N(mu | mu_N, sigma_N^2)
        """
        X = np.asarray(X, dtype=float).ravel()
        N = len(X)
        sample_mean = np.mean(X) if N > 0 else 0.0

        prec_0 = 1.0 / prior_var
        prec_noise = 1.0 / noise_var
        prec_N = prec_0 + N * prec_noise
        post_var = 1.0 / prec_N

        post_mean = post_var * (prec_0 * prior_mean + prec_noise * N * sample_mean)
        return Gaussian1D(mu=post_mean, var=post_var)

    @staticmethod
    def bayesian_precision_update(X, prior_a, prior_b, true_mean):
        """
        平均既知における精度 lambda = 1/sigma^2 の共役ベイズ更新 (PRML 式 2.149 - 2.151)
        事前分布: p(lambda) = Gam(lambda | a_0, b_0)
        事後分布: p(lambda | X) = Gam(lambda | a_N, b_N)
        """
        X = np.asarray(X, dtype=float).ravel()
        N = len(X)
        sq_err = np.sum((X - true_mean)**2)
        post_a = prior_a + 0.5 * N
        post_b = prior_b + 0.5 * sq_err
        return GammaDistribution(a=post_a, b=post_b)

    @staticmethod
    def predictive_distribution(prior_mean, prior_var, noise_var, N):
        """
        事後予測分布 p(x_new | X) (PRML 式 2.143)
        p(x_new | X) = N(x_new | mu_N, sigma_N^2 + sigma^2)
        """
        prec_0 = 1.0 / prior_var
        prec_noise = 1.0 / noise_var
        post_var = 1.0 / (prec_0 + N * prec_noise)
        pred_var = post_var + noise_var
        return pred_var


class MultivariateGaussian:
    """
    多変量ガウス分布 (Multivariate Gaussian Distribution) - PRML Section 2.3
    
    平均ベクトル mu, 共分散行列 Sigma に対する最尤推定、
    条件付き分布 p(x_a | x_b) の解析的導出 (式 2.81-2.82)、
    周辺分布 p(x_a) の導出 (式 2.98-2.99)、
    線形ガウスモデル (Linear Gaussian Systems, 式 2.113-2.115) を完備。
    """
    def __init__(self, mean=None, cov=None):
        if mean is not None:
            self.mean = np.asarray(mean, dtype=float)
            self.dim = self.mean.shape[0]
        else:
            self.mean = None
            self.dim = None

        if cov is not None:
            self.cov = np.asarray(cov, dtype=float)
            self._update_cov()
        else:
            self.cov = None
            self.prec = None

    def _update_cov(self):
        self.cov_det = np.linalg.det(self.cov)
        self.prec = np.linalg.inv(self.cov)

    def fit(self, X):
        """N x D の観測データ行列から MLE 平均ベクトルと共分散行列を算出"""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        self.dim = X.shape[1]
        self.mean = np.mean(X, axis=0)
        diff = X - self.mean
        self.cov = (diff.T @ diff) / X.shape[0]
        self._update_cov()
        return self

    def mahalanobis_distance(self, X):
        """マハラノビス二乗距離 Delta^2 = (x - mu)^T Sigma^-1 (x - mu) (PRML 式 2.44)"""
        X = np.asarray(X, dtype=float)
        single = (X.ndim == 1)
        if single:
            X = X[np.newaxis, :]
        diff = X - self.mean
        dist_sq = np.sum((diff @ self.prec) * diff, axis=1)
        return dist_sq[0] if single else dist_sq

    def pdf(self, X):
        """確率密度関数 N(x | mu, Sigma)"""
        X = np.asarray(X, dtype=float)
        single = (X.ndim == 1)
        if single:
            X = X[np.newaxis, :]
        D = self.dim
        norm = 1.0 / np.sqrt(((2.0 * np.pi) ** D) * np.maximum(self.cov_det, 1e-30))
        dist_sq = self.mahalanobis_distance(X)
        res = norm * np.exp(-0.5 * dist_sq)
        return res[0] if single else res

    def log_pdf(self, X):
        """対数確率密度関数"""
        X = np.asarray(X, dtype=float)
        single = (X.ndim == 1)
        if single:
            X = X[np.newaxis, :]
        D = self.dim
        dist_sq = self.mahalanobis_distance(X)
        res = -0.5 * (D * np.log(2.0 * np.pi) + np.log(np.maximum(self.cov_det, 1e-30)) + dist_sq)
        return res[0] if single else res

    def condition(self, x_b, idx_a, idx_b):
        """
        条件付きガウス分布 p(x_a | x_b) の解析的算出 (PRML 式 2.81 - 2.82)
        mu_{a|b} = mu_a + Sigma_{ab} Sigma_{bb}^-1 (x_b - mu_b)
        Sigma_{a|b} = Sigma_{aa} - Sigma_{ab} Sigma_{bb}^-1 Sigma_{ba}
        """
        idx_a = np.asarray(idx_a, dtype=int)
        idx_b = np.asarray(idx_b, dtype=int)
        x_b = np.asarray(x_b, dtype=float).ravel()

        mu_a = self.mean[idx_a]
        mu_b = self.mean[idx_b]

        Sigma_aa = self.cov[np.ix_(idx_a, idx_a)]
        Sigma_ab = self.cov[np.ix_(idx_a, idx_b)]
        Sigma_ba = self.cov[np.ix_(idx_b, idx_a)]
        Sigma_bb = self.cov[np.ix_(idx_b, idx_b)]

        Sigma_bb_inv = np.linalg.inv(Sigma_bb)
        cond_mean = mu_a + Sigma_ab @ Sigma_bb_inv @ (x_b - mu_b)
        cond_cov = Sigma_aa - Sigma_ab @ Sigma_bb_inv @ Sigma_ba

        return MultivariateGaussian(mean=cond_mean, cov=cond_cov)

    def marginalize(self, idx_a):
        """
        周辺ガウス分布 p(x_a) の算出 (PRML 式 2.98 - 2.99)
        E[x_a] = mu_a
        cov[x_a] = Sigma_{aa}
        """
        idx_a = np.asarray(idx_a, dtype=int)
        marg_mean = self.mean[idx_a]
        marg_cov = self.cov[np.ix_(idx_a, idx_a)]
        return MultivariateGaussian(mean=marg_mean, cov=marg_cov)

    @staticmethod
    def linear_gaussian_system(mu_x, sigma_x, A, b, L_cov):
        """
        線形ガウスモデル (Linear Gaussian Systems) - PRML 式 2.113 - 2.115
        p(x) = N(x | mu_x, sigma_x)
        p(y | x) = N(y | A x + b, L_cov)
        
        返り値:
        - marginal_y: 周辺分布 p(y) = N(y | A mu_x + b, L_cov + A sigma_x A^T)
        - posterior_solver: 観測 y を受け取って事後分布 p(x | y) = N(x | mu_{x|y}, Sigma_{x|y}) を返す関数
        """
        mu_x = np.asarray(mu_x, dtype=float)
        sigma_x = np.asarray(sigma_x, dtype=float)
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        L_cov = np.asarray(L_cov, dtype=float)

        # Marginal p(y)
        mu_y = A @ mu_x + b
        cov_y = L_cov + A @ sigma_x @ A.T
        marginal_y = MultivariateGaussian(mean=mu_y, cov=cov_y)

        # Posterior covariance Sigma_{x|y} = (sigma_x^-1 + A^T L^-1 A)^-1
        sigma_x_inv = np.linalg.inv(sigma_x)
        L_inv = np.linalg.inv(L_cov)
        sigma_x_given_y = np.linalg.inv(sigma_x_inv + A.T @ L_inv @ A)

        def posterior_given_y(y):
            y = np.asarray(y, dtype=float)
            mu_x_given_y = sigma_x_given_y @ (A.T @ L_inv @ (y - b) + sigma_x_inv @ mu_x)
            return MultivariateGaussian(mean=mu_x_given_y, cov=sigma_x_given_y)

        return marginal_y, posterior_given_y


class BetaDistribution:
    """
    ベータ分布 (Beta Distribution) - PRML Section 2.1.3
    
    Beta(mu | a, b) = Gamma(a+b)/(Gamma(a)Gamma(b)) * mu^(a-1) * (1-mu)^(b-1)
    二項尤度に対する共役事前分布、逐次ベイズ更新および事後予測分布をサポート。
    """
    def __init__(self, a=1.0, b=1.0):
        self.a = float(a)
        self.b = float(b)

    @property
    def mean(self):
        """期待値 E[mu] = a / (a + b) (式 2.15)"""
        return self.a / (self.a + self.b)

    @property
    def variance(self):
        """分散 var[mu] = ab / ((a+b)^2 (a+b+1)) (式 2.16)"""
        ab = self.a + self.b
        return (self.a * self.b) / (ab**2 * (ab + 1.0))

    @property
    def mode(self):
        """最頻値 (a > 1 かつ b > 1 のとき (a-1)/(a+b-2))"""
        if self.a > 1.0 and self.b > 1.0:
            return (self.a - 1.0) / (self.a + self.b - 2.0)
        return np.nan

    def pdf(self, mu):
        """確率密度関数"""
        mu = np.asarray(mu, dtype=float)
        valid = (mu >= 0.0) & (mu <= 1.0)
        res = np.zeros_like(mu)
        log_b = sp.gammaln(self.a) + sp.gammaln(self.b) - sp.gammaln(self.a + self.b)
        log_p = -log_b + (self.a - 1.0) * np.log(np.maximum(mu, 1e-12)) + (self.b - 1.0) * np.log(np.maximum(1.0 - mu, 1e-12))
        res[valid] = np.exp(log_p[valid])
        return res

    def bayesian_update(self, n_heads, n_tails):
        """
        観測データ (表 n_heads 回, 裏 n_tails 回) による事後分布更新 (PRML 式 2.18)
        Beta(mu | a + m, b + l)
        """
        return BetaDistribution(a=self.a + n_heads, b=self.b + n_tails)

    def predictive_probability(self):
        """事後予測確率 p(x=1 | D) = a / (a + b) (PRML 式 2.19)"""
        return self.mean


class DirichletDistribution:
    """
    ディリクレ分布 (Dirichlet Distribution) - PRML Section 2.2.1
    
    Dir(mu | alpha) = 1/B(alpha) * prod_{k=1}^K mu_k^{alpha_k - 1}
    多項尤度に対する共役事前分布、モーメント、最頻値、ベイズ事後更新をサポート。
    """
    def __init__(self, alpha):
        self.alpha = np.asarray(alpha, dtype=float)
        self.alpha_0 = np.sum(self.alpha)
        self.K = len(self.alpha)

    @property
    def mean(self):
        """期待値 E[mu_k] = alpha_k / alpha_0 (PRML 式 2.39)"""
        return self.alpha / self.alpha_0

    @property
    def variance(self):
        """各成分の分散 var[mu_k] = alpha_k (alpha_0 - alpha_k) / (alpha_0^2 (alpha_0 + 1))"""
        return (self.alpha * (self.alpha_0 - self.alpha)) / (self.alpha_0**2 * (self.alpha_0 + 1.0))

    @property
    def covariance_matrix(self):
        """共分散行列 cov[mu_j, mu_k]"""
        cov = np.zeros((self.K, self.K))
        denom = (self.alpha_0**2 * (self.alpha_0 + 1.0))
        for j in range(self.K):
            for k in range(self.K):
                if j == k:
                    cov[j, k] = (self.alpha[j] * (self.alpha_0 - self.alpha[j])) / denom
                else:
                    cov[j, k] = -(self.alpha[j] * self.alpha[k]) / denom
        return cov

    @property
    def mode(self):
        """最頻値 (all alpha_k > 1 のとき (alpha_k - 1) / (alpha_0 - K)) (Exercise 2.12)"""
        if np.all(self.alpha > 1.0):
            return (self.alpha - 1.0) / (self.alpha_0 - self.K)
        return None

    def bayesian_update(self, counts):
        """多項観測頻度 m による事後分布更新: Dir(mu | alpha + m) (PRML 式 2.41)"""
        counts = np.asarray(counts, dtype=float)
        return DirichletDistribution(alpha=self.alpha + counts)

    def predictive_probability(self):
        """事後予測確率 p(x_k = 1 | D) = alpha_k / alpha_0"""
        return self.mean


class GammaDistribution:
    """
    ガンマ分布 (Gamma Distribution) - PRML Section 2.3.6 (式 2.146)
    
    Gam(lambda | a, b) = 1/Gamma(a) * b^a * lambda^(a-1) * exp(-b * lambda)
    ガウス分布の精度パラメータ lambda に対する共役事前分布。
    """
    def __init__(self, a=1.0, b=1.0):
        self.a = float(a)
        self.b = float(b)

    @property
    def mean(self):
        """期待値 E[lambda] = a / b (式 2.147)"""
        return self.a / self.b

    @property
    def variance(self):
        """分散 var[lambda] = a / b^2 (式 2.148)"""
        return self.a / (self.b**2)

    @property
    def mode(self):
        """最頻値 (a >= 1 のとき (a - 1) / b)"""
        if self.a >= 1.0:
            return (self.a - 1.0) / self.b
        return np.nan

    def pdf(self, lam):
        """確率密度関数"""
        lam = np.asarray(lam, dtype=float)
        valid = (lam > 0)
        res = np.zeros_like(lam)
        log_p = self.a * np.log(self.b) - sp.gammaln(self.a) + (self.a - 1.0) * np.log(np.maximum(lam, 1e-12)) - self.b * lam
        res[valid] = np.exp(log_p[valid])
        return res


class StudentsTDistribution:
    """
    スチューデントのt分布 (Student's t-Distribution) - PRML Section 2.3.7
    
    無限個のガウス分布のガンマ混合（精度母数の積分消去）として導出されるロバスト分布。
    外れ値 (outliers) に対してガウス分布よりも極めて頑健な特性を持つ。
    """
    def __init__(self, mu=0.0, lam=1.0, nu=1.0):
        self.mu = float(mu)
        self.lam = float(lam)  # scale/precision parameter
        self.nu = float(nu)    # degrees of freedom (自由度)

    @property
    def mean(self):
        """平均 (nu > 1 のとき mu)"""
        return self.mu if self.nu > 1.0 else np.nan

    @property
    def variance(self):
        """分散 (nu > 2 のとき nu / (lam * (nu - 2))) (PRML 式 2.161)"""
        return (self.nu / (self.lam * (self.nu - 2.0))) if self.nu > 2.0 else np.inf

    @property
    def mode(self):
        """最頻値 mu"""
        return self.mu

    def pdf(self, x):
        """確率密度関数 (式 2.159)"""
        return student_t_pdf(x, self.mu, self.lam, self.nu)


class VonMisesDistribution:
    """
    フォン・ミーゼス分布 (Von Mises Distribution) - PRML Section 2.3.8
    
    周期変数 theta in [0, 2*pi) のための円周分布（正規分布の円周版）。
    p(theta | mu, kappa) = 1 / (2*pi*I_0(kappa)) * exp(kappa * cos(theta - mu))
    """
    def __init__(self, mu=0.0, kappa=1.0):
        self.mu = float(mu)
        self.kappa = float(kappa)

    def pdf(self, theta):
        """確率密度関数"""
        return von_mises_pdf(theta, self.mu, self.kappa)

    def fit(self, thetas):
        """
        周期データ thetas から平均方向 mu と集中度 kappa の最尤推定 (PRML 式 2.168 - 2.171)
        bar{r} = A(kappa) = I_1(kappa) / I_0(kappa)
        """
        thetas = np.asarray(thetas, dtype=float)
        bar_x = np.mean(np.cos(thetas))
        bar_y = np.mean(np.sin(thetas))
        self.mu = float(np.arctan2(bar_y, bar_x)) % (2.0 * np.pi)

        bar_r = float(np.sqrt(bar_x**2 + bar_y**2))

        # A(kappa) = bar_r を解く
        if bar_r < 1e-6:
            self.kappa = 0.0
        elif bar_r > 0.999:
            self.kappa = 100.0
        else:
            def obj(k):
                return sp.i1(k) / sp.i0(k) - bar_r
            sol = root_scalar(obj, bracket=[1e-4, 50.0], method='brentq')
            self.kappa = float(sol.root)
        return self


class KernelDensityEstimator:
    """
    カーネル密度推定量 (Kernel Density Estimator / Parzen Window) - PRML Section 2.5.1
    
    p(x) = 1/N * sum_{n=1}^N 1/h^D * k((x - x_n)/h)
    サポートカーネル: 'gaussian', 'box' (tophat), 'epanechnikov'
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = float(bandwidth)
        self.kernel = kernel.lower()
        self.X_train = None
        self.N = None
        self.D = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        self.X_train = X
        self.N, self.D = X.shape
        return self

    def _kernel_val(self, u_sq):
        """u_sq = ||(x - x_n)/h||^2 に対応するカーネル値"""
        if self.kernel == 'gaussian':
            norm = (2.0 * np.pi) ** (-self.D / 2.0)
            return norm * np.exp(-0.5 * u_sq)
        elif self.kernel in ['box', 'tophat']:
            # 球内 (u <= 1) で均一密度
            vol = (np.pi ** (self.D / 2.0)) / sp.gamma(self.D / 2.0 + 1.0)
            return np.where(u_sq <= 1.0, 1.0 / vol, 0.0)
        elif self.kernel == 'epanechnikov':
            # 1 - u^2 (放物線)
            vol = (np.pi ** (self.D / 2.0)) / sp.gamma(self.D / 2.0 + 1.0)
            factor = (self.D + 2.0) / (2.0 * vol)
            return np.where(u_sq <= 1.0, factor * (1.0 - u_sq), 0.0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def score_samples(self, X_eval):
        """評価点での確率密度 p(x) を算出"""
        X_eval = np.asarray(X_eval, dtype=float)
        single = (X_eval.ndim == 1 and self.D > 1)
        if X_eval.ndim == 1:
            if self.D == 1:
                X_eval = X_eval[:, np.newaxis]
            else:
                X_eval = X_eval[np.newaxis, :]

        # (M, N, D) の差分
        diff = (X_eval[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) / self.bandwidth
        u_sq = np.sum(diff**2, axis=-1)  # (M, N)
        k_vals = self._kernel_val(u_sq)   # (M, N)
        densities = np.mean(k_vals, axis=1) / (self.bandwidth ** self.D)
        return densities[0] if single else densities

    @staticmethod
    def silverman_bandwidth(X):
        """シルバマンの経験則 (Silverman's rule of thumb for 1D Gaussian KDE)"""
        X = np.asarray(X, dtype=float).ravel()
        std = np.std(X, ddof=1)
        N = len(X)
        return 1.06 * std * (N ** (-1.0 / 5.0))


class KNearestNeighborsDensity:
    """
    K近傍密度推定量 (K-Nearest-Neighbours Density Estimator) - PRML Section 2.5.2
    
    p(x) = K / (N * V(x))
    各点 x を中心として K 個の訓練点を含む超球の体積 V(x) を用いて適応的に密度推定を行う。
    """
    def __init__(self, k=5):
        self.k = int(k)
        self.X_train = None
        self.N = None
        self.D = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        self.X_train = X
        self.N, self.D = X.shape
        if self.k > self.N:
            raise ValueError(f"k ({self.k}) cannot be larger than sample size N ({self.N})")
        return self

    def score_samples(self, X_eval):
        """評価点での密度 p(x) = K / (N * V_k)"""
        X_eval = np.asarray(X_eval, dtype=float)
        single = (X_eval.ndim == 1 and self.D > 1)
        if X_eval.ndim == 1:
            if self.D == 1:
                X_eval = X_eval[:, np.newaxis]
            else:
                X_eval = X_eval[np.newaxis, :]

        # ユークリッド距離計算
        dists = np.linalg.norm(X_eval[:, np.newaxis, :] - self.X_train[np.newaxis, :, :], axis=-1)
        # 各行をソートして K 番目に近い距離 R_k を取得 (0-indexed: dists_sorted[:, k-1])
        dists_sorted = np.sort(dists, axis=1)
        r_k = dists_sorted[:, self.k - 1]
        r_k = np.maximum(r_k, 1e-10)

        # D次元球の体積 V = pi^(D/2) / Gamma(D/2 + 1) * R^D
        unit_vol = (np.pi ** (self.D / 2.0)) / sp.gamma(self.D / 2.0 + 1.0)
        v_k = unit_vol * (r_k ** self.D)

        densities = self.k / (self.N * v_k)
        return densities[0] if single else densities


class RobbinsMonro:
    """
    ロビンス・モンロー逐次確率近似アルゴリズム (Robbins-Monro Algorithm) - PRML Section 2.3.5
    
    回帰関数 f(theta) = E[z | theta] = 0 となる根 theta* を逐次的に推定するアルゴリズム。
    更新式: theta^{(N)} = theta^{(N-1)} - a_{N-1} z(x_N, theta^{(N-1)})
    係数列の収束条件: sum a_N = inf, sum a_N^2 < inf
    """
    def __init__(self, a_coeff=1.0):
        self.a_coeff = float(a_coeff)
        self.step_count = 0
        self.theta = None

    def reset(self, init_theta=0.0):
        self.theta = float(init_theta)
        self.step_count = 0
        return self

    def step(self, z):
        """1ステップのパラメータ更新"""
        self.step_count += 1
        a_n = self.a_coeff / self.step_count
        self.theta = self.theta - a_n * z
        return self.theta

    def estimate_mean(self, X, init_mu=0.0):
        """
        ガウス平均推定への適用 (z = theta - x)
        E[z | theta] = theta - mu_true = 0
        """
        X = np.asarray(X, dtype=float).ravel()
        self.reset(init_mu)
        history = [self.theta]
        for x in X:
            # z = theta - x
            z = self.theta - x
            self.step(z)
            history.append(self.theta)
        return self.theta, np.array(history)
