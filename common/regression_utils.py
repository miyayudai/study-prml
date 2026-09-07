"""
PRML Chapter 3 Linear Models for Regression Utilities
Bishop「パターン認識と機械学習」第3章に基づく厳密かつ高精度なNumPy実装
基底関数、最尤推定、重み付き最小二乗、L2正則化(Ridge)、L1正則化(Lasso)、オンライン逐次学習(LMS)、
ベイズ線形回帰、共役正規ガンマモデル、等価カーネル、エビデンス近似、バイアス・バリアンス分解、
局所重み付き回帰、ロバスト回帰 (Huber IRLS)
"""

import numpy as np
from scipy.special import gamma as gamma_func


class PolynomialBasis:
    """多項式基底関数: phi_j(x) = x^j (j=0, ..., M)"""
    def __init__(self, degree):
        self.degree = int(degree)

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        Phi = np.hstack([X**i for i in range(self.degree + 1)])
        return Phi

    def transform(self, X):
        return self(X)


class GaussianBasis:
    """ガウス基底関数: phi_j(x) = exp(- (x - mu_j)^2 / (2 * s^2))"""
    def __init__(self, centers, scale):
        self.centers = np.asarray(centers, dtype=float)
        self.scale = float(scale)

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        Phi = [np.ones((len(X), 1))]
        for mu in self.centers:
            diff = X - mu
            dist_sq = np.sum(diff**2, axis=-1, keepdims=True)
            phi_j = np.exp(-0.5 * dist_sq / (self.scale ** 2))
            Phi.append(phi_j)
        return np.hstack(Phi)

    def transform(self, X):
        return self(X)


class SigmoidalBasis:
    """シグモイド基底関数: phi_j(x) = sigma((x - mu_j) / s)"""
    def __init__(self, centers, scale):
        self.centers = np.asarray(centers, dtype=float)
        self.scale = float(scale)

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        Phi = [np.ones((len(X), 1))]
        for mu in self.centers:
            a = (X - mu) / self.scale
            if a.ndim > 1 and a.shape[1] > 1:
                a = np.sum(a, axis=1, keepdims=True)
            phi_j = 1.0 / (1.0 + np.exp(-np.clip(a, -50.0, 50.0)))
            Phi.append(phi_j)
        return np.hstack(Phi)

    def transform(self, X):
        return self(X)


class FourierBasis:
    """フーリエ基底関数: phi(x) = [1, cos(k omega x), sin(k omega x), ...]"""
    def __init__(self, n_components=2, period=2.0, n_harmonics=None):
        self.n_components = int(n_harmonics if n_harmonics is not None else n_components)
        self.period = float(period)

    def __call__(self, X):
        X = np.asarray(X, dtype=float).ravel()
        omega = 2.0 * np.pi / self.period
        Phi = [np.ones((len(X), 1))]
        for k in range(1, self.n_components + 1):
            Phi.append(np.cos(k * omega * X)[:, None])
            Phi.append(np.sin(k * omega * X)[:, None])
        return np.hstack(Phi)

    def transform(self, X):
        return self(X)


class LinearRegression:
    """PRML 3.1.1節 & 3.1.5節に基づく線形回帰モデル（最尤推定 / 最小二乗解）
    単一出力および多次元目的変数 T in R^(N x K) の双方に完全対応
    """
    def __init__(self):
        self.w = None
        self.sigma2_ = None

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float)
        # w_ML = pinv(Phi) @ t (正規方程式解)
        self.w = np.linalg.pinv(Phi) @ t
        
        residuals = t - Phi @ self.w
        if t.ndim == 1:
            self.sigma2_ = float(np.mean(residuals**2))
        else:
            self.sigma2_ = np.mean(residuals**2, axis=0)
        return self

    def predict(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ self.w

    def score(self, Phi, y):
        """決定係数 R^2 スコア"""
        Phi = np.asarray(Phi, dtype=float)
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(Phi)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y, axis=0))**2)
        return float(1.0 - ss_res / max(ss_tot, 1e-12))

    def log_likelihood(self, Phi, t):
        """最尤推定における対数尤度 ln p(t | w_ML, beta_ML)"""
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        N = len(t)
        residuals = t - self.predict(Phi)
        sigma2 = float(np.mean(residuals**2))
        sigma2 = max(sigma2, 1e-12)
        return float(-0.5 * N * (np.log(2.0 * np.pi * sigma2) + 1.0))

    def aic(self, Phi, y):
        """赤池情報量規準 (Akaike Information Criterion): AIC = 2k - 2 ln(L)"""
        Phi = np.asarray(Phi, dtype=float)
        y = np.asarray(y, dtype=float)
        N, M = Phi.shape
        residuals = y - self.predict(Phi)
        sigma2 = float(np.mean(residuals**2))
        sigma2 = max(sigma2, 1e-12)
        log_lik = -0.5 * N * (np.log(2.0 * np.pi * sigma2) + 1.0)
        k = M + 1  # 重み M 個 + 分散パラメータ 1 個
        return float(2.0 * k - 2.0 * log_lik)

    def bic(self, Phi, y):
        """ベイズ情報量規準 (Bayesian Information Criterion): BIC = k ln(N) - 2 ln(L)"""
        Phi = np.asarray(Phi, dtype=float)
        y = np.asarray(y, dtype=float)
        N, M = Phi.shape
        residuals = y - self.predict(Phi)
        sigma2 = float(np.mean(residuals**2))
        sigma2 = max(sigma2, 1e-12)
        log_lik = -0.5 * N * (np.log(2.0 * np.pi * sigma2) + 1.0)
        k = M + 1
        return float(k * np.log(N) - 2.0 * log_lik)


MultivariateLinearRegression = LinearRegression


class WeightedLinearRegression:
    """PRML 3.1.1節 演習 3.3: 重み付き最小二乗法 (Weighted Least Squares)
    w = (Phi^T W Phi)^-1 Phi^T W t
    """
    def __init__(self):
        self.w = None

    def fit(self, Phi, t, weights=None):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        if weights is None:
            weights = np.ones(len(t))
        weights = np.asarray(weights, dtype=float).ravel()
        W = np.diag(weights)
        self.w = np.linalg.pinv(Phi.T @ W @ Phi) @ (Phi.T @ W @ t)
        return self

    def predict(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ self.w


MultivariateLinearRegression = LinearRegression


class RidgeRegression:
    """PRML 3.1.4節 式 (3.28) に基づく L2 正則化線形回帰 (Ridge 回帰)"""
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)  # 正則化係数 lambda
        self.w = None
        self.M_ = None

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float)
        self.M_ = Phi.shape[1]
        A = self.alpha * np.eye(self.M_) + Phi.T @ Phi
        self.w = np.linalg.solve(A, Phi.T @ t)
        return self

    def predict(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ self.w

    def effective_degrees_of_freedom(self, Phi):
        """有効パラメータ自由度 gamma = tr(Phi (Phi^T Phi + alpha I)^-1 Phi^T) = sum_i s_i^2 / (s_i^2 + alpha)"""
        Phi = np.asarray(Phi, dtype=float)
        _, s, _ = np.linalg.svd(Phi, full_matrices=False)
        eigenvalues = s**2
        return float(np.sum(eigenvalues / (eigenvalues + self.alpha)))

    def shrinkage_factors(self, Phi):
        """主成分ごとの縮小係数 s_i^2 / (s_i^2 + alpha)"""
        Phi = np.asarray(Phi, dtype=float)
        _, s, _ = np.linalg.svd(Phi, full_matrices=False)
        return s**2 / (s**2 + self.alpha)


class LassoRegression:
    """PRML 3.1.4節 式 (3.29) (q=1) に基づく L1 正則化線形回帰 (Lasso 回帰)
    座標降下法 (Coordinate Descent) とソフト閾値処理 (Soft-thresholding) による厳密なスパース推定
    """
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-5):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.w = None

    @staticmethod
    def _soft_threshold(z, gamma):
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0.0

    @staticmethod
    def soft_threshold(z, lam):
        return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        N, D = X.shape

        self.w = np.linalg.pinv(X) @ y
        col_norms = np.sum(X**2, axis=0)
        col_norms = np.where(col_norms < 1e-12, 1e-12, col_norms)

        for _ in range(self.max_iter):
            w_old = self.w.copy()
            for j in range(D):
                r_j = y - (X @ self.w - X[:, j] * self.w[j])
                rho_j = float(X[:, j] @ r_j)
                if rho_j < -self.alpha:
                    self.w[j] = (rho_j + self.alpha) / col_norms[j]
                elif rho_j > self.alpha:
                    self.w[j] = (rho_j - self.alpha) / col_norms[j]
                else:
                    self.w[j] = 0.0

            if np.max(np.abs(self.w - w_old)) < self.tol:
                break
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w

    def get_sparsity(self, tol=1e-4):
        """ゼロ係数の割合 (0.0 〜 1.0)"""
        if self.w is None:
            return 0.0
        return float(np.mean(np.abs(self.w) < tol))


class LeastMeanSquares:
    """PRML 式 (3.22) に基づく逐次勾配降下学習 (LMS / Robbins-Monro 型オンライン学習)
    w^(tau+1) = w^(tau) + eta * (t_n - w^T phi_n) * phi_n
    """
    def __init__(self, learning_rate=0.05, max_iter=200, decay_rate=0.001, eta_decay=None):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.decay_rate = float(decay_rate if eta_decay is None else eta_decay)
        self.w = None
        self.weight_history = []
        self.history = []

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        N, M = Phi.shape
        self.w = np.zeros(M)
        self.weight_history = [self.w.copy()]
        self.history = [self.w.copy()]

        step = 0
        indices = np.arange(N)
        for _ in range(self.max_iter):
            np.random.shuffle(indices)
            for i in indices:
                step += 1
                eta = self.learning_rate / (1.0 + self.decay_rate * step)
                phi_i = Phi[i]
                error = t[i] - float(phi_i @ self.w)
                self.w += eta * error * phi_i
                self.weight_history.append(self.w.copy())
                self.history.append(self.w.copy())
        return self

    def update(self, phi_n, t_n, learning_rate=None):
        lr = learning_rate or self.learning_rate
        if self.w is None:
            self.w = np.zeros(len(phi_n))
        error = t_n - float(np.dot(self.w, phi_n))
        self.w += lr * error * phi_n
        return self

    def predict(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ self.w


# エイリアス
LMS = LeastMeanSquares
SequentialLinearRegression = LeastMeanSquares


class BayesianLinearRegression:
    """PRML 3.3節 式 (3.49) - (3.60) に基づくベイズ線形回帰モデル"""
    def __init__(self, alpha=2.0, beta=25.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.m_N = None
        self.S_N = None
        self.Phi = None
        self.t = None

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        M = Phi.shape[1]
        self.Phi = Phi
        self.t = t
        # S_N^-1 = alpha * I + beta * Phi^T @ Phi (式 3.54)
        S_N_inv = self.alpha * np.eye(M) + self.beta * (Phi.T @ Phi)
        self.S_N = np.linalg.inv(S_N_inv)
        # m_N = beta * S_N @ Phi^T @ t (式 3.53)
        self.m_N = self.beta * (self.S_N @ Phi.T @ t)
        return self

    def update(self, phi_n, t_n):
        """新規1データ点 (phi_n, t_n) による逐次ベイズ更新 (PRML 式 3.50 - 3.51)"""
        phi_n = np.asarray(phi_n, dtype=float).ravel()
        t_n = float(t_n)
        M = len(phi_n)
        if self.S_N is None:
            self.m_N = np.zeros(M)
            self.S_N = (1.0 / self.alpha) * np.eye(M)

        S_N_inv_old = np.linalg.inv(self.S_N)
        S_N_inv_new = S_N_inv_old + self.beta * np.outer(phi_n, phi_n)
        self.S_N = np.linalg.inv(S_N_inv_new)
        self.m_N = self.S_N @ (S_N_inv_old @ self.m_N + self.beta * t_n * phi_n)
        return self

    def predict(self, Phi_new):
        """予測平均と予測標準偏差を返す (PRML 式 3.58 - 3.59)"""
        Phi_new = np.asarray(Phi_new, dtype=float)
        y_mean = Phi_new @ self.m_N
        y_var = (1.0 / self.beta) + np.sum((Phi_new @ self.S_N) * Phi_new, axis=1)
        return y_mean, np.sqrt(np.maximum(y_var, 1e-12))

    def sample_weights(self, n_samples=5):
        """事後分布 p(w|t) からパラメータベクトルをサンプリング"""
        return np.random.multivariate_normal(self.m_N, self.S_N, size=n_samples)

    def sample_functions(self, Phi_eval, n_samples=5):
        """事後分布からサンプリングしたパラメータに基づく回帰関数群 y(x)"""
        w_samples = self.sample_weights(n_samples=n_samples)
        return Phi_eval @ w_samples.T

    def equivalent_kernel(self, Phi_eval):
        """PRML 式 (3.62) に基づく等価カーネル行列 k(x, x_n) = beta * phi(x)^T S_N phi(x_n)"""
        Phi_eval = np.asarray(Phi_eval, dtype=float)
        return self.beta * (Phi_eval @ self.S_N @ self.Phi.T)

    def log_marginal_likelihood(self):
        """エビデンス関数 (対数周辺尤度) ln p(t | alpha, beta) (PRML 式 3.86)"""
        N, M = self.Phi.shape
        A = self.alpha * np.eye(M) + self.beta * (self.Phi.T @ self.Phi)
        E_mN = (self.beta / 2.0) * np.sum((self.t - self.Phi @ self.m_N)**2) + (self.alpha / 2.0) * np.sum(self.m_N**2)
        _, log_det_A = np.linalg.slogdet(A)
        log_evidence = (M / 2.0) * np.log(self.alpha) + (N / 2.0) * np.log(self.beta) \
                       - E_mN - 0.5 * log_det_A - (N / 2.0) * np.log(2.0 * np.pi)
        return float(log_evidence)


class NormalGammaLinearRegression:
    """PRML Exercises 3.12, 3.13 に基づく未知のノイズ精度 beta を持つ共役正規ガンマモデル
    事前分布: p(w, beta) = N(w | m_0, (beta * S_0)^-1) * Gam(beta | a_0, b_0)
    事後分布: p(w, beta | t) = N(w | m_N, (beta * S_N)^-1) * Gam(beta | a_N, b_N)
    予測分布: Student's t 分布 St(t | m_N^T phi(x), scale^2, nu)
    """
    def __init__(self, m0=None, S0_inv=None, a0=1.0, b0=1.0):
        self.m0 = m0
        self.S0_inv = S0_inv
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.m_N = None
        self.S_N = None
        self.a_N = None
        self.b_N = None
        self.Phi = None
        self.t = None

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        self.Phi = Phi
        self.t = t
        N, M = Phi.shape

        S0_inv = np.eye(M) if self.S0_inv is None else np.asarray(self.S0_inv, dtype=float)
        m0 = np.zeros(M) if self.m0 is None else np.asarray(self.m0, dtype=float)

        # 更新式 (PRML 式 3.116 - 3.117)
        S_N_inv = S0_inv + Phi.T @ Phi
        self.S_N = np.linalg.inv(S_N_inv)
        self.m_N = self.S_N @ (S0_inv @ m0 + Phi.T @ t)

        self.a_N = self.a0 + N / 2.0
        quad_prior = float(m0 @ S0_inv @ m0)
        quad_post = float(self.m_N @ S_N_inv @ self.m_N)
        self.b_N = self.b0 + 0.5 * (float(np.dot(t, t)) + quad_prior - quad_post)
        return self

    def predict(self, Phi_new):
        """予測分布 (Student's t 分布) の平均、スケール標準偏差、自由度 nu"""
        Phi_new = np.asarray(Phi_new, dtype=float)
        mu = Phi_new @ self.m_N
        nu = 2.0 * self.a_N
        factor = 1.0 + np.sum((Phi_new @ self.S_N) * Phi_new, axis=1)
        scale = np.sqrt((self.b_N / self.a_N) * factor)
        return mu, scale, nu

    def model_evidence(self):
        """PRML 式 (3.118) による厳密モデルエビデンス p(t)"""
        N, M = self.Phi.shape
        S0_inv = np.eye(M) if self.S0_inv is None else np.asarray(self.S0_inv, dtype=float)
        det_ratio = np.sqrt(np.linalg.det(self.S_N) * np.linalg.det(S0_inv))
        c = (1.0 / (2.0 * np.pi) ** (N / 2.0))
        gamma_ratio = gamma_func(self.a_N) / gamma_func(self.a0)
        b_ratio = (self.b0 ** self.a0) / (self.b_N ** self.a_N)
        return float(c * b_ratio * gamma_ratio * det_ratio)


class EvidenceApproximation:
    """PRML 3.5節: エビデンス近似（経験ベイズ法 / Empirical Bayes）
    ハイパーパラメータ alpha (事前精度) と beta (ノイズ精度) の自動最尤推定
    """
    def __init__(self, max_iter=100, tol=1e-5):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.m_N = None
        self.S_N = None
        self.history = []

    def fit(self, Phi, t, init_alpha=1.0, init_beta=1.0):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        N, M = Phi.shape
        alpha = float(init_alpha)
        beta = float(init_beta)

        PhiT_Phi = Phi.T @ Phi
        eigenvalues = np.linalg.eigvalsh(PhiT_Phi)

        self.history = []
        for it in range(self.max_iter):
            S_N_inv = alpha * np.eye(M) + beta * PhiT_Phi
            S_N = np.linalg.inv(S_N_inv)
            m_N = beta * (S_N @ Phi.T @ t)

            lambdas = beta * eigenvalues
            gamma = float(np.sum(lambdas / (alpha + lambdas)))

            w_sq = float(np.sum(m_N**2))
            new_alpha = gamma / max(w_sq, 1e-12)

            err_sq = float(np.sum((t - Phi @ m_N)**2))
            denom = max(N - gamma, 1e-12)
            new_beta = denom / max(err_sq, 1e-12)

            self.history.append({'iter': it, 'alpha': alpha, 'beta': beta, 'gamma': gamma})

            if abs(new_alpha - alpha) < self.tol and abs(new_beta - beta) < self.tol:
                alpha, beta = new_alpha, new_beta
                break
            alpha, beta = new_alpha, new_beta

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.m_N = m_N
        self.S_N = S_N
        return self


class LocallyWeightedRegression:
    """局所重み付き線形回帰 (Locally Weighted Regression / LWR)
    各クエリ点 x_0 ごとに近傍距離重み W(x_0) を適用して局所 OLS を解く
    w(x_0) = (Phi^T W Phi)^-1 Phi^T W t
    """
    def __init__(self, tau=0.5):
        self.tau = float(tau)  # バンド幅
        self.X_train = None
        self.t_train = None

    def fit(self, X, t):
        self.X_train = np.asarray(X, dtype=float)
        self.t_train = np.asarray(t, dtype=float).ravel()
        return self

    def predict(self, X_eval):
        X_eval = np.asarray(X_eval, dtype=float)
        if self.X_train.ndim == 1:
            X_tr = self.X_train[:, None]
        else:
            X_tr = self.X_train
        if X_eval.ndim == 1:
            X_ev = X_eval[:, None]
        else:
            X_ev = X_eval

        N = len(X_tr)
        Phi_tr = np.hstack([np.ones((N, 1)), X_tr])
        predictions = []

        for x0 in X_ev:
            phi0 = np.hstack([1.0, x0])
            dists_sq = np.sum((X_tr - x0)**2, axis=1)
            weights = np.exp(-dists_sq / (2.0 * self.tau**2))
            W = np.diag(weights)
            A = Phi_tr.T @ W @ Phi_tr + 1e-8 * np.eye(Phi_tr.shape[1])
            w_local = np.linalg.solve(A, Phi_tr.T @ W @ self.t_train)
            predictions.append(float(phi0 @ w_local))

        return np.array(predictions)


class RobustLinearRegression:
    """Huber 損失関数に基づくロバスト線形回帰（外れ値に強いIRLS法）"""
    def __init__(self, delta=1.345, df=None, max_iter=100, tol=1e-5):
        self.delta = float(delta)
        self.df = df
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.w = None

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        N, M = Phi.shape

        self.w = np.linalg.pinv(Phi) @ t  # 初期解は OLS

        for _ in range(self.max_iter):
            residuals = np.abs(t - Phi @ self.w)
            if self.df is not None:
                scale = np.median(residuals) / 0.6745 if np.median(residuals) > 1e-6 else 1.0
                weights = (float(self.df) + 1.0) / (float(self.df) + (residuals / scale)**2)
            else:
                # Huber 重み: |r| <= delta なら 1, それ以上は delta / |r|
                weights = np.where(residuals <= self.delta, 1.0, self.delta / np.maximum(residuals, 1e-10))
            W = np.diag(weights)
            w_new = np.linalg.solve(Phi.T @ W @ Phi + 1e-8 * np.eye(M), Phi.T @ W @ t)
            if np.max(np.abs(w_new - self.w)) < self.tol:
                self.w = w_new
                break
            self.w = w_new
        return self

    def predict(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ self.w


HuberRegression = RobustLinearRegression


class EquivalentKernel:
    """PRML 3.3.3節: 等価カーネル (Equivalent Kernel)
    k(x, x') = beta * phi(x)^T S_N phi(x')
    """
    def __init__(self, blr_model):
        self.blr = blr_model

    def __call__(self, x_eval):
        return self.blr.equivalent_kernel(x_eval)


def orthogonal_projection_matrix(Phi):
    """PRML 3.1.2節 式 (3.25): 直交射影行列 P = Phi (Phi^T Phi)^-1 Phi^T"""
    Phi = np.asarray(Phi, dtype=float)
    return Phi @ np.linalg.pinv(Phi)


def equivalent_kernel_matrix(Phi_eval, Phi_train, S_N=None, beta=1.0, alpha=1.0):
    """PRML 3.3.3節 式 (3.62): 等価カーネル行列 K = beta * Phi_eval @ S_N @ Phi_train^T"""
    Phi_eval = np.asarray(Phi_eval, dtype=float)
    Phi_train = np.asarray(Phi_train, dtype=float)
    if S_N is None:
        M = Phi_train.shape[1]
        S_N = np.linalg.pinv(alpha * np.eye(M) + beta * (Phi_train.T @ Phi_train))
    return float(beta) * (Phi_eval @ S_N @ Phi_train.T)


def bias_variance_decomposition(model_class, model_kwargs, X_train_sets, y_train_sets, X_test, y_test_true, noise_var=0.0):
    """PRML 3.2節: L個のデータセットによるバイアス・バリアンス・期待損失の厳密な数値分解
    E_D[(y - h)^2] = (bias)^2 + variance
    """
    L = len(X_train_sets)
    predictions = []

    for l in range(L):
        model = model_class(**model_kwargs)
        model.fit(X_train_sets[l], y_train_sets[l])
        preds = model.predict(X_test)
        if isinstance(preds, tuple):
            preds = preds[0]
        predictions.append(preds)

    predictions = np.array(predictions)
    y_bar = np.mean(predictions, axis=0)

    bias_sq = float(np.mean((y_bar - y_test_true)**2))
    variance = float(np.mean(np.var(predictions, axis=0)))
    total_error = float(np.mean((predictions - y_test_true)**2)) + float(noise_var)

    return {
        'bias_squared': bias_sq,
        'variance': variance,
        'noise': float(noise_var),
        'expected_loss': float(bias_sq + variance + noise_var),
        'total_empirical_error': total_error,
        'y_bar': y_bar,
        'predictions': predictions
    }


class BiasVarianceSimulator:
    """PRML 3.2節: モンテカルロシミュレーションによるバイアス-バリアンス分解"""
    def __init__(self, L=100, N=25, sigma_noise=0.3, h_func=None):
        self.L = int(L)
        self.N = int(N)
        self.sigma_noise = float(sigma_noise)
        self.h_func = h_func or (lambda x: np.sin(2.0 * np.pi * x))

    def generate_datasets(self, seed=42):
        rng = np.random.RandomState(seed)
        X_list, t_list = [], []
        for _ in range(self.L):
            x = rng.uniform(0.0, 1.0, self.N)
            t = self.h_func(x) + rng.normal(0.0, self.sigma_noise, self.N)
            X_list.append(x)
            t_list.append(t)
        return X_list, t_list

    def simulate(self, model_class, model_kwargs, X_test=None, seed=42):
        if X_test is None:
            X_test = np.linspace(0.0, 1.0, 100)
        y_test_true = self.h_func(X_test)
        X_list, t_list = self.generate_datasets(seed=seed)
        return bias_variance_decomposition(
            model_class, model_kwargs, X_list, t_list, X_test, y_test_true, noise_var=self.sigma_noise**2
        )


__all__ = [
    'PolynomialBasis', 'GaussianBasis', 'SigmoidalBasis', 'FourierBasis',
    'LinearRegression', 'MultivariateLinearRegression', 'WeightedLinearRegression',
    'RidgeRegression', 'LassoRegression', 'LeastMeanSquares', 'LMS',
    'SequentialLinearRegression', 'BayesianLinearRegression', 'NormalGammaLinearRegression',
    'EvidenceApproximation', 'LocallyWeightedRegression', 'RobustLinearRegression',
    'HuberRegression', 'EquivalentKernel',
    'orthogonal_projection_matrix', 'equivalent_kernel_matrix',
    'bias_variance_decomposition', 'BiasVarianceSimulator', 'bayesian_model_evidence',
]

class EvidenceResult(dict):
    """対数エビデンス結果オブジェクト (辞書としても数値 float としても動作)"""
    def __init__(self, log_evidence, log_occam_factor=0.0, **kwargs):
        super().__init__(log_evidence=float(log_evidence), log_occam_factor=float(log_occam_factor), **kwargs)
        self.log_evidence = float(log_evidence)
        self.log_occam_factor = float(log_occam_factor)

    def __float__(self):
        return self.log_evidence

    def __add__(self, other):
        return float(self) + float(other)

    def __radd__(self, other):
        return float(other) + float(self)

    def __sub__(self, other):
        return float(self) - float(other)

    def __rsub__(self, other):
        return float(other) - float(self)

    def __mul__(self, other):
        return float(self) * float(other)

    def __rmul__(self, other):
        return float(other) * float(self)

    def __truediv__(self, other):
        return float(self) / float(other)

    def __rtruediv__(self, other):
        return float(other) / float(self)

    def __neg__(self):
        return -float(self)


def bayesian_model_evidence(Phi, t, alpha=2.0, beta=25.0):
    """PRML 3.5節 式 (3.86): 対数エビデンス関数 ln p(t | alpha, beta) およびオッカム因子
    ln p(t | alpha, beta) = M/2 ln(alpha) + N/2 ln(beta) - E(m_N) - 1/2 ln|A| - N/2 ln(2 pi)
    """
    Phi = np.asarray(Phi, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    N, M = Phi.shape
    A = alpha * np.eye(M) + beta * (Phi.T @ Phi)
    S_N = np.linalg.inv(A)
    m_N = beta * S_N @ Phi.T @ t
    E_mN = 0.5 * beta * np.sum((t - Phi @ m_N)**2) + 0.5 * alpha * np.dot(m_N, m_N)
    _, log_det_A = np.linalg.slogdet(A)
    log_evidence = 0.5 * M * np.log(alpha) + 0.5 * N * np.log(beta) - E_mN - 0.5 * log_det_A - 0.5 * N * np.log(2.0 * np.pi)
    log_occam_factor = 0.5 * M * np.log(alpha) - 0.5 * log_det_A
    return EvidenceResult(log_evidence, log_occam_factor=log_occam_factor, E_mN=float(E_mN))

