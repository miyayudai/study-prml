"""
Script to build robust and complete common/regression_utils.py for PRML Chapter 3
"""

content = '''"""
PRML Chapter 3: Linear Models for Regression - Foundation Utilities
線形基底関数モデル、最尤推定、正則化 (Ridge / Lasso)、逐次学習 (LMS)、
ベイズ線形回帰、等価カーネル、エビデンス近似、バイアス-バリアンス分解を包括的に実装
"""

import numpy as np


class PolynomialBasis:
    """多項式基底関数: phi_j(x) = x^j (j=0, ..., degree)"""
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
    """ガウス基底関数 (動径基底関数 / RBF):
    phi_0(x) = 1 (バイアス項)
    phi_j(x) = exp(- (x - mu_j)^2 / (2 * s^2)) (j=1, ..., M-1)
    """
    def __init__(self, centers, scale):
        self.centers = np.asarray(centers, dtype=float)
        if self.centers.ndim == 1:
            self.centers = self.centers[:, None]
        self.scale = float(scale)

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        N = len(X)
        Phi = [np.ones((N, 1))]
        for mu in self.centers:
            diff_sq = np.sum((X - mu)**2, axis=1, keepdims=True)
            phi_j = np.exp(-0.5 * diff_sq / (self.scale**2))
            Phi.append(phi_j)
        return np.hstack(Phi)

    def transform(self, X):
        return self(X)


class SigmoidalBasis:
    """シグモイド基底関数:
    phi_0(x) = 1 (バイアス項)
    phi_j(x) = sigma((x - mu_j) / s) = 1 / (1 + exp(-(x - mu_j) / s))
    """
    def __init__(self, centers, scale):
        self.centers = np.asarray(centers, dtype=float)
        if self.centers.ndim == 1:
            self.centers = self.centers[:, None]
        self.scale = float(scale)

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        N = len(X)
        Phi = [np.ones((N, 1))]
        for mu in self.centers:
            a = np.sum((X - mu), axis=1, keepdims=True) / self.scale
            a_clipped = np.clip(a, -500.0, 500.0)
            phi_j = 1.0 / (1.0 + np.exp(-a_clipped))
            Phi.append(phi_j)
        return np.hstack(Phi)

    def transform(self, X):
        return self(X)


class FourierBasis:
    """フーリエ基底関数:
    phi_0(x) = 1 (バイアス項)
    phi_{2j-1}(x) = sin(2 pi j x / L)
    phi_{2j}(x) = cos(2 pi j x / L)
    """
    def __init__(self, n_components, period=1.0):
        self.n_components = int(n_components)
        self.period = float(period)

    def __call__(self, X):
        X = np.asarray(X, dtype=float).ravel()
        N = len(X)
        Phi = [np.ones((N, 1))]
        for j in range(1, self.n_components + 1):
            omega = 2.0 * np.pi * j / self.period
            Phi.append(np.sin(omega * X)[:, None])
            Phi.append(np.cos(omega * X)[:, None])
        return np.hstack(Phi)

    def transform(self, X):
        return self(X)


class LinearRegression:
    """PRML 式 (3.15) に基づく線形回帰（最尤推定 / 最小二乗解）
    単一出力および複数出力 (式 3.31-3.33) に完全対応
    """
    def __init__(self):
        self.w = None
        self.sigma2_ = None
        self.sigma2_unbiased_ = None
        self.N_ = None
        self.M_ = None

    def fit(self, Phi, t):
        """正規方程式 / ムーア・ペンローズ擬似逆行列による重みの学習
        w_ML = (Phi^T Phi)^-1 Phi^T t = pinv(Phi) @ t (式 3.15, 3.17)
        """
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float)
        self.N_, self.M_ = Phi.shape

        self.w = np.linalg.pinv(Phi) @ t

        residuals = t - Phi @ self.w
        if t.ndim == 1:
            self.sigma2_ = float(np.mean(residuals**2))
            df = max(self.N_ - self.M_, 1)
            self.sigma2_unbiased_ = float(np.sum(residuals**2) / df)
        else:
            self.sigma2_ = np.mean(residuals**2, axis=0)
            df = max(self.N_ - self.M_, 1)
            self.sigma2_unbiased_ = np.sum(residuals**2, axis=0) / df
        return self

    def predict(self, Phi):
        """予測値 y = Phi @ w"""
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ self.w

    def score(self, Phi, t):
        """決定係数 R^2 (Coefficient of Determination)"""
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float)
        y_pred = self.predict(Phi)
        ss_res = np.sum((t - y_pred)**2)
        ss_tot = np.sum((t - np.mean(t, axis=0))**2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def projection_matrix(self, Phi):
        """直交射影行列 (Hat Matrix) P = Phi (Phi^T Phi)^-1 Phi^T = Phi pinv(Phi) (式 3.25)"""
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ np.linalg.pinv(Phi)

    def log_likelihood(self, Phi, t):
        """対数尤度 ln p(t | w, beta_ML) (式 3.12)"""
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float)
        residuals = t - Phi @ self.w
        if t.ndim == 1:
            beta_ml = 1.0 / max(self.sigma2_, 1e-12)
            ll = 0.5 * self.N_ * np.log(beta_ml) - 0.5 * self.N_ * np.log(2.0 * np.pi) - 0.5 * beta_ml * np.sum(residuals**2)
            return float(ll)
        else:
            beta_ml = 1.0 / np.maximum(self.sigma2_, 1e-12)
            K = t.shape[1]
            ll = 0.5 * self.N_ * np.sum(np.log(beta_ml)) - 0.5 * self.N_ * K * np.log(2.0 * np.pi) - 0.5 * np.sum(beta_ml * np.sum(residuals**2, axis=0))
            return float(ll)

    def aic(self, Phi, t):
        """赤池情報量基準 AIC = 2 * M - 2 * ln L"""
        ll = self.log_likelihood(Phi, t)
        k = self.M_ + 1
        return float(2.0 * k - 2.0 * ll)

    def bic(self, Phi, t):
        """ベイズ情報量基準 BIC = k * ln(N) - 2 * ln L"""
        ll = self.log_likelihood(Phi, t)
        k = self.M_ + 1
        return float(k * np.log(self.N_) - 2.0 * ll)


class MultivariateLinearRegression:
    """PRML 3.1.5節 式 (3.30) - (3.36) に基づく多変量目的変数の線形回帰
    T in R^{N x K}, W in R^{M x K}
    """
    def __init__(self):
        self.W = None
        self.Sigma = None
        self.Phi = None
        self.T = None

    def fit(self, Phi, T):
        Phi = np.asarray(Phi, dtype=float)
        T = np.asarray(T, dtype=float)
        if T.ndim == 1:
            T = T[:, None]
        N, M = Phi.shape
        self.Phi = Phi
        self.T = T

        self.W = np.linalg.pinv(Phi) @ T
        residuals = T - Phi @ self.W
        self.Sigma = (residuals.T @ residuals) / max(N, 1)
        return self

    def predict(self, Phi_new):
        Phi_new = np.asarray(Phi_new, dtype=float)
        return Phi_new @ self.W

    def predict_covariance(self, phi_x):
        phi_x = np.asarray(phi_x, dtype=float).ravel()
        mean = phi_x @ self.W
        PhiT_Phi_inv = np.linalg.pinv(self.Phi.T @ self.Phi)
        leverage = float(phi_x @ PhiT_Phi_inv @ phi_x)
        cov = self.Sigma * (1.0 + leverage)
        return mean, cov


class RidgeRegression:
    """PRML 式 (3.28) に基づく L2 正則化線形回帰 (Weight Decay / Ridge 回帰)"""
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.w = None
        self.N_ = None
        self.M_ = None

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float)
        self.N_, self.M_ = Phi.shape
        A = self.alpha * np.eye(self.M_) + Phi.T @ Phi
        self.w = np.linalg.solve(A, Phi.T @ t)
        return self

    def predict(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ self.w

    def effective_degrees_of_freedom(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        _, s, _ = np.linalg.svd(Phi, full_matrices=False)
        eigenvalues = s**2
        return float(np.sum(eigenvalues / (eigenvalues + self.alpha)))


class LassoRegression:
    """PRML 3.1.4節に基づく L1 正則化線形回帰 (Lasso 回帰)
    座標降下法 (Coordinate Descent) とソフトしきい値作用素 (Soft-Thresholding)
    """
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-5):
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.cost_history = []

    @staticmethod
    def _soft_threshold(z, gamma):
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0.0

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        N, M = Phi.shape

        col_norm_sq = np.sum(Phi**2, axis=0)
        col_norm_sq = np.where(col_norm_sq == 0, 1e-12, col_norm_sq)

        self.w = np.linalg.solve(0.1 * np.eye(M) + Phi.T @ Phi, Phi.T @ t)
        self.cost_history = []

        for it in range(self.max_iter):
            w_prev = self.w.copy()
            for j in range(M):
                r_j = t - (Phi @ self.w) + self.w[j] * Phi[:, j]
                rho_j = float(Phi[:, j] @ r_j)
                self.w[j] = self._soft_threshold(rho_j, self.alpha) / col_norm_sq[j]

            cost = 0.5 * np.sum((t - Phi @ self.w)**2) + self.alpha * np.sum(np.abs(self.w))
            self.cost_history.append(cost)

            if np.max(np.abs(self.w - w_prev)) < self.tol:
                break

        return self

    def predict(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ self.w

    def get_sparsity(self):
        if self.w is None:
            return 0.0
        return float(np.mean(np.abs(self.w) < 1e-6))


class LeastMeanSquares:
    """PRML 3.1.3節 式 (3.22) & (3.23): 逐次学習アルゴリズム (Widrow-Hoff / LMS 法)"""
    def __init__(self, learning_rate=0.01, max_iter=100, decay_rate=0.0):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.decay_rate = float(decay_rate)
        self.w = None
        self.weight_history = []

    def partial_fit(self, phi_n, t_n, eta=None):
        if eta is None:
            eta = self.learning_rate
        phi_n = np.asarray(phi_n, dtype=float)
        err = float(t_n - phi_n @ self.w)
        self.w += eta * err * phi_n
        return self

    def fit(self, Phi, t, w_init=None):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        N, M = Phi.shape

        if w_init is not None:
            self.w = np.asarray(w_init, dtype=float).copy()
        else:
            self.w = np.zeros(M, dtype=float)

        self.weight_history = [self.w.copy()]
        step = 0

        for epoch in range(self.max_iter):
            indices = np.arange(N)
            for idx in indices:
                step += 1
                eta = self.learning_rate / (1.0 + self.decay_rate * step)
                self.partial_fit(Phi[idx], t[idx], eta=eta)
                self.weight_history.append(self.w.copy())

        return self

    def predict(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ self.w


SequentialLinearRegression = LeastMeanSquares


class BayesianLinearRegression:
    """PRML 式 (3.49) - (3.60) に基づくベイズ線形回帰モデル"""
    def __init__(self, alpha=2.0, beta=25.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.m_N = None
        self.S_N = None
        self.Phi = None
        self.t = None
        self.M = None

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        N, M = Phi.shape
        self.Phi = Phi
        self.t = t
        self.M = M

        S_N_inv = self.alpha * np.eye(M) + self.beta * (Phi.T @ Phi)
        self.S_N = np.linalg.inv(S_N_inv)
        self.m_N = self.beta * (self.S_N @ Phi.T @ t)
        return self

    def update(self, phi_new, t_new):
        phi_new = np.asarray(phi_new, dtype=float).reshape(-1, 1)
        t_new = float(t_new)

        if self.S_N is None:
            self.M = len(phi_new)
            S_prev_inv = self.alpha * np.eye(self.M)
            m_prev = np.zeros(self.M)
        else:
            S_prev_inv = np.linalg.inv(self.S_N)
            m_prev = self.m_N

        S_next_inv = S_prev_inv + self.beta * (phi_new @ phi_new.T)
        self.S_N = np.linalg.inv(S_next_inv)
        self.m_N = (self.S_N @ (S_prev_inv @ m_prev + self.beta * t_new * phi_new.ravel()))
        return self

    def predict(self, Phi_new, return_std=True):
        Phi_new = np.asarray(Phi_new, dtype=float)
        y_mean = Phi_new @ self.m_N
        y_var = (1.0 / self.beta) + np.sum((Phi_new @ self.S_N) * Phi_new, axis=1)
        if return_std:
            return y_mean, np.sqrt(y_var)
        return y_mean, y_var

    def predict_cov(self, Phi_new):
        Phi_new = np.asarray(Phi_new, dtype=float)
        K = len(Phi_new)
        return (1.0 / self.beta) * np.eye(K) + Phi_new @ self.S_N @ Phi_new.T

    def sample_weights(self, n_samples=5):
        return np.random.multivariate_normal(self.m_N, self.S_N, size=n_samples)

    def sample_functions(self, Phi_eval, n_samples=5):
        w_samples = self.sample_weights(n_samples=n_samples)
        return Phi_eval @ w_samples.T

    def equivalent_kernel(self, Phi_eval):
        if self.Phi is None:
            raise ValueError("Training data Phi is required to evaluate the equivalent kernel.")
        Phi_eval = np.asarray(Phi_eval, dtype=float)
        return self.beta * (Phi_eval @ self.S_N @ self.Phi.T)

    def log_marginal_likelihood(self):
        if self.Phi is None or self.t is None:
            raise ValueError("Model must be fitted before computing marginal likelihood.")
        N, M = self.Phi.shape
        A = self.alpha * np.eye(M) + self.beta * (self.Phi.T @ self.Phi)
        E_mN = (self.beta / 2.0) * np.sum((self.t - self.Phi @ self.m_N)**2) + (self.alpha / 2.0) * np.sum(self.m_N**2)
        _, log_det_A = np.linalg.slogdet(A)
        log_evidence = (M / 2.0) * np.log(self.alpha) + (N / 2.0) * np.log(self.beta) \\
                       - E_mN - 0.5 * log_det_A - (N / 2.0) * np.log(2.0 * np.pi)
        return float(log_evidence)


class NormalGammaLinearRegression:
    """PRML Exercises 3.12, 3.13 に基づく未知のノイズ精度 beta を持つ共役正規ガンマモデル"""
    def __init__(self, m0=None, S0_inv=None, a0=1.0, b0=1.0):
        self.m0 = m0
        self.S0_inv = S0_inv
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.m_N = None
        self.S_N = None
        self.a_N = None
        self.b_N = None

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        N, M = Phi.shape

        if self.m0 is None:
            m0 = np.zeros(M)
        else:
            m0 = np.asarray(self.m0, dtype=float)

        if self.S0_inv is None:
            S0_inv = 1e-4 * np.eye(M)
        else:
            S0_inv = np.asarray(self.S0_inv, dtype=float)

        S_N_inv = S0_inv + Phi.T @ Phi
        self.S_N = np.linalg.inv(S_N_inv)
        self.m_N = self.S_N @ (S0_inv @ m0 + Phi.T @ t)

        self.a_N = self.a0 + 0.5 * N
        quad = (t @ t) + (m0 @ S0_inv @ m0) - (self.m_N @ S_N_inv @ self.m_N)
        self.b_N = self.b0 + 0.5 * quad
        return self

    def predict(self, Phi_new):
        Phi_new = np.asarray(Phi_new, dtype=float)
        mean = Phi_new @ self.m_N
        nu = 2.0 * self.a_N
        quad_phi = np.sum((Phi_new @ self.S_N) * Phi_new, axis=1)
        scale_sq = (self.b_N / self.a_N) * (1.0 + quad_phi)

        if nu > 2:
            variance = (nu / (nu - 2.0)) * scale_sq
        else:
            variance = np.full_like(mean, np.nan)

        return mean, np.sqrt(variance), nu


class EvidenceApproximation:
    """PRML 3.5節に基づくエビデンス近似（経験ベイズ / Empirical Bayes）による alpha, beta の自動最適化"""
    def __init__(self, max_iter=100, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol
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

            w_sq = float(m_N @ m_N)
            new_alpha = gamma / max(w_sq, 1e-12)

            err_sq = float(np.sum((t - Phi @ m_N)**2))
            denom = max(N - gamma, 1e-12)
            new_beta = denom / max(err_sq, 1e-12)

            A = alpha * np.eye(M) + beta * PhiT_Phi
            E_mN = 0.5 * beta * err_sq + 0.5 * alpha * w_sq
            _, log_det_A = np.linalg.slogdet(A)
            log_evidence = 0.5 * M * np.log(alpha) + 0.5 * N * np.log(beta) - E_mN - 0.5 * log_det_A - 0.5 * N * np.log(2 * np.pi)

            self.history.append({
                'iter': it,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'log_evidence': float(log_evidence)
            })

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

    def predict(self, Phi_new):
        Phi_new = np.asarray(Phi_new, dtype=float)
        mean = Phi_new @ self.m_N
        var = (1.0 / self.beta) + np.sum((Phi_new @ self.S_N) * Phi_new, axis=1)
        return mean, np.sqrt(var)


class LocallyWeightedRegression:
    """局所重み付き線形回帰 (Locally Weighted Linear Regression / LOWESS)"""
    def __init__(self, tau=0.2, reg=1e-4):
        self.tau = float(tau)
        self.reg = float(reg)
        self.X_train = None
        self.Phi_train = None
        self.t_train = None
        self.basis_fn = None

    def fit(self, X, t, basis_fn=None):
        self.X_train = np.asarray(X, dtype=float)
        if self.X_train.ndim == 1:
            self.X_train = self.X_train[:, None]
        self.t_train = np.asarray(t, dtype=float).ravel()
        if basis_fn is None:
            self.Phi_train = np.hstack([np.ones((len(self.X_train), 1)), self.X_train])
        else:
            self.Phi_train = basis_fn(self.X_train)
        self.basis_fn = basis_fn
        return self

    def predict_point(self, x0):
        x0 = np.asarray(x0, dtype=float)
        if x0.ndim == 0:
            x0 = x0.reshape(1, 1)
        elif x0.ndim == 1:
            x0 = x0.reshape(1, -1)

        diff_sq = np.sum((self.X_train - x0)**2, axis=1)
        weights = np.exp(-0.5 * diff_sq / (self.tau**2))
        W = np.diag(weights)

        M = self.Phi_train.shape[1]
        A = self.Phi_train.T @ W @ self.Phi_train + self.reg * np.eye(M)
        b = self.Phi_train.T @ W @ self.t_train
        w_x0 = np.linalg.solve(A, b)

        if self.basis_fn is None:
            phi_x0 = np.hstack([np.ones((len(x0), 1)), x0])
        else:
            phi_x0 = self.basis_fn(x0)
        return float(phi_x0 @ w_x0)

    def predict(self, X_query):
        X_query = np.asarray(X_query, dtype=float)
        if X_query.ndim == 1:
            X_query = X_query[:, None]
        preds = [self.predict_point(x0) for x0 in X_query]
        return np.array(preds)


class RobustLinearRegression:
    """Huber 損失および Student-t ノイズモデルに基づくロバスト線形回帰 (IRLS)"""
    def __init__(self, delta=1.345, df=4.0, max_iter=50, tol=1e-5):
        self.delta = float(delta)
        self.df = float(df)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.w = None
        self.sigma2 = 1.0

    def fit(self, Phi, t):
        Phi = np.asarray(Phi, dtype=float)
        t = np.asarray(t, dtype=float).ravel()
        N, M = Phi.shape

        self.w = np.linalg.pinv(Phi) @ t

        for it in range(self.max_iter):
            w_prev = self.w.copy()
            residuals = t - Phi @ self.w
            abs_r = np.abs(residuals)
            weights = np.where(abs_r <= self.delta, 1.0, self.delta / np.maximum(abs_r, 1e-12))
            W = np.diag(weights)

            A = Phi.T @ W @ Phi + 1e-8 * np.eye(M)
            b = Phi.T @ W @ t
            self.w = np.linalg.solve(A, b)

            if np.max(np.abs(self.w - w_prev)) < self.tol:
                break
        return self

    def predict(self, Phi):
        Phi = np.asarray(Phi, dtype=float)
        return Phi @ self.w


def bias_variance_decomposition(estimator_fn=None, X_train_sets=None, y_train_sets=None, X_eval=None, y_eval_true=None, noise_var=0.0, **kwargs):
    """PRML 3.2節 式 (3.37)-(3.41) に基づくバイアス-バリアンス分解シミュレーション"""
    if estimator_fn is None and 'model_class' in kwargs:
        model_class = kwargs['model_class']
        model_kwargs = kwargs.get('model_kwargs', {})
        estimator_fn = lambda X, y: model_class(**model_kwargs).fit(X, y)
    elif callable(estimator_fn) and not hasattr(estimator_fn, 'fit'):
        pass
    elif hasattr(estimator_fn, 'fit'):
        est_cls = estimator_fn
        estimator_fn = lambda X, y: est_cls().fit(X, y)

    if X_eval is None and 'X_test' in kwargs:
        X_eval = kwargs['X_test']
    if y_eval_true is None and 'y_test_true' in kwargs:
        y_eval_true = kwargs['y_test_true']

    L = len(X_train_sets)
    predictions = []

    for l in range(L):
        model = estimator_fn(X_train_sets[l], y_train_sets[l])
        preds = model.predict(X_eval)
        if isinstance(preds, tuple):
            preds = preds[0]
        predictions.append(preds)

    predictions = np.array(predictions)
    y_bar = np.mean(predictions, axis=0)

    bias_squared_pointwise = (y_bar - y_eval_true)**2
    bias_squared = float(np.mean(bias_squared_pointwise))

    variance_pointwise = np.mean((predictions - y_bar)**2, axis=0)
    variance = float(np.mean(variance_pointwise))
    total_empirical_error = float(np.mean((predictions - y_eval_true)**2)) + noise_var

    return {
        'bias_squared': bias_squared,
        'variance': variance,
        'noise': float(noise_var),
        'expected_loss': float(bias_squared + variance + noise_var),
        'total_empirical_error': total_empirical_error,
        'y_bar': y_bar,
        'predictions': predictions
    }


def orthogonal_projection_matrix(Phi):
    """PRML 3.1.2節 式 (3.25): 直交射影行列 P = Phi (Phi^T Phi)^-1 Phi^T"""
    Phi = np.asarray(Phi, dtype=float)
    return Phi @ np.linalg.pinv(Phi)


def equivalent_kernel_matrix(Phi_eval, Phi_train, S_N, beta):
    """PRML 3.3.3節 式 (3.62): 等価カーネル行列 K = beta * Phi_eval @ S_N @ Phi_train^T"""
    Phi_eval = np.asarray(Phi_eval, dtype=float)
    Phi_train = np.asarray(Phi_train, dtype=float)
    return float(beta) * (Phi_eval @ S_N @ Phi_train.T)
'''

with open("common/regression_utils.py", "w", encoding="utf-8") as f:
    f.write(content.strip() + "\\n")
print("common/regression_utils.py built successfully.")
