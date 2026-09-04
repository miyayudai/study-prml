import numpy as np

class PolynomialBasis:
    """多項式基底関数: phi_j(x) = x^j (j=0, ..., M)"""
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        Phi = np.hstack([X**i for i in range(self.degree + 1)])
        return Phi

class GaussianBasis:
    """ガウス基底関数: phi_j(x) = exp(- (x - mu_j)^2 / (2 * s^2))"""
    def __init__(self, centers, scale):
        self.centers = np.asarray(centers)
        self.scale = scale

    def __call__(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        # x_0 = 1 (バイアス項)
        Phi = [np.ones((len(X), 1))]
        for mu in self.centers:
            phi_j = np.exp(-0.5 * ((X - mu) / self.scale) ** 2)
            Phi.append(phi_j)
        return np.hstack(Phi)

class SigmoidalBasis:
    """シグモイド基底関数: phi_j(x) = sigma((x - mu_j) / s)"""
    def __init__(self, centers, scale):
        self.centers = np.asarray(centers)
        self.scale = scale

    def __call__(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        Phi = [np.ones((len(X), 1))]
        for mu in self.centers:
            a = (X - mu) / self.scale
            phi_j = 1.0 / (1.0 + np.exp(-a))
            Phi.append(phi_j)
        return np.hstack(Phi)

class LinearRegression:
    """PRML 式 (3.15) に基づく線形回帰（最尤推定 / 最小二乗解）"""
    def __init__(self):
        self.w = None

    def fit(self, Phi, t):
        """正規方程式 / ムーア・ペンローズ擬似逆行列による重みの学習"""
        # w_ML = (Phi^T @ Phi)^-1 @ Phi^T @ t = pinv(Phi) @ t
        self.w = np.linalg.pinv(Phi) @ t
        return self

    def predict(self, Phi):
        """予測値 y = Phi @ w"""
        return Phi @ self.w

class RidgeRegression:
    """PRML 式 (3.28) に基づく L2 正則化線形回帰 (Ridge 回帰)"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 正則化係数 lambda
        self.w = None

    def fit(self, Phi, t):
        """正則化正規方程式による重みの学習"""
        M = Phi.shape[1]
        self.w = np.linalg.solve(self.alpha * np.eye(M) + Phi.T @ Phi, Phi.T @ t)
        return self

    def predict(self, Phi):
        """予測値 y = Phi @ w"""
        return Phi @ self.w

class BayesianLinearRegression:
    """PRML 式 (3.49) - (3.60) に基づくベイズ線形回帰モデル"""
    def __init__(self, alpha=2.0, beta=25.0):
        self.alpha = alpha  # 事前分布の精度
        self.beta = beta    # ノイズの精度 (1/variance)
        self.m_N = None     # 事後平均
        self.S_N = None     # 事後共分散
        self.Phi = None     # 学習データの計画行列
        self.t = None

    def fit(self, Phi, t):
        """学習データ Phi (N x M) と目標値 t (N,) による事後分布の計算"""
        M = Phi.shape[1]
        self.Phi = Phi
        self.t = t
        # S_N^-1 = alpha * I + beta * Phi^T @ Phi
        S_N_inv = self.alpha * np.eye(M) + self.beta * (Phi.T @ Phi)
        self.S_N = np.linalg.inv(S_N_inv)
        # m_N = beta * S_N @ Phi^T @ t
        self.m_N = self.beta * (self.S_N @ Phi.T @ t)
        return self

    def predict(self, Phi_new):
        """新しい計画行列 Phi_new に対する予測分布の平均と標準偏差"""
        y_mean = Phi_new @ self.m_N
        # var = 1 / beta + diag(Phi_new @ S_N @ Phi_new^T)
        y_var = (1.0 / self.beta) + np.sum((Phi_new @ self.S_N) * Phi_new, axis=1)
        return y_mean, np.sqrt(y_var)

    def sample_weights(self, n_samples=5):
        """事後分布 p(w|t) からパラメータ w のサンプルを生成"""
        return np.random.multivariate_normal(self.m_N, self.S_N, size=n_samples)

    def equivalent_kernel(self, Phi_eval):
        """PRML 式 (3.62) に基づく等価カーネル k(x, x_n) の計算
        k(x, x') = beta * phi(x)^T S_N phi(x')
        Phi_eval: (K, M), self.Phi: (N, M)
        返り値: (K, N)
        """
        return self.beta * (Phi_eval @ self.S_N @ self.Phi.T)

    def log_marginal_likelihood(self):
        """エビデンス関数 (対数周辺尤度) ln p(t | alpha, beta) の計算 (PRML 式 (3.86))"""
        N, M = self.Phi.shape
        A = self.alpha * np.eye(M) + self.beta * (self.Phi.T @ self.Phi)
        E_mN = (self.beta / 2.0) * np.sum((self.t - self.Phi @ self.m_N)**2) + (self.alpha / 2.0) * np.sum(self.m_N**2)
        log_det_A = np.linalg.slogdet(A)[1]
        log_evidence = (M / 2.0) * np.log(self.alpha) + (N / 2.0) * np.log(self.beta) \
                       - E_mN - 0.5 * log_det_A - (N / 2.0) * np.log(2.0 * np.pi)
        return log_evidence

class EvidenceApproximation:
    """PRML 3.5節に基づくエビデンス近似（経験ベイズ）による alpha, beta の自動最適化"""
    def __init__(self, max_iter=100, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.history = []

    def fit(self, Phi, t, init_alpha=1.0, init_beta=1.0):
        N, M = Phi.shape
        alpha = init_alpha
        beta = init_beta

        # Phi^T @ Phi の固有値分解 (PRML 式 3.87)
        PhiT_Phi = Phi.T @ Phi
        eigenvalues = np.linalg.eigvalsh(PhiT_Phi)

        self.history = []
        for it in range(self.max_iter):
            # 事後共分散行列 S_N と平均 m_N
            S_N_inv = alpha * np.eye(M) + beta * PhiT_Phi
            S_N = np.linalg.inv(S_N_inv)
            m_N = beta * (S_N @ Phi.T @ t)

            # lambda_i は beta * (Phi^T Phi) の固有値
            lambdas = beta * eigenvalues
            gamma = np.sum(lambdas / (alpha + lambdas))

            # 新しい alpha (式 3.92)
            w_sq = np.sum(m_N**2)
            new_alpha = gamma / max(w_sq, 1e-12)

            # 新しい beta (式 3.95)
            err_sq = np.sum((t - Phi @ m_N)**2)
            denom = N - gamma
            new_beta = denom / max(err_sq, 1e-12)

            self.history.append({'iter': it, 'alpha': alpha, 'beta': beta, 'gamma': gamma})

            if abs(new_alpha - alpha) < self.tol and abs(new_beta - beta) < self.tol:
                alpha, beta = new_alpha, new_beta
                break
            alpha, beta = new_alpha, new_beta

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.m_N = m_N
        self.S_N = S_N
        return self
