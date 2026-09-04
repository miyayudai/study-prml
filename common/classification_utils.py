import numpy as np
import scipy.special as sp_special
import scipy.stats as stats
import matplotlib.pyplot as plt

def sigmoid(a):
    """ロジスティックシグモイド関数 sigma(a) = 1 / (1 + exp(-a))"""
    return sp_special.expit(a)

def softmax(a, axis=-1):
    """数値的に安定なソフトマックス関数"""
    a_max = np.max(a, axis=axis, keepdims=True)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a, axis=axis, keepdims=True)

class Perceptron:
    """パーセプトロン学習アルゴリズム (PRML 4.1.7節)"""
    def __init__(self, max_iter=1000, lr=1.0):
        self.max_iter = max_iter
        self.lr = lr
        self.w = None
        self.history = []

    def fit(self, X, y):
        """
        X: shape (N, D)
        y: shape (N,) - ラベル {-1, +1}
        """
        N, D = X.shape
        # バイアス項の追加
        Phi = np.column_stack([np.ones(N), X])
        self.w = np.zeros(D + 1)
        self.history = [self.w.copy()]

        for epoch in range(self.max_iter):
            converged = True
            for n in range(N):
                # 誤分類条件: w^T phi_n * y_n <= 0
                if np.dot(self.w, Phi[n]) * y[n] <= 0:
                    self.w += self.lr * Phi[n] * y[n]
                    self.history.append(self.w.copy())
                    converged = False
            if converged:
                break
        return self

    def predict(self, X):
        Phi = np.column_stack([np.ones(len(X)), X])
        return np.sign(Phi @ self.w)


class FisherLinearDiscriminant:
    """フィッシャーの線形判別分析 (Fisher's Linear Discriminant, PRML 4.1.4節)"""
    def __init__(self):
        self.w = None
        self.w0 = None

    def fit(self, X, y):
        """
        X: shape (N, D)
        y: shape (N,) - ラベル {0, 1} または {-1, 1}
        """
        classes = np.unique(y)
        assert len(classes) == 2, "2クラス判別のみ対応"
        X1 = X[y == classes[0]]
        X2 = X[y == classes[1]]

        m1 = np.mean(X1, axis=0)
        m2 = np.mean(X2, axis=0)

        # クラス内共分散行列 S_W
        S_W = np.zeros((X.shape[1], X.shape[1]))
        for x in X1:
            diff = (x - m1).reshape(-1, 1)
            S_W += diff @ diff.T
        for x in X2:
            diff = (x - m2).reshape(-1, 1)
            S_W += diff @ diff.T

        # 最適な重みベクトル w \propto S_W^-1 (m2 - m1)
        self.w = np.linalg.solve(S_W, (m2 - m1))
        self.w /= np.linalg.norm(self.w)

        # しきい値 w0 (正規分布仮定または中間点)
        self.w0 = -0.5 * np.dot(self.w, (m1 + m2))
        return self

    def project(self, X):
        """1次元部分空間への射影 y = w^T x"""
        return X @ self.w

    def predict(self, X):
        return (self.project(X) + self.w0 >= 0).astype(int)


class GaussianGenerativeClassifier:
    """ガウス生成識別モデル (LDA / QDA, PRML 4.2.1-4.2.2節)"""
    def __init__(self, shared_cov=True):
        self.shared_cov = shared_cov
        self.classes = None
        self.priors = None
        self.means = None
        self.covs = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        K = len(self.classes)
        N, D = X.shape

        self.priors = np.zeros(K)
        self.means = np.zeros((K, D))
        self.covs = []

        for k, c in enumerate(self.classes):
            X_k = X[y == c]
            N_k = len(X_k)
            self.priors[k] = N_k / N
            self.means[k] = np.mean(X_k, axis=0)

        if self.shared_cov:
            # 共通共分散行列 (LDA)
            S = np.zeros((D, D))
            for k, c in enumerate(self.classes):
                X_k = X[y == c]
                diff = X_k - self.means[k]
                S += diff.T @ diff
            Sigma = S / N
            self.covs = [Sigma for _ in range(K)]
        else:
            # 個別共分散行列 (QDA)
            for k, c in enumerate(self.classes):
                X_k = X[y == c]
                diff = X_k - self.means[k]
                self.covs.append((diff.T @ diff) / len(X_k))
        return self

    def predict_proba(self, X):
        N = len(X)
        K = len(self.classes)
        log_joint = np.zeros((N, K))

        for k in range(K):
            cov = self.covs[k] + 1e-6 * np.eye(X.shape[1]) # 正則化
            dist = stats.multivariate_normal(mean=self.means[k], cov=cov)
            log_joint[:, k] = np.log(self.priors[k] + 1e-12) + dist.logpdf(X)

        return softmax(log_joint, axis=-1)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=-1)]


class LogisticRegression:
    """ロジスティック回帰 (IRLS アルゴリズム, PRML 4.3.2-4.3.3節)"""
    def __init__(self, alpha=0.0, max_iter=100, tol=1e-6):
        self.alpha = alpha # L2 正則化係数
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.history = []

    def fit(self, Phi, t):
        """
        Phi: shape (N, M) - 計画行列（バイアス項を含む）
        t: shape (N,) - 二値ラベル {0, 1}
        """
        N, M = Phi.shape
        self.w = np.zeros(M)
        self.history = [self.w.copy()]

        for i in range(self.max_iter):
            # 予測確率 y_n = sigma(w^T phi_n)
            a = Phi @ self.w
            y = sigmoid(a)

            # 重み行列 R_nn = y_n (1 - y_n)
            r = y * (1.0 - y)
            r = np.clip(r, 1e-12, np.inf)

            # 勾配: g = Phi^T (y - t) + alpha * w
            g = Phi.T @ (y - t) + self.alpha * self.w

            # ヘッセ行列: H = Phi^T R Phi + alpha * I
            R_mat = np.diag(r)
            H = Phi.T @ (r[:, None] * Phi) + self.alpha * np.eye(M)

            # ニュートン・ラフソン更新: w_new = w - H^-1 g
            delta_w = np.linalg.solve(H, g)
            self.w -= delta_w
            self.history.append(self.w.copy())

            if np.linalg.norm(delta_w) < self.tol:
                break
        return self

    def predict_proba(self, Phi):
        return sigmoid(Phi @ self.w)

    def predict(self, Phi):
        return (self.predict_proba(Phi) >= 0.5).astype(int)


class MulticlassLogisticRegression:
    """多クラスロジスティック回帰 / ソフトマックス回帰 (PRML 4.3.4節)"""
    def __init__(self, alpha=0.0, lr=0.1, max_iter=500, tol=1e-5):
        self.alpha = alpha
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.W = None # shape (M, K)

    def fit(self, Phi, T):
        """
        Phi: shape (N, M)
        T: shape (N, K) - 1-of-K 表現
        """
        N, M = Phi.shape
        K = T.shape[1]
        self.W = np.zeros((M, K))

        for it in range(self.max_iter):
            # Y: shape (N, K)
            A = Phi @ self.W
            Y = softmax(A, axis=-1)

            # 勾配: G = Phi^T (Y - T) + alpha * W
            grad = Phi.T @ (Y - T) + self.alpha * self.W

            self.W -= self.lr * grad
            if np.linalg.norm(grad) < self.tol:
                break
        return self

    def predict_proba(self, Phi):
        return softmax(Phi @ self.W, axis=-1)

    def predict(self, Phi):
        return np.argmax(self.predict_proba(Phi), axis=-1)


class BayesianLogisticRegression:
    """ベイズロジスティック回帰 (ラプラス近似 & プロビット近似, PRML 4.5節)"""
    def __init__(self, alpha=1.0, max_iter=100):
        self.alpha = alpha # 事前分布の精度 p(w) = N(0, alpha^-1 I)
        self.max_iter = max_iter
        self.w_map = None
        self.S_N = None

    def fit(self, Phi, t):
        """
        Phi: shape (N, M)
        t: shape (N,) - ラベル {0, 1}
        """
        # MAP 推定 (L2 正則化付きロジスティック回帰)
        lr_model = LogisticRegression(alpha=self.alpha, max_iter=self.max_iter).fit(Phi, t)
        self.w_map = lr_model.w

        # 事後共分散行列 S_N = H^-1
        y = sigmoid(Phi @ self.w_map)
        r = y * (1.0 - y)
        H = Phi.T @ (r[:, None] * Phi) + self.alpha * np.eye(Phi.shape[1])
        self.S_N = np.linalg.inv(H)
        return self

    def predict_proba(self, Phi):
        """
        予測分布 p(C1 | phi) \approx sigma(kappa(sigma_a^2) * mu_a)
        PRML 式 (4.153), (4.154)
        """
        mu_a = Phi @ self.w_map
        # 分散 sigma_a^2 = phi^T S_N phi
        sigma_a2 = np.sum((Phi @ self.S_N) * Phi, axis=1)
        # kappa(sigma^2) = (1 + pi * sigma^2 / 8)^(-1/2)
        kappa = (1.0 + np.pi * sigma_a2 / 8.0) ** (-0.5)
        return sigmoid(kappa * mu_a)

    def predict(self, Phi):
        return (self.predict_proba(Phi) >= 0.5).astype(int)


def plot_decision_boundary_2d(model, X, y, ax=None, h=0.02, title="Decision Boundary"):
    """2次元データに対する決定境界の可視化ヘルパー"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(model, 'predict_proba'):
        # 確率予測可能な場合
        if hasattr(model, 'w') and len(model.w) == 3:
            # バイアス項を追加
            grid_phi = np.column_stack([np.ones(len(grid_pts)), grid_pts])
            Z = model.predict_proba(grid_phi)
        else:
            Z = model.predict_proba(grid_pts)
        if Z.ndim > 1 and Z.shape[1] == 2:
            Z = Z[:, 1]
        elif Z.ndim > 1:
            Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)
        contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2)
    else:
        # 決定関数予測
        Z = model.predict(grid_pts)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap='RdBu_r', alpha=0.5)
        ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)

    # データ点のプロット
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=45, zorder=5)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
