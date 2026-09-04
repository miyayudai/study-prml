import numpy as np
import scipy.optimize as opt
from common.kernel_utils import rbf_kernel
from common.classification_utils import sigmoid

class SupportVectorClassifier:
    """
    サポートベクトルマシン分類器 (Support Vector Classifier: SVC, PRML 7.1節)
    双対二次計画法による厳密解法
    """
    def __init__(self, C=1.0, kernel=rbf_kernel, **kernel_kwargs):
        self.C = C
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.X_train = None
        self.t_train = None
        self.a = None
        self.b = 0.0
        self.sv_indices = None
        self.sv_X = None
        self.sv_t = None
        self.sv_a = None

    def fit(self, X, t):
        self.X_train = np.atleast_2d(X)
        self.t_train = np.asarray(t, dtype=float).ravel() # t in {-1, +1}
        # 0/1 の場合は -1/+1 に変換
        if set(np.unique(self.t_train)).issubset({0, 1}):
            self.t_train = 2.0 * self.t_train - 1.0
            
        N = len(self.X_train)
        K = self.kernel(self.X_train, self.X_train, **self.kernel_kwargs)
        # P_nm = t_n * t_m * K_nm
        P = np.outer(self.t_train, self.t_train) * K
        P = 0.5 * (P + P.T) + 1e-8 * np.eye(N) # 数値的正定値化
        
        # 目的関数: 0.5 * a^T P a - sum(a)
        def objective(a):
            return 0.5 * a @ P @ a - np.sum(a)
        
        def obj_grad(a):
            return P @ a - 1.0
        
        # 制約条件: sum(a_n * t_n) = 0
        constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, self.t_train), 'jac': lambda a: self.t_train}
        # ボックス境界: 0 <= a_n <= C
        bounds = [(0.0, self.C) for _ in range(N)]
        
        a0 = np.zeros(N)
        res = opt.minimize(objective, a0, jac=obj_grad, constraints=constraints, bounds=bounds, method='SLSQP', options={'maxiter': 500, 'ftol': 1e-7})
        self.a = res.x
        
        # サポートベクトルの抽出 (a_n > 1e-5)
        self.sv_indices = np.where(self.a > 1e-5)[0]
        self.sv_X = self.X_train[self.sv_indices]
        self.sv_t = self.t_train[self.sv_indices]
        self.sv_a = self.a[self.sv_indices]
        
        # バイアス b の計算 (マージン上のサポートベクトル 1e-5 < a_n < C - 1e-5 の平均)
        margin_sv = np.where((self.a > 1e-5) & (self.a < self.C - 1e-5))[0]
        if len(margin_sv) > 0:
            # t_n - sum_m a_m t_m K_nm
            b_vals = []
            for n in margin_sv:
                k_n = self.kernel(self.X_train, self.X_train[n:n+1], **self.kernel_kwargs).ravel()
                b_val = self.t_train[n] - np.sum(self.a * self.t_train * k_n)
                b_vals.append(b_val)
            self.b = np.mean(b_vals)
        else:
            # マージン上の点がなければ全サポートベクトルでの平均
            if len(self.sv_indices) > 0:
                b_vals = [self.t_train[n] - np.sum(self.a * self.t_train * self.kernel(self.X_train, self.X_train[n:n+1], **self.kernel_kwargs).ravel()) for n in self.sv_indices]
                self.b = np.mean(b_vals)
            else:
                self.b = 0.0
                
        return self

    def decision_function(self, X):
        X = np.atleast_2d(X)
        if len(self.sv_indices) == 0:
            return np.zeros(len(X))
        K_cross = self.kernel(X, self.sv_X, **self.kernel_kwargs) # (N_test, N_sv)
        return K_cross @ (self.sv_a * self.sv_t) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

class RelevanceVectorRegressor:
    """
    関連ベクトルマシン回帰 (Relevance Vector Machine for Regression: RVM, PRML 7.2.1節)
    エビデンスフレームワークによる超パラメータ alpha_i の自動剪定 (Sparsity)
    """
    def __init__(self, kernel=rbf_kernel, alpha_threshold=1e4, max_iter=500, tol=1e-4, **kernel_kwargs):
        self.kernel = kernel
        self.alpha_threshold = alpha_threshold
        self.max_iter = max_iter
        self.tol = tol
        self.kernel_kwargs = kernel_kwargs
        self.X_train = None
        self.t_train = None
        self.alpha = None
        self.beta = 1.0
        self.mu = None
        self.Sigma = None
        self.rv_indices = None
        self.rv_X = None

    def fit(self, X, t):
        self.X_train = np.atleast_2d(X)
        self.t_train = np.asarray(t, dtype=float).ravel()
        N = len(self.X_train)
        
        # デザイン行列 Phi: (N, N+1) [バイアス項 + 各学習サンプルとのカーネル]
        K = self.kernel(self.X_train, self.X_train, **self.kernel_kwargs)
        Phi = np.column_stack([np.ones(N), K])
        M = Phi.shape[1]
        
        # 初期ハイパーパラメータ
        self.alpha = np.ones(M) * 1.0
        self.beta = 1.0 / (np.var(self.t_train) + 1e-4)
        active = np.ones(M, dtype=bool)
        
        for iteration in range(self.max_iter):
            # アクティブな基底関数のみで計算
            Phi_act = Phi[:, active]
            alpha_act = self.alpha[active]
            
            # 事後共分散 Sigma と平均 mu (PRML 式 7.82, 7.83)
            # A = diag(alpha)
            Sigma_inv = np.diag(alpha_act) + self.beta * (Phi_act.T @ Phi_act)
            Sigma = np.linalg.pinv(Sigma_inv)
            mu = self.beta * Sigma @ Phi_act.T @ self.t_train
            
            # 有効パラメータ比率 gamma_i = 1 - alpha_i * Sigma_ii (PRML 式 7.89)
            gamma = 1.0 - alpha_act * np.diag(Sigma)
            
            # alpha_i の更新 (PRML 式 7.87)
            alpha_new = gamma / (mu**2 + 1e-12)
            
            # beta の更新 (PRML 式 7.88)
            residuals = self.t_train - Phi_act @ mu
            beta_new = (N - np.sum(gamma)) / (np.sum(residuals**2) + 1e-12)
            
            # スパース化: alpha が閾値を超えた基底を枝刈り
            active_idx = np.where(active)[0]
            prune = alpha_new > self.alpha_threshold
            active[active_idx[prune]] = False
            self.alpha[active_idx[~prune]] = alpha_new[~prune]
            self.alpha[active_idx[prune]] = np.inf
            
            diff = np.max(np.abs(self.beta - beta_new))
            self.beta = beta_new
            
            if diff < self.tol and not np.any(prune):
                break
                
        # 最終アクティブ基底の決定
        Phi_act = Phi[:, active]
        alpha_act = self.alpha[active]
        Sigma_inv = np.diag(alpha_act) + self.beta * (Phi_act.T @ Phi_act)
        self.Sigma = np.linalg.pinv(Sigma_inv)
        self.mu = self.beta * self.Sigma @ Phi_act.T @ self.t_train
        
        self.active_mask = active
        # バイアス (インデックス 0) を除いた関連ベクトル
        rv_in_K = active[1:]
        self.rv_indices = np.where(rv_in_K)[0]
        self.rv_X = self.X_train[self.rv_indices]
        return self

    def predict(self, X, return_std=True):
        X = np.atleast_2d(X)
        N_test = len(X)
        K_test = self.kernel(X, self.X_train, **self.kernel_kwargs)
        Phi_test = np.column_stack([np.ones(N_test), K_test])
        Phi_test_act = Phi_test[:, self.active_mask]
        
        # 予測平均 (PRML 式 7.90)
        y_mean = Phi_test_act @ self.mu
        # 予測分散: sigma^2(x) = 1/beta + phi(x)^T Sigma phi(x)
        var = 1.0 / self.beta + np.sum((Phi_test_act @ self.Sigma) * Phi_test_act, axis=1)
        if return_std:
            return y_mean, np.sqrt(var)
        return y_mean

class RelevanceVectorClassifier:
    """
    関連ベクトルマシン分類器 (Relevance Vector Machine for Classification, PRML 7.2.3節)
    ラプラス近似 + 超パラメータ alpha_i の自動推定
    """
    def __init__(self, kernel=rbf_kernel, alpha_threshold=1e4, max_iter=100, **kernel_kwargs):
        self.kernel = kernel
        self.alpha_threshold = alpha_threshold
        self.max_iter = max_iter
        self.kernel_kwargs = kernel_kwargs
        self.X_train = None
        self.t_train = None
        self.alpha = None
        self.w = None
        self.Sigma = None
        self.rv_indices = None
        self.rv_X = None

    def fit(self, X, t):
        self.X_train = np.atleast_2d(X)
        self.t_train = np.asarray(t, dtype=float).ravel() # t in {0, 1}
        if set(np.unique(self.t_train)).issubset({-1, 1}):
            self.t_train = (self.t_train + 1.0) / 2.0
            
        N = len(self.X_train)
        K = self.kernel(self.X_train, self.X_train, **self.kernel_kwargs)
        Phi = np.column_stack([np.ones(N), K])
        M = Phi.shape[1]
        
        self.alpha = np.ones(M) * 1.0
        active = np.ones(M, dtype=bool)
        w = np.zeros(M)
        
        for iteration in range(self.max_iter):
            Phi_act = Phi[:, active]
            alpha_act = self.alpha[active]
            w_act = w[active]
            
            # IRLS (Newton-Raphson) による重み w_act の MAP 推定 (PRML 式 7.110, 7.111)
            for _ in range(15):
                y = sigmoid(Phi_act @ w_act)
                W_diag = y * (1.0 - y)
                H = np.diag(alpha_act) + (Phi_act.T * W_diag) @ Phi_act
                H_inv = np.linalg.pinv(H)
                grad = Phi_act.T @ (self.t_train - y) - alpha_act * w_act
                w_step = H_inv @ grad
                w_act = w_act + w_step
                if np.max(np.abs(w_step)) < 1e-4:
                    break
                    
            # 事後共分散
            y = sigmoid(Phi_act @ w_act)
            W_diag = y * (1.0 - y)
            Sigma = np.linalg.pinv(np.diag(alpha_act) + (Phi_act.T * W_diag) @ Phi_act)
            
            # alpha_i の更新 (PRML 式 7.112)
            gamma = 1.0 - alpha_act * np.diag(Sigma)
            alpha_new = gamma / (w_act**2 + 1e-12)
            
            # スパース化
            active_idx = np.where(active)[0]
            prune = alpha_new > self.alpha_threshold
            active[active_idx[prune]] = False
            self.alpha[active_idx[~prune]] = alpha_new[~prune]
            self.alpha[active_idx[prune]] = np.inf
            
            w[active] = w_act[~prune]
            w[~active] = 0.0
            
            if np.sum(prune) == 0 and np.max(np.abs(alpha_act - alpha_new)) < 1e-2:
                break
                
        # 最終状態
        Phi_act = Phi[:, active]
        alpha_act = self.alpha[active]
        w_act = w[active]
        y = sigmoid(Phi_act @ w_act)
        W_diag = y * (1.0 - y)
        self.Sigma = np.linalg.pinv(np.diag(alpha_act) + (Phi_act.T * W_diag) @ Phi_act)
        self.w = w_act
        self.active_mask = active
        
        rv_in_K = active[1:]
        self.rv_indices = np.where(rv_in_K)[0]
        self.rv_X = self.X_train[self.rv_indices]
        return self

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        N_test = len(X)
        K_test = self.kernel(X, self.X_train, **self.kernel_kwargs)
        Phi_test = np.column_stack([np.ones(N_test), K_test])
        Phi_test_act = Phi_test[:, self.active_mask]
        
        # 決定関数の平均と分散
        mu_a = Phi_test_act @ self.w
        sigma_a2 = np.sum((Phi_test_act @ self.Sigma) * Phi_test_act, axis=1)
        # プロビット畳み込み
        kappa = 1.0 / np.sqrt(1.0 + np.pi * sigma_a2 / 8.0)
        return sigmoid(kappa * mu_a)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
