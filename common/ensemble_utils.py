import numpy as np

class DecisionStump:
    """
    1次元軸に平行な閾値による2値弱分類器 (PRML 14.3節)
    t in {-1, +1}
    """
    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.polarity = 1 # 1なら x >= thr が +1, -1なら x < thr が +1

    def fit(self, X, y, sample_weight):
        N, D = X.shape
        min_err = float('inf')
        
        for d in range(D):
            x_col = X[:, d]
            sorted_x = np.sort(np.unique(x_col))
            # 閾値候補: 隣接する値の中点
            if len(sorted_x) == 1:
                thresholds = sorted_x
            else:
                thresholds = (sorted_x[:-1] + sorted_x[1:]) / 2.0
                
            for thr in thresholds:
                for polarity in [1, -1]:
                    preds = np.ones(N)
                    if polarity == 1:
                        preds[x_col < thr] = -1
                    else:
                        preds[x_col >= thr] = -1
                        
                    err = np.sum(sample_weight[preds != y])
                    if err < min_err:
                        min_err = err
                        self.feature_idx = d
                        self.threshold = thr
                        self.polarity = polarity
                        
        return self

    def predict(self, X):
        N = X.shape[0]
        preds = np.ones(N)
        x_col = X[:, self.feature_idx]
        if self.polarity == 1:
            preds[x_col < self.threshold] = -1
        else:
            preds[x_col >= self.threshold] = -1
        return preds


class AdaBoostClassifier:
    """
    AdaBoost アルゴリズム (PRML 14.3節)
    指数損失関数の逐次最小化
    """
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []
        self.weights_history = []

    def fit(self, X, y):
        N = X.shape[0]
        # 初期重み w_n = 1 / N (式 14.14)
        w = np.ones(N) / N
        self.models = []
        self.alphas = []
        self.weights_history = [w.copy()]
        
        for m in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, sample_weight=w)
            preds = stump.predict(X)
            
            # 誤分類率 epsilon_m (式 14.16)
            err = np.sum(w[preds != y]) / np.sum(w)
            err = np.clip(err, 1e-10, 1.0 - 1e-10)
            
            # 分類器重み alpha_m (式 14.17)
            alpha = 0.5 * np.log((1.0 - err) / err)
            
            # データ点重みの更新 (式 14.18)
            w = w * np.exp(-alpha * y * preds)
            w /= np.sum(w) # 正規化
            
            self.models.append(stump)
            self.alphas.append(alpha)
            self.weights_history.append(w.copy())
            
        return self

    def predict(self, X):
        """符号による複合予測 (式 14.19)"""
        final_preds = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final_preds += alpha * model.predict(X)
        return np.sign(final_preds)

    def staged_predict(self, X):
        """各ブースティングステップでの予測値の推移"""
        current_preds = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            current_preds += alpha * model.predict(X)
            yield np.sign(current_preds)


class MixtureOfLinearRegressions:
    """
    線形回帰混合モデル (PRML 14.5.1節)
    EM アルゴリズムによる最尤推定
    """
    def __init__(self, n_components=2, max_iter=80, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.weights_ = None # (K, D)
        self.vars_ = None    # (K,)
        self.pi_ = None      # (K,)

    def fit(self, X, t):
        np.random.seed(self.random_state)
        N = X.shape[0]
        K = self.n_components
        
        # バイアス項の追加 [1, x]
        Phi = np.hstack([np.ones((N, 1)), X])
        D = Phi.shape[1]
        
        # 初期化
        self.pi_ = np.ones(K) / K
        self.vars_ = np.ones(K) * np.var(t)
        self.weights_ = np.random.randn(K, D) * 0.5
        
        for it in range(self.max_iter):
            # Eステップ: 負担率 gamma_nk (式 14.36)
            responsibilities = np.zeros((N, K))
            for k in range(K):
                diff = t - Phi @ self.weights_[k]
                dens = (1.0 / np.sqrt(2.0 * np.pi * self.vars_[k])) * np.exp(-0.5 * (diff**2) / self.vars_[k])
                responsibilities[:, k] = self.pi_[k] * np.maximum(dens, 1e-300)
                
            resp_sum = np.sum(responsibilities, axis=1, keepdims=True)
            gamma = responsibilities / (resp_sum + 1e-300)
            
            # Mステップ: パラメータ更新 (式 14.38, 14.39, 14.40)
            N_k = np.sum(gamma, axis=0) # (K,)
            self.pi_ = N_k / N
            
            for k in range(K):
                # 重み付き最小二乗 w_k = (Phi^T R_k Phi)^-1 Phi^T R_k t
                R_k = np.diag(gamma[:, k])
                self.weights_[k] = np.linalg.solve(Phi.T @ R_k @ Phi + 1e-6 * np.eye(D), Phi.T @ (gamma[:, k] * t))
                
                diff = t - Phi @ self.weights_[k]
                self.vars_[k] = np.sum(gamma[:, k] * (diff**2)) / N_k[k]
                self.vars_[k] = max(self.vars_[k], 1e-4)
                
        return self

    def predict_density(self, x_grid, t_grid):
        """予測条件付き密度 p(t|x) の評価 (PRML Figure 14.9)"""
        # x_grid: (Nx,), t_grid: (Nt,)
        Nx, Nt = len(x_grid), len(t_grid)
        density = np.zeros((Nt, Nx))
        
        Phi_grid = np.column_stack([np.ones(Nx), x_grid])
        
        for k in range(self.n_components):
            mu_k = Phi_grid @ self.weights_[k] # (Nx,)
            sigma_k = np.sqrt(self.vars_[k])
            for i in range(Nx):
                diff = t_grid - mu_k[i]
                comp_pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma_k)) * np.exp(-0.5 * (diff / sigma_k)**2)
                density[:, i] += self.pi_[k] * comp_pdf
                
        return density

    def predict(self, X):
        """期待値予測 y(x) = sum_k pi_k * (w_k^T phi(x))"""
        X = np.atleast_2d(X)
        N, D = X.shape
        Phi = np.column_stack([np.ones(N), X])
        y_pred = np.zeros(N)
        for k in range(self.n_components):
            y_pred += self.pi_[k] * (Phi @ self.weights_[k])
        return y_pred

