import numpy as np

class PCA:
    """
    標準的主成分分析 (PRML 12.1節)
    最大分散 / 最小再構成誤差の固有値問題 S u = lambda u
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None # (M, D)
        self.explained_variance_ = None # (M,)
        self.explained_variance_ratio_ = None # (M,)

    def fit(self, X):
        N, D = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # サンプル共分散行列 S (D, D)
        # N << D の場合は X @ X.T の N x N 行列で計算可能 (PRML 12.1.4)
        if N < D:
            K_mat = (X_centered @ X_centered.T) / N # (N, N)
            vals, vecs = np.linalg.eigh(K_mat)
            idx = np.argsort(vals)[::-1]
            vals = np.maximum(vals[idx], 0.0)
            # v_i = X.T @ u_i / sqrt(N * lambda_i)
            U = []
            for i in range(min(self.n_components, N)):
                if vals[i] > 1e-10:
                    u = (X_centered.T @ vecs[:, idx[i]]) / np.sqrt(N * vals[i])
                    U.append(u / np.linalg.norm(u))
                else:
                    U.append(np.zeros(D))
            self.components_ = np.array(U)
            self.explained_variance_ = vals[:self.n_components]
        else:
            S = (X_centered.T @ X_centered) / N
            vals, vecs = np.linalg.eigh(S)
            idx = np.argsort(vals)[::-1]
            vals = np.maximum(vals[idx], 0.0)
            self.components_ = vecs[:, idx[:self.n_components]].T # (M, D)
            self.explained_variance_ = vals[:self.n_components]
            
        total_var = np.sum(vals)
        self.explained_variance_ratio_ = self.explained_variance_ / (total_var + 1e-12)
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T # (N, M)

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_ # (N, D)


class ProbabilisticPCA:
    """
    確率的主成分分析 (PPCA, PRML 12.2節)
    x = W z + mu + eps, z ~ N(0, I), eps ~ N(0, sigma^2 I)
    """
    def __init__(self, n_components=2, method='closed_form', max_iter=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.mean_ = None
        self.W_ = None        # (D, M)
        self.sigma2_ = None   # スカラーノイズ分散
        self.M_inv_ = None    # M = W^T W + sigma^2 I の逆行列

    def fit(self, X):
        N, D = X.shape
        M = self.n_components
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        S = (X_centered.T @ X_centered) / N
        
        if self.method == 'closed_form':
            # クローズドフォーム最尤解 (PRML 式 12.45, 12.46)
            vals, vecs = np.linalg.eigh(S)
            idx = np.argsort(vals)[::-1]
            vals = np.maximum(vals[idx], 1e-10)
            vecs = vecs[:, idx]
            
            # sigma_ML^2 = 1/(D-M) * sum_{i=M+1}^D lambda_i
            if D > M:
                self.sigma2_ = np.sum(vals[M:]) / (D - M)
            else:
                self.sigma2_ = 1e-6
                
            # W_ML = U_M (L_M - sigma^2 I)^{1/2}
            U_M = vecs[:, :M]
            L_M = np.diag(vals[:M])
            diff = np.maximum(L_M - self.sigma2_ * np.eye(M), 0.0)
            self.W_ = U_M @ np.sqrt(diff)
            
        elif self.method == 'em':
            # EMアルゴリズムによる解法 (PRML 12.2.2節)
            np.random.seed(self.random_state)
            self.W_ = np.random.randn(D, M)
            self.sigma2_ = 1.0
            
            for it in range(self.max_iter):
                # Eステップ: M_mat = W^T W + sigma^2 I
                M_mat = self.W_.T @ self.W_ + self.sigma2_ * np.eye(M)
                M_inv = np.linalg.inv(M_mat)
                
                # E[z_n] = M_inv W^T (x_n - mu)
                # E[z_n z_n^T] = sigma^2 M_inv + E[z_n] E[z_n]^T
                E_z = (M_inv @ self.W_.T @ X_centered.T).T # (N, M)
                E_zz = N * self.sigma2_ * M_inv + E_z.T @ E_z # (M, M)
                
                # Mステップ (式 12.56, 12.57)
                W_new = (X_centered.T @ E_z) @ np.linalg.inv(E_zz)
                
                diff_norm = np.sum(X_centered**2)
                term2 = -2 * np.trace(E_z @ W_new.T @ X_centered.T)
                term3 = np.trace(E_zz @ (W_new.T @ W_new))
                sigma2_new = (diff_norm + term2 + term3) / (N * D)
                sigma2_new = max(sigma2_new, 1e-10)
                
                if np.allclose(self.W_, W_new, atol=self.tol):
                    self.W_ = W_new
                    self.sigma2_ = sigma2_new
                    break
                self.W_ = W_new
                self.sigma2_ = sigma2_new
                
        # 事後分布用 M_inv
        M_mat = self.W_.T @ self.W_ + self.sigma2_ * np.eye(M)
        self.M_inv_ = np.linalg.inv(M_mat)
        return self

    def transform(self, X):
        # 潜在変数の事後平均 E[z | x] (PRML 式 12.42)
        X_centered = X - self.mean_
        return (self.M_inv_ @ self.W_.T @ X_centered.T).T # (N, M)

    def inverse_transform(self, Z):
        # 再構成 W E[z|x] + mu
        return Z @ self.W_.T + self.mean_


class KernelPCA:
    """
    カーネル主成分分析 (Kernel PCA, PRML 12.3節)
    グラム行列 K_tilde = K - 1_N K - K 1_N + 1_N K 1_N
    """
    def __init__(self, n_components=2, kernel='rbf', gamma=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.X_fit_ = None
        self.alphas_ = None # (N, M)
        self.lambdas_ = None # (M,)

    def _pairwise_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        if self.kernel == 'rbf':
            diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            return np.exp(-self.gamma * dist_sq)
        elif self.kernel == 'linear':
            return X @ Y.T
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X):
        self.X_fit_ = X
        N = len(X)
        K = self._pairwise_kernel(X)
        
        # グラム行列の中心化 (PRML 式 12.80)
        one_N = np.ones((N, N)) / N
        K_tilde = K - one_N @ K - K @ one_N + one_N @ K @ one_N
        
        vals, vecs = np.linalg.eigh(K_tilde)
        idx = np.argsort(vals)[::-1]
        vals = np.maximum(vals[idx], 0.0)
        vecs = vecs[:, idx]
        
        # 固有ベクトルの正規化: lambda_i * (a_i^T a_i) = 1 (式 12.81)
        valid_m = min(self.n_components, N)
        self.lambdas_ = vals[:valid_m]
        self.alphas_ = np.zeros((N, valid_m))
        for i in range(valid_m):
            if self.lambdas_[i] > 1e-10:
                self.alphas_[:, i] = vecs[:, i] / np.sqrt(self.lambdas_[i])
                
        return self

    def transform(self, X):
        K_test = self._pairwise_kernel(X, self.X_fit_) # (N_test, N_train)
        N_train = len(self.X_fit_)
        N_test = len(X)
        one_N_test = np.ones((N_test, N_train)) / N_train
        one_N_train = np.ones((N_train, N_train)) / N_train
        
        K_fit = self._pairwise_kernel(self.X_fit_)
        K_test_tilde = K_test - one_N_test @ K_fit - K_test @ one_N_train + one_N_test @ K_fit @ one_N_train
        return K_test_tilde @ self.alphas_
