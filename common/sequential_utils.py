import numpy as np
from scipy.stats import multivariate_normal

class GaussianHMM:
    """
    ガウス放出隠れマルコフモデル (PRML 13.2節)
    スケーリング係数 c_n を用いた数値安定な Forward-Backward および Baum-Welch EM
    """
    def __init__(self, n_components=3, n_iter=50, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        
        self.pi_ = None      # 初期確率 (K,)
        self.A_ = None       # 遷移確率行列 (K, K) A_jk = p(z_k | z_j)
        self.means_ = None   # 放出平均 (K, D)
        self.covs_ = None    # 放出共分散 (K, D, D)

    def _emission_probs(self, X):
        N, D = X.shape
        K = self.n_components
        B = np.zeros((N, K))
        for k in range(K):
            # 正則化を加えて特異行列を防止
            cov = self.covs_[k] + 1e-6 * np.eye(D)
            dist = multivariate_normal(mean=self.means_[k], cov=cov)
            B[:, k] = dist.pdf(X)
        return np.maximum(B, 1e-300)

    def _forward(self, B):
        N, K = B.shape
        alpha = np.zeros((N, K))
        c = np.zeros(N)
        
        # 初期ステップ n=0
        alpha[0] = self.pi_ * B[0]
        c[0] = np.sum(alpha[0])
        alpha[0] /= c[0]
        
        # 再帰ステップ (式 13.59)
        for n in range(1, N):
            alpha[n] = (alpha[n-1] @ self.A_) * B[n]
            c[n] = np.sum(alpha[n])
            alpha[n] /= (c[n] + 1e-300)
            
        return alpha, c

    def _backward(self, B, c):
        N, K = B.shape
        beta = np.zeros((N, K))
        
        # 最終ステップ n=N-1 (式 13.62)
        beta[-1] = 1.0
        
        # 後ろ向き再帰 (式 13.62)
        for n in range(N - 2, -1, -1):
            beta[n] = self.A_ @ (B[n+1] * beta[n+1])
            beta[n] /= (c[n+1] + 1e-300)
            
        return beta

    def fit(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        K = self.n_components
        
        # 初期化
        self.pi_ = np.ones(K) / K
        self.A_ = np.ones((K, K)) / K
        # K-means 類似の初期平均
        indices = np.random.choice(N, K, replace=False)
        self.means_ = X[indices].copy()
        self.covs_ = np.array([np.cov(X.T) + 0.1 * np.eye(D) for _ in range(K)])
        
        prev_log_lik = -np.inf
        
        for it in range(self.n_iter):
            B = self._emission_probs(X)
            alpha, c = self._forward(B)
            beta = self._backward(B, c)
            
            # 対数尤度 ln p(X) = sum_n ln c_n (式 13.63)
            log_lik = np.sum(np.log(c + 1e-300))
            if abs(log_lik - prev_log_lik) < self.tol:
                break
            prev_log_lik = log_lik
            
            # Eステップ: gamma(z_n) と xi(z_{n-1}, z_n) の計算 (式 13.64, 13.65)
            gamma = alpha * beta # (N, K)
            gamma /= np.sum(gamma, axis=1, keepdims=True)
            
            xi = np.zeros((N - 1, K, K))
            for n in range(N - 1):
                xi[n] = (alpha[n][:, np.newaxis] * self.A_) * (B[n+1] * beta[n+1])
                xi[n] /= (c[n+1] + 1e-300)
            
            # Mステップ: パラメータ更新 (式 13.18, 13.19, 13.20, 13.21)
            self.pi_ = gamma[0] / np.sum(gamma[0])
            self.A_ = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]
            self.A_ /= np.sum(self.A_, axis=1, keepdims=True)
            
            gamma_sum = np.sum(gamma, axis=0) # (K,)
            for k in range(K):
                self.means_[k] = np.sum(gamma[:, k:k+1] * X, axis=0) / gamma_sum[k]
                diff = X - self.means_[k]
                self.covs_[k] = (gamma[:, k:k+1] * diff).T @ diff / gamma_sum[k]
                self.covs_[k] += 1e-4 * np.eye(D)
                
        return self

    def predict(self, X):
        """
        ビタビアルゴリズムによる最確潜在状態系列の探索 (PRML 13.2.5節)
        """
        N, D = X.shape
        K = self.n_components
        B = self._emission_probs(X)
        
        # 対数領域での動的計画法
        log_pi = np.log(self.pi_ + 1e-300)
        log_A = np.log(self.A_ + 1e-300)
        log_B = np.log(B + 1e-300)
        
        omega = np.zeros((N, K))
        psi = np.zeros((N, K), dtype=int)
        
        # 初期化 n=0
        omega[0] = log_pi + log_B[0]
        
        # 前向き再帰
        for n in range(1, N):
            for k in range(K):
                trans_probs = omega[n-1] + log_A[:, k]
                psi[n, k] = np.argmax(trans_probs)
                omega[n, k] = trans_probs[psi[n, k]] + log_B[n, k]
                
        # バックトラッキング
        best_path = np.zeros(N, dtype=int)
        best_path[-1] = np.argmax(omega[-1])
        for n in range(N - 2, -1, -1):
            best_path[n] = psi[n+1, best_path[n+1]]
            
        return best_path

    def sample(self, n_samples=100, random_state=42):
        """HMM からのサンプリング系列生成 (PRML Figure 13.8)"""
        np.random.seed(random_state)
        states = []
        obs = []
        
        # 初期状態
        current_state = np.random.choice(self.n_components, p=self.pi_)
        states.append(current_state)
        obs.append(np.random.multivariate_normal(self.means_[current_state], self.covs_[current_state]))
        
        for _ in range(n_samples - 1):
            current_state = np.random.choice(self.n_components, p=self.A_[current_state])
            states.append(current_state)
            obs.append(np.random.multivariate_normal(self.means_[current_state], self.covs_[current_state]))
            
        return np.array(states), np.array(obs)


class KalmanFilter:
    """
    線形動的システム (LDS) のカルマンフィルタ & スムーザ (PRML 13.3節)
    z_n = A z_{n-1} + w_n,  w_n ~ N(0, Gamma)
    x_n = C z_n + v_n,      v_n ~ N(0, Sigma)
    """
    def __init__(self, A, C, Gamma, Sigma, mu_0, V_0):
        self.A = np.array(A, dtype=float)
        self.C = np.array(C, dtype=float)
        self.Gamma = np.array(Gamma, dtype=float)
        self.Sigma = np.array(Sigma, dtype=float)
        self.mu_0 = np.array(mu_0, dtype=float)
        self.V_0 = np.array(V_0, dtype=float)

    def filter(self, X):
        """カルマンフィルタ (前向き再帰, PRML 式 13.89 - 13.92)"""
        N = len(X)
        M = len(self.mu_0)
        
        mu = np.zeros((N, M))
        V = np.zeros((N, M, M))
        
        # 予測と更新
        mu_prev = self.mu_0
        V_prev = self.V_0
        
        for n in range(N):
            # 予測ステップ (Prior)
            P_n = self.A @ V_prev @ self.A.T + self.Gamma
            mu_pred = self.A @ mu_prev
            
            # カルマンゲイン K_n (式 13.92)
            S_n = self.C @ P_n @ self.C.T + self.Sigma
            K_n = P_n @ self.C.T @ np.linalg.inv(S_n)
            
            # 観測更新ステップ (Posterior)
            y_res = X[n] - self.C @ mu_pred
            mu[n] = mu_pred + K_n @ y_res
            V[n] = (np.eye(M) - K_n @ self.C) @ P_n
            
            mu_prev = mu[n]
            V_prev = V[n]
            
        return mu, V

    def smooth(self, X):
        """カルマンスムーザ (RTS 後ろ向き再帰, PRML 式 13.98 - 13.102)"""
        mu_filt, V_filt = self.filter(X)
        N = len(X)
        M = len(self.mu_0)
        
        mu_smooth = np.zeros((N, M))
        V_smooth = np.zeros((N, M, M))
        
        mu_smooth[-1] = mu_filt[-1]
        V_smooth[-1] = V_filt[-1]
        
        for n in range(N - 2, -1, -1):
            P_next = self.A @ V_filt[n] @ self.A.T + self.Gamma
            J_n = V_filt[n] @ self.A.T @ np.linalg.inv(P_next)
            
            mu_smooth[n] = mu_filt[n] + J_n @ (mu_smooth[n+1] - self.A @ mu_filt[n])
            V_smooth[n] = V_filt[n] + J_n @ (V_smooth[n+1] - P_next) @ J_n.T
            
        return mu_smooth, V_smooth
