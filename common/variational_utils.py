import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import norm, gamma as gamma_dist

def variational_gaussian_1d(X, mu_0=0.0, lambda_0=0.0, a_0=0.0, b_0=0.0, max_iter=20):
    """
    1変量ガウス分布の平均 mu と精度 tau に対する変分推論 (PRML 10.1.3節, Figure 10.4)
    p(D | mu, tau) = prod_n N(x_n | mu, tau^-1)
    事前分布: p(mu | tau) = N(mu | mu_0, (lambda_0 tau)^-1), p(tau) = Gam(tau | a_0, b_0)
    因数分解近似: q(mu, tau) = q_mu(mu) * q_tau(tau)
    """
    N = len(X)
    x_bar = np.mean(X)
    
    # パラメータ初期化
    mu_N = mu_0
    lambda_N = lambda_0 + N
    a_N = a_0 + (N + 1) / 2.0
    
    # E[tau] の初期値
    E_tau = 1.0
    
    history = []
    
    for iteration in range(max_iter):
        # 1. q_mu(mu) の更新 (式 10.26, 10.27)
        mu_N = (lambda_0 * mu_0 + N * x_bar) / (lambda_0 + N)
        lambda_N = (lambda_0 + N) * E_tau
        
        # 2. q_tau(tau) の更新 (式 10.29, 10.30)
        E_mu = mu_N
        E_mu_sq = mu_N**2 + 1.0 / lambda_N
        
        # E[(x_n - mu)^2] = x_n^2 - 2 x_n E[mu] + E[mu^2]
        term1 = np.sum(X**2) - 2 * np.sum(X) * E_mu + N * E_mu_sq
        term2 = lambda_0 * (E_mu_sq - 2 * mu_0 * E_mu + mu_0**2)
        b_N = b_0 + 0.5 * (term1 + term2)
        
        E_tau = a_N / b_N
        history.append((mu_N, lambda_N, a_N, b_N))
        
    return history

class VariationalGaussianMixture:
    """
    変分ベイズ混合ガウスモデル (PRML 10.2節)
    ディリクレ・ガウス・ウィシャート共役事前分布によるフルベイズ推論
    自動適合クラスタ数判定 (ARD 的消去)
    """
    def __init__(self, n_components=6, alpha_0=1e-3, beta_0=1.0, nu_0=None, max_iter=100, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.responsibilities_ = None # (N, K)
        self.alpha_ = None            # (K,)
        self.m_ = None                # (K, D)
        self.W_ = None                # (K, D, D)
        self.nu_ = None               # (K,)

    def fit(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        K = self.n_components
        
        # 事前パラメータ設定
        if self.nu_0 is None:
            self.nu_0 = D
        self.m_0 = np.mean(X, axis=0)
        self.W_0 = np.eye(D)
        self.W_0_inv = np.linalg.inv(self.W_0)
        
        # 初期化: 負担率をランダムに初期化 (ディリクレ分布)
        self.responsibilities_ = np.random.dirichlet(np.ones(K), size=N)
        
        for iteration in range(self.max_iter):
            # Mステップ相当 (変分事後パラメータの更新, PRML 10.51 - 10.63)
            N_k = np.sum(self.responsibilities_, axis=0) + 1e-10 # (K,)
            x_bar_k = (self.responsibilities_.T @ X) / N_k[:, np.newaxis] # (K, D)
            
            # S_k の計算
            S_k = np.zeros((K, D, D))
            for k in range(K):
                diff = X - x_bar_k[k]
                S_k[k] = (self.responsibilities_[:, k:k+1] * diff).T @ diff / N_k[k]
                
            # alpha_k (ディリクレ事後パラメータ)
            self.alpha_ = self.alpha_0 + N_k
            
            # beta_k
            beta_k = self.beta_0 + N_k
            
            # m_k (ガウス平均事後パラメータ)
            self.m_ = (self.beta_0 * self.m_0 + N_k[:, np.newaxis] * x_bar_k) / beta_k[:, np.newaxis]
            
            # W_k_inv, nu_k (ウィシャート事後パラメータ)
            self.nu_ = self.nu_0 + N_k
            self.W_ = np.zeros((K, D, D))
            for k in range(K):
                mean_diff = x_bar_k[k] - self.m_0
                W_k_inv = (self.W_0_inv + N_k[k] * S_k[k] + 
                           (self.beta_0 * N_k[k] / beta_k[k]) * np.outer(mean_diff, mean_diff))
                self.W_[k] = np.linalg.inv(W_k_inv)
                
            # Eステップ相当 (負担率 r_{nk} の更新, PRML 式 10.64 - 10.67)
            # E[ln pi_k] = psi(alpha_k) - psi(sum alpha)
            E_ln_pi = digamma(self.alpha_) - digamma(np.sum(self.alpha_))
            
            # E[ln |Lambda_k|]
            E_ln_det_Lambda = np.zeros(K)
            for k in range(K):
                sign, logdet = np.linalg.slogdet(self.W_[k])
                psi_sum = np.sum([digamma((self.nu_[k] + 1 - (i + 1)) / 2.0) for i in range(D)])
                E_ln_det_Lambda[k] = psi_sum + D * np.log(2.0) + logdet
                
            # E[(x - mu_k)^T Lambda_k (x - mu_k)]
            ln_rho = np.zeros((N, K))
            for k in range(K):
                diff_m = X - self.m_[k] # (N, D)
                quad = np.sum((diff_m @ self.W_[k]) * diff_m, axis=1) * self.nu_[k] # (N,)
                quad += D / beta_k[k]
                ln_rho[:, k] = E_ln_pi[k] + 0.5 * E_ln_det_Lambda[k] - 0.5 * D * np.log(2 * np.pi) - 0.5 * quad
                
            # 負担率 r_{nk} = softmax(ln_rho)
            max_ln_rho = np.max(ln_rho, axis=1, keepdims=True)
            rho = np.exp(ln_rho - max_ln_rho)
            self.responsibilities_ = rho / np.sum(rho, axis=1, keepdims=True)
            
        return self

def jaakkola_jordan_lambda(xi):
    """
    Jaakkola & Jordan ロジスティック変分境界関数 (PRML 式 10.144)
    lambda(xi) = (1 / 2xi) * (sigma(xi) - 1/2) = (1 / 4xi) * tanh(xi / 2)
    """
    xi = np.asarray(xi, dtype=float)
    # xi -> 0 での極限値は 1/8
    result = np.zeros_like(xi)
    zero_mask = np.isclose(xi, 0.0, atol=1e-8)
    result[zero_mask] = 0.125
    
    nz = ~zero_mask
    result[nz] = (1.0 / (4.0 * xi[nz])) * np.tanh(xi[nz] / 2.0)
    return result
