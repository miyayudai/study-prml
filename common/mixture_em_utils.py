import numpy as np
from scipy.stats import multivariate_normal

class KMeans:
    """
    K-means クラスタリングアルゴリズム (PRML 9.1節)
    歪み尺度 J = sum_n sum_k r_{nk} ||x_n - mu_k||^2 を最小化
    """
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.cost_history_ = []

    def fit(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        K = self.n_clusters
        
        # 初期クラスタ中心のランダム選択
        init_indices = np.random.choice(N, K, replace=False)
        self.cluster_centers_ = X[init_indices].copy()
        
        self.cost_history_ = []
        
        for iteration in range(self.max_iter):
            # 1. 割り当てステップ (r_nk の計算)
            # 各データ点から全クラスタ中心への二乗距離 (N, K)
            diff = X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :] # (N, K, D)
            dist_sq = np.sum(diff**2, axis=-1) # (N, K)
            self.labels_ = np.argmin(dist_sq, axis=1) # (N,)
            
            # 歪み尺度 J の記録
            cost = np.sum(np.min(dist_sq, axis=1))
            self.cost_history_.append(cost)
            
            # 2. 中心更新ステップ
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(K):
                cluster_pts = X[self.labels_ == k]
                if len(cluster_pts) > 0:
                    new_centers[k] = np.mean(cluster_pts, axis=0)
                else:
                    new_centers[k] = self.cluster_centers_[k]
                    
            if np.allclose(self.cluster_centers_, new_centers, atol=self.tol):
                break
            self.cluster_centers_ = new_centers
            
        return self

    def predict(self, X):
        diff = X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        return np.argmin(dist_sq, axis=1)


class GaussianMixtureModel:
    """
    混合ガウスモデル (GMM) と EMアルゴリズム (PRML 9.2節, 9.3節)
    p(x) = sum_k pi_k N(x | mu_k, Sigma_k)
    """
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.weights_ = None # (K,)
        self.means_ = None   # (K, D)
        self.covariances_ = None # (K, D, D)
        self.responsibilities_ = None # (N, K)
        self.log_likelihood_history_ = []

    def fit(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        K = self.n_components
        
        # 初期化: K-means を用いて中心を初期化
        km = KMeans(n_clusters=K, max_iter=20, random_state=self.random_state)
        km.fit(X)
        self.means_ = km.cluster_centers_.copy()
        self.weights_ = np.ones(K) / K
        self.covariances_ = np.array([np.cov(X, rowvar=False) + self.reg_covar * np.eye(D) for _ in range(K)])
        
        self.log_likelihood_history_ = []
        
        for iteration in range(self.max_iter):
            # 1. Eステップ: 負担率 gamma(z_nk) の評価
            weighted_densities = np.zeros((N, K))
            for k in range(K):
                # 共分散の正定値性を保証
                cov_k = self.covariances_[k] + self.reg_covar * np.eye(D)
                norm = multivariate_normal(mean=self.means_[k], cov=cov_k, allow_singular=True)
                weighted_densities[:, k] = self.weights_[k] * norm.pdf(X)
                
            total_densities = np.sum(weighted_densities, axis=1, keepdims=True)
            total_densities[total_densities < 1e-300] = 1e-300
            self.responsibilities_ = weighted_densities / total_densities
            
            # 対数尤度の記録
            log_likelihood = np.sum(np.log(np.sum(weighted_densities, axis=1) + 1e-300))
            self.log_likelihood_history_.append(log_likelihood)
            
            # 収束判定
            if iteration > 0 and abs(self.log_likelihood_history_[-1] - self.log_likelihood_history_[-2]) < self.tol:
                break
                
            # 2. Mステップ: パラメータ pi_k, mu_k, Sigma_k の再推定
            N_k = np.sum(self.responsibilities_, axis=0) # (K,)
            
            # 混合比 pi_k
            self.weights_ = N_k / N
            
            # 平均 mu_k
            self.means_ = (self.responsibilities_.T @ X) / N_k[:, np.newaxis]
            
            # 共分散 Sigma_k
            for k in range(K):
                diff = X - self.means_[k] # (N, D)
                self.covariances_[k] = (self.responsibilities_[:, k:k+1] * diff).T @ diff / N_k[k]
                self.covariances_[k] += self.reg_covar * np.eye(D)
                
        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        N, D = X.shape
        K = self.n_components
        weighted_densities = np.zeros((N, K))
        for k in range(K):
            cov_k = self.covariances_[k] + self.reg_covar * np.eye(D)
            norm = multivariate_normal(mean=self.means_[k], cov=cov_k, allow_singular=True)
            weighted_densities[:, k] = self.weights_[k] * norm.pdf(X)
        total_densities = np.sum(weighted_densities, axis=1, keepdims=True)
        total_densities[total_densities < 1e-300] = 1e-300
        return weighted_densities / total_densities


class BernoulliMixtureModel:
    """
    ベルヌーイ混合モデル (Bernoulli Mixture Model: BMM, PRML 9.3.3節)
    p(x | mu_k) = prod_{i=1}^D mu_{ki}^{x_i} (1 - mu_{ki})^{1 - x_i}
    手書き数字の二値画像データ等に適用
    """
    def __init__(self, n_components=3, max_iter=50, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights_ = None # (K,)
        self.means_ = None   # (K, D) [0, 1]
        self.responsibilities_ = None # (N, K)
        self.log_likelihood_history_ = []

    def fit(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        K = self.n_components
        
        # 初期化: 指定がなければ平均を [0.25, 0.75] でランダムに初期化
        if self.means_ is None:
            self.means_ = np.random.uniform(0.25, 0.75, size=(K, D))
        if self.weights_ is None:
            self.weights_ = np.ones(K) / K
        self.log_likelihood_history_ = []
        
        for iteration in range(self.max_iter):
            # 1. Eステップ: 対数尤度領域で計算して数値安定化
            # ln p(x | mu_k) = sum_i [ x_i ln mu_{ki} + (1 - x_i) ln (1 - mu_{ki}) ]
            eps = 1e-12
            mu = np.clip(self.means_, eps, 1.0 - eps)
            log_p_x_given_k = np.zeros((N, K))
            for k in range(K):
                log_p_x_given_k[:, k] = X @ np.log(mu[k]) + (1.0 - X) @ np.log(1.0 - mu[k])
                
            log_joint = log_p_x_given_k + np.log(self.weights_) # (N, K)
            # logsumexp で規格化
            max_log = np.max(log_joint, axis=1, keepdims=True)
            exp_term = np.exp(log_joint - max_log)
            sum_exp = np.sum(exp_term, axis=1, keepdims=True)
            self.responsibilities_ = exp_term / sum_exp
            
            ll = np.sum(max_log + np.log(sum_exp))
            self.log_likelihood_history_.append(ll)
            
            if iteration > 0 and abs(self.log_likelihood_history_[-1] - self.log_likelihood_history_[-2]) < self.tol:
                break
                
            # 2. Mステップ
            N_k = np.sum(self.responsibilities_, axis=0) # (K,)
            self.weights_ = N_k / N
            self.means_ = (self.responsibilities_.T @ X) / (N_k[:, np.newaxis] + eps)
            
        return self

    def predict(self, X):
        eps = 1e-12
        mu = np.clip(self.means_, eps, 1.0 - eps)
        N, D = X.shape
        K = self.n_components
        log_joint = np.zeros((N, K))
        return np.argmax(log_joint, axis=1)


def mixture_moments(weights, means, covariances):
    """
    混合分布の全平均と全共分散の計算 (PRML 式 9.49, 9.50, 演習 9.12)
    E[x] = sum_k pi_k mu_k
    cov[x] = sum_k pi_k { Sigma_k + (mu_k - E[x])(mu_k - E[x])^T }
    """
    weights = np.asarray(weights)
    means = np.asarray(means)
    covariances = np.asarray(covariances)
    
    K, D = means.shape
    overall_mean = np.sum(weights[:, np.newaxis] * means, axis=0)
    overall_cov = np.zeros((D, D))
    for k in range(K):
        diff = means[k] - overall_mean
        overall_cov += weights[k] * (covariances[k] + np.outer(diff, diff))
        
    return overall_mean, overall_cov


def incremental_em_update(x_m, gamma_old_m, gamma_new_m, N_k_old, mu_old, cov_old, N_total):
    """
    GMM インクリメンタル EM アルゴリズムの単一データ点更新 (PRML 式 9.78, 9.79, 演習 9.26, 9.27)
    """
    x_m = np.asarray(x_m)
    gamma_old_m = np.asarray(gamma_old_m)
    gamma_new_m = np.asarray(gamma_new_m)
    N_k_old = np.asarray(N_k_old)
    mu_old = np.asarray(mu_old)
    cov_old = np.asarray(cov_old)
    
    delta_gamma = gamma_new_m - gamma_old_m
    N_k_new = N_k_old + delta_gamma
    weights_new = N_k_new / N_total
    
    K, D = mu_old.shape
    mu_new = np.zeros_like(mu_old)
    cov_new = np.zeros_like(cov_old)
    
    for k in range(K):
        mu_new[k] = mu_old[k] + (delta_gamma[k] / N_k_new[k]) * (x_m - mu_old[k])
        S_old = N_k_old[k] * (cov_old[k] + np.outer(mu_old[k], mu_old[k]))
        S_new = S_old + delta_gamma[k] * np.outer(x_m, x_m)
        cov_new[k] = S_new / N_k_new[k] - np.outer(mu_new[k], mu_new[k])
        
    return N_k_new, mu_new, cov_new, weights_new

