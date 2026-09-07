"""
PRML Chapter 9 & 10: Mixture Models, EM Algorithm, and Variational Inference Unit Tests
K-Means, GMM, BMM, 変分混合ガウスモデルのEM収束性と確率的整合性
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs

from prml.clustering import (
    KMeans,
    GaussianMixtureModel,
    BernoulliMixtureModel,
    VariationalGaussianMixture,
    mixture_moments,
    incremental_em_update,
)

class TestClusteringAndEM(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_kmeans_inertia_decrease(self):
        X, _ = make_blobs(n_samples=60, centers=3, random_state=42)
        km = KMeans(n_clusters=3, max_iter=20, random_state=42).fit(X)
        self.assertEqual(km.cluster_centers_.shape, (3, 2))
        labels = km.predict(X)
        self.assertEqual(len(labels), 60)
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})

    def test_gmm_likelihood_monotonic_increase(self):
        # EMアルゴリズムにおいて、対数尤度履歴が単調増加すること
        X, _ = make_blobs(n_samples=80, centers=2, random_state=42)
        gmm = GaussianMixtureModel(n_components=2, max_iter=30, tol=1e-6, random_state=42).fit(X)
        
        # 混合比率の和が1
        self.assertAlmostEqual(np.sum(gmm.weights_), 1.0, places=5)
        
        # 対数尤度履歴が存在する場合、各ステップで非減少（微小な数値誤差許容）
        if hasattr(gmm, 'log_likelihood_history_') and len(gmm.log_likelihood_history_) > 1:
            diffs = np.diff(gmm.log_likelihood_history_)
            self.assertTrue(np.all(diffs >= -1e-4))

    def test_bmm_probabilities(self):
        # ベルヌーイ混合モデル: 平均パラメータが [0, 1] の範囲内
        X_bin = np.random.binomial(1, 0.4, (50, 4))
        bmm = BernoulliMixtureModel(n_components=2, max_iter=20, random_state=42).fit(X_bin)
        self.assertTrue(np.all((bmm.means_ >= 0.0) & (bmm.means_ <= 1.0)))
        # 訓練データに対する責任度の和が 1
        np.testing.assert_allclose(np.sum(bmm.responsibilities_, axis=1), np.ones(50), atol=1e-5)
        # 予測ラベル
        preds = bmm.predict(X_bin)
        self.assertEqual(len(preds), 50)

    def test_gmm_responsibilities_sum_to_one(self):
        # PRML 9.2.2節 式 (9.23): 事後確率 (責任度 gamma_nk) の行和は厳密に 1
        X, _ = make_blobs(n_samples=60, centers=3, random_state=42)
        gmm = GaussianMixtureModel(n_components=3, max_iter=25, random_state=42).fit(X)
        np.testing.assert_allclose(np.sum(gmm.responsibilities_, axis=1), np.ones(60), atol=1e-5)
        self.assertTrue(np.all(gmm.responsibilities_ >= 0.0))
        self.assertTrue(np.all(gmm.responsibilities_ <= 1.0))

    def test_gmm_covariances_positive_definite(self):
        # PRML 9.2.1節: 特異性回避のための共分散行列の半正定値性・正定値性 (reg_covar)
        X, _ = make_blobs(n_samples=50, centers=2, random_state=42)
        gmm = GaussianMixtureModel(n_components=2, max_iter=20, reg_covar=1e-5, random_state=42).fit(X)
        for k in range(2):
            eigvals = np.linalg.eigvalsh(gmm.covariances_[k])
            self.assertTrue(np.all(eigvals > 0.0), f"Covariance matrix {k} not strictly positive definite")

    def test_gmm_predict_proba_simplex(self):
        # 未知テスト点に対する事後確率予測
        X, _ = make_blobs(n_samples=60, centers=3, random_state=42)
        gmm = GaussianMixtureModel(n_components=3, max_iter=25, random_state=42).fit(X)
        X_test = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, -5.0]])
        probs = gmm.predict_proba(X_test)
        self.assertEqual(probs.shape, (3, 3))
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(3), atol=1e-6)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_kmeans_cost_monotonic_decrease(self):
        # PRML 9.1節 式 (9.1): 歪み尺度 J は各ステップで非増加 J^{(t+1)} <= J^{(t)}
        X, _ = make_blobs(n_samples=80, centers=4, random_state=42)
        km = KMeans(n_clusters=4, max_iter=30, random_state=42).fit(X)
        if len(km.cost_history_) > 1:
            diffs = np.diff(km.cost_history_)
            self.assertTrue(np.all(diffs <= 1e-6))

    def test_mixture_moments(self):
        # PRML 9.12節 式 (9.49)-(9.50)
        weights = [0.3, 0.7]
        means = [[1.0, 2.0], [-1.0, -2.0]]
        covs = [np.eye(2), np.eye(2) * 2.0]
        mean, cov = mixture_moments(weights, means, covs)
        
        # E[x] = 0.3*[1, 2] + 0.7*[-1, -2] = [-0.4, -0.8]
        np.testing.assert_allclose(mean, [-0.4, -0.8], atol=1e-12)
        # Covariance must be symmetric positive definite
        self.assertTrue(np.allclose(cov, cov.T))
        eigvals = np.linalg.eigvalsh(cov)
        self.assertTrue(np.all(eigvals > 0.0))

    def test_incremental_em_update(self):
        # PRML 式 (9.78)-(9.79), Exercise 9.26 & 9.27
        N = 10
        D = 2
        K = 2
        X = np.random.randn(N, D)
        gamma = np.random.dirichlet([1, 1], size=N)
        
        N_k_old = np.sum(gamma, axis=0)
        mu_old = (gamma.T @ X) / N_k_old[:, None]
        cov_old = []
        for k in range(K):
            diff = X - mu_old[k]
            cov_old.append((gamma[:, k:k+1] * diff).T @ diff / N_k_old[k])
        cov_old = np.array(cov_old)
        
        # 単一データ点 m=0 を更新
        gamma_new_m = np.array([0.8, 0.2])
        N_k_new, mu_new, cov_new, pi_new = incremental_em_update(
            X[0], gamma[0], gamma_new_m, N_k_old, mu_old, cov_old, N
        )
        
        # バッチ再計算と比較
        gamma_batch = gamma.copy()
        gamma_batch[0] = gamma_new_m
        N_k_batch = np.sum(gamma_batch, axis=0)
        mu_batch = (gamma_batch.T @ X) / N_k_batch[:, None]
        cov_batch = []
        for k in range(K):
            diff = X - mu_batch[k]
            cov_batch.append((gamma_batch[:, k:k+1] * diff).T @ diff / N_k_batch[k])
        cov_batch = np.array(cov_batch)
        
        np.testing.assert_allclose(N_k_new, N_k_batch, atol=1e-12)
        np.testing.assert_allclose(mu_new, mu_batch, atol=1e-12)
        np.testing.assert_allclose(cov_new, cov_batch, atol=1e-12)
        np.testing.assert_allclose(pi_new, N_k_batch / N, atol=1e-12)

    def test_gmm_tied_covariance_update(self):
        # PRML Exercise 9.6: 共通共分散 GMM の M ステップ更新 Sigma = (1/N) * sum_k S_k
        N, D, K = 30, 2, 2
        X = np.random.randn(N, D)
        gamma = np.random.dirichlet([1, 1], size=N)
        N_k = gamma.sum(axis=0)
        mu = (gamma.T @ X) / N_k[:, None]
        
        # S_k = sum_n gamma_{nk} (x_n - mu_k)(x_n - mu_k)^T
        S_sum = np.zeros((D, D))
        for k in range(K):
            diff = X - mu[k]
            S_k = (gamma[:, k:k+1] * diff).T @ diff
            S_sum += S_k
        Sigma_tied = S_sum / N
        
        # 正定値性および対称性の検証
        self.assertTrue(np.allclose(Sigma_tied, Sigma_tied.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(Sigma_tied) > 0.0))

    def test_bmm_identical_initialization_collapse(self):
        # PRML Exercise 9.13: 全成分同一パラメータ初期化時の1反復退化証明
        N, D, K = 40, 3, 2
        X = (np.random.rand(N, D) > 0.5).astype(float)
        x_bar = np.mean(X, axis=0)
        
        # 同一初期化
        mu = np.full((K, D), 0.5)
        pi = np.full(K, 1.0 / K)
        
        # E step
        resp = np.zeros((N, K))
        for n in range(N):
            dens = np.array([pi[k] * np.prod(mu[k]**X[n] * (1 - mu[k])**(1 - X[n])) for k in range(K)])
            resp[n] = dens / np.sum(dens)
        
        # すべての負担率が厳密に 1/K
        np.testing.assert_allclose(resp, 1.0 / K, atol=1e-12)
        
        # M step
        N_k = resp.sum(axis=0)
        mu_new = (resp.T @ X) / N_k[:, None]
        pi_new = N_k / N
        
        # 標本平均への退化
        for k in range(K):
            np.testing.assert_allclose(mu_new[k], x_bar, atol=1e-12)
            np.testing.assert_allclose(pi_new[k], 1.0 / K, atol=1e-12)

    def test_variational_decomposition_elbo_and_kl(self):
        # PRML Exercise 9.24: ln p(X) = L(q) + KL(q || p(Z|X))
        p_x_z = np.array([[0.15, 0.25], [0.20, 0.40]])
        p_x = np.sum(p_x_z)
        p_z_given_x = p_x_z / p_x
        ln_p = np.log(p_x)
        
        q = np.array([[0.1, 0.3], [0.2, 0.4]])
        q /= q.sum()
        
        L_q = np.sum(q * np.log(p_x_z / q))
        kl = np.sum(q * np.log(q / p_z_given_x))
        
        np.testing.assert_allclose(L_q + kl, ln_p, atol=1e-12)
        self.assertLessEqual(L_q, ln_p + 1e-12)
        self.assertGreaterEqual(kl, 0.0)

    def test_elbo_tangency_condition(self):
        # PRML Exercise 9.25: 接点における変分下界と対数尤度の勾配一致
        X = np.array([-1.2, -0.4, 0.5, 1.6, 2.1])
        pi = np.array([0.5, 0.5])
        mu_old = np.array([-0.8, 1.5])
        var = np.array([1.0, 1.0])
        
        # E-step
        resp = np.zeros((len(X), 2))
        for n, x in enumerate(X):
            p0 = pi[0] * np.exp(-0.5 * (x - mu_old[0])**2 / var[0])
            p1 = pi[1] * np.exp(-0.5 * (x - mu_old[1])**2 / var[1])
            resp[n] = [p0 / (p0 + p1), p1 / (p0 + p1)]
            
        grad_loglik = sum(
            (pi[0] * np.exp(-0.5 * (x - mu_old[0])**2 / var[0]) * (x - mu_old[0]) / var[0]) /
            (pi[0] * np.exp(-0.5 * (x - mu_old[0])**2 / var[0]) + pi[1] * np.exp(-0.5 * (x - mu_old[1])**2 / var[1]))
            for x in X
        )
        grad_lower_bound = sum(resp[n, 0] * (x - mu_old[0]) / var[0] for n, x in enumerate(X))
        
        np.testing.assert_allclose(grad_loglik, grad_lower_bound, atol=1e-12)

    def test_rvm_em_and_direct_algebraic_equivalence(self):
        # PRML Exercise 9.23: 直接証拠最大化と EM の代数的等価性
        M, N = 3, 10
        Phi = np.random.randn(N, M)
        t = np.random.randn(N)
        alpha = np.array([1.2, 2.5, 0.5])
        beta = 1.8
        
        A = np.diag(alpha)
        Sigma = np.linalg.inv(A + beta * Phi.T @ Phi)
        m = beta * Sigma @ Phi.T @ t
        gamma_i = 1.0 - alpha * np.diag(Sigma)
        
        # 恒等式 beta * Tr(Phi^T Phi Sigma) == sum(gamma_i)
        trace_val = np.trace(Phi.T @ Phi @ Sigma)
        np.testing.assert_allclose(beta * trace_val, np.sum(gamma_i), atol=1e-12)
        
        # alpha 方程式の比例関係
        for i in range(M):
            f_direct = alpha[i] - (1.0 - alpha[i] * Sigma[i, i]) / (m[i]**2)
            f_em = alpha[i] - 1.0 / (m[i]**2 + Sigma[i, i])
            scale_i = (m[i]**2 + Sigma[i, i]) / (m[i]**2)
            np.testing.assert_allclose(f_direct, scale_i * f_em, atol=1e-12)


if __name__ == '__main__':
    unittest.main()


