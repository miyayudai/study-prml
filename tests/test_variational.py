import unittest
import numpy as np
import prml


class TestVariationalInference(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_variational_gaussian_1d_cavi(self):
        true_mu = 2.5
        true_sigma = 1.2
        true_tau = 1.0 / (true_sigma ** 2)
        X = np.random.normal(loc=true_mu, scale=true_sigma, size=300)
        
        cavi = prml.VariationalGaussian1D(max_iter=30)
        cavi.fit(X)
        
        self.assertIsNotNone(cavi.mean_mu)
        self.assertIsNotNone(cavi.mean_tau)
        self.assertAlmostEqual(cavi.mean_mu, np.mean(X), places=3)
        self.assertGreater(cavi.mean_tau, 0.0)
        self.assertAlmostEqual(cavi.mean_tau, 1.0 / np.var(X), delta=0.1)

    def test_jaakkola_jordan_bound(self):
        # xi = 0 should give 1/8 = 0.125
        val_0 = prml.jaakkola_jordan_lambda(0.0)
        self.assertAlmostEqual(float(val_0), 0.125)
        
        # Symmetry: lambda(xi) == lambda(-xi)
        xis = np.array([0.5, 1.0, 2.5, -1.0])
        lambdas = prml.jaakkola_jordan_lambda(xis)
        self.assertAlmostEqual(float(lambdas[1]), float(lambdas[3]))


    def test_variational_gaussian_mixture_pruning(self):
        # PRML 10.2.5節: 変分ベイズGMMにおける過剰クラスタの自動消去 (ARD特性)
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=100, centers=2, cluster_std=0.6, random_state=42)
        
        # 6個のクラスタ成分を指定して初期化
        vbgmm = prml.VariationalGaussianMixture(n_components=6, max_iter=40, random_state=42).fit(X)
        
        # 事後混合パラメータ alpha_k は正
        self.assertEqual(len(vbgmm.alpha_), 6)
        self.assertTrue(np.all(vbgmm.alpha_ > 0))
        
        # 各サンプルの責任度和は1
        np.testing.assert_allclose(np.sum(vbgmm.responsibilities_, axis=1), np.ones(100), atol=1e-5)
        
        # 有効なクラスタ数 (支配的なサンプル数を持つクラスタ) は真のクラスタ数 (2個程度) に絞られる
        effective_clusters = np.sum(vbgmm.alpha_ > 5.0)
        self.assertLessEqual(effective_clusters, 4)

    def test_variational_gaussian_1d_direct_updates(self):
        # PRML 10.1.3節: variational_gaussian_1d 直接更新公式 (式 10.26 - 10.30)
        from common.variational_utils import variational_gaussian_1d
        X = np.random.normal(loc=1.0, scale=0.5, size=200) # tau = 4.0
        history = variational_gaussian_1d(X, mu_0=0.0, lambda_0=1.0, a_0=1.0, b_0=1.0, max_iter=25)
        self.assertEqual(len(history), 25)
        mu_final, lambda_final, a_final, b_final = history[-1]
        
        # 平均 mu_N は事前分布と尤度の精度加重平均 (式 10.26) と厳密に一致
        expected_mu = (1.0 * 0.0 + len(X) * np.mean(X)) / (1.0 + len(X))
        self.assertAlmostEqual(mu_final, expected_mu, places=5)
        # E[tau] = a_N / b_N は真の精度 4.0 (誤差0.6以内)
        e_tau = a_final / b_final
        self.assertAlmostEqual(e_tau, 4.0, delta=0.6)


if __name__ == '__main__':
    unittest.main()

