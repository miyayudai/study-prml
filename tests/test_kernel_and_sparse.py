"""
PRML Chapter 6 & 7: Kernel Methods and Sparse Kernel Machines Unit Tests
ガウス過程、カーネルリッジ、SVM、RVM の厳密な検証
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs

from prml.kernel import (
    KernelRidgeRegression,
    NadarayaWatsonRegressor,
    GaussianProcessRegressor,
    GaussianProcessClassifier,
    SupportVectorClassifier,
    SupportVectorRegressor,
    RelevanceVectorRegressor,
    RelevanceVectorClassifier,
)

class TestKernelMethods(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_kernel_ridge_rbf(self):
        X = np.linspace(-2, 2, 25)[:, np.newaxis]
        y = np.cos(X).ravel()
        krr = KernelRidgeRegression(kernel='rbf', gamma=1.0, reg_lambda=0.01).fit(X, y)
        preds = krr.predict(X)
        mse = np.mean((preds - y)**2)
        self.assertLess(mse, 0.05)

    def test_nadaraya_watson(self):
        X = np.linspace(0, 3, 20)[:, np.newaxis]
        y = np.sin(X).ravel()
        nw = NadarayaWatsonRegressor(kernel='gaussian', h=0.5).fit(X, y)
        preds = nw.predict(X)
        self.assertEqual(len(preds), 20)

    def test_gaussian_process_exact_interpolation(self):
        # GP において観測ノイズ beta が極めて大きいとき、訓練点を高精度で補間する
        X = np.array([[-1.0], [0.0], [1.0]])
        y = np.array([0.5, 1.2, -0.8])
        
        gpr = GaussianProcessRegressor(beta=1e4, theta0=1.0, theta1=1.0).fit(X, y)
        mean, var = gpr.predict(X)
        np.testing.assert_allclose(mean, y, atol=1e-2)
        # 訓練点での事後分散は微小
        self.assertTrue(np.all(var < 0.05))

    def test_gaussian_process_classifier(self):
        X, y = make_blobs(n_samples=30, n_features=2, centers=2, random_state=42)
        gpc = GaussianProcessClassifier(gamma=1.0).fit(X, y)
        probs = gpc.predict_proba(X)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_gram_matrix_positive_semidefinite(self):
        # PRML 6.2節 式 (6.13): 有効なカーネルのグラム行列は常に半正定値 K >= 0 (Mercer's theorem)
        X = np.random.randn(20, 3)
        
        # 1. ガウス / RBF カーネル
        from common.kernel_utils import rbf_kernel, linear_kernel
        K_rbf = rbf_kernel(X, X, length_scale=1.5)
        eigvals_rbf = np.linalg.eigvalsh(K_rbf)
        self.assertTrue(np.all(eigvals_rbf >= -1e-10))

        # 2. 線形カーネル K = X @ X.T
        K_lin = linear_kernel(X, X)
        eigvals_lin = np.linalg.eigvalsh(K_lin)
        self.assertTrue(np.all(eigvals_lin >= -1e-10))

    def test_nadaraya_watson_partition_of_unity(self):
        # PRML 6.3.1節 式 (6.43): Nadaraya-Watson 核回帰の有効重みの総和は 1
        # y(x) = sum_n k(x, x_n) t_n where sum_n k(x, x_n) == 1
        X = np.linspace(-1, 1, 15)[:, np.newaxis]
        y = X.ravel()**2
        nw = NadarayaWatsonRegressor(kernel='gaussian', h=0.3).fit(X, y)
        
        X_test = np.array([[-0.5], [0.0], [0.7]])
        # 各テスト点に対する全訓練サンプルへのカーネル重みの総和を計算
        dists_sq = (X_test - X.T)**2
        weights = np.exp(-0.5 * dists_sq / (0.3**2))
        normalized_weights = weights / np.sum(weights, axis=1, keepdims=True)
        np.testing.assert_allclose(np.sum(normalized_weights, axis=1), np.ones(3), atol=1e-10)

        preds = nw.predict(X_test)
        preds_direct = normalized_weights @ y
        np.testing.assert_allclose(preds, preds_direct, atol=1e-10)


class TestSparseKernelMachines(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_svc_dual_margin(self):
        X, y = make_blobs(n_samples=40, centers=2, cluster_std=0.8, random_state=42)
        y_pm = np.where(y == 0, -1, 1)
        
        svc = SupportVectorClassifier(C=10.0, kernel='rbf', gamma=0.5).fit(X, y_pm)
        preds = svc.predict(X)
        acc = np.mean(preds == y_pm)
        self.assertGreaterEqual(acc, 0.95)
        # サポートベクトルが存在すること
        self.assertGreater(len(svc.sv_indices), 0)

    def test_svc_kkt_complementarity(self):
        # PRML 7.1.1節 式 (7.21): KKT 相補性条件 a_n (t_n y(x_n) - 1 + xi_n) = 0
        X, y = make_blobs(n_samples=30, centers=2, cluster_std=0.7, random_state=42)
        y_pm = np.where(y == 0, -1, 1)
        
        svc = SupportVectorClassifier(C=5.0, kernel='linear').fit(X, y_pm)
        # 非サポートベクトル (a_n == 0) ではマージン条件 t_n y(x_n) >= 1
        decision = svc.decision_function(X)
        functional_margin = y_pm * decision
        
        non_sv_idx = np.where(svc.a < 1e-5)[0]
        self.assertTrue(np.all(functional_margin[non_sv_idx] >= 1.0 - 1e-3))

    def test_rvm_sparsity(self):
        # RVM のスパース性（関連ベクトル数が全サンプル数より著しく少ないこと）
        X = np.linspace(-2, 2, 30)[:, np.newaxis]
        y = np.sinc(X).ravel() + np.random.normal(0, 0.05, 30)
        
        rvr = RelevanceVectorRegressor(kernel='rbf', gamma=1.0, max_iter=80).fit(X, y)
        preds = rvr.predict(X, return_std=False)
        self.assertEqual(len(preds), 30)
        # 関連ベクトル数が30未満に刈り込まれる
        self.assertLess(len(rvr.rv_indices), 30)

    def test_rvc_classification_sparsity(self):
        # PRML 7.2.3節: 関連ベクトル分類器 (RVC) によるスパース決定境界
        X, y = make_blobs(n_samples=30, centers=2, cluster_std=1.0, random_state=42)
        rvc = RelevanceVectorClassifier(kernel='rbf', gamma=0.5, max_iter=50).fit(X, y)
        probs = rvc.predict_proba(X)
        self.assertEqual(len(probs), 30)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        self.assertLess(len(rvc.rv_indices), len(X))

    def test_svr_epsilon_tube_and_sparsity(self):
        # PRML 7.1.4節: サポートベクトル回帰 (SVR)
        X = np.linspace(-2, 2, 25)[:, np.newaxis]
        y = np.cos(X).ravel()
        svr = SupportVectorRegressor(C=10.0, epsilon=0.1, kernel='rbf', gamma=0.5).fit(X, y)
        preds = svr.predict(X)
        mse = np.mean((preds - y)**2)
        self.assertLess(mse, 0.05)
        # a_n * a_hat_n == 0 (PRML Ex 7.10)
        prod = svr.a * svr.a_hat
        np.testing.assert_allclose(prod, np.zeros_like(prod), atol=1e-5)
        # sum(a - a_hat) == 0 (PRML Eq 7.66)
        self.assertAlmostEqual(np.sum(svr.a - svr.a_hat), 0.0, places=4)


if __name__ == '__main__':
    unittest.main()

