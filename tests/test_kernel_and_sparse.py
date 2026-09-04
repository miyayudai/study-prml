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

    def test_rvm_sparsity(self):
        # RVM のスパース性（関連ベクトル数が全サンプル数より著しく少ないこと）
        X = np.linspace(-2, 2, 30)[:, np.newaxis]
        y = np.sinc(X).ravel() + np.random.normal(0, 0.05, 30)
        
        rvr = RelevanceVectorRegressor(kernel='rbf', gamma=1.0, max_iter=80).fit(X, y)
        preds = rvr.predict(X, return_std=False)
        self.assertEqual(len(preds), 30)
        # 関連ベクトル数が30未満に刈り込まれる
        self.assertLess(len(rvr.rv_indices), 30)

if __name__ == '__main__':
    unittest.main()
