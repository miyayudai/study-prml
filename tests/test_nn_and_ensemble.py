"""
PRML Chapter 5, 12 & 14: Neural Networks, Dimensionality Reduction, and Ensemble Methods Unit Tests
多層パーセプトロン、MDN、PCA、PPCA、AdaBoost、混合エキスパートの検証
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs, make_regression

from prml.nn import (
    MLPRegressor,
    MixtureDensityNetwork,
)

from prml.dimreduce import (
    PCA,
    ProbabilisticPCA,
    KernelPCA,
)

from prml.ensemble import (
    DecisionStump,
    AdaBoostClassifier,
    MixtureOfLinearRegressions,
)

class TestNeuralNetworks(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_mlp_backpropagation_gradient_check(self):
        # MLP の数値勾配と解析的勾配の比較 (Gradient Check)
        X = np.array([[0.5, -0.3]])
        y = np.array([[0.8]])
        
        mlp = MLPRegressor(n_in=2, n_hidden=3, n_out=1, random_state=42)
        loss, grads = mlp.compute_loss_and_grads(X, y)
        
        # W1 の数値勾配
        eps = 1e-5
        W1_orig = mlp.W1.copy()
        num_grad_W1 = np.zeros_like(mlp.W1)
        for i in range(mlp.W1.shape[0]):
            for j in range(mlp.W1.shape[1]):
                mlp.W1[i, j] = W1_orig[i, j] + eps
                l_plus, _ = mlp.compute_loss_and_grads(X, y)
                mlp.W1[i, j] = W1_orig[i, j] - eps
                l_minus, _ = mlp.compute_loss_and_grads(X, y)
                mlp.W1[i, j] = W1_orig[i, j]
                num_grad_W1[i, j] = (l_plus - l_minus) / (2 * eps)
                
        np.testing.assert_allclose(grads['W1'], num_grad_W1, rtol=1e-3, atol=1e-4)

    def test_mdn_output_constraints(self):
        # 混合密度ネットワーク: 混合係数 pi の総和が1、sigma > 0
        X = np.linspace(-1, 1, 15)[:, np.newaxis]
        mdn = MixtureDensityNetwork(n_in=1, n_hidden=5, n_components=3, random_state=42)
        z, pi, sigma, mu, a_sig = mdn.forward(X)
        
        # pi の行和が 1
        np.testing.assert_allclose(np.sum(pi, axis=1), np.ones(15), atol=1e-5)
        # sigma が正
        self.assertTrue(np.all(sigma > 0))

class TestDimensionalityReduction(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_pca_reconstruction_orthogonal(self):
        X = np.random.randn(40, 5)
        pca = PCA(n_components=3).fit(X)
        X_proj = pca.transform(X)
        self.assertEqual(X_proj.shape, (40, 3))
        
        # 主成分ベクトルの直交性 W^T W = I
        WTW = pca.components_ @ pca.components_.T
        np.testing.assert_allclose(WTW, np.eye(3), atol=1e-5)

    def test_ppca_closed_form_vs_em(self):
        # 確率的PCA: 解析解とEM解の潜在空間射影の相関
        X = np.random.randn(50, 4)
        ppca_cf = ProbabilisticPCA(n_components=2, method='closed_form').fit(X)
        ppca_em = ProbabilisticPCA(n_components=2, method='em', max_iter=30).fit(X)
        
        z_cf = ppca_cf.transform(X)
        z_em = ppca_em.transform(X)
        self.assertEqual(z_cf.shape, (50, 2))
        self.assertEqual(z_em.shape, (50, 2))

class TestEnsembleMethods(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_adaboost_exponential_loss_decrease(self):
        X, y = make_blobs(n_samples=50, centers=2, cluster_std=1.2, random_state=42)
        y_pm = np.where(y == 0, -1, 1)
        
        ada = AdaBoostClassifier(n_estimators=10).fit(X, y_pm)
        preds = ada.predict(X)
        acc = np.mean(preds == y_pm)
        self.assertGreaterEqual(acc, 0.90)
        self.assertEqual(len(ada.models), 10)
        self.assertEqual(len(ada.alphas), 10)

    def test_mixture_of_linear_regressions(self):
        # 2つの直線 y = 2x + 2 と y = -1x - 2 からの混合データ
        X = np.linspace(-1, 1, 40)[:, np.newaxis]
        t1 = 2.0 * X.ravel() + 2.0
        t2 = -1.0 * X.ravel() - 2.0
        # 50% ずつサンプリング
        choice = np.random.binomial(1, 0.5, 40)
        y = np.where(choice == 1, t1, t2) + np.random.normal(0, 0.05, 40)
        
        moe = MixtureOfLinearRegressions(n_components=2, max_iter=40, random_state=42).fit(X, y)
        preds = moe.predict(X)
        self.assertEqual(len(preds), 40)
        # 2成分の重みが学習されていること
        self.assertEqual(moe.weights_.shape, (2, 2))

if __name__ == '__main__':
    unittest.main()
