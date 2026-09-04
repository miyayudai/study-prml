"""
PRML Chapter 3 & 4: Linear Models Unit Test Suite
線形回帰および線形分類モデルの厳密な数理的性質・境界条件・収束性テスト
"""

import unittest
import numpy as np
from sklearn.datasets import make_regression, make_blobs

from prml.linear import (
    LinearRegression,
    RidgeRegression,
    BayesianLinearRegression,
    EvidenceApproximation,
    PolynomialBasis,
    GaussianBasis,
    Perceptron,
    FisherLinearDiscriminant,
    GaussianGenerativeClassifier,
    LogisticRegression,
    MulticlassLogisticRegression,
    BayesianLogisticRegression,
)

class TestLinearRegressionModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_ols_exact_fit(self):
        # 決定論的データでの OLS (バイアス項を含む計画行列 Phi)
        X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
        Phi = np.hstack([np.ones((len(X), 1)), X])
        true_w = np.array([3.0, 1.5, -2.0])
        t = Phi @ true_w
        
        lr = LinearRegression().fit(Phi, t)
        preds = lr.predict(Phi)
        np.testing.assert_allclose(preds, t, atol=1e-5)
        np.testing.assert_allclose(lr.w, true_w, atol=1e-5)

    def test_ridge_shrinkage(self):
        # Ridge 正則化の収縮特性 (alpha が大 -> 重みノルム縮小)
        X, y = make_regression(n_samples=40, n_features=5, noise=1.0, random_state=42)
        Phi = np.hstack([np.ones((len(X), 1)), X])
        ridge_small = RidgeRegression(alpha=0.01).fit(Phi, y)
        ridge_large = RidgeRegression(alpha=100.0).fit(Phi, y)
        
        norm_small = np.linalg.norm(ridge_small.w)
        norm_large = np.linalg.norm(ridge_large.w)
        self.assertGreater(norm_small, norm_large)

    def test_bayesian_linear_uncertainty(self):
        # ベイズ線形回帰: 予測分散 s^2(x) >= beta^{-1}
        X = np.linspace(-1, 1, 15)[:, np.newaxis]
        y = np.sin(np.pi * X).ravel() + np.random.normal(0, 0.1, 15)
        
        beta = 100.0
        blr = BayesianLinearRegression(alpha=1.0, beta=beta).fit(X, y)
        
        # 訓練データ範囲内と範囲外の点
        X_test = np.array([[-0.5], [0.0], [2.5]])  # 2.5 は外挿点
        mean, var = blr.predict(X_test)
        
        # 予測分散は観測ノイズ以上
        self.assertTrue(np.all(var >= 1.0 / beta - 1e-8))
        # データから遠い外挿領域 (x=2.5) では不確実性がデータ中心部 (x=0) より大
        self.assertGreater(var[2], var[1])

    def test_basis_expansions(self):
        X = np.linspace(-1, 1, 10)[:, np.newaxis]
        poly = PolynomialBasis(degree=3)
        Phi_poly = poly.transform(X)
        self.assertEqual(Phi_poly.shape, (10, 4))
        
        centers = np.array([[-0.5], [0.0], [0.5]])
        gauss = GaussianBasis(centers=centers, scale=0.5)
        Phi_gauss = gauss.transform(X)
        self.assertEqual(Phi_gauss.shape, (10, 4))  # intercept + 3 centers

class TestLinearClassificationModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_perceptron_convergence(self):
        # 線形分離可能データでのパーセプトロン収束
        X = np.array([[2.0, 2.0], [3.0, 3.0], [-2.0, -2.0], [-3.0, -3.0]])
        y = np.array([1, 1, -1, -1])
        pct = Perceptron(max_iter=50).fit(X, y)
        preds = pct.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_fisher_linear_discriminant(self):
        X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.0, random_state=42)
        fld = FisherLinearDiscriminant().fit(X, y)
        acc = np.mean(fld.predict(X) == y)
        self.assertGreaterEqual(acc, 0.95)

    def test_logistic_regression_irls(self):
        # IRLS (Newton-Raphson) ロジスティック回帰
        X, y = make_blobs(n_samples=60, n_features=2, centers=2, random_state=42)
        lr = LogisticRegression(max_iter=30).fit(X, y)
        probs = lr.predict_proba(X)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        acc = np.mean(lr.predict(X) == y)
        self.assertGreater(acc, 0.90)

    def test_bayesian_logistic_laplace(self):
        # ラプラス近似ベイズロジスティック回帰
        X, y = make_blobs(n_samples=50, n_features=2, centers=2, random_state=42)
        Phi = np.hstack([np.ones((len(X), 1)), X])
        blr = BayesianLogisticRegression(alpha=1.0, max_iter=20).fit(Phi, y)
        preds = blr.predict(Phi)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.90)
        self.assertEqual(blr.S_N.shape, (3, 3))  # 2 features + bias

if __name__ == '__main__':
    unittest.main()
