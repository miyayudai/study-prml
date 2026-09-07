"""
PRML Chapter 3 Comprehensive Regression Test Suite
線形回帰モデル、正則化、ベイズ推定、エビデンス近似、バイアス-バリアンス分解の完全テスト
"""

import unittest
import numpy as np

from prml.linear import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    LeastMeanSquares,
    BayesianLinearRegression,
    NormalGammaLinearRegression,
    EvidenceApproximation,
    LocallyWeightedRegression,
    RobustLinearRegression,
    PolynomialBasis,
    GaussianBasis,
    SigmoidalBasis,
    FourierBasis,
    orthogonal_projection_matrix,
    equivalent_kernel_matrix,
    bias_variance_decomposition,
)


class TestComprehensiveRegressionSuite(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_orthogonal_projection_complement(self):
        # 射影行列 P と直交補空間射影 (I - P) の数学的性質
        N, M = 30, 5
        Phi = np.random.randn(N, M)
        P = orthogonal_projection_matrix(Phi)
        I = np.eye(N)
        P_perp = I - P

        # 1. P_perp のべき等性と対称性
        np.testing.assert_allclose(P_perp @ P_perp, P_perp, atol=1e-10)
        np.testing.assert_allclose(P_perp.T, P_perp, atol=1e-10)

        # 2. P と P_perp の直交性: P @ P_perp = 0
        np.testing.assert_allclose(P @ P_perp, np.zeros((N, N)), atol=1e-10)

        # 3. トレース定理: Tr(P) = M, Tr(P_perp) = N - M
        self.assertAlmostEqual(float(np.trace(P)), float(M), places=6)
        self.assertAlmostEqual(float(np.trace(P_perp)), float(N - M), places=6)

    def test_linear_regression_log_likelihood_and_score(self):
        # 最尤推定の対数尤度と R^2 決定係数のテスト
        X = np.linspace(0, 2, 40)[:, None]
        poly = PolynomialBasis(degree=2)
        Phi = poly.transform(X)
        true_w = np.array([1.0, -2.0, 1.5])
        noise_std = 0.05
        t = Phi @ true_w + np.random.normal(0, noise_std, size=len(X))

        lr = LinearRegression().fit(Phi, t)
        r2 = lr.score(Phi, t)
        self.assertGreater(r2, 0.98)

        log_lik = lr.log_likelihood(Phi, t)
        self.assertIsInstance(log_lik, float)
        self.assertFalse(np.isnan(log_lik))

    def test_ridge_shrinkage_monotonicity(self):
        # 正則化パラメータ alpha が増加すると有効自由度が単調減少する
        X = np.random.randn(50, 6)
        Phi = np.hstack([np.ones((50, 1)), X])
        y = np.random.randn(50)

        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        dfs = []
        for a in alphas:
            ridge = RidgeRegression(alpha=a).fit(Phi, y)
            dfs.append(ridge.effective_degrees_of_freedom(Phi))

        # 単調減少の検証
        for i in range(len(dfs) - 1):
            self.assertGreater(dfs[i], dfs[i+1])

    def test_lasso_exact_zero_coefficients(self):
        # Lasso による不要な特徴量の完全ゼロ化 (L1 スパース性)
        N = 80
        X = np.random.randn(N, 8)
        true_w = np.array([3.0, -2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        t = X @ true_w + np.random.normal(0, 0.05, size=N)

        lasso = LassoRegression(alpha=6.0, max_iter=1000).fit(X, t)
        sparsity = lasso.get_sparsity()
        self.assertGreaterEqual(sparsity, 0.6)
        np.testing.assert_allclose(lasso.w[2:], np.zeros(6), atol=1e-3)

    def test_bayesian_linear_function_sampling(self):
        # ベイズ線形回帰: 事後分布からのパラメータサンプルおよび関数サンプルの次元と値
        X = np.linspace(-1, 1, 25)[:, None]
        poly = PolynomialBasis(degree=2)
        Phi = poly.transform(X)
        t = 0.5 * X.ravel()**2 - 0.2 + np.random.normal(0, 0.1, size=len(X))

        blr = BayesianLinearRegression(alpha=1.0, beta=20.0).fit(Phi, t)
        samples_w = blr.sample_weights(n_samples=7)
        self.assertEqual(samples_w.shape, (7, 3))

        X_eval = np.linspace(-1.2, 1.2, 50)[:, None]
        Phi_eval = poly.transform(X_eval)
        sample_funcs = blr.sample_functions(Phi_eval, n_samples=7)
        self.assertEqual(sample_funcs.shape, (50, 7))

    def test_bayesian_linear_equivalent_kernel_partition_of_unity(self):
        # PRML 式 (3.64): 定数項 phi_0(x)=1 を含む基底関数において、
        # 無情報事前分布 (alpha -> 0) または alpha がデータ規模に対して十分小さい極限で
        # sum_{n=1}^N k(x, x_n) = 1 が成立する (局所重み付き平均の性質)
        X_train = np.linspace(-1, 1, 30)[:, None]
        poly = PolynomialBasis(degree=3)
        Phi_train = poly.transform(X_train)
        y_train = np.cos(np.pi * X_train).ravel()

        blr = BayesianLinearRegression(alpha=1e-5, beta=15.0).fit(Phi_train, y_train)

        X_eval = np.array([[-0.8], [-0.3], [0.0], [0.5], [0.9]])
        Phi_eval = poly.transform(X_eval)
        K = blr.equivalent_kernel(Phi_eval)  # shape: (5, 30)

        kernel_sums = np.sum(K, axis=1)
        np.testing.assert_allclose(kernel_sums, np.ones(5), atol=1e-4)

    def test_bayesian_marginal_likelihood_evidence(self):
        # 対数エビデンス ln p(t | alpha, beta) の妥当性
        X = np.linspace(0, 1, 20)[:, None]
        poly = PolynomialBasis(degree=2)
        Phi = poly.transform(X)
        t = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, size=len(X))

        blr = BayesianLinearRegression(alpha=1.0, beta=100.0).fit(Phi, t)
        log_ev = blr.log_marginal_likelihood()
        self.assertIsInstance(log_ev, float)
        self.assertFalse(np.isnan(log_ev))

    def test_evidence_approximation_gamma_bounds(self):
        # エビデンス近似における有効パラメータ数 gamma <= M
        X = np.linspace(-1, 1, 35)[:, None]
        poly = PolynomialBasis(degree=4)  # M = 5
        Phi = poly.transform(X)
        t = X.ravel()**3 + np.random.normal(0, 0.2, size=len(X))

        eb = EvidenceApproximation(max_iter=60).fit(Phi, t, init_alpha=2.0, init_beta=10.0)
        self.assertGreater(eb.gamma, 0.0)
        self.assertLessEqual(eb.gamma, 5.0)

    def test_normal_gamma_predictive_student_t(self):
        # Normal-Gamma モデルの Student-t 予測分布
        X = np.linspace(0, 2, 25)[:, None]
        Phi = np.hstack([np.ones((25, 1)), X])
        t = 2.0 + 1.5 * X.ravel() + np.random.normal(0, 0.3, size=25)

        ng = NormalGammaLinearRegression(a0=1.0, b0=1.0).fit(Phi, t)
        X_test = np.array([[1.0], [3.0]])
        Phi_test = np.hstack([np.ones((2, 1)), X_test])
        mu, std, nu = ng.predict(Phi_test)

        self.assertAlmostEqual(nu, 1.0 * 2 + 25, places=5)
        # 外挿点 (x=3.0) の不確実性が内挿点 (x=1.0) より大きい
        self.assertGreater(std[1], std[0])

    def test_robust_regression_resilience_to_outliers(self):
        # 外れ値が存在するデータにおけるロバスト線形回帰と OLS の比較
        np.random.seed(42)
        X = np.linspace(-2, 2, 30)[:, None]
        Phi = np.hstack([np.ones((30, 1)), X])
        true_w = np.array([1.0, 2.0])
        t = Phi @ true_w + np.random.normal(0, 0.1, size=30)

        # 3点を極端な外れ値に汚染
        t[5] += 20.0
        t[15] -= 25.0
        t[25] += 30.0

        ols = LinearRegression().fit(Phi, t)
        robust = RobustLinearRegression(df=2.0).fit(Phi, t)

        # ロバスト回帰の推定傾きが OLS よりも真値 (2.0) にはるかに近い
        err_ols = abs(ols.w[1] - true_w[1])
        err_robust = abs(robust.w[1] - true_w[1])
        self.assertLess(err_robust, err_ols)

    def test_locally_weighted_regression_non_linear_fit(self):
        # 局所重み付き回帰による非線形関数のフィッティング
        X = np.linspace(-np.pi, np.pi, 40)[:, None]
        t = np.sin(X).ravel() + np.random.normal(0, 0.05, size=40)

        lwr = LocallyWeightedRegression(tau=0.4).fit(X, t)
        X_test = np.linspace(-2.5, 2.5, 20)[:, None]
        preds = lwr.predict(X_test)
        true_vals = np.sin(X_test).ravel()

        np.testing.assert_allclose(preds, true_vals, atol=0.2)

    def test_basis_functions_shapes(self):
        # 各基底関数の形状テスト
        X = np.linspace(0, 1, 12)[:, None]
        
        # 多項式
        poly = PolynomialBasis(degree=4)
        self.assertEqual(poly(X).shape, (12, 5))

        # ガウス (RBF)
        centers = [0.2, 0.5, 0.8]
        gauss = GaussianBasis(centers=centers, scale=0.2)
        self.assertEqual(gauss(X).shape, (12, 4))  # 1 bias + 3 centers

        # シグモイド
        sig = SigmoidalBasis(centers=centers, scale=0.2)
        self.assertEqual(sig(X).shape, (12, 4))

        # フーリエ
        fourier = FourierBasis(n_components=3, period=1.0)
        self.assertEqual(fourier(X).shape, (12, 7))  # 1 bias + 3 sin + 3 cos


if __name__ == '__main__':
    unittest.main()
