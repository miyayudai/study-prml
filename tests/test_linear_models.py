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
    LassoRegression,
    LeastMeanSquares,
    BayesianLinearRegression,
    NormalGammaLinearRegression,
    EvidenceApproximation,
    PolynomialBasis,
    GaussianBasis,
    SigmoidalBasis,
    FourierBasis,
    bias_variance_decomposition,
    orthogonal_projection_matrix,
    equivalent_kernel_matrix,
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
        self.assertLess(lr.sigma2_, 1e-10)

    def test_multi_output_linear_regression(self):
        # PRML 3.1.5節: 多次元目的変数 T in R^(N x K)
        N = 30
        Phi = np.hstack([np.ones((N, 1)), np.random.randn(N, 3)])
        true_W = np.array([
            [1.0, -2.0],
            [0.5, 3.0],
            [-1.5, 0.0],
            [2.0, 1.0]
        ])
        T = Phi @ true_W + np.random.normal(0, 0.01, size=(N, 2))
        
        lr = LinearRegression().fit(Phi, T)
        preds = lr.predict(Phi)
        self.assertEqual(preds.shape, (N, 2))
        np.testing.assert_allclose(lr.w, true_W, atol=0.05)
        self.assertEqual(len(lr.sigma2_), 2)

    def test_orthogonal_projection_matrix_properties(self):
        # PRML 3.1.2節 式 (3.25): 直交射影行列 P = Phi (Phi^T Phi)^-1 Phi^T
        Phi = np.random.randn(20, 4)
        t = np.random.randn(20)
        
        P = orthogonal_projection_matrix(Phi)
        # 1. べき等性 P^2 = P
        np.testing.assert_allclose(P @ P, P, atol=1e-10)
        # 2. 対称性 P^T = P
        np.testing.assert_allclose(P.T, P, atol=1e-10)
        # 3. 射影 y = P t と残差 (I - P) t の直交性
        y = P @ t
        residual = (np.eye(20) - P) @ t
        self.assertAlmostEqual(float(y @ residual), 0.0, places=8)
        # 4. 残差ベクトルは基底 Phi の各列と直交する (Phi^T (t - y) = 0)
        np.testing.assert_allclose(Phi.T @ residual, np.zeros(4), atol=1e-8)

    def test_ridge_shrinkage_and_df(self):
        # Ridge 正則化の収縮特性 (alpha が大 -> 重みノルム縮小 & 有効自由度低下)
        X, y = make_regression(n_samples=40, n_features=5, noise=1.0, random_state=42)
        Phi = np.hstack([np.ones((len(X), 1)), X])
        ridge_small = RidgeRegression(alpha=0.01).fit(Phi, y)
        ridge_large = RidgeRegression(alpha=100.0).fit(Phi, y)
        
        norm_small = np.linalg.norm(ridge_small.w)
        norm_large = np.linalg.norm(ridge_large.w)
        self.assertGreater(norm_small, norm_large)

        df_small = ridge_small.effective_degrees_of_freedom(Phi)
        df_large = ridge_large.effective_degrees_of_freedom(Phi)
        self.assertGreater(df_small, df_large)
        self.assertLessEqual(df_small, 6.0)

    def test_lasso_coordinate_descent_sparsity(self):
        # PRML 3.1.4節: Lasso による重みの厳密なスパース化
        np.random.seed(42)
        N = 60
        X = np.random.randn(N, 6)
        # 真の重みは 2 つだけ非ゼロ、残り 4 つは 0
        true_w = np.array([2.5, -3.0, 0.0, 0.0, 0.0, 0.0])
        y = X @ true_w + np.random.normal(0, 0.1, size=N)

        lasso = LassoRegression(alpha=5.0, max_iter=500).fit(X, y)
        sparsity = lasso.get_sparsity()
        self.assertGreaterEqual(sparsity, 0.5)  # 少なくとも半数の係数がゼロ
        self.assertGreater(abs(lasso.w[0]), 0.5)
        self.assertGreater(abs(lasso.w[1]), 0.5)

    def test_lms_sequential_convergence(self):
        # PRML 3.1.3節: LMS 法のオンライン更新による OLS 解への収束
        np.random.seed(42)
        N = 100
        Phi = np.hstack([np.ones((N, 1)), np.random.randn(N, 2)])
        true_w = np.array([1.0, 2.0, -1.0])
        t = Phi @ true_w

        lms = LeastMeanSquares(learning_rate=0.05, max_iter=200, decay_rate=0.001)
        lms.fit(Phi, t)
        np.testing.assert_allclose(lms.w, true_w, atol=0.1)

    def test_bayesian_linear_uncertainty(self):
        # ベイズ線形回帰: 予測分散 s^2(x) >= beta^{-1}
        X = np.linspace(-1, 1, 15)[:, np.newaxis]
        y = np.sin(np.pi * X).ravel() + np.random.normal(0, 0.1, 15)
        
        beta = 100.0
        blr = BayesianLinearRegression(alpha=1.0, beta=beta).fit(X, y)
        
        # 訓練データ範囲内と範囲外の点
        X_test = np.array([[-0.5], [0.0], [2.5]])  # 2.5 は外挿点
        mean, std = blr.predict(X_test)
        var = std**2
        
        # 予測分散は観測ノイズ以上
        self.assertTrue(np.all(var >= 1.0 / beta - 1e-8))
        # データから遠い外挿領域 (x=2.5) では不確実性がデータ中心部 (x=0) より大
        self.assertGreater(var[2], var[1])

    def test_bayesian_linear_sequential_update_consistency(self):
        # 逐次ベイズ更新 (point-by-point) と一括ベイズ更新 (batch) の事後分布が厳密に一致する検証
        np.random.seed(42)
        N = 10
        Phi = np.hstack([np.ones((N, 1)), np.random.randn(N, 2)])
        t = np.random.randn(N)
        alpha, beta = 2.0, 5.0

        # 一括学習
        batch_blr = BayesianLinearRegression(alpha=alpha, beta=beta).fit(Phi, t)

        # 逐次学習 (1点ずつ更新)
        seq_blr = BayesianLinearRegression(alpha=alpha, beta=beta)
        for i in range(N):
            seq_blr.update(Phi[i], t[i])

        np.testing.assert_allclose(seq_blr.m_N, batch_blr.m_N, atol=1e-10)
        np.testing.assert_allclose(seq_blr.S_N, batch_blr.S_N, atol=1e-10)

    def test_equivalent_kernel_properties(self):
        # PRML 3.3.3節: 等価カーネルの性質 sum_n k(x, x_n) = 1 (バイアス項を含む場合)
        np.random.seed(42)
        X_train = np.linspace(-1, 1, 15)[:, np.newaxis]
        poly = PolynomialBasis(degree=3)
        Phi_train = poly.transform(X_train)
        y_train = np.sin(np.pi * X_train).ravel()

        blr = BayesianLinearRegression(alpha=1.0, beta=10.0).fit(Phi_train, y_train)

        X_eval = np.array([[-0.7], [0.1], [0.8]])
        Phi_eval = poly.transform(X_eval)
        K = blr.equivalent_kernel(Phi_eval)  # (3, 15)

        # 予測平均 y(x) = sum_n k(x, x_n) t_n (式 3.61)
        mean_from_kernel = K @ y_train
        mean_direct, _ = blr.predict(Phi_eval)
        np.testing.assert_allclose(mean_from_kernel, mean_direct, atol=1e-10)

    def test_evidence_approximation_convergence(self):
        # PRML 3.5節: エビデンス近似（経験ベイズ）による alpha, beta 最適化
        np.random.seed(42)
        X = np.linspace(0, 1, 30)[:, np.newaxis]
        poly = PolynomialBasis(degree=3)
        Phi = poly.transform(X)
        t = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.2, size=30)

        eb = EvidenceApproximation(max_iter=50).fit(Phi, t, init_alpha=1.0, init_beta=1.0)
        self.assertGreater(eb.alpha, 0)
        self.assertGreater(eb.beta, 0)
        self.assertGreater(eb.gamma, 0)
        self.assertLessEqual(eb.gamma, 4.0)  # 有効パラメータ数は総パラメータ数 M=4 以下

    def test_normal_gamma_students_t_predictive(self):
        # PRML Exercises 3.12-3.13: 未知の精度 beta を持つ共役正規ガンマモデル
        np.random.seed(42)
        X = np.linspace(-1, 1, 20)[:, np.newaxis]
        Phi = np.hstack([np.ones((20, 1)), X])
        t = 1.5 + 2.0 * X.ravel() + np.random.normal(0, 0.2, size=20)

        ng = NormalGammaLinearRegression(a0=2.0, b0=2.0).fit(Phi, t)
        X_test = np.array([[0.0], [2.0]])
        Phi_test = np.hstack([np.ones((2, 1)), X_test])
        mean, std, nu = ng.predict(Phi_test)

        self.assertEqual(len(mean), 2)
        self.assertGreater(nu, 2.0)
        self.assertGreater(std[1], std[0])  # 外挿点の不確実性が大

    def test_bias_variance_decomposition(self):
        # PRML 3.2節: バイアス-バリアンス分解シミュレーション
        np.random.seed(42)
        L = 20
        N = 25
        X_test = np.linspace(0, 1, 30)[:, np.newaxis]
        h_test = np.sin(2 * np.pi * X_test).ravel()

        X_train_sets = []
        y_train_sets = []
        for _ in range(L):
            X = np.random.uniform(0, 1, N)[:, np.newaxis]
            y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, size=N)
            poly = PolynomialBasis(degree=3)
            X_train_sets.append(poly.transform(X))
            y_train_sets.append(y)

        Phi_test = PolynomialBasis(degree=3).transform(X_test)
        decomp = bias_variance_decomposition(
            model_class=RidgeRegression,
            model_kwargs={'alpha': 0.1},
            X_train_sets=X_train_sets,
            y_train_sets=y_train_sets,
            X_test=Phi_test,
            y_test_true=h_test,
            noise_var=0.01
        )
        self.assertGreater(decomp['bias_squared'], 0.0)
        self.assertGreater(decomp['variance'], 0.0)
        self.assertAlmostEqual(decomp['expected_loss'], decomp['bias_squared'] + decomp['variance'] + decomp['noise'], places=6)

    def test_basis_expansions(self):
        X = np.linspace(-1, 1, 10)[:, np.newaxis]
        poly = PolynomialBasis(degree=3)
        Phi_poly = poly.transform(X)
        self.assertEqual(Phi_poly.shape, (10, 4))
        
        centers = np.array([[-0.5], [0.0], [0.5]])
        gauss = GaussianBasis(centers=centers, scale=0.5)
        Phi_gauss = gauss.transform(X)
        self.assertEqual(Phi_gauss.shape, (10, 4))  # intercept + 3 centers

        fourier = FourierBasis(n_components=2, period=2.0)
        Phi_fourier = fourier.transform(X)
        self.assertEqual(Phi_fourier.shape, (10, 5))  # 1 + 2 * cos + 2 * sin

    def test_locally_weighted_regression(self):
        # 局所重み付き線形回帰 (LOWESS) の非線形フィッティング能力
        from prml.linear import LocallyWeightedRegression
        np.random.seed(42)
        X = np.linspace(-1, 1, 30)[:, np.newaxis]
        y_true = np.sin(np.pi * X).ravel()
        y = y_true + np.random.normal(0, 0.05, size=30)

        # 通常の OLS (直線) vs 局所重み付き回帰
        ols = LinearRegression().fit(np.hstack([np.ones((30, 1)), X]), y)
        ols_preds = ols.predict(np.hstack([np.ones((30, 1)), X]))

        lowess = LocallyWeightedRegression(tau=0.2).fit(X, y)
        lowess_preds = lowess.predict(X)

        ols_mse = np.mean((ols_preds - y_true)**2)
        lowess_mse = np.mean((lowess_preds - y_true)**2)
        self.assertLess(lowess_mse, ols_mse * 0.2)  # LOWESS は OLS より 5倍以上高精度

    def test_robust_linear_regression_outliers(self):
        # Huber 損失によるロバスト回帰の外れ値耐性
        from prml.linear import RobustLinearRegression
        np.random.seed(42)
        N = 30
        X = np.linspace(-2, 2, N)[:, np.newaxis]
        Phi = np.hstack([np.ones((N, 1)), X])
        true_w = np.array([1.0, 2.0])  # y = 1 + 2x
        t = Phi @ true_w + np.random.normal(0, 0.1, size=N)

        # 3点を極端な外れ値に改変
        t[5] += 15.0
        t[15] -= 18.0
        t[25] += 20.0

        ols = LinearRegression().fit(Phi, t)
        robust = RobustLinearRegression(delta=1.345).fit(Phi, t)

        # ロバスト回帰の推定誤差が OLS より劇的に小さいことを検証
        ols_err = np.linalg.norm(ols.w - true_w)
        robust_err = np.linalg.norm(robust.w - true_w)
        self.assertLess(robust_err, ols_err * 0.3)
        self.assertAlmostEqual(robust.w[1], 2.0, places=0)

    def test_model_selection_criteria(self):
        # AIC, BIC, R^2 スコアの計算
        np.random.seed(42)
        X = np.linspace(-1, 1, 40)[:, np.newaxis]
        y = 2.0 * X.ravel() + 1.0 + np.random.normal(0, 0.2, size=40)
        Phi1 = np.hstack([np.ones((40, 1)), X])
        lr1 = LinearRegression().fit(Phi1, y)

        self.assertGreater(lr1.score(Phi1, y), 0.8)
        aic1 = lr1.aic(Phi1, y)
        bic1 = lr1.bic(Phi1, y)
        self.assertTrue(np.isfinite(aic1))
        self.assertTrue(np.isfinite(bic1))
        # N=40 > e^2=7.39 なので ln(N) > 2 となり BIC > AIC
        self.assertGreater(bic1, aic1)


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
        X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.0, random_state=42)
        lr = LogisticRegression(max_iter=30).fit(X, y)
        probs = lr.predict_proba(X)
        self.assertEqual(probs.shape, (50,))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        acc = np.mean(lr.predict(X) == y)
        self.assertGreaterEqual(acc, 0.95)

    def test_bayesian_logistic_laplace(self):
        X, y = make_blobs(n_samples=40, n_features=2, centers=2, cluster_std=1.0, random_state=42)
        blr = BayesianLogisticRegression(alpha=1.0, max_iter=30).fit(X, y)
        probs = blr.predict_proba(X)
        self.assertEqual(probs.shape, (40,))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_multiclass_fisher_linear_discriminant(self):
        from prml.linear import MulticlassFisherLinearDiscriminant
        X, y = make_blobs(n_samples=60, n_features=4, centers=3, cluster_std=1.0, random_state=42)
        mfld = MulticlassFisherLinearDiscriminant(n_components=2).fit(X, y)
        Z = mfld.project(X)
        self.assertEqual(Z.shape, (60, 2))
        preds = mfld.predict(X)
        acc = np.mean(preds == y)
        self.assertGreaterEqual(acc, 0.90)

    def test_probit_regression(self):
        from prml.linear import ProbitRegression
        X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.0, random_state=42)
        Phi = np.column_stack([np.ones(len(X)), X])
        pr = ProbitRegression(max_iter=50, lr=0.05).fit(Phi, y)
        probs = pr.predict_proba(Phi)
        self.assertEqual(probs.shape, (50,))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        preds = pr.predict(Phi)
        acc = np.mean(preds == y)
        self.assertGreaterEqual(acc, 0.90)

    def test_laplace_approximation_and_evidence(self):
        from prml.linear import LaplaceApproximation, bayesian_model_evidence
        # 1. LaplaceApproximation on quadratic bowl
        # E(w) = 0.5 * (w - w*)^T H (w - w*)
        w_true = np.array([1.5, -2.0])
        H_true = np.array([[3.0, 0.5], [0.5, 2.0]])
        energy = lambda w: 0.5 * (w - w_true) @ H_true @ (w - w_true)
        grad = lambda w: H_true @ (w - w_true)
        laplace = LaplaceApproximation(energy, grad_fn=grad).fit(np.zeros(2))
        np.testing.assert_allclose(laplace.mode, w_true, atol=1e-3)
        self.assertGreater(laplace.cov.shape[0], 0)
        self.assertTrue(np.isfinite(laplace.log_evidence()))

        # 2. bayesian_model_evidence
        X = np.linspace(-1, 1, 20)[:, None]
        t = 2.0 * X.ravel() + np.random.normal(0, 0.1, 20)
        Phi = np.hstack([np.ones((20, 1)), X])
        res = bayesian_model_evidence(Phi, t, alpha=1.0, beta=25.0)
        self.assertIn('log_evidence', res)
        self.assertIn('log_occam_factor', res)
        self.assertTrue(np.isfinite(res['log_evidence']))

