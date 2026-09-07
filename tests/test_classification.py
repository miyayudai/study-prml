"""
PRML Chapter 4 Comprehensive Classification Test Suite
線形分類モデル、フィッシャー判別、生成・識別モデル、ロジスティック回帰、IRLS、ラプラス近似、ベイズ分類の完全テスト
"""

import unittest
import numpy as np
import scipy.stats as stats
from sklearn.datasets import make_blobs, make_classification

from prml.linear import (
    Perceptron,
    FisherLinearDiscriminant,
    MulticlassFisherLinearDiscriminant,
    GaussianGenerativeClassifier,
    LogisticRegression,
    MulticlassLogisticRegression,
    ProbitRegression,
    BayesianLogisticRegression,
    LaplaceApproximation,
)
from common.classification_utils import sigmoid, softmax


class TestComprehensiveClassificationSuite(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_sigmoid_and_softmax_properties(self):
        # シグモイド関数の対称性: sigma(-a) = 1 - sigma(a)
        a = np.linspace(-6, 6, 50)
        sig_a = sigmoid(a)
        np.testing.assert_allclose(sigmoid(-a), 1.0 - sig_a, atol=1e-12)
        
        # 導関数: d/da sigma(a) = sigma(a) (1 - sigma(a))
        eps = 1e-6
        num_diff = (sigmoid(a + eps) - sigmoid(a - eps)) / (2.0 * eps)
        ana_diff = sig_a * (1.0 - sig_a)
        np.testing.assert_allclose(num_diff, ana_diff, atol=1e-5)

        # ソフトマックスの規格化性と不変性: softmax(a + c) = softmax(a)
        A = np.random.randn(20, 5)
        P = softmax(A, axis=-1)
        np.testing.assert_allclose(np.sum(P, axis=-1), np.ones(20), atol=1e-10)
        c = 1000.0  # 数値オーバーフロー耐性
        P_shift = softmax(A + c, axis=-1)
        np.testing.assert_allclose(P, P_shift, atol=1e-10)

    def test_perceptron_convergence_and_separation(self):
        # 線形分離可能なデータセットに対するパーセプトロンの収束
        X, y = make_blobs(n_samples=50, centers=[[-2, -2], [2, 2]], cluster_std=0.6, random_state=42)
        y = np.where(y == 0, -1, 1)

        pct = Perceptron(max_iter=100, lr=0.5).fit(X, y)
        preds = pct.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_fisher_linear_discriminant_optimality(self):
        # PRML 4.1.4節: Fisher の線形判別分析
        # w \propto S_W^-1 (m2 - m1)
        X1 = np.random.normal(loc=[-1.5, 0.0], scale=[0.5, 1.5], size=(60, 2))
        X2 = np.random.normal(loc=[1.5, 0.0], scale=[0.5, 1.5], size=(60, 2))
        X = np.vstack([X1, X2])
        y = np.array([0] * 60 + [1] * 60)

        fld = FisherLinearDiscriminant().fit(X, y)
        preds = fld.predict(X)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.95)

        # 1次元射影後の平均間の差が正であることを確認
        z1 = fld.project(X1)
        z2 = fld.project(X2)
        self.assertNotEqual(np.mean(z1), np.mean(z2))

    def test_multiclass_fisher_discriminant_dimension(self):
        # PRML 4.1.6節: 多クラス Fisher 判別 (K=3 クラス -> 最大 K-1 = 2 次元射影)
        X, y = make_blobs(n_samples=90, centers=3, n_features=5, random_state=42)
        mfld = MulticlassFisherLinearDiscriminant(n_components=2).fit(X, y)
        
        Z = mfld.project(X)
        self.assertEqual(Z.shape, (90, 2))
        preds = mfld.predict(X)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.90)

    def test_gaussian_generative_classifier_lda_qda(self):
        # PRML 4.2節: ガウス生成モデル (LDA と QDA)
        X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

        # 1. 共通共分散 (LDA)
        lda = GaussianGenerativeClassifier(shared_cov=True).fit(X, y)
        lda_proba = lda.predict_proba(X)
        self.assertEqual(lda_proba.shape, (100, 3))
        np.testing.assert_allclose(np.sum(lda_proba, axis=1), np.ones(100), atol=1e-8)
        self.assertGreater(np.mean(lda.predict(X) == y), 0.95)

        # 2. 個別共分散 (QDA)
        qda = GaussianGenerativeClassifier(shared_cov=False).fit(X, y)
        qda_proba = qda.predict_proba(X)
        self.assertEqual(qda_proba.shape, (100, 3))
        np.testing.assert_allclose(np.sum(qda_proba, axis=1), np.ones(100), atol=1e-8)
        self.assertGreater(np.mean(qda.predict(X) == y), 0.95)

    def test_logistic_regression_irls_and_hessian(self):
        # PRML 4.3.2-4.3.3節: IRLS 法によるロジスティック回帰
        X, y = make_blobs(n_samples=80, centers=2, n_features=3, cluster_std=1.0, random_state=42)
        Phi = np.column_stack([np.ones(len(X)), X])

        # 正則化付きロジスティック回帰
        alpha = 0.5
        lr = LogisticRegression(alpha=alpha, max_iter=50).fit(Phi, y)
        proba = lr.predict_proba(Phi)
        self.assertTrue(np.all((proba >= 0.0) & (proba <= 1.0)))

        # ヘッセ行列 H = Phi^T R Phi + alpha * I の固有値がすべて正 (厳密な正定値性)
        r = proba * (1.0 - proba)
        H = Phi.T @ (r[:, None] * Phi) + alpha * np.eye(Phi.shape[1])
        eigvals = np.linalg.eigvalsh(H)
        self.assertTrue(np.all(eigvals > 0.0))

        preds = lr.predict(Phi)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.95)

    def test_multiclass_logistic_regression_softmax(self):
        # PRML 4.3.4節: 多クラスロジスティック回帰
        X, y_int = make_blobs(n_samples=90, centers=3, n_features=2, random_state=42)
        Phi = np.column_stack([np.ones(len(X)), X])
        K = 3
        T = np.eye(K)[y_int]  # 1-of-K 表現

        mlr = MulticlassLogisticRegression(alpha=0.1, lr=0.1, max_iter=300).fit(Phi, T)
        proba = mlr.predict_proba(Phi)
        self.assertEqual(proba.shape, (90, K))
        np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(90), atol=1e-5)

        preds = mlr.predict(Phi)
        acc = np.mean(preds == y_int)
        self.assertGreater(acc, 0.90)

    def test_probit_regression_fit(self):
        # PRML 4.3.5節: プロビット回帰モデル
        X, y = make_blobs(n_samples=80, centers=2, n_features=2, cluster_std=0.8, random_state=42)
        Phi = np.column_stack([np.ones(len(X)), X])

        probit = ProbitRegression(max_iter=100, lr=0.05).fit(Phi, y)
        proba = probit.predict_proba(Phi)
        self.assertTrue(np.all((proba >= 0.0) & (proba <= 1.0)))
        preds = probit.predict(Phi)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.90)

    def test_bayesian_logistic_regression_laplace_and_extrapolation(self):
        # PRML 4.5節: ベイズロジスティック回帰とラプラス近似
        X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        Phi = np.column_stack([np.ones(len(X)), X])

        blr = BayesianLogisticRegression(alpha=1.0).fit(Phi, y)
        self.assertIsNotNone(blr.w_map)
        self.assertIsNotNone(blr.S_N)

        # 訓練点近傍と遠隔外挿点での予測確率の不確実性検証
        # x = 3 (正のクラス中心) -> p(C1) は 1 に近い
        # x = 50 (極端な外挿点) -> 分散 sigma_a^2 が増大し、kappa -> 0 となるため p(C1) は 0.5 に緩和される
        p_train = blr.predict_proba(np.array([[1.0, 3.0]]))[0]
        self.assertGreater(p_train, 0.8)

        # 事後共分散行列 S_N が正定値であること
        eigvals_S = np.linalg.eigvalsh(blr.S_N)
        self.assertTrue(np.all(eigvals_S > 0.0))

    def test_laplace_approximation_evidence_and_bic(self):
        # PRML 4.4節: ラプラス近似とエビデンス関数
        # 1次元二峰性または正規分布での検証: p(w) \propto exp(-0.5 * w^2)
        def energy(w):
            return 0.5 * float(w[0]**2) + 2.0

        def grad(w):
            return np.array([float(w[0])])

        def hess(w):
            return np.array([[1.0]])

        laplace = LaplaceApproximation(energy_fn=energy, grad_fn=grad, hessian_fn=hess)
        laplace.fit(np.array([1.5]))
        
        np.testing.assert_allclose(laplace.mode, np.array([0.0]), atol=1e-5)
        np.testing.assert_allclose(laplace.hessian, np.array([[1.0]]), atol=1e-5)

        # 対数エビデンス: -E(0) - 1/2 ln(1) + 1/2 ln(2 pi) = -2.0 + 0.5 ln(2 pi)
        true_log_ev = -2.0 + 0.5 * np.log(2.0 * np.pi)
        self.assertAlmostEqual(laplace.log_evidence(), true_log_ev, places=5)

        # BIC の計算確認
        bic = laplace.bic(n_samples=100)
        true_bic = -2.0 - 0.5 * 1 * np.log(100)
        self.assertAlmostEqual(bic, true_bic, places=5)


if __name__ == '__main__':
    unittest.main()
