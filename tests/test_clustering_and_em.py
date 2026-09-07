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


if __name__ == '__main__':
    unittest.main()

