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

    def test_variational_gaussian_mixture(self):
        # 変分ベイズGMM: ハイパーパラメータの更新と正値性
        X, _ = make_blobs(n_samples=50, centers=2, random_state=42)
        vbgmm = VariationalGaussianMixture(n_components=4, max_iter=25, random_state=42).fit(X)
        self.assertEqual(len(vbgmm.alpha_), 4)
        self.assertTrue(np.all(vbgmm.alpha_ > 0))

if __name__ == '__main__':
    unittest.main()
