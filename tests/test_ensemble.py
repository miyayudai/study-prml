"""
PRML Chapter 14: Combining Models & Ensemble Methods Unit Tests
決定株 (Decision Stump), AdaBoost, 線形回帰混合モデル (Mixture of Linear Regressions) の数理的検証
"""

import unittest
import numpy as np

from prml.ensemble import (
    DecisionStump,
    AdaBoostClassifier,
    MixtureOfLinearRegressions,
)


class TestEnsembleMethods(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_decision_stump_weighted_fit(self):
        # PRML 14.3節: 重み付き決定株の最適閾値探索
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([-1, -1, 1, 1, 1])
        
        # 均等重みでの適合
        w = np.ones(5) / 5.0
        stump = DecisionStump().fit(X, y, sample_weight=w)
        preds = stump.predict(X)
        
        # 閾値は 2.0 と 3.0 の間 (2.5) に選ばれるべき
        self.assertAlmostEqual(stump.threshold, 2.5, places=5)
        np.testing.assert_array_equal(preds, y)

    def test_adaboost_exponential_loss_and_convergence(self):
        # PRML 14.3節 式 (14.16)-(14.19): AdaBoost の逐次重み更新と訓練誤差の減少
        # PRML Figure 14.2 準拠の2クラス2次元データセット
        N_pts = 40
        X_pos = np.random.randn(N_pts // 2, 2) * 0.4 + np.array([0.4, 0.4])
        X_neg = np.random.randn(N_pts // 2, 2) * 0.4 + np.array([-0.4, -0.4])
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * (N_pts // 2) + [-1] * (N_pts // 2))
        
        adaboost = AdaBoostClassifier(n_estimators=10).fit(X, y)
        
        # 1. 各弱学習器の重み alpha_m が有限で正（弱学習器の性能がランダムより良い場合）
        self.assertEqual(len(adaboost.alphas), 10)
        self.assertTrue(all(np.isfinite(a) for a in adaboost.alphas))
        
        # 2. データ点重み w の総和が常に 1 に規格化されていること
        for w_hist in adaboost.weights_history:
            self.assertAlmostEqual(np.sum(w_hist), 1.0, places=6)
            self.assertTrue(np.all(w_hist >= 0.0))
            
        # 3. 複合モデルによる予測精度
        final_preds = adaboost.predict(X)
        np.testing.assert_array_equal(final_preds, y)
        
        # 4. staged_predict の確認
        staged_errors = []
        for p in adaboost.staged_predict(X):
            staged_errors.append(np.mean(p != y))
        self.assertEqual(len(staged_errors), 10)
        # 最終エラーが初期エラー以下
        self.assertLessEqual(staged_errors[-1], staged_errors[0])

    def test_mixture_of_linear_regressions_em(self):
        # PRML 14.5.1節 式 (14.34)-(14.40): 線形回帰混合モデルの EM 推定
        # 2つの異なる傾きを持つ直線からデータを生成
        N = 100
        x = np.random.uniform(-2, 2, N)
        # 潜在クラスタ z in {0, 1}
        z = np.random.binomial(1, 0.5, N)
        
        # クラスタ 0: y = 2x + 1 + noise
        # クラスタ 1: y = -2x - 1 + noise
        y = np.zeros(N)
        y[z == 0] = 2.0 * x[z == 0] + 1.0 + np.random.normal(0, 0.1, np.sum(z == 0))
        y[z == 1] = -2.0 * x[z == 1] - 1.0 + np.random.normal(0, 0.1, np.sum(z == 1))
        
        X = x[:, np.newaxis]
        moe = MixtureOfLinearRegressions(n_components=2, max_iter=60, random_state=42).fit(X, y)
        
        # 混合比率 pi の和が 1
        self.assertAlmostEqual(np.sum(moe.pi_), 1.0, places=5)
        # 分散が正
        self.assertTrue(np.all(moe.vars_ > 0))
        
        # 傾きの推定値の絶対値が約 2.0 であること (クラスタの順序は不問)
        slopes = np.abs(moe.weights_[:, 1])
        np.testing.assert_allclose(np.sort(slopes), [2.0, 2.0], atol=0.4)
        
        # 予測密度の次元と非負性
        x_grid = np.linspace(-1, 1, 10)
        t_grid = np.linspace(-3, 3, 20)
        dens = moe.predict_density(x_grid, t_grid)
        self.assertEqual(dens.shape, (20, 10))
        self.assertTrue(np.all(dens >= 0.0))
        
        # 期待値予測 predict(X)
        y_preds = moe.predict(X)
        self.assertEqual(len(y_preds), N)
        self.assertTrue(np.all(np.isfinite(y_preds)))

    def test_decision_stump_2d_negative_polarity(self):
        # 2次元空間における決定株の境界分離
        X = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, -1.0], [0.0, -2.0]])
        y = np.array([1, 1, -1, -1])
        stump = DecisionStump().fit(X, y, sample_weight=np.ones(4) / 4.0)
        self.assertEqual(stump.feature_idx, 1) # y軸で分離
        preds = stump.predict(X)
        np.testing.assert_array_equal(preds, y)


if __name__ == '__main__':
    unittest.main()
