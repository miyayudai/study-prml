"""
PRML Chapter 8: Graphical Models Unit Tests
d-分離 (d-separation)、因子グラフ和積アルゴリズム、ICM によるMRF画像ノイズ除去の厳密な数理検証
"""

import unittest
import numpy as np

from prml.graphical import (
    check_d_separation,
    SimpleFactorGraphChain,
)
from common.graphical_models_utils import denoise_image_icm


class TestGraphicalModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_d_separation_head_to_head(self):
        # PRML 8.2.2節 合流型 (head-to-head / collider): a -> c <- b
        adj = {
            'a': ['c'],
            'b': ['c'],
            'c': []
        }
        # c が未観測のとき、a と b は d-分離される（独立）
        self.assertTrue(check_d_separation(adj, ['a'], ['b'], []))
        
        # c が観測されているとき、a と b は d-分離されない（条件付き従属: explaining away）
        self.assertFalse(check_d_separation(adj, ['a'], ['b'], ['c']))

    def test_d_separation_tail_to_tail(self):
        # PRML 8.2.2節 分岐型 (tail-to-tail): a <- c -> b
        adj = {
            'c': ['a', 'b'],
            'a': [],
            'b': []
        }
        # c が未観測のとき、a と b は d-分離されない（従属）
        self.assertFalse(check_d_separation(adj, ['a'], ['b'], []))
        
        # c が観測されているとき、a と b は d-分離される（条件付き独立）
        self.assertTrue(check_d_separation(adj, ['a'], ['b'], ['c']))

    def test_d_separation_head_to_tail(self):
        # PRML 8.2.2節 連鎖型 (head-to-tail / chain): a -> c -> b
        adj = {
            'a': ['c'],
            'c': ['b'],
            'b': []
        }
        # c が未観測のとき、a と b は d-分離されない（従属）
        self.assertFalse(check_d_separation(adj, ['a'], ['b'], []))
        
        # c が観測されているとき、a と b は d-分離される（条件付き独立）
        self.assertTrue(check_d_separation(adj, ['a'], ['b'], ['c']))

    def test_d_separation_descendant_collider(self):
        # 合流型の子孫ノードが観測されるケース: a -> c <- b, c -> d
        adj = {
            'a': ['c'],
            'b': ['c'],
            'c': ['d'],
            'd': []
        }
        # d も c も未観測 -> a と b は独立
        self.assertTrue(check_d_separation(adj, ['a'], ['b'], []))
        
        # c の子孫である d が観測された場合 -> a と b は条件付き従属
        self.assertFalse(check_d_separation(adj, ['a'], ['b'], ['d']))

    def test_factor_graph_chain_sum_product_marginals(self):
        # PRML 8.4.1節 一次元連鎖グラフに対する和積 (Sum-Product) アルゴリズム
        # 3ノード (N=3), 各ノードは2値 (K=2)
        K = 2
        # 遷移確率行列 (K, K)
        T01 = np.array([[0.8, 0.2], [0.3, 0.7]])
        T12 = np.array([[0.6, 0.4], [0.1, 0.9]])
        trans = [T01, T12]
        
        # 観測・事前ポテンシャル (K,)
        phi0 = np.array([0.5, 0.5])
        phi1 = np.array([0.9, 0.1])
        phi2 = np.array([0.2, 0.8])
        emiss = [phi0, phi1, phi2]
        
        chain = SimpleFactorGraphChain(node_states=K, transition_matrices=trans, emission_potentials=emiss)
        marginals = chain.forward_backward_marginals()
        
        # 各ノードの周辺確率の和が 1
        self.assertEqual(marginals.shape, (3, 2))
        np.testing.assert_allclose(np.sum(marginals, axis=1), np.ones(3), atol=1e-10)
        
        # ブルートフォースによる全結合結合確率 p(x0, x1, x2) との厳密な照合
        # p(x0, x1, x2) = phi0(x0) * T01(x0, x1) * phi1(x1) * T12(x1, x2) * phi2(x2)
        joint = np.zeros((2, 2, 2))
        for x0 in range(2):
            for x1 in range(2):
                for x2 in range(2):
                    joint[x0, x1, x2] = phi0[x0] * T01[x0, x1] * phi1[x1] * T12[x1, x2] * phi2[x2]
        joint /= np.sum(joint) # 規格化
        
        true_m0 = np.sum(joint, axis=(1, 2))
        true_m1 = np.sum(joint, axis=(0, 2))
        true_m2 = np.sum(joint, axis=(0, 1))
        
        np.testing.assert_allclose(marginals[0], true_m0, atol=1e-7)
        np.testing.assert_allclose(marginals[1], true_m1, atol=1e-7)
        np.testing.assert_allclose(marginals[2], true_m2, atol=1e-7)

    def test_denoise_image_icm_energy_decrease(self):
        # PRML 8.3.3節 ICM (Iterated Conditional Modes) による画像ノイズ除去
        # 単純な 8x8 パターン
        clean = np.ones((8, 8))
        clean[:4, :] = -1.0
        
        # ノイズ付加 (10% の確率で符号反転)
        noisy = clean.copy()
        flip_mask = np.random.rand(8, 8) < 0.15
        noisy[flip_mask] *= -1.0
        
        denoised, energies = denoise_image_icm(noisy, h=0.0, beta=1.0, eta=2.1, max_iter=10)
        
        # 各反復でエネルギーは単調非増加（収束）すること
        self.assertGreater(len(energies), 1)
        diffs = np.diff(energies)
        self.assertTrue(np.all(diffs <= 1e-6))
        
        # ノイズ除去後の正答率がノイズ画像より改善すること
        noisy_acc = np.mean(noisy == clean)
        denoised_acc = np.mean(denoised == clean)
        self.assertGreaterEqual(denoised_acc, noisy_acc)


if __name__ == '__main__':
    unittest.main()
