"""
PRML Chapter 8: Graphical Models Unit Tests
d-分離 (d-separation)、因子グラフ和積アルゴリズム、ICM によるMRF画像ノイズ除去の厳密な数理検証
"""

import unittest
import numpy as np

from prml.graphical import (
    check_d_separation,
    SimpleFactorGraphChain,
    denoise_image_icm,
    noisy_or,
    linear_gaussian_moments,
)


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

    def test_noisy_or_gate_behavior(self):
        # PRML Ex 8.6: Noisy-OR
        # p(y=1 | x) = 1 - (1 - mu_0) * prod((1 - mu_i)**x_i)
        mu_0 = 0.05
        mu = np.array([0.8, 0.6, 0.4])
        # x = [0, 0, 0] -> p(y=1) = mu_0
        p_zero = 1.0 - (1.0 - mu_0) * np.prod((1.0 - mu)**[0, 0, 0])
        self.assertAlmostEqual(p_zero, mu_0)

        # x = [1, 0, 0]
        p_x0 = 1.0 - (1.0 - mu_0) * (1.0 - mu[0])
        self.assertAlmostEqual(p_x0, 1.0 - 0.95 * 0.2)

        # Monotonicity: adding active causes increases activation probability
        p_x01 = 1.0 - (1.0 - mu_0) * (1.0 - mu[0]) * (1.0 - mu[1])
        self.assertGreater(p_x01, p_x0)

    def test_marginal_mode_vs_joint_max_discrepancy_ex8_27(self):
        # PRML Ex 8.27: argmax p(x) and argmax p(y) can have joint probability zero
        P = np.array([
            [0.00, 0.20, 0.20],
            [0.25, 0.00, 0.10],
            [0.25, 0.10, 0.00]
        ])
        p_x = P.sum(axis=1) # [0.40, 0.35, 0.35]
        p_y = P.sum(axis=0) # [0.50, 0.30, 0.30]
        x_star = np.argmax(p_x)
        y_star = np.argmax(p_y)
        self.assertEqual(x_star, 0)
        self.assertEqual(y_star, 0)
        self.assertEqual(P[x_star, y_star], 0.0)

    def test_table_8_2_conditional_independence_ex8_3(self):
        # PRML Ex 8.3 & 8.4: Table 8.2
        table = {
            (0, 0, 0): 0.192, (0, 0, 1): 0.144,
            (0, 1, 0): 0.048, (0, 1, 1): 0.216,
            (1, 0, 0): 0.192, (1, 0, 1): 0.064,
            (1, 1, 0): 0.048, (1, 1, 1): 0.096
        }
        # Marginal dependence: p(a=1) * p(b=1) != p(a=1, b=1)
        p_a1 = sum(v for (a, b, c), v in table.items() if a == 1)
        p_b1 = sum(v for (a, b, c), v in table.items() if b == 1)
        p_a1_b1 = sum(v for (a, b, c), v in table.items() if a == 1 and b == 1)
        self.assertFalse(np.isclose(p_a1 * p_b1, p_a1_b1))

        # Conditional independence: p(a, b | c) = p(a | c) * p(b | c) for c in {0, 1}
        for c in [0, 1]:
            p_c = sum(v for (x, y, z), v in table.items() if z == c)
            for a in [0, 1]:
                for b in [0, 1]:
                    p_ab_given_c = table[(a, b, c)] / p_c
                    p_a_given_c = sum(v for (x, y, z), v in table.items() if x == a and z == c) / p_c
                    p_b_given_c = sum(v for (x, y, z), v in table.items() if y == b and z == c) / p_c
                    np.testing.assert_allclose(p_ab_given_c, p_a_given_c * p_b_given_c, atol=1e-7)

    def test_car_fuel_explaining_away_ex8_11(self):
        # PRML Ex 8.11: Explaining away
        p_B = {1: 0.9, 0: 0.1}
        p_F = {1: 0.9, 0: 0.1}
        p_G = {
            (1, 1): {1: 0.8, 0: 0.2},
            (1, 0): {1: 0.2, 0: 0.8},
            (0, 1): {1: 0.2, 0: 0.8},
            (0, 0): {1: 0.1, 0: 0.9},
        }
        p_D = {1: {1: 0.9, 0: 0.1}, 0: {1: 0.1, 0: 0.9}}
        joint = {}
        for b in [0, 1]:
            for f in [0, 1]:
                for g in [0, 1]:
                    for d in [0, 1]:
                        joint[(b, f, g, d)] = p_B[b] * p_F[f] * p_G[(b, f)][g] * p_D[g][d]

        p_D0 = sum(v for (b, f, g, d), v in joint.items() if d == 0)
        p_F0_given_D0 = sum(v for (b, f, g, d), v in joint.items() if d == 0 and f == 0) / p_D0

        p_D0_B0 = sum(v for (b, f, g, d), v in joint.items() if d == 0 and b == 0)
        p_F0_given_D0_B0 = sum(v for (b, f, g, d), v in joint.items() if d == 0 and b == 0 and f == 0) / p_D0_B0

        # Explaining away: observing battery is flat explains empty gauge, lowering p(fuel empty)
        self.assertLess(p_F0_given_D0_B0, p_F0_given_D0)

    def test_linear_gaussian_moments_recursion(self):
        # PRML Eq (8.15)-(8.18): linear Gaussian DAG moments
        W = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.3, -0.4, 0.0]
        ])
        b = np.array([1.0, 2.0, -1.0])
        v = np.array([0.5, 0.8, 0.2])

        mean_rec, cov_rec = linear_gaussian_moments(W, b, v)

        # Closed form: x = (I - W)^{-1} (b + eps)
        I_W_inv = np.linalg.inv(np.eye(3) - W)
        mean_exact = I_W_inv @ b
        cov_exact = I_W_inv @ np.diag(v) @ I_W_inv.T

        np.testing.assert_allclose(mean_rec, mean_exact, atol=1e-12)
        np.testing.assert_allclose(cov_rec, cov_exact, atol=1e-12)

    def test_noisy_or_utility_function(self):
        # Noisy-OR utility function test
        mu = [0.9, 0.7]
        prob_00 = noisy_or([0, 0], mu, mu_0=0.1)
        self.assertAlmostEqual(prob_00, 0.1)

        prob_10 = noisy_or([1, 0], mu, mu_0=0.0)
        self.assertAlmostEqual(prob_10, 0.9)

        # Logical OR recovery
        sharp_mu = [0.99999, 0.99999]
        self.assertAlmostEqual(noisy_or([0, 0], sharp_mu, mu_0=0.0), 0.0, places=4)
        self.assertAlmostEqual(noisy_or([1, 0], sharp_mu, mu_0=0.0), 1.0, places=4)
        self.assertAlmostEqual(noisy_or([0, 1], sharp_mu, mu_0=0.0), 1.0, places=4)
        self.assertAlmostEqual(noisy_or([1, 1], sharp_mu, mu_0=0.0), 1.0, places=4)


if __name__ == '__main__':
    unittest.main()

