"""
PRML Chapter 5: Neural Networks Comprehensive Unit Tests
ニューラルネットワーク、誤差逆伝播法、ヘッセ行列、正則化、MDN、ベイズニューラルネットの厳密な数理検証
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs, make_regression

from prml.nn import (
    MLPRegressor,
    MixtureDensityNetwork,
)
from common.classification_utils import sigmoid, softmax
from common.nn_utils import tanh


class TestNeuralNetworksComprehensive(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_tanh_sigmoid_linear_equivalence(self):
        # PRML 5.1節 式 (5.24) & 演習 5.1: tanh(a) = 2*sigma(2a) - 1
        # シグモイド隠れユニットと tanh 隠れユニットの線形変換による恒等性
        a = np.linspace(-3.0, 3.0, 50)
        tanh_val = tanh(a)
        sig_val = 2.0 * sigmoid(2.0 * a) - 1.0
        np.testing.assert_allclose(tanh_val, sig_val, atol=1e-12)

        # 2層ネットワークの等価パラメータ変換
        D, M, K = 2, 3, 1
        W1_sig = np.array([[1.0, -0.5, 0.2], [0.3, 0.8, -1.0]])
        b1_sig = np.array([0.1, -0.2, 0.5])
        W2_sig = np.array([[0.5], [-1.2], [0.8]])
        b2_sig = np.array([0.3])

        # 等価な tanh ネットワークのパラメータ
        W1_tanh = 0.5 * W1_sig
        b1_tanh = 0.5 * b1_sig
        W2_tanh = 0.5 * W2_sig
        b2_tanh = b2_sig + 0.5 * np.sum(W2_sig, axis=0)

        X = np.random.randn(10, D)
        out_sig = sigmoid(X @ W1_sig + b1_sig) @ W2_sig + b2_sig
        out_tanh = tanh(X @ W1_tanh + b1_tanh) @ W2_tanh + b2_tanh
        np.testing.assert_allclose(out_sig, out_tanh, atol=1e-12)

    def test_canonical_link_universal_delta(self):
        # PRML 5.2.2節 & 演習 5.6-5.9: 正準連結関数におけるデルタ公式 dE_n/da_k = y_k - t_k
        eps = 1e-6
        # 1. 二乗和誤差 + 恒等活性化
        a_id = 1.5
        t_id = 2.0
        y_id = a_id
        dE_da_id = y_id - t_id
        num_grad_id = (0.5 * (a_id + eps - t_id)**2 - 0.5 * (a_id - eps - t_id)**2) / (2 * eps)
        self.assertAlmostEqual(dE_da_id, num_grad_id, places=5)

        # 2. 二値交差エントロピー + ロジスティックシグモイド
        a_sig = 0.8
        t_sig = 1.0
        y_sig = sigmoid(a_sig)
        dE_da_sig = y_sig - t_sig
        def bce_loss(a):
            y = sigmoid(a)
            return - (t_sig * np.log(y) + (1.0 - t_sig) * np.log(1.0 - y))
        num_grad_sig = (bce_loss(a_sig + eps) - bce_loss(a_sig - eps)) / (2 * eps)
        self.assertAlmostEqual(dE_da_sig, num_grad_sig, places=5)

        # 3. 多クラス交差エントロピー + ソフトマックス
        a_soft = np.array([1.0, 2.0, -0.5])
        t_soft = np.array([0.0, 1.0, 0.0])
        y_soft = softmax(a_soft)
        dE_da_soft = y_soft - t_soft
        def cce_loss(a):
            y = softmax(a)
            return - np.sum(t_soft * np.log(y))
        num_grad_soft = np.zeros(3)
        for k in range(3):
            e_k = np.zeros(3)
            e_k[k] = eps
            num_grad_soft[k] = (cce_loss(a_soft + e_k) - cce_loss(a_soft - e_k)) / (2 * eps)
        np.testing.assert_allclose(dE_da_soft, num_grad_soft, atol=1e-5)

    def test_mlp_full_gradient_check(self):
        # PRML 5.3節: 誤差逆伝播法の数値勾配チェック (W1, b1, W2, b2)
        X = np.random.randn(5, 3)
        y = np.random.randn(5, 2)
        mlp = MLPRegressor(n_in=3, n_hidden=4, n_out=2, random_state=42)
        loss, grads = mlp.compute_loss_and_grads(X, y)

        eps = 1e-6
        for param_name in ['W1', 'b1', 'W2', 'b2']:
            param = getattr(mlp, param_name)
            analytic_grad = grads[param_name]
            numeric_grad = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                orig_val = param[idx]
                param[idx] = orig_val + eps
                l_plus, _ = mlp.compute_loss_and_grads(X, y)
                param[idx] = orig_val - eps
                l_minus, _ = mlp.compute_loss_and_grads(X, y)
                param[idx] = orig_val
                numeric_grad[idx] = (l_plus - l_minus) / (2 * eps)
                it.iternext()
            np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-3, atol=1e-4)

    def test_gauss_newton_hessian_outer_product(self):
        # PRML 5.4.2節 式 (5.84): ガウス・ニュートン外積近似 H ~ sum_n nabla y_n nabla y_n^T
        # 非線形活性化関数における真のヘッセ行列と外積近似の非負定値性
        M = 4
        J = np.random.randn(10, M)  # ヤコビアン行列
        H_gn = J.T @ J
        # ガウス・ニュートン近似行列は常に半正定値 (固有値 >= 0)
        eigvals = np.linalg.eigvalsh(H_gn)
        self.assertTrue(np.all(eigvals >= -1e-10))

    def test_weight_decay_regularization_effect(self):
        # PRML 5.5.1節 式 (5.112): 重み減衰 (Weight Decay / L2 正則化) による過学習抑制
        X_train = np.linspace(-1, 1, 8)[:, np.newaxis]
        y_train = np.sin(np.pi * X_train).ravel() + np.random.normal(0, 0.1, 8)
        
        # 正則化なし vs 正則化あり
        mlp_unreg = MLPRegressor(n_in=1, n_hidden=15, n_out=1, weight_decay=0.0, lr=0.01, random_state=42)
        mlp_unreg.fit(X_train, y_train[:, np.newaxis], n_epochs=200)

        mlp_reg = MLPRegressor(n_in=1, n_hidden=15, n_out=1, weight_decay=0.1, lr=0.01, random_state=42)
        mlp_reg.fit(X_train, y_train[:, np.newaxis], n_epochs=200)

        # 正則化ありモデルの重みノルムは正則化なしより小さい
        norm_unreg = np.sum(mlp_unreg.W1**2) + np.sum(mlp_unreg.W2**2)
        norm_reg = np.sum(mlp_reg.W1**2) + np.sum(mlp_reg.W2**2)
        self.assertLess(norm_reg, norm_unreg)

    def test_mdn_total_variance_decomposition(self):
        # PRML 5.6節 式 (5.158): 全分散の法則による不確実性分解
        # var[t|x] = sum_k pi_k sigma_k^2 + sum_k pi_k (mu_k - E[t|x])^2
        pi = np.array([0.3, 0.7])
        mu = np.array([-1.5, 2.0])
        sigma = np.array([0.4, 0.8])

        # 条件付き期待値 E[t|x]
        mean = np.sum(pi * mu)
        # 全分散
        intrinsic_var = np.sum(pi * (sigma**2))
        modal_spread = np.sum(pi * (mu - mean)**2)
        total_var = intrinsic_var + modal_spread

        # サンプリングによるモンテカルロ分散推定量との一致
        n_samples = 100000
        z = np.random.choice(2, size=n_samples, p=pi)
        samples = np.where(z == 0, np.random.normal(mu[0], sigma[0], size=n_samples),
                                  np.random.normal(mu[1], sigma[1], size=n_samples))
        sample_mean = np.mean(samples)
        sample_var = np.var(samples)

        self.assertAlmostEqual(mean, sample_mean, delta=0.03)
        self.assertAlmostEqual(total_var, sample_var, delta=0.05)

    def test_mdn_multimodal_inverse_problem(self):
        # PRML 5.6節 図 5.19-5.21: 逆問題における多峰性予測 (y = x + 0.3*sin(2*pi*x) の逆)
        N = 100
        t = np.random.uniform(0, 1, N)
        x = t + 0.3 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.05, N)

        mdn = MixtureDensityNetwork(n_in=1, n_hidden=10, n_components=3, random_state=42)
        mdn.fit(x[:, np.newaxis], t[:, np.newaxis], n_epochs=100, lr=0.05)

        # 各コンポーネントの重み pi が正かつ総和が1
        x_eval = np.array([[0.5]])
        z, pi, sigma, mu, a_sig = mdn.forward(x_eval)
        np.testing.assert_allclose(np.sum(pi, axis=1), [1.0], atol=1e-5)
        self.assertTrue(np.all(sigma > 0))


if __name__ == '__main__':
    unittest.main()
