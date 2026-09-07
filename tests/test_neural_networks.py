"""
PRML Chapter 5: Neural Networks Comprehensive Unit Tests
ニューラルネットワーク、誤差逆伝播法、ヘッセ行列、正則化、MDN、ベイズニューラルネットの厳密な数理検証
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs, make_regression

from prml.nn import (
    MLPRegressor,
    MLPClassifier,
    MixtureDensityNetwork,
    BayesianMLPRegressor,
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

    def test_mlp_classifier_binary_and_multiclass(self):
        # PRML 5.2-5.3節: 二値分類 (シグモイド) および多クラス分類 (ソフトマックス) の検証
        X_b, y_b = make_blobs(n_samples=60, n_features=2, centers=2, random_state=42)
        clf_b = MLPClassifier(2, 6, n_classes=2, lr=0.05, random_state=42)
        clf_b.fit(X_b, y_b, n_epochs=300)
        acc_b = np.mean(clf_b.predict(X_b) == y_b)
        self.assertGreaterEqual(acc_b, 0.95)

        X_m, y_m = make_blobs(n_samples=60, n_features=2, centers=3, random_state=42)
        clf_m = MLPClassifier(2, 8, n_classes=3, lr=0.05, random_state=42)
        clf_m.fit(X_m, y_m, n_epochs=400)
        acc_m = np.mean(clf_m.predict(X_m) == y_m)
        self.assertGreaterEqual(acc_m, 0.95)

    def test_hessian_exact_and_gauss_newton(self):
        # PRML 5.4節: 厳密ヘッセ行列とガウス・ニュートン外積近似の検証
        mlp = MLPRegressor(2, 3, 1, random_state=42)
        X = np.random.randn(8, 2)
        T = np.random.randn(8, 1)
        H_ex = mlp.compute_hessian_exact(X, T)
        H_gn = mlp.compute_hessian_gauss_newton(X)
        self.assertEqual(H_ex.shape, (13, 13))
        self.assertEqual(H_gn.shape, (13, 13))

        # ガウス・ニュートン近似は常に半正定値 (固有値 >= 0)
        eigvals_gn = np.linalg.eigvalsh(H_gn)
        self.assertTrue(np.all(eigvals_gn >= -1e-10))

    def test_hessian_vector_product_pearlmutter(self):
        # PRML 5.4.7節: Pearlmutter の R{.} 演算子による高速ヘッセ・ベクトル積 H v
        mlp = MLPRegressor(2, 3, 1, random_state=42)
        X = np.random.randn(8, 2)
        T = np.random.randn(8, 1)
        H_ex = mlp.compute_hessian_exact(X, T)
        v = np.random.randn(13)
        Hv_ex = H_ex @ v
        Hv_rop = mlp.hessian_vector_product(X, T, v)
        np.testing.assert_allclose(Hv_ex, Hv_rop, rtol=1e-3, atol=1e-4)

    def test_bayesian_mlp_regressor_predictive_distribution_and_evidence(self):
        # PRML 5.7節: ベイズニューラルネットワークのラプラス近似・予測分布・エビデンス検証
        X_r = np.linspace(-1, 1, 15)[:, np.newaxis]
        T_r = np.sin(np.pi * X_r) + np.random.normal(0, 0.05, (15, 1))
        bnn = BayesianMLPRegressor(1, 4, 1, alpha=1.0, beta=10.0, random_state=42)
        bnn.fit(X_r, T_r, n_epochs=300)
        mean, std = bnn.predict(X_r)
        self.assertEqual(len(std), 15)
        self.assertTrue(np.all(std > 0))
        ev = bnn.compute_evidence(X_r, T_r)
        self.assertFalse(np.isnan(ev))

    def test_weight_space_symmetries_tanh(self):
        # PRML 5.1.1節 & 演習 5.21: tanh 隠れユニットの符号反転対称性
        mlp = MLPRegressor(2, 3, 1, random_state=42)
        x_test = np.random.randn(5, 2)
        y_orig = mlp.predict(x_test)

        # 隠れユニット0の符号反転
        mlp_sym = MLPRegressor(2, 3, 1, random_state=42)
        mlp_sym.W1[:, 0] *= -1
        mlp_sym.b1[0] *= -1
        mlp_sym.W2[0, :] *= -1
        y_sym = mlp_sym.predict(x_test)
        np.testing.assert_allclose(y_orig, y_sym, atol=1e-12)

    def test_forward_jacobian_propagation(self):
        # PRML 演習 5.15: ヤコビ行列の前向き伝播漸化式と数値微分の一致
        D, M, K = 3, 4, 2
        W1 = np.random.randn(D, M); b1 = np.random.randn(M)
        W2 = np.random.randn(M, K); b2 = np.random.randn(K)
        x = np.random.randn(D)

        a1 = x @ W1 + b1
        z1 = np.tanh(a1)
        # 前向き伝播による解析的ヤコビ行列 J (K, D)
        J_forward = (W2.T * (1.0 - z1**2)) @ W1.T

        eps = 1e-6
        J_num = np.zeros((K, D))
        for i in range(D):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps; xm[i] -= eps
            yp = np.tanh(xp @ W1 + b1) @ W2 + b2
            ym = np.tanh(xm @ W1 + b1) @ W2 + b2
            J_num[:, i] = (yp - ym) / (2 * eps)

        np.testing.assert_allclose(J_forward, J_num, atol=1e-6)

    def test_sherman_morrison_hessian_update(self):
        # PRML 演習 5.21: 外積ヘッセ更新に対する Sherman-Morrison 公式
        P = 5
        H_prev = np.random.randn(P, P)
        H_prev = H_prev.T @ H_prev + np.eye(P)
        H_inv_prev = np.linalg.inv(H_prev)

        b_N = np.random.randn(P)
        H_new = H_prev + np.outer(b_N, b_N)
        H_inv_direct = np.linalg.inv(H_new)

        v = H_inv_prev @ b_N
        H_inv_sm = H_inv_prev - np.outer(v, v) / (1.0 + b_N @ v)
        np.testing.assert_allclose(H_inv_direct, H_inv_sm, atol=1e-10)

    def test_affine_transformation_invariance(self):
        # PRML 演習 5.24: 入力のアフィン変換に対する重み調整と出力の完全保存
        D, M, K = 3, 4, 2
        W1 = np.random.randn(D, M); b1 = np.random.randn(M)
        W2 = np.random.randn(M, K); b2 = np.random.randn(K)

        A = np.array([2.5, -1.2, 0.8])
        B = np.array([0.4, 1.1, -0.6])

        W1_tilde = W1 / A[:, None]
        b1_tilde = b1 - np.sum(W1 * (B[:, None] / A[:, None]), axis=0)

        X = np.random.randn(10, D)
        X_tilde = X * A + B

        out_orig = np.tanh(X @ W1 + b1) @ W2 + b2
        out_trans = np.tanh(X_tilde @ W1_tilde + b1_tilde) @ W2 + b2
        np.testing.assert_allclose(out_orig, out_trans, atol=1e-12)

    def test_soft_weight_sharing_gradients(self):
        # PRML 演習 5.29-5.32: ソフト重み共有の各パラメータに関する解析勾配と数値微分の一致
        import scipy.stats as stats
        w_vec = np.array([-0.5, 0.2, 1.1])
        pi_k = np.array([0.4, 0.6])
        mu_k = np.array([-0.3, 0.9])
        sig_k = np.array([0.4, 0.7])

        dens = np.array([stats.norm.pdf(w, loc=mu_k, scale=sig_k) for w in w_vec])
        gamma = (dens * pi_k) / np.sum(dens * pi_k, axis=1, keepdims=True)

        # 1. dOmega / d w_i (式 5.141)
        ana_grad_w = np.sum(gamma * (w_vec[:, None] - mu_k) / (sig_k**2), axis=1)
        eps = 1e-6
        for i in range(len(w_vec)):
            wp, wm = w_vec.copy(), w_vec.copy()
            wp[i] += eps; wm[i] -= eps
            lp = -np.log(np.sum(pi_k * stats.norm.pdf(wp[i], loc=mu_k, scale=sig_k)))
            lm = -np.log(np.sum(pi_k * stats.norm.pdf(wm[i], loc=mu_k, scale=sig_k)))
            num_g = (lp - lm) / (2 * eps)
            self.assertAlmostEqual(ana_grad_w[i], num_g, places=5)

        # 2. dOmega / d mu_j (式 5.142)
        ana_grad_mu0 = np.sum(gamma[:, 0] * (mu_k[0] - w_vec) / (sig_k[0]**2))
        mup = mu_k.copy(); mup[0] += eps
        mum = mu_k.copy(); mum[0] -= eps
        lp = -np.sum(np.log(np.sum(pi_k * np.array([stats.norm.pdf(w, loc=mup, scale=sig_k) for w in w_vec]), axis=1)))
        lm = -np.sum(np.log(np.sum(pi_k * np.array([stats.norm.pdf(w, loc=mum, scale=sig_k) for w in w_vec]), axis=1)))
        num_g_mu = (lp - lm) / (2 * eps)
        self.assertAlmostEqual(ana_grad_mu0, num_g_mu, places=5)

    def test_mdn_activation_gradients(self):
        # PRML 演習 5.34-5.36: MDN 出力活性化 a_pi, a_mu, a_sig に関するデルタ公式
        import scipy.stats as stats
        a_pi = np.array([0.3, -0.4, 0.8])
        pi = softmax(a_pi)
        mu = np.array([0.2, 1.1, -1.0])
        a_sig = np.array([-0.5, 0.1, 0.4])
        sigma = np.exp(a_sig)
        t = 0.5

        dens = stats.norm.pdf(t, loc=mu, scale=sigma)
        gamma = (pi * dens) / np.sum(pi * dens)

        # dE/da_pi = pi - gamma (式 5.155)
        ana_grad_pi = pi - gamma
        # dE/da_mu = gamma * (mu - t) / sigma^2 (式 5.156)
        ana_grad_mu = gamma * (mu - t) / (sigma**2)
        # dE/da_sig = gamma * (1 - (t - mu)^2 / sigma^2) (式 5.157)
        ana_grad_sig = gamma * (1.0 - ((t - mu)**2) / (sigma**2))

        eps = 1e-6
        # 数値微分チェック
        for k in range(3):
            ap, am = a_pi.copy(), a_pi.copy()
            ap[k] += eps; am[k] -= eps
            lp = -np.log(np.sum(softmax(ap) * dens))
            lm = -np.log(np.sum(softmax(am) * dens))
            self.assertAlmostEqual(ana_grad_pi[k], (lp - lm)/(2*eps), places=5)

            mup, mum = mu.copy(), mu.copy()
            mup[k] += eps; mum[k] -= eps
            lp_m = -np.log(np.sum(pi * stats.norm.pdf(t, loc=mup, scale=sigma)))
            lm_m = -np.log(np.sum(pi * stats.norm.pdf(t, loc=mum, scale=sigma)))
            self.assertAlmostEqual(ana_grad_mu[k], (lp_m - lm_m)/(2*eps), places=5)

            asp, asm = a_sig.copy(), a_sig.copy()
            asp[k] += eps; asm[k] -= eps
            lp_s = -np.log(np.sum(pi * stats.norm.pdf(t, loc=mu, scale=np.exp(asp))))
            lm_s = -np.log(np.sum(pi * stats.norm.pdf(t, loc=mu, scale=np.exp(asm))))
            self.assertAlmostEqual(ana_grad_sig[k], (lp_s - lm_s)/(2*eps), places=5)


if __name__ == '__main__':
    unittest.main()
