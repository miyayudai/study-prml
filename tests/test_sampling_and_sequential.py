"""
PRML Chapter 11 & 13: Sampling and Sequential Data Models Unit Tests
MCMC、HMC、HMM、カルマンフィルタの統計的検証
"""

import unittest
import numpy as np

from prml.sampling import (
    rejection_sample,
    metropolis_hastings,
    gibbs_sampler_2d,
    hamiltonian_monte_carlo,
)

from prml.sequential import (
    GaussianHMM,
    KalmanFilter,
)

class TestSamplingMethods(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_rejection_sample_normal(self):
        # 標準正規分布を提案分布（一様分布）からサンプリング
        target = lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        proposal_s = lambda: np.random.uniform(-4, 4)
        proposal_p = lambda x: 1.0 / 8.0
        
        samples, acc = rejection_sample(target, proposal_s, proposal_p, k=4.0, n_samples=200)
        self.assertEqual(len(samples), 200)
        self.assertAlmostEqual(np.mean(samples), 0.0, delta=0.3)

    def test_gibbs_sampler_correlation(self):
        # 相関 rho=0.6 の2変量正規分布のギブスサンプリング
        rho = 0.6
        cond_x = lambda y: np.random.normal(rho * y, np.sqrt(1 - rho**2))
        cond_y = lambda x: np.random.normal(rho * x, np.sqrt(1 - rho**2))
        
        traj = gibbs_sampler_2d(cond_x, cond_y, initial_state=[0.0, 0.0], n_samples=500)
        burn_in = traj[100:]
        sample_cov = np.corrcoef(burn_in[:, 0], burn_in[:, 1])[0, 1]
        self.assertAlmostEqual(sample_cov, rho, delta=0.15)

    def test_hamiltonian_monte_carlo_harmonic(self):
        # 1次元調和振動子 (標準正規分布 U(q) = 0.5 * q^2) に対する HMC
        potential_energy = lambda q: 0.5 * (q[0]**2)
        grad_potential = lambda q: np.array([q[0]])
        
        samples, acc = hamiltonian_monte_carlo(
            potential_energy=potential_energy,
            grad_potential=grad_potential,
            initial_state=np.array([0.0]),
            n_samples=100,
            step_size=0.1,
            n_leapfrog=10
        )
        self.assertEqual(len(samples), 100)
        self.assertGreater(acc, 0.5)

    def test_hmc_leapfrog_reversibility_and_energy(self):
        # PRML 11.5.1節: リープフロッグ積分器の時間反転対称性とハミルトニアン H(q, r) の保存性
        potential_energy = lambda q: 0.5 * (q[0]**2)
        grad_potential = lambda q: np.array([q[0]])
        
        q0 = np.array([1.5])
        r0 = np.array([-0.8])
        step_size = 0.05
        n_leapfrog = 20
        
        # 順方向リープフロッグ
        q = q0.copy()
        r = r0.copy()
        r -= 0.5 * step_size * grad_potential(q)
        for step in range(n_leapfrog):
            q += step_size * r
            if step != n_leapfrog - 1:
                r -= step_size * grad_potential(q)
        r -= 0.5 * step_size * grad_potential(q)
        
        # エネルギー保存性: リープフロッグ法はハミルトニアン H(q, r) を O(step_size^2) の精度で保存
        H_init = potential_energy(q0) + 0.5 * np.sum(r0**2)
        H_final = potential_energy(q) + 0.5 * np.sum(r**2)
        self.assertAlmostEqual(H_init, H_final, delta=0.01)
        
        # 逆方向反転 (r -> -r で再度積分すると厳密に初期状態 (q0, -r0) に戻る)
        r_rev = -r
        for step in range(n_leapfrog):
            if step == 0:
                r_rev -= 0.5 * step_size * grad_potential(q)
            q += step_size * r_rev
            if step != n_leapfrog - 1:
                r_rev -= step_size * grad_potential(q)
            else:
                r_rev -= 0.5 * step_size * grad_potential(q)
        
        np.testing.assert_allclose(q, q0, atol=1e-7)

    def test_metropolis_hastings_detailed_balance(self):
        # PRML 11.2.2節 式 (11.34): メトロポリス・ヘイスティングス法の詳細釣り合い条件
        # p(x) q(x'|x) alpha(x, x') = p(x') q(x|x') alpha(x', x)
        target = lambda x: np.exp(-0.5 * (x - 1.0)**2 / 0.5)
        
        # 2点 x1, x2 における対称ガウス提案 q(x'|x) = q(x|x')
        x1, x2 = 0.5, 1.5
        p1, p2 = target(x1), target(x2)
        alpha12 = min(1.0, p2 / p1)
        alpha21 = min(1.0, p1 / p2)
        
        # 詳細釣り合いの成立
        lhs = p1 * alpha12
        rhs = p2 * alpha21
        self.assertAlmostEqual(lhs, rhs, places=7)

        # M-H サンプラーの実行と平均推定
        target_log = lambda x: -0.5 * (x[0] - 1.0)**2 / 0.5
        proposal_sample = lambda x: x + np.random.normal(0.0, 0.5, size=x.shape)
        samples, accepted_flags = metropolis_hastings(
            target_log_pdf=target_log,
            initial_state=np.array([0.0]),
            proposal_sampler=proposal_sample,
            n_samples=800,
            random_state=42
        )
        self.assertEqual(len(samples), 800)
        acc_rate = np.mean(accepted_flags)
        self.assertGreater(acc_rate, 0.2)
        self.assertAlmostEqual(float(np.mean(samples[150:, 0])), 1.0, delta=0.25)


class TestSequentialData(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_kalman_filter_smoothing_variance_reduction(self):
        # カルマンフィルタ: 平滑化事後共分散 <= フィルタリング共分散
        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        C = np.array([[1.0, 0.0]])
        Gamma = np.eye(2) * 0.05
        Sigma = np.eye(1) * 0.2
        mu_0 = np.array([0.0, 1.0])
        V_0 = np.eye(2)
        
        kf = KalmanFilter(A, C, Gamma, Sigma, mu_0, V_0)
        obs_seq = np.linspace(0, 5, 20)[:, np.newaxis] + np.random.normal(0, 0.2, (20, 1))
        
        means_filt, covs_filt = kf.filter(obs_seq)
        means_smooth, covs_smooth = kf.smooth(obs_seq)
        
        # 系列中盤 (t=10) において平滑化分散 <= フィルタリング分散
        var_filt = covs_filt[10, 0, 0]
        var_smooth = covs_smooth[10, 0, 0]
        self.assertLessEqual(var_smooth, var_filt + 1e-6)

    def test_gaussian_hmm_sampling_and_pred(self):
        hmm = GaussianHMM(n_components=2, n_iter=5)
        hmm.pi_ = np.array([0.5, 0.5])
        hmm.A_ = np.array([[0.8, 0.2], [0.2, 0.8]])
        hmm.means_ = np.array([[-2.0], [2.0]])
        hmm.covs_ = np.array([[[0.3]], [[0.3]]])
        
        states, obs = hmm.sample(n_samples=50)
        preds = hmm.predict(obs)
        self.assertEqual(len(preds), 50)
        # 状態予測の一致率が高いこと
        acc = np.mean(preds == states)
        self.assertGreater(acc, 0.70)

    def test_hmm_forward_backward_consistency(self):
        # PRML 13.2.2節 式 (13.36)-(13.42): 前向き変数 alpha_n(k) による観測尤度 p(X)
        hmm = GaussianHMM(n_components=2, n_iter=5)
        hmm.pi_ = np.array([0.6, 0.4])
        hmm.A_ = np.array([[0.7, 0.3], [0.4, 0.6]])
        hmm.means_ = np.array([[-1.0], [1.0]])
        hmm.covs_ = np.array([[[0.5]], [[0.5]]])
        
        obs = np.array([[-0.8], [-0.9], [0.9], [1.1], [0.8]])
        B = hmm._emission_probs(obs)
        # 前向きメッセージの対数尤度
        alpha, c = hmm._forward(B)
        self.assertEqual(alpha.shape, (5, 2))
        # 各時刻での正規化条件 (行和が 1)
        np.testing.assert_allclose(np.sum(alpha, axis=1), np.ones(5), atol=1e-10)
        # スケーリング係数 c_n は正
        self.assertTrue(np.all(c > 0.0))

    def test_kalman_filter_gain_and_variance_monotonicity(self):
        # PRML 13.3.1節 式 (13.90), (13.92): 観測更新による共分散低減 V_n <= P_n
        A = np.array([[1.0]])
        C = np.array([[1.0]])
        Gamma = np.array([[0.1]]) # 過程ノイズ
        Sigma = np.array([[0.5]]) # 観測ノイズ
        mu_0 = np.array([0.0])
        V_0 = np.array([[1.0]])
        
        kf = KalmanFilter(A, C, Gamma, Sigma, mu_0, V_0)
        obs = np.array([[1.2], [1.5], [1.1], [1.8]])
        means_filt, covs_filt = kf.filter(obs)
        
        # 各時刻において事後分散 V_n は事前予測分散 P_n 未満
        P_prev = V_0
        for n in range(len(obs)):
            P_n = A @ P_prev @ A.T + Gamma
            V_n = covs_filt[n]
            # V_n <= P_n
            self.assertLessEqual(V_n[0, 0], P_n[0, 0] + 1e-10)
            P_prev = V_n


if __name__ == '__main__':
    unittest.main()

