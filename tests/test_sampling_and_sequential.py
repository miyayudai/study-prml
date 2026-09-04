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

if __name__ == '__main__':
    unittest.main()
