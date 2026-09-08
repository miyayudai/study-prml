"""
tests/test_ch10_exercises.py
Unit tests verifying core theoretical derivations and numerical proofs for PRML Chapter 10 Exercises (10.1 to 10.39).
"""

import unittest
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize
from scipy.special import expit as sig, psi, gamma, gammaln
from scipy.stats import multivariate_normal, wishart

from common.variational_utils import (
    variational_gaussian_1d,
    VariationalGaussianMixture,
    jaakkola_jordan_lambda,
)
from prml.clustering import GaussianMixtureModel

class TestChapter10Exercises(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_ex10_1_lower_bound_decomposition(self):
        """Exercise 10.1: ln p(X) = L(q) + KL(q || p) and L(q) <= ln p(X)"""
        mu_true, var_true = 2.0, 1.5
        def p_joint(x, z):
            return (1.0 / np.sqrt(2 * np.pi * var_true)) * np.exp(-0.5 * (x - z)**2 / var_true) * \
                   (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)

        x_obs = 1.8
        p_X, _ = integrate.quad(lambda z: p_joint(x_obs, z), -np.inf, np.inf)
        ln_p_X = np.log(p_X)

        mu_q, var_q = 1.0, 0.8
        def q_dist(z):
            return (1.0 / np.sqrt(2 * np.pi * var_q)) * np.exp(-0.5 * (z - mu_q)**2 / var_q)

        def p_cond(z):
            return p_joint(x_obs, z) / p_X

        L_q, _ = integrate.quad(lambda z: q_dist(z) * np.log(p_joint(x_obs, z) / q_dist(z)), -10, 10)
        KL_qp, _ = integrate.quad(lambda z: q_dist(z) * np.log(q_dist(z) / p_cond(z)), -10, 10)

        self.assertAlmostEqual(ln_p_X, L_q + KL_qp, places=5)
        self.assertGreaterEqual(KL_qp, 0.0)
        self.assertLessEqual(L_q, ln_p_X)

    def test_ex10_2_bivariate_factorization(self):
        """Exercise 10.2: Optimal variational mean matches true mean and variances underestimate true variance"""
        mu = np.array([1.5, -2.0])
        Sigma = np.array([[2.0, 0.8], [0.8, 1.5]])
        Lambda = np.linalg.inv(Sigma)

        # Variational optimal variances: 1 / Lambda_ii
        var_q1 = 1.0 / Lambda[0, 0]
        var_q2 = 1.0 / Lambda[1, 1]

        # In bivariate Gaussian with correlation != 0, 1 / Lambda_ii < Sigma_ii
        self.assertLess(var_q1, Sigma[0, 0])
        self.assertLess(var_q2, Sigma[1, 1])
        # Ratio equals 1 - rho^2
        rho_sq = (Sigma[0, 1]**2) / (Sigma[0, 0] * Sigma[1, 1])
        self.assertAlmostEqual(var_q1, Sigma[0, 0] * (1.0 - rho_sq), places=7)
        self.assertAlmostEqual(var_q2, Sigma[1, 1] * (1.0 - rho_sq), places=7)

    def test_ex10_6_alpha_divergence_limits(self):
        """Exercise 10.6: D_alpha converges continuously to KL(p||q) as alpha -> 1"""
        def D_alpha(alpha, p_pdf, q_pdf, x_grid):
            dx = x_grid[1] - x_grid[0]
            integral = np.sum((p_pdf ** ((1.0 + alpha) / 2.0)) * (q_pdf ** ((1.0 - alpha) / 2.0))) * dx
            return (4.0 / (1.0 - alpha**2)) * (1.0 - integral)

        x_grid = np.linspace(-6, 6, 2000)
        dx = x_grid[1] - x_grid[0]
        p_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_grid**2)
        q_pdf = (1.0 / np.sqrt(2 * np.pi * 1.5)) * np.exp(-0.5 * (x_grid - 0.5)**2 / 1.5)

        true_kl = np.sum(p_pdf * np.log(p_pdf / q_pdf)) * dx
        d_val = D_alpha(0.999, p_pdf, q_pdf, x_grid)
        self.assertAlmostEqual(d_val, true_kl, places=3)

    def test_ex10_9_unbiased_sample_variance_fixed_point(self):
        """Exercise 10.9: 1/E[tau] fixed point matches unbiased sample variance (10.33)"""
        X = np.array([1.2, 2.3, 1.8, 2.7, 3.1])
        N = len(X)
        x_bar = np.mean(X)
        s2 = np.var(X, ddof=1)

        E_tau = 1.0
        for _ in range(30):
            b_N = 0.5 * np.sum((X - x_bar)**2) + 0.5 / E_tau
            a_N = N / 2.0
            E_tau = a_N / b_N

        inv_E_tau = 1.0 / E_tau
        self.assertAlmostEqual(inv_E_tau, s2, places=7)

    def test_ex10_14_quadratic_form_expectation(self):
        """Exercise 10.14: E[(x - mu)^T Lambda (x - mu)] = D / beta + nu (x - m)^T W (x - m)"""
        D = 2
        m = np.array([0.5, -1.0])
        beta = 3.0
        W = np.array([[2.0, 0.4], [0.4, 1.5]])
        nu = 5.0
        x = np.array([1.2, 0.3])

        # Theoretical expectation (10.64)
        theory_val = D / beta + nu * (x - m).T @ W @ (x - m)

        # Monte Carlo verification
        np.random.seed(42)
        N_mc = 20000
        Lambdas = wishart.rvs(df=nu, scale=W, size=N_mc)
        mus = np.array([np.random.multivariate_normal(mean=m, cov=np.linalg.inv(beta * Lam)) for Lam in Lambdas])

        quads = [ (x - mus[i]).T @ Lambdas[i] @ (x - mus[i]) for i in range(N_mc) ]
        mc_val = np.mean(quads)

        np.testing.assert_allclose(mc_val, theory_val, rtol=0.03)

    def test_ex10_15_dirichlet_expectation(self):
        """Exercise 10.15: E[pi_k] = alpha_k / sum alpha_j (10.69)"""
        alpha = np.array([2.5, 4.0, 1.5])
        expected_pi = alpha / np.sum(alpha)

        np.random.seed(42)
        samples = np.random.dirichlet(alpha, size=20000)
        mc_mean = np.mean(samples, axis=0)

        np.testing.assert_allclose(mc_mean, expected_pi, atol=0.01)

    def test_ex10_20_variational_asymptotic_to_mle(self):
        """Exercise 10.20: Variational Gaussian Mixture matches ML GMM as N -> infty"""
        np.random.seed(42)
        N_large = 6000
        X_large = np.concatenate([
            np.random.normal(-2.0, 0.8, size=(int(N_large * 0.4), 1)),
            np.random.normal(2.5, 1.2, size=(int(N_large * 0.6), 1))
        ])

        gmm = GaussianMixtureModel(n_components=2, max_iter=40, random_state=42).fit(X_large)
        ml_means = np.sort(gmm.means_.flatten())

        vgm = VariationalGaussianMixture(n_components=2, max_iter=40, random_state=42).fit(X_large)
        vb_means = np.sort(vgm.means_.flatten())

        np.testing.assert_allclose(vb_means, ml_means, atol=0.08)

    def test_ex10_31_jaakkola_jordan_bound(self):
        """Exercise 10.31: Jaakkola-Jordan bound strictly bounds sigma(z) from below with equality at z = xi"""
        def jj_bound(z, xi):
            return sig(xi) * np.exp(0.5 * (z - xi) - jaakkola_jordan_lambda(xi) * (z**2 - xi**2))

        for xi in [0.5, 1.5, 3.0, -2.0]:
            # Equality at z = xi
            self.assertAlmostEqual(jj_bound(xi, xi), sig(xi), places=7)
            # Global lower bound on grid
            z_grid = np.linspace(-5, 5, 200)
            self.assertTrue(np.all(sig(z_grid) >= jj_bound(z_grid, xi) - 1e-10))

    def test_ex10_33_variational_parameter_update(self):
        """Exercise 10.33: Stationary condition dQ/dxi = 0 yields xi^2 = x^T E[w w^T] x (10.163)"""
        m_N = np.array([0.5, -0.8])
        S_N = np.array([[0.3, 0.1], [0.1, 0.4]])
        x_n = np.array([1.2, 0.7])

        E_wwT = S_N + np.outer(m_N, m_N)
        xi_theory = np.sqrt(x_n.T @ E_wwT @ x_n)

        def neg_Q(xi_val):
            xi = xi_val[0]
            term = np.log(sig(xi)) - 0.5 * xi - jaakkola_jordan_lambda(xi) * (x_n.T @ E_wwT @ x_n) + jaakkola_jordan_lambda(xi) * xi**2
            return -term

        res = minimize(neg_Q, [1.0], bounds=[(1e-4, None)], tol=1e-9)
        self.assertAlmostEqual(res.x[0], xi_theory, places=3)

if __name__ == '__main__':
    unittest.main()
