"""
Tests for PRML Chapter 2 Probability Distributions Module (prml.distributions)
"""

import unittest
import numpy as np
import scipy.integrate as integrate

from prml.distributions import (
    Gaussian1D,
    MultivariateGaussian,
    BetaDistribution,
    DirichletDistribution,
    GammaDistribution,
    StudentsTDistribution,
    VonMisesDistribution,
    KernelDensityEstimator,
    KNearestNeighborsDensity,
    RobbinsMonro,
    simplex_to_xy,
    student_t_pdf,
    von_mises_pdf,
)


class TestDistributions(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_gaussian_1d_mle_and_pdf(self):
        data = np.random.normal(loc=3.5, scale=1.2, size=1000)
        g = Gaussian1D().fit(data)
        self.assertAlmostEqual(g.mu, 3.5, delta=0.1)
        self.assertAlmostEqual(np.sqrt(g.var), 1.2, delta=0.1)
        self.assertGreater(g.unbiased_var, g.var)

        # PDF normalization check by numerical integration
        integral, _ = integrate.quad(g.pdf, -10, 15)
        self.assertAlmostEqual(integral, 1.0, places=4)

    def test_gaussian_1d_bayesian_updates(self):
        # Bayesian mean update
        prior_mean, prior_var = 0.0, 4.0
        noise_var = 1.0
        X = np.array([2.0, 3.0, 2.5, 3.5])  # N=4, mean = 2.75
        post_g = Gaussian1D.bayesian_mean_update(X, prior_mean, prior_var, noise_var)
        
        # Theoretical posterior precision: 1/4 + 4/1 = 4.25 -> var = 1/4.25 ~ 0.2353
        expected_var = 1.0 / (1.0 / 4.0 + 4.0 / 1.0)
        expected_mean = expected_var * (0.0 / 4.0 + (4.0 * 2.75) / 1.0)
        self.assertAlmostEqual(post_g.var, expected_var, places=4)
        self.assertAlmostEqual(post_g.mu, expected_mean, places=4)

        # Bayesian precision update with Gamma prior
        prior_a, prior_b = 2.0, 2.0
        post_gamma = Gaussian1D.bayesian_precision_update(X, prior_a, prior_b, true_mean=2.75)
        self.assertEqual(post_gamma.a, prior_a + 2.0)
        self.assertGreater(post_gamma.b, prior_b)

    def test_multivariate_gaussian_mle_and_mahalanobis(self):
        mean_true = np.array([1.5, -2.0])
        cov_true = np.array([[2.0, 0.8], [0.8, 1.5]])
        X = np.random.multivariate_normal(mean_true, cov_true, size=2000)

        mg = MultivariateGaussian().fit(X)
        np.testing.assert_allclose(mg.mean, mean_true, atol=0.15)
        np.testing.assert_allclose(mg.cov, cov_true, atol=0.2)

        # Mahalanobis distance at mean should be 0
        dist_at_mean = mg.mahalanobis_distance(mg.mean)
        self.assertAlmostEqual(float(dist_at_mean), 0.0, places=5)

    def test_multivariate_gaussian_conditioning_and_marginalization(self):
        # 3D Gaussian
        mu = np.array([1.0, 2.0, 3.0])
        cov = np.array([
            [2.0, 0.5, 0.2],
            [0.5, 1.5, 0.3],
            [0.2, 0.3, 1.0]
        ])
        mg = MultivariateGaussian(mean=mu, cov=cov)

        # Marginal p(x_0, x_1)
        marg = mg.marginalize(idx_a=[0, 1])
        np.testing.assert_allclose(marg.mean, [1.0, 2.0])
        np.testing.assert_allclose(marg.cov, cov[:2, :2])

        # Conditional p(x_0 | x_1=2.5, x_2=3.5)
        cond = mg.condition(x_b=[2.5, 3.5], idx_a=[0], idx_b=[1, 2])
        self.assertEqual(cond.dim, 1)
        self.assertIsInstance(cond.mean[0], float)
        self.assertGreater(cond.cov[0, 0], 0.0)

    def test_linear_gaussian_system(self):
        # Bishop PRML eq 2.113 - 2.115
        mu_x = np.array([0.5])
        sigma_x = np.array([[1.0]])
        A = np.array([[2.0]])
        b = np.array([1.0])
        L_cov = np.array([[0.5]])

        marginal_y, posterior_solver = MultivariateGaussian.linear_gaussian_system(
            mu_x, sigma_x, A, b, L_cov
        )
        # E[y] = A mu_x + b = 2*0.5 + 1 = 2.0
        self.assertAlmostEqual(marginal_y.mean[0], 2.0, places=5)
        # var[y] = L + A sigma_x A^T = 0.5 + 2*1*2 = 4.5
        self.assertAlmostEqual(marginal_y.cov[0, 0], 4.5, places=5)

        # Posterior given y=2.0 (matches prior mean)
        post = posterior_solver([2.0])
        self.assertAlmostEqual(post.mean[0], 0.5, places=5)
        # Posterior variance: (1/1 + 4/0.5)^-1 = (1 + 8)^-1 = 1/9
        self.assertAlmostEqual(post.cov[0, 0], 1.0 / 9.0, places=5)

    def test_beta_distribution(self):
        beta = BetaDistribution(a=3.0, b=5.0)
        self.assertAlmostEqual(beta.mean, 3.0 / 8.0)
        expected_var = (3.0 * 5.0) / (8.0**2 * 9.0)
        self.assertAlmostEqual(beta.variance, expected_var)
        self.assertAlmostEqual(beta.mode, (3.0 - 1.0) / (3.0 + 5.0 - 2.0))

        # Conjugate update: 7 heads, 3 tails
        post = beta.bayesian_update(n_heads=7, n_tails=3)
        self.assertEqual(post.a, 10.0)
        self.assertEqual(post.b, 8.0)
        self.assertAlmostEqual(post.predictive_probability(), 10.0 / 18.0)

    def test_dirichlet_distribution(self):
        alpha = np.array([2.0, 3.0, 5.0])
        diri = DirichletDistribution(alpha=alpha)
        np.testing.assert_allclose(diri.mean, [0.2, 0.3, 0.5])
        
        # Off-diagonal covariances must be negative
        cov = diri.covariance_matrix
        self.assertLess(cov[0, 1], 0.0)
        self.assertLess(cov[1, 2], 0.0)

        # Mode
        np.testing.assert_allclose(diri.mode, (alpha - 1.0) / (10.0 - 3.0))

        # Simplex mapping
        x, y = simplex_to_xy([0.0, 1.0, 0.0])
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)

    def test_gamma_and_students_t(self):
        # Gamma
        gam = GammaDistribution(a=4.0, b=2.0)
        self.assertAlmostEqual(gam.mean, 2.0)
        self.assertAlmostEqual(gam.variance, 1.0)
        self.assertAlmostEqual(gam.mode, 1.5)

        # Student's t
        st = StudentsTDistribution(mu=1.0, lam=2.0, nu=5.0)
        self.assertEqual(st.mean, 1.0)
        # var = nu / (lam * (nu - 2)) = 5 / (2 * 3) = 5/6
        self.assertAlmostEqual(st.variance, 5.0 / 6.0)

        # Robustness against outliers: Student's t tails are heavier than Gaussian
        st_heavy = StudentsTDistribution(mu=0.0, lam=1.0, nu=1.0) # Cauchy
        g = Gaussian1D(mu=0.0, var=1.0)
        self.assertGreater(st_heavy.pdf(10.0), g.pdf(10.0))

    def test_von_mises(self):
        vm = VonMisesDistribution(mu=np.pi / 2.0, kappa=3.0)
        self.assertGreater(vm.pdf(np.pi / 2.0), vm.pdf(0.0))

        # Fit to data clustered around pi
        thetas = np.random.normal(loc=np.pi, scale=0.3, size=500) % (2.0 * np.pi)
        vm_fit = VonMisesDistribution().fit(thetas)
        self.assertAlmostEqual(vm_fit.mu, np.pi, delta=0.2)
        self.assertGreater(vm_fit.kappa, 1.0)

    def test_kde_and_knn(self):
        X = np.random.normal(loc=0.0, scale=1.0, size=200)
        # KDE with Gaussian and Epanechnikov
        kde_gauss = KernelDensityEstimator(bandwidth=0.5, kernel='gaussian').fit(X)
        kde_epan = KernelDensityEstimator(bandwidth=0.5, kernel='epanechnikov').fit(X)

        # Center density should be significantly higher than tail density
        self.assertGreater(kde_gauss.score_samples([0.0]), kde_gauss.score_samples([3.5]))
        self.assertGreater(kde_epan.score_samples([0.0]), kde_epan.score_samples([3.5]))

        # Silverman's bandwidth rule
        h_silver = KernelDensityEstimator.silverman_bandwidth(X)
        self.assertGreater(h_silver, 0.0)

        # KNN Density
        knn = KNearestNeighborsDensity(k=10).fit(X)
        p_center = knn.score_samples([0.0])
        p_tail = knn.score_samples([3.0])
        self.assertGreater(p_center, p_tail)

    def test_robbins_monro(self):
        mu_true = 5.0
        X = np.random.normal(loc=mu_true, scale=2.0, size=2000)
        rm = RobbinsMonro(a_coeff=1.0)
        final_theta, history = rm.estimate_mean(X, init_mu=0.0)
        self.assertAlmostEqual(final_theta, mu_true, delta=0.2)


if __name__ == '__main__':
    unittest.main()
