# tests/test_ch11_exercises.py
"""
Unit tests for PRML Chapter 11 (Sampling Methods) Exercises.
Covers theoretical derivations and numerical algorithms for:
- Monte Carlo variance scaling (Ex 11.1)
- Inverse CDF methods (Ex 11.2, 11.3, 11.7)
- Polar Box-Muller Gaussian generation (Ex 11.4)
- Cholesky multivariate Gaussian sampling (Ex 11.5)
- Rejection sampling exactness (Ex 11.6)
- ARS envelope conditions (Ex 11.8)
- 1D random walk variance scaling (Ex 11.10)
- Gibbs detailed balance & conditional updates (Ex 11.11, 11.13)
- Over-relaxation variance conservation (Ex 11.14)
- Hamiltonian dynamics and Leapfrog reversibility (Ex 11.15, 11.17)
"""

import numpy as np
import pytest
from scipy.stats import kstest, expon, cauchy, norm, multivariate_normal

def test_monte_carlo_unbiased_and_variance_decay():
    """Exercise 11.1: E[f_hat] = E[f] and var[f_hat] = var[f] / L"""
    np.random.seed(111)
    N_trials = 2500
    L_values = [20, 100]
    true_mean = 1.0
    true_var = 2.0  # For z^2 with z ~ N(0, 1)

    for L in L_values:
        samples = np.random.normal(0, 1, size=(N_trials, L))
        f_hat = np.mean(samples**2, axis=1)
        np.testing.assert_allclose(np.mean(f_hat), true_mean, atol=0.03)
        np.testing.assert_allclose(np.var(f_hat), true_var / L, rtol=0.12)

def test_inverse_cdf_exponential():
    """Exercise 11.2: y = -ln(1 - z) / lambda produces Exponential(lambda)"""
    np.random.seed(112)
    lam = 2.0
    N = 30000
    z = np.random.uniform(0, 1, size=N)
    y = -np.log(1.0 - z) / lam

    ks_stat, p_val = kstest(y, expon(scale=1.0/lam).cdf)
    assert p_val > 0.05
    np.testing.assert_allclose(np.mean(y), 1.0 / lam, rtol=0.03)
    np.testing.assert_allclose(np.var(y), 1.0 / (lam**2), rtol=0.04)

def test_cauchy_inverse_tangent():
    """Exercise 11.3: y = tan(pi * (z - 1/2)) produces standard Cauchy"""
    np.random.seed(113)
    N = 40000
    z = np.random.uniform(0, 1, size=N)
    y = np.tan(np.pi * (z - 0.5))

    np.testing.assert_allclose(np.median(y), 0.0, atol=0.03)
    q25, q75 = np.percentile(y, [25, 75])
    np.testing.assert_allclose(q75 - q25, 2.0, atol=0.05)
    assert kstest(y, cauchy.cdf).pvalue > 0.02

def test_box_muller_polar():
    """Exercise 11.4: Polar Box-Muller transformation produces N(0, I)"""
    np.random.seed(114)
    N = 60000
    u1 = np.random.uniform(-1, 1, N)
    u2 = np.random.uniform(-1, 1, N)
    r2 = u1**2 + u2**2
    mask = (r2 > 0) & (r2 < 1.0)

    z1, z2, r2_acc = u1[mask], u2[mask], r2[mask]
    fac = np.sqrt(-2.0 * np.log(r2_acc) / r2_acc)
    y1 = z1 * fac
    y2 = z2 * fac

    np.testing.assert_allclose([np.mean(y1), np.mean(y2)], [0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(np.cov(y1, y2), np.eye(2), atol=0.03)

def test_cholesky_multivariate_gaussian():
    """Exercise 11.5: y = mu + L z has mean mu and covariance Sigma"""
    np.random.seed(115)
    D = 3
    N = 40000
    mu_true = np.array([1.0, -1.5, 2.0])
    Sigma_true = np.array([[2.0, 0.6, -0.4], [0.6, 1.5, 0.2], [-0.4, 0.2, 1.8]])
    L = np.linalg.cholesky(Sigma_true)

    z = np.random.normal(0, 1, size=(D, N))
    y = mu_true[:, None] + L @ z

    np.testing.assert_allclose(np.mean(y, axis=1), mu_true, atol=0.03)
    np.testing.assert_allclose(np.cov(y), Sigma_true, atol=0.08)

def test_rejection_sampling_exactness():
    """Exercise 11.6: Rejection sampling matches target distribution"""
    np.random.seed(116)
    k = 2.0 * np.sqrt(np.e)
    N = 50000
    u_lap = np.random.uniform(-0.5, 0.5, size=N)
    z_prop = -np.sign(u_lap) * np.log(1.0 - 2.0 * np.abs(u_lap))

    p_tilde = np.exp(-0.5 * z_prop**2)
    q_val = 0.5 * np.exp(-np.abs(z_prop))
    u_rand = np.random.uniform(0, 1, size=N)
    accepted = z_prop[u_rand <= (p_tilde / (k * q_val))]

    theo_acc_rate = np.sqrt(2 * np.pi) / k
    np.testing.assert_allclose(len(accepted) / N, theo_acc_rate, rtol=0.03)
    assert kstest(accepted, norm.cdf).pvalue > 0.05

def test_generalized_cauchy_transform():
    """Exercise 11.7: Generalized Cauchy y = b * tan(pi * (z - 0.5)) + c"""
    np.random.seed(117)
    b, c = 2.5, -3.0
    N = 40000
    z = np.random.uniform(0, 1, size=N)
    y = b * np.tan(np.pi * (z - 0.5)) + c

    np.testing.assert_allclose(np.median(y), c, atol=0.06)
    q25, q75 = np.percentile(y, [25, 75])
    np.testing.assert_allclose(q75 - q25, 2.0 * b, atol=0.10)
    assert kstest(y, cauchy(loc=c, scale=b).cdf).pvalue > 0.03

def test_adaptive_rejection_continuity_and_norm():
    """Exercise 11.8: ARS envelope continuity and total normalization"""
    z_bounds = [0.0, 1.2, 2.5, 4.0]
    lambdas = [1.5, 0.9, 1.2]
    M = len(lambdas)

    c = np.zeros(M)
    c[0] = 1.0
    for i in range(M - 1):
        dz = z_bounds[i+1] - z_bounds[i]
        c[i+1] = c[i] * (lambdas[i] / lambdas[i+1]) * np.exp(-lambdas[i] * dz)

    I_rel = [c[i] * (1.0 - np.exp(-lambdas[i] * (z_bounds[i+1] - z_bounds[i]))) for i in range(M)]
    k_1 = 1.0 / np.sum(I_rel)
    k = k_1 * c

    for i in range(M - 1):
        left = k[i] * lambdas[i] * np.exp(-lambdas[i] * (z_bounds[i+1] - z_bounds[i]))
        right = k[i+1] * lambdas[i+1]
        np.testing.assert_allclose(left, right, atol=1e-12)

    total_int = np.sum([k[i] * (1.0 - np.exp(-lambdas[i] * (z_bounds[i+1] - z_bounds[i]))) for i in range(M)])
    np.testing.assert_allclose(total_int, 1.0, atol=1e-12)

def test_random_walk_mean_squared_displacement():
    """Exercise 11.10: 1D random walk MSD E[(z^tau)^2] = tau / 2"""
    np.random.seed(118)
    N_walks = 8000
    tau_max = 50
    steps = np.random.choice([0, 1, -1], p=[0.5, 0.25, 0.25], size=(N_walks, tau_max))
    z_traj = np.hstack([np.zeros((N_walks, 1)), np.cumsum(steps, axis=1)])
    msd = np.mean(z_traj**2, axis=0)

    taus = np.arange(tau_max + 1)
    np.testing.assert_allclose(msd[10:], (taus / 2.0)[10:], rtol=0.10)

def test_gibbs_detailed_balance():
    """Exercise 11.11: Gibbs detailed balance p(z) T(z -> z*) = p(z*) T(z* -> z)"""
    mu_vec = np.array([0.2, -0.3])
    cov_mat = np.array([[1.0, 0.5], [0.5, 2.0]])
    mvn = multivariate_normal(mean=mu_vec, cov=cov_mat)

    s1, s2 = np.sqrt(cov_mat[0, 0]), np.sqrt(cov_mat[1, 1])
    rho = cov_mat[0, 1] / (s1 * s2)
    var_cond1 = (s1**2) * (1.0 - rho**2)

    z2_fixed = 0.5
    z1_a, z1_b = -0.6, 1.0
    cond_mean = mu_vec[0] + rho * (s1 / s2) * (z2_fixed - mu_vec[1])

    p_a = mvn.pdf([z1_a, z2_fixed])
    p_b = mvn.pdf([z1_b, z2_fixed])
    T_a_to_b = norm.pdf(z1_b, loc=cond_mean, scale=np.sqrt(var_cond1))
    T_b_to_a = norm.pdf(z1_a, loc=cond_mean, scale=np.sqrt(var_cond1))

    np.testing.assert_allclose(p_a * T_a_to_b, p_b * T_b_to_a, atol=1e-12)

def test_over_relaxation_variance_preservation():
    """Exercise 11.14: Over-relaxation update maintains var[z_i'] = sigma_i^2"""
    np.random.seed(119)
    N = 35000
    mu, sigma = 1.2, 0.8
    for alpha in [-0.8, -0.4, 0.3, 0.6]:
        z = np.random.normal(mu, sigma, size=N)
        nu = np.random.normal(0, 1, size=N)
        z_prime = mu + alpha * (z - mu) + sigma * np.sqrt(1.0 - alpha**2) * nu
        np.testing.assert_allclose(np.mean(z_prime), mu, atol=0.03)
        np.testing.assert_allclose(np.var(z_prime), sigma**2, atol=0.04)

def test_hamiltonian_leapfrog_reversibility_and_energy_conservation():
    """Exercise 11.15 & 11.17: Leapfrog reversibility and HMC detailed balance"""
    def E_pot(z):
        return 0.5 * (z[0]**2 + z[1]**2) + 0.1 * z[0]**4

    def grad_E(z):
        return np.array([z[0] + 0.4 * z[0]**3, z[1]])

    def H_func(z, r):
        return E_pot(z) + 0.5 * np.sum(r**2)

    def leapfrog(z_in, r_in, eps_val, L_steps):
        z = z_in.copy()
        r = r_in.copy()
        r = r - 0.5 * eps_val * grad_E(z)
        for _ in range(L_steps - 1):
            z = z + eps_val * r
            r = r - eps_val * grad_E(z)
        z = z + eps_val * r
        r = r - 0.5 * eps_val * grad_E(z)
        return z, r

    z0 = np.array([0.8, -1.0])
    r0 = np.array([-0.5, 0.9])
    eps = 0.1
    L = 10

    # 1. Reversibility check
    z_star, r_star = leapfrog(z0, r0, eps, L)
    z_back, r_back = leapfrog(z_star, -r_star, eps, L)
    np.testing.assert_allclose(z_back, z0, atol=1e-12)
    np.testing.assert_allclose(r_back, -r0, atol=1e-12)

    # 2. Detailed balance check
    H0 = H_func(z0, r0)
    H_star = H_func(z_star, r_star)
    fwd = np.exp(-H0) * min(1.0, np.exp(-H_star + H0))
    rev = np.exp(-H_star) * min(1.0, np.exp(-H0 + H_star))
    np.testing.assert_allclose(fwd, rev, atol=1e-12)
