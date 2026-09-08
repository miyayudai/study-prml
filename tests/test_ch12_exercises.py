# tests/test_ch12_exercises.py
"""
Unit tests for PRML Chapter 12 (Continuous Latent Variables) Exercises.
Covers theoretical derivations and numerical algorithms for:
- PCA variance maximization & induction (Ex 12.1)
- Distortion measure and residual eigenvalues (Ex 12.2)
- High-dimensional snapshot method eigenvector normalization (Ex 12.3)
- Latent prior affine reparameterization (Ex 12.4)
- Gaussian affine transformation and rank (Ex 12.5)
- PPCA conditional independence / Naive Bayes structure (Ex 12.6)
- Marginal distribution p(x) (Ex 12.7)
- Posterior distribution p(z | x) (Ex 12.8)
- Sample mean MLE gradient & negative definite Hessian (Ex 12.9, 12.10)
- Noise-free limit sigma^2 -> 0 orthogonal projection (Ex 12.11)
- Shrinkage towards origin for sigma^2 > 0 (Ex 12.12)
- PPCA parameter counting (Ex 12.14)
- PPCA EM M-step equations (Ex 12.15)
- Alternating least squares subspace convergence (Ex 12.17)
- Factor analysis parameter counting & rotation invariance (Ex 12.18, 12.19)
- Factor analysis EM E-step (Ex 12.21)
- Multivariate Student-t EM outlier downweighting (Ex 12.24)
- FA scaling covariance and PPCA rotation covariance (Ex 12.25)
- Kernel PCA null-space invariance (Ex 12.26)
- Linear Kernel PCA equivalence to standard PCA (Ex 12.27)
- Nonlinear density transformation ODE (Ex 12.28)
- Zero correlation with deterministic dependence (Ex 12.29)
"""

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.stats import norm, expon
from sklearn.decomposition import PCA, KernelPCA

def test_ex12_1_pca_induction_orthogonal_variance():
    """Exercise 12.1: Sequential orthogonal variance maximization yields S eigenvectors"""
    np.random.seed(121)
    D = 4
    N = 800
    X = np.random.randn(N, D) @ np.diag([4.0, 2.5, 1.5, 0.8])
    S = np.cov(X, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]

    found_u = []
    for m in range(2):
        def objective(u):
            return -u.T @ S @ u

        constraints = [{'type': 'eq', 'fun': lambda u: u.T @ u - 1.0}]
        for prev_u in found_u:
            constraints.append({'type': 'eq', 'fun': lambda u, pu=prev_u: u.T @ pu})

        res = minimize(objective, np.random.randn(D), constraints=constraints, method='SLSQP')
        assert res.success
        u_opt = res.x
        var_opt = -res.fun
        found_u.append(u_opt)
        np.testing.assert_allclose(var_opt, eigvals[m], rtol=1e-3)

def test_ex12_2_distortion_equals_discarded_eigenvalues():
    """Exercise 12.2: Minimal distortion J strictly equals sum of discarded eigenvalues"""
    np.random.seed(122)
    D = 5
    M = 2
    N = 300
    X = np.random.randn(N, D)
    S = np.cov(X, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)
    U_discarded = eigvecs[:, idx[:D-M]]
    discarded_sum = np.sum(eigvals[idx[:D-M]])

    J = np.trace(U_discarded.T @ S @ U_discarded)
    np.testing.assert_allclose(J, discarded_sum, atol=1e-10)

def test_ex12_3_snapshot_eigenvector_normalization():
    """Exercise 12.3: Snapshot formula u_i = (1/sqrt(N lambda)) X^T v has unit norm"""
    np.random.seed(123)
    N = 15
    D = 80
    X = np.random.randn(N, D)
    X = X - np.mean(X, axis=0)

    K_gram = (1.0 / N) * (X @ X.T)
    eigvals_K, eigvecs_K = np.linalg.eigh(K_gram)
    pos_idx = np.where(eigvals_K > 1e-10)[0]

    for idx_i in pos_idx:
        lam = eigvals_K[idx_i]
        v = eigvecs_K[:, idx_i]
        u = (1.0 / np.sqrt(N * lam)) * (X.T @ v)
        np.testing.assert_allclose(u.T @ u, 1.0, atol=1e-12)

def test_ex12_4_latent_prior_reparameterization():
    """Exercise 12.4: General latent prior N(m, Sigma) gives identical marginal distribution"""
    np.random.seed(124)
    D = 4
    M = 2
    W = np.random.randn(D, M)
    mu = np.array([1.0, -0.5, 2.0, 0.0])
    sigma_sq = 0.3
    m = np.array([0.8, -1.2])
    A = np.random.randn(M, M)
    Sigma = A @ A.T + 0.2 * np.eye(M)
    L = np.linalg.cholesky(Sigma)

    mean_orig = mu + W @ m
    Cov_orig = W @ Sigma @ W.T + sigma_sq * np.eye(D)
    W_tilde = W @ L
    Cov_reparam = W_tilde @ W_tilde.T + sigma_sq * np.eye(D)

    np.testing.assert_allclose(mean_orig, mu + W @ m, atol=1e-12)
    np.testing.assert_allclose(Cov_orig, Cov_reparam, atol=1e-12)

def test_ex12_5_affine_transform_rank():
    """Exercise 12.5: Gaussian affine transform rank bounded by min(M, D)"""
    np.random.seed(125)
    D = 3
    Sigma = np.diag([2.0, 1.0, 0.5])
    M_high = 6
    A_high = np.random.randn(M_high, D)
    Cov_high = A_high @ Sigma @ A_high.T
    assert np.linalg.matrix_rank(Cov_high) <= D

def test_ex12_6_ppca_conditional_independence():
    """Exercise 12.6: Components x_d conditionally independent given z"""
    np.random.seed(126)
    D = 4
    M = 2
    W = np.random.randn(D, M)
    sigma_sq = 0.25
    C_marginal = W @ W.T + sigma_sq * np.eye(D)
    off_diag = np.max(np.abs(C_marginal - np.diag(np.diag(C_marginal))))
    assert off_diag > 0.1  # Marginally correlated

    # Fixed z Monte Carlo
    z_fixed = np.array([0.5, -0.8])
    N_mc = 25000
    eps = np.random.normal(0, np.sqrt(sigma_sq), size=(N_mc, D))
    x_samples = z_fixed @ W.T + eps
    cov_cond = np.cov(x_samples, rowvar=False)
    np.testing.assert_allclose(cov_cond, sigma_sq * np.eye(D), atol=0.02)

def test_ex12_7_marginal_distribution_formula():
    """Exercise 12.7: Marginals match mu and C = W W^T + sigma^2 I"""
    np.random.seed(127)
    D = 3
    M = 2
    W = np.random.randn(D, M)
    mu = np.array([1.0, -2.0, 0.5])
    sigma_sq = 0.4
    C_true = W @ W.T + sigma_sq * np.eye(D)

    N_mc = 40000
    z = np.random.randn(N_mc, M)
    eps = np.random.normal(0, np.sqrt(sigma_sq), size=(N_mc, D))
    x = mu + z @ W.T + eps

    np.testing.assert_allclose(np.mean(x, axis=0), mu, atol=0.03)
    np.testing.assert_allclose(np.cov(x, rowvar=False), C_true, atol=0.04)

def test_ex12_8_posterior_distribution_formula():
    """Exercise 12.8: Closed-form posterior p(z | x) matches joint conditioning"""
    np.random.seed(128)
    D = 3
    M = 2
    W = np.random.randn(D, M)
    mu = np.array([0.5, 1.0, -0.5])
    sigma_sq = 0.3
    M_mat = W.T @ W + sigma_sq * np.eye(M)
    M_inv = np.linalg.inv(M_mat)

    x_test = np.array([1.5, -0.2, 0.8])
    post_mean_theory = M_inv @ W.T @ (x_test - mu)
    post_cov_theory = sigma_sq * M_inv

    C_mat = W @ W.T + sigma_sq * np.eye(D)
    C_inv = np.linalg.inv(C_mat)
    post_mean_joint = W.T @ C_inv @ (x_test - mu)
    post_cov_joint = np.eye(M) - W.T @ C_inv @ W

    np.testing.assert_allclose(post_mean_theory, post_mean_joint, atol=1e-12)
    np.testing.assert_allclose(post_cov_theory, post_cov_joint, atol=1e-12)

def test_ex12_9_and_10_mle_gradient_and_negative_definite_hessian():
    """Exercise 12.9 & 12.10: Log-likelihood gradient is zero at x_bar and Hessian is negative definite"""
    np.random.seed(129)
    N = 100
    D = 4
    M = 2
    X = np.random.randn(N, D)
    x_bar = np.mean(X, axis=0)
    W = np.random.randn(D, M)
    sigma_sq = 0.5
    C = W @ W.T + sigma_sq * np.eye(D)
    C_inv = np.linalg.inv(C)

    # Gradient at x_bar
    grad_mu = C_inv @ np.sum(X - x_bar, axis=0)
    np.testing.assert_allclose(grad_mu, np.zeros(D), atol=1e-12)

    # Hessian negative definiteness
    Hessian = -N * C_inv
    eigvals_H = np.linalg.eigvalsh(Hessian)
    assert np.all(eigvals_H < -1e-6)

def test_ex12_11_noise_free_limit_orthogonal_projection():
    """Exercise 12.11: As sigma^2 -> 0, PPCA reconstruction converges to orthogonal projection"""
    np.random.seed(1211)
    D = 4
    M = 2
    W = np.random.randn(D, M)
    mu = np.array([0.5, -0.5, 1.0, 2.0])
    x = np.array([2.0, 1.0, -1.5, 0.0])

    P_ortho = W @ np.linalg.inv(W.T @ W) @ W.T
    x_ortho = mu + P_ortho @ (x - mu)

    # Small sigma^2
    sigma_sq = 1e-7
    M_mat = W.T @ W + sigma_sq * np.eye(M)
    z_post = np.linalg.inv(M_mat) @ W.T @ (x - mu)
    x_ppca = mu + W @ z_post

    np.testing.assert_allclose(x_ppca, x_ortho, atol=1e-5)

def test_ex12_12_shrinkage_factor():
    """Exercise 12.12: Shrinkage factor strictly s_j^2 / (s_j^2 + sigma^2) < 1"""
    np.random.seed(1212)
    D = 4
    M = 3
    W = np.random.randn(D, M)
    sigma_sq = 0.6
    U, S_sing, Vt = np.linalg.svd(W, full_matrices=False)
    factors = S_sing**2 / (S_sing**2 + sigma_sq)
    assert np.all(factors < 1.0)
    assert np.all(factors > 0.0)

def test_ex12_14_parameter_counting():
    """Exercise 12.14: PPCA parameter counting matches isotropic (M=0) and full covariance (M=D-1)"""
    def count_ppca(D, M):
        return D * M + 1 - (M * (M - 1)) // 2

    for D in [3, 5, 8]:
        assert count_ppca(D, 0) == 1
        assert count_ppca(D, D - 1) == (D * (D + 1)) // 2

def test_ex12_15_ppca_em_m_step():
    """Exercise 12.15: M-step updates W_new, sigma^2_new zero the complete Q gradient"""
    np.random.seed(1215)
    N = 60
    D = 3
    M = 2
    X = np.random.randn(N, D)
    mu = np.mean(X, axis=0)
    X_cent = X - mu

    Ez = np.random.randn(N, M)
    Cov_z = [0.1 * np.eye(M) for _ in range(N)]
    Ezz = np.zeros((M, M))
    for n in range(N):
        Ezz += np.outer(Ez[n], Ez[n]) + Cov_z[n]

    W_new = (X_cent.T @ Ez) @ np.linalg.inv(Ezz)
    grad_W = np.zeros((D, M))
    for n in range(N):
        grad_W += np.outer(X_cent[n], Ez[n]) - W_new @ (np.outer(Ez[n], Ez[n]) + Cov_z[n])
    np.testing.assert_allclose(grad_W, np.zeros((D, M)), atol=1e-12)

def test_ex12_17_alternating_least_squares_pca():
    """Exercise 12.17: Alternating least squares on J converges to PCA principal subspace"""
    np.random.seed(1217)
    N = 250
    D = 4
    M = 2
    X = np.random.randn(N, D) @ np.diag([3.0, 2.0, 0.8, 0.3])
    X_cent = X - np.mean(X, axis=0)

    # True principal subspace via SVD
    _, _, Vt_true = np.linalg.svd(X_cent, full_matrices=False)
    P_true = Vt_true[:M].T @ Vt_true[:M]

    W = np.random.randn(D, M)
    for _ in range(25):
        Z = X_cent @ W @ np.linalg.inv(W.T @ W)
        W = (X_cent.T @ Z) @ np.linalg.inv(Z.T @ Z)

    P_W = W @ np.linalg.inv(W.T @ W) @ W.T
    np.testing.assert_allclose(P_W, P_true, atol=1e-5)

def test_ex12_18_and_19_factor_analysis_properties():
    """Exercise 12.18 & 12.19: FA parameter counting and latent rotation invariance"""
    D, M = 5, 2
    n_params = D * (M + 1) - (M * (M - 1)) // 2
    assert n_params == 14
    assert n_params <= (D * (D + 1)) // 2

    # Rotation invariance
    W = np.random.randn(D, M)
    Psi = np.diag([0.2, 0.4, 0.3, 0.5, 0.1])
    C_orig = W @ W.T + Psi

    theta = 1.1
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    W_rot = W @ R.T
    C_rot = W_rot @ W_rot.T + Psi
    np.testing.assert_allclose(C_rot, C_orig, atol=1e-14)

def test_ex12_21_factor_analysis_e_step():
    """Exercise 12.21: FA E-step formula matches joint Gaussian conditioning"""
    np.random.seed(1221)
    D = 4
    M = 2
    W = np.random.randn(D, M)
    Psi = np.diag([0.3, 0.4, 0.2, 0.5])
    Psi_inv = np.diag(1.0 / np.diag(Psi))
    x_cent = np.array([0.8, -1.2, 0.5, 2.0])

    G = np.linalg.inv(np.eye(M) + W.T @ Psi_inv @ W)
    Ez_formula = G @ W.T @ Psi_inv @ x_cent

    C = W @ W.T + Psi
    Ez_joint = W.T @ np.linalg.inv(C) @ x_cent
    np.testing.assert_allclose(Ez_formula, Ez_joint, atol=1e-12)

def test_ex12_24_student_t_em_outlier_robustness():
    """Exercise 12.24: Student-t EM downweights outliers and yields robust mean"""
    np.random.seed(1224)
    N_clean = 80
    D = 2
    X_clean = np.random.randn(N_clean, D)
    X_outliers = np.random.uniform(20, 30, size=(10, D))
    X = np.vstack([X_clean, X_outliers])
    N = len(X)

    mu = np.median(X, axis=0)
    Sigma = np.cov(X, rowvar=False)
    nu = 4.0

    for _ in range(20):
        Sigma_inv = np.linalg.inv(Sigma)
        diff = X - mu
        delta_sq = np.sum(diff @ Sigma_inv * diff, axis=1)
        u = (nu + D) / (nu + delta_sq)
        mu = np.sum(u[:, None] * X, axis=0) / np.sum(u)
        diff_new = X - mu
        Sigma = (diff_new.T @ (u[:, None] * diff_new)) / N

    # Outliers should have much smaller weights than clean points
    assert np.mean(u[N_clean:]) < 0.1 * np.mean(u[:N_clean])
    np.testing.assert_allclose(mu, np.zeros(D), atol=0.25)

def test_ex12_25_affine_covariance_properties():
    """Exercise 12.25: FA covariant under diagonal scaling; PPCA covariant under rotation"""
    np.random.seed(1225)
    D = 3
    Psi = np.diag([0.3, 0.5, 0.7])
    A_diag = np.diag([2.0, -1.5, 0.8])
    transformed_Psi = A_diag @ Psi @ A_diag.T
    np.testing.assert_allclose(transformed_Psi, np.diag(np.diag(transformed_Psi)))

    sigma_sq = 0.5
    Q, _ = np.linalg.qr(np.random.randn(D, D))
    transformed_isotropic = Q @ (sigma_sq * np.eye(D)) @ Q.T
    np.testing.assert_allclose(transformed_isotropic, sigma_sq * np.eye(D), atol=1e-12)

def test_ex12_26_kernel_pca_null_space_invariance():
    """Exercise 12.26: Null space eigenvector addition preserves K a and K^2 a"""
    np.random.seed(1226)
    N = 25
    D = 2
    X = np.random.randn(N, D)
    K = X @ X.T  # rank <= 2

    eigvals, eigvecs = np.linalg.eigh(K)
    lam_max = eigvals[-1]
    a_max = eigvecs[:, -1]
    v_null = eigvecs[:, 0]  # zero eigenvalue

    c = 4.2
    a_prime = a_max + c * v_null
    np.testing.assert_allclose(K @ a_prime, K @ a_max, atol=1e-10)
    np.testing.assert_allclose(K @ K @ a_prime, lam_max * (K @ a_prime), atol=1e-8)

def test_ex12_27_linear_kernel_pca_equals_standard_pca():
    """Exercise 12.27: Linear Kernel PCA recovers standard linear PCA projections"""
    np.random.seed(1227)
    N = 80
    D = 4
    X = np.random.randn(N, D)
    X = X - np.mean(X, axis=0)

    pca = PCA(n_components=2)
    proj_std = pca.fit_transform(X)

    kpca = KernelPCA(n_components=2, kernel='linear')
    proj_kpca = kpca.fit_transform(X)

    for m in range(2):
        r = np.abs(np.corrcoef(proj_std[:, m], proj_kpca[:, m])[0, 1])
        np.testing.assert_allclose(r, 1.0, atol=1e-7)

def test_ex12_28_density_transform_ode():
    """Exercise 12.28: ODE f'(x) = q(x)/p(f(x)) matches inverse CDF composition"""
    def odefun(x, y):
        return [norm.pdf(x) / max(expon.pdf(y[0]), 1e-12)]

    x_eval = np.linspace(0.0, 1.5, 30)
    sol = solve_ivp(odefun, [0.0, 1.5], [np.log(2.0)], t_eval=x_eval, rtol=1e-7, atol=1e-7)
    y_ode = sol.y[0]
    y_theory = -np.log(1.0 - norm.cdf(x_eval))
    np.testing.assert_allclose(y_ode, y_theory, atol=1e-4)

def test_ex12_29_uncorrelated_but_dependent():
    """Exercise 12.29: y2 = y1^2 has zero correlation with y1 yet deterministic dependence"""
    np.random.seed(1229)
    N = 80000
    y1 = np.random.uniform(-1, 1, size=N)
    y2 = y1**2
    cov_12 = np.cov(y1, y2)[0, 1]
    np.testing.assert_allclose(cov_12, 0.0, atol=0.01)
    assert np.var(y2) > 0.05
