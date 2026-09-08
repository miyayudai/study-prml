"""
tests/test_ch14_exercises.py
Unit tests verifying core theoretical derivations and numerical algorithms for
PRML Chapter 14 (Combining Models) Exercises (14.1 to 14.17).
"""

import os
import nbformat
import numpy as np
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from scipy.optimize import minimize_scalar

NOTEBOOK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../14/14_Exercises.ipynb"))


def test_ch14_exercises_notebook():
    """Verify that Chapter 14 notebook exists, has all cells, and executes cleanly."""
    assert os.path.exists(NOTEBOOK_PATH), f"Notebook not found at {NOTEBOOK_PATH}"

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    assert len(nb.cells) >= 35, f"Expected at least 35 cells, found {len(nb.cells)}"

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(NOTEBOOK_PATH)}})
    except Exception as e:
        pytest.fail(f"Notebook execution failed: {e}")


# =========================================================================
# 個別演習問題ユニットテスト (Exercises 14.1 - 14.17 アルゴリズム検証)
# =========================================================================

def test_ex14_1_committee_error_decomposition():
    """Exercise 14.1: Committee squared error decomposition and 1/M reduction for uncorrelated errors"""
    N = 3000
    M = 4
    np.random.seed(141)
    errors = np.random.randn(N, M)

    E_AV = np.mean(np.mean(errors**2, axis=0))
    E_COM = np.mean(np.mean(errors, axis=1)**2)

    # 2次モーメント積による厳密分解公式
    second_moments = (errors.T @ errors) / N
    diag_part = np.trace(second_moments) / (M**2)
    offdiag_part = (np.sum(second_moments) - np.trace(second_moments)) / (M**2)
    decomp_val = diag_part + offdiag_part

    np.testing.assert_allclose(E_COM, decomp_val, atol=1e-12)
    np.testing.assert_allclose(E_COM / E_AV, 1.0 / M, atol=0.03)


def test_ex14_2_jensen_inequality_committee_upper_bound():
    """Exercise 14.2: Jensen's inequality guarantees E_COM <= E_AV strictly for diverse models"""
    N = 1000
    M = 5
    errors = np.random.uniform(-1, 1, size=(N, M))
    E_AV = np.mean(errors**2)
    E_COM = np.mean(np.mean(errors, axis=1)**2)

    assert E_COM < E_AV

    # 全モデル同一の場合は等号成立
    ident_errors = np.tile(np.random.randn(N, 1), (1, M))
    np.testing.assert_allclose(np.mean(ident_errors**2), np.mean(np.mean(ident_errors, axis=1)**2), atol=1e-12)


def test_ex14_3_convex_loss_ensemble_superiority():
    """Exercise 14.3: Ensemble error is <= average model error for general convex loss functions"""
    N = 500
    M = 3
    t_true = np.random.randn(N)
    y_models = np.random.randn(N, M) + t_true[:, None]
    y_com = np.mean(y_models, axis=1)

    # L1 損失 (絶対値損失)
    loss_l1_com = np.mean(np.abs(y_com - t_true))
    loss_l1_av = np.mean([np.mean(np.abs(y_models[:, m] - t_true)) for m in range(M)])
    assert loss_l1_com <= loss_l1_av


def test_ex14_4_adaboost_optimal_alpha():
    """Exercise 14.4: Analytical alpha_m = 0.5 * ln((1 - eps) / eps) minimizes exponential loss"""
    for eps in [0.15, 0.3, 0.45]:
        alpha_analytical = 0.5 * np.log((1.0 - eps) / eps)

        def exp_loss(alpha):
            return (1.0 - eps) * np.exp(-alpha) + eps * np.exp(alpha)

        res = minimize_scalar(exp_loss, bounds=(-5, 5), method='bounded', options={'xatol': 1e-10})
        np.testing.assert_allclose(alpha_analytical, res.x, atol=1e-5)


def test_ex14_5_adaboost_weight_reduction_factor():
    """Exercise 14.5: Total weight is scaled by factor 2 * sqrt(eps * (1 - eps))"""
    N = 100
    w = np.random.uniform(0.5, 1.5, size=N)
    t = np.random.choice([-1, 1], size=N)
    y = t.copy()
    y[:30] *= -1  # 誤分類率約0.3

    W_m = np.sum(w)
    eps = np.sum(w[t != y]) / W_m
    alpha = 0.5 * np.log((1.0 - eps) / eps)

    w_next = w * np.exp(-alpha * t * y)
    factor_analytical = 2.0 * np.sqrt(eps * (1.0 - eps))

    np.testing.assert_allclose(np.sum(w_next), factor_analytical * W_m, atol=1e-12)


def test_ex14_6_adaboost_exponential_convergence_bound():
    """Exercise 14.6: Training error bound product_m 2*sqrt(eps*(1-eps)) <= exp(-2*gamma^2*M)"""
    gamma = 0.12
    M = 50
    eps = 0.5 - gamma
    factor = 2.0 * np.sqrt(eps * (1.0 - eps))
    product_bound = factor**M
    exp_bound = np.exp(-2.0 * (gamma**2) * M)

    assert factor < 1.0
    assert product_bound <= exp_bound + 1e-12
    assert product_bound < 0.3  # M=50 で十分に指数収束


def test_ex14_7_exponential_loss_population_log_odds():
    """Exercise 14.7: Population minimizer y*(x) = 0.5 * ln(p(1|x) / p(-1|x))"""
    for p1 in [0.2, 0.5, 0.8]:
        p_minus1 = 1.0 - p1
        y_opt = 0.5 * np.log(p1 / p_minus1)

        def conditional_loss(y):
            return np.exp(-y) * p1 + np.exp(y) * p_minus1

        res = minimize_scalar(conditional_loss, bounds=(-5, 5), method='bounded', options={'xatol': 1e-10})
        np.testing.assert_allclose(y_opt, res.x, atol=1e-5)


def test_ex14_9_10_decision_tree_impurity_concavity():
    """Exercises 14.9, 14.10: Gini impurity and entropy are strictly concave with maximum at p=0.5"""
    p_grid = np.linspace(0.01, 0.99, 100)
    gini = 2.0 * p_grid * (1.0 - p_grid)
    entropy = -p_grid * np.log(p_grid) - (1.0 - p_grid) * np.log(1.0 - p_grid)

    # 2階微分（有限差分）が全域で負（凹関数）
    d2_gini = np.diff(gini, n=2)
    d2_entropy = np.diff(entropy, n=2)
    assert np.all(d2_gini < 0)
    assert np.all(d2_entropy < 0)

    # 最大値が p=0.5
    assert np.isclose(gini[49], 0.5, atol=0.01)
    assert np.isclose(entropy[49], np.log(2.0), atol=0.01)


def test_ex14_11_cost_complexity_monotonicity():
    """Exercise 14.11: Optimal tree size |T| is monotonically non-increasing in penalty alpha"""
    trees = [
        {"R": 5.0, "size": 8},
        {"R": 10.0, "size": 4},
        {"R": 18.0, "size": 2},
        {"R": 30.0, "size": 1},
    ]
    alphas = [0.0, 1.0, 3.0, 6.0, 15.0]
    sizes = [trees[np.argmin([t["R"] + a * t["size"] for t in trees])]["size"] for a in alphas]
    assert np.all(np.diff(sizes) <= 0)


def test_ex14_14_mixture_of_experts_wls_update():
    """Exercise 14.14: Expert regression weights satisfy Weighted Least Squares equations"""
    N, D = 50, 2
    np.random.seed(1414)
    X = np.random.randn(N, D)
    gamma = np.random.uniform(0.1, 1.0, size=N)
    t = X @ np.array([2.0, -1.0]) + np.random.randn(N) * 0.1

    R = np.diag(gamma)
    w_opt = np.linalg.solve(X.T @ R @ X, X.T @ R @ t)

    # WLS 勾配ゼロの検証
    grad = X.T @ R @ (t - X @ w_opt)
    np.testing.assert_allclose(grad, np.zeros(D), atol=1e-12)


def test_ex14_15_gating_softmax_gradient():
    """Exercise 14.15: Analytical gating gradient grad = sum_n (gamma_{nk} - pi_{nk}) x_n"""
    N, D, K = 30, 2, 2
    np.random.seed(1415)
    X = np.random.randn(N, D)
    eta = np.random.randn(K, D)
    gamma = np.random.dirichlet(np.ones(K), size=N)

    logits = X @ eta.T
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    pi = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    # 解析的勾配
    grad_ana = np.zeros((K, D))
    for k in range(K):
        grad_ana[k] = np.sum((gamma[:, k] - pi[:, k])[:, None] * X, axis=0)

    # 有限差分法による数値勾配
    eps_fd = 1e-6
    grad_num = np.zeros((K, D))
    for k in range(K):
        for d in range(D):
            eta_plus = eta.copy()
            eta_plus[k, d] += eps_fd
            l_plus = X @ eta_plus.T
            pi_plus = np.exp(l_plus - np.max(l_plus, axis=1, keepdims=True))
            pi_plus /= np.sum(pi_plus, axis=1, keepdims=True)

            val_base = np.sum(gamma * np.log(pi + 1e-15))
            val_plus = np.sum(gamma * np.log(pi_plus + 1e-15))
            grad_num[k, d] = (val_plus - val_base) / eps_fd

    np.testing.assert_allclose(grad_ana, grad_num, atol=1e-5)


def test_ex14_16_conditional_mean_failure_on_multimodal():
    """Exercise 14.16: Conditional expectation E[t|x] fails catastrophically on bimodal distributions"""
    x = 3.0
    mode1 = np.random.normal(loc=x, scale=0.1, size=400)
    mode2 = np.random.normal(loc=-x, scale=0.1, size=400)
    t = np.concatenate([mode1, mode2])

    cond_mean = np.mean(t)
    # 平均値周辺 (|t| < 1) にはデータが存在しない
    count_near_mean = np.sum(np.abs(t - cond_mean) < 1.0)
    assert count_near_mean == 0
    assert np.isclose(cond_mean, 0.0, atol=0.1)


def test_ex14_17_hme_marginal_consistency():
    """Exercise 14.17: Hierarchical Mixture of Experts marginal responsibility consistency"""
    J, K = 2, 3
    pi_top = np.array([0.3, 0.7])
    pi_sub = np.array([[0.4, 0.4, 0.2],
                       [0.1, 0.6, 0.3]])
    p_leaf = np.array([[1.0, 0.5, 0.2],
                       [0.8, 1.5, 0.4]])

    pi_joint = pi_top[:, None] * pi_sub
    np.testing.assert_allclose(np.sum(pi_joint), 1.0, atol=1e-12)

    joint_unnorm = pi_joint * p_leaf
    gamma_joint = joint_unnorm / np.sum(joint_unnorm)
    np.testing.assert_allclose(np.sum(gamma_joint), 1.0, atol=1e-12)

    # 上位ノードの周辺負担率
    gamma_top = np.sum(gamma_joint, axis=1)
    assert len(gamma_top) == J
    assert np.isclose(np.sum(gamma_top), 1.0, atol=1e-12)
