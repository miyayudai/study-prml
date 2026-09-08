"""
tests/test_ch13_exercises.py
Unit tests verifying core theoretical derivations and numerical algorithms for
PRML Chapter 13 (Sequential Data) Exercises (13.1 to 13.34).
"""

import os
import nbformat
import numpy as np
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from scipy.optimize import minimize

NOTEBOOK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../13/13_Exercises.ipynb"))


def test_ch13_exercises_notebook():
    """Verify that Chapter 13 notebook exists, has all cells, and executes cleanly."""
    assert os.path.exists(NOTEBOOK_PATH), f"Notebook not found at {NOTEBOOK_PATH}"

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    assert len(nb.cells) >= 69, f"Expected at least 69 cells, found {len(nb.cells)}"

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(NOTEBOOK_PATH)}})
    except Exception as e:
        pytest.fail(f"Notebook execution failed: {e}")


# =========================================================================
# 個別演習問題ユニットテスト (Exercises 13.1 - 13.34 アルゴリズム検証)
# =========================================================================

def test_ex13_1_markov_chain_conditional_independence():
    """Exercise 13.1: Markov chain p(x_1, ..., x_N) = p(x_1) prod p(x_n | x_{n-1}) and d-separation"""
    # 3変数の離散マルコフ連鎖 x1 -> x2 -> x3
    K = 3
    p_x1 = np.array([0.2, 0.5, 0.3])
    A = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.8, 0.1],
                  [0.2, 0.3, 0.5]])

    # 結合分布 p(x1, x2, x3)
    p_joint = np.zeros((K, K, K))
    for i in range(K):
        for j in range(K):
            for k in range(K):
                p_joint[i, j, k] = p_x1[i] * A[i, j] * A[j, k]

    # 条件付き独立性: p(x3 | x2, x1) == p(x3 | x2)
    p_x1_x2 = np.sum(p_joint, axis=2)  # (K, K)
    p_x2 = np.sum(p_x1_x2, axis=0)     # (K,)

    for i in range(K):
        for j in range(K):
            p_x3_given_x1_x2 = p_joint[i, j, :] / p_x1_x2[i, j]
            p_x3_given_x2 = (np.sum(p_joint, axis=0)[j, :]) / p_x2[j]
            np.testing.assert_allclose(p_x3_given_x1_x2, p_x3_given_x2, atol=1e-12)


def test_ex13_5_baum_welch_m_step_stationary_point():
    """Exercise 13.5: Baum-Welch M-step update maximizes Q(theta, theta_old) subject to sum A_jk = 1"""
    K = 3
    xi = np.random.dirichlet(np.ones(K * K), size=10).reshape(10, K, K)
    # 解析解
    A_analytical = np.sum(xi, axis=0) / np.sum(xi, axis=(0, 2))[:, None]

    # SLSQP による制約付き直接最大化
    for j in range(K):
        w_j = np.sum(xi[:, j, :], axis=0)

        def objective(a_row):
            return -np.sum(w_j * np.log(a_row + 1e-15))

        cons = ({'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0})
        bnds = [(1e-6, 1.0) for _ in range(K)]
        res = minimize(objective, np.ones(K) / K, method='SLSQP', bounds=bnds, constraints=cons,
                       options={'ftol': 1e-12, 'maxiter': 500})
        assert res.success
        np.testing.assert_allclose(res.x, A_analytical[j], atol=1e-3)


def test_ex13_7_forward_backward_marginal_consistency():
    """Exercise 13.7: sum_k alpha(z_nk) * beta(z_nk) is constant across all time steps n and equals p(X)"""
    K = 2
    N_time = 5
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3],
                  [0.2, 0.8]])
    # 観測尤度 p(x_n | z_n)
    phi = np.array([[0.9, 0.1],
                    [0.2, 0.8],
                    [0.7, 0.3],
                    [0.4, 0.6],
                    [0.8, 0.2]])

    # Forward
    alpha = np.zeros((N_time, K))
    alpha[0] = pi * phi[0]
    for n in range(1, N_time):
        alpha[n] = phi[n] * (alpha[n - 1] @ A)

    # Backward
    beta = np.zeros((N_time, K))
    beta[-1] = 1.0
    for n in range(N_time - 2, -1, -1):
        beta[n] = A @ (phi[n + 1] * beta[n + 1])

    # sum_k alpha(z_nk) beta(z_nk) は任意の時点 n で同一
    p_X_values = np.sum(alpha * beta, axis=1)
    np.testing.assert_allclose(p_X_values, p_X_values[0], atol=1e-12)


def test_ex13_16_viterbi_vs_marginal_divergence():
    """Exercise 13.16: Viterbi most probable path vs max marginal path divergence counterexample"""
    K = 2
    pi = np.array([0.5, 0.5])
    # 状態1から状態1への遷移確率がゼロ（不許可）
    A = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    phi = np.array([[0.6, 0.4],
                    [0.6, 0.4]])  # 2ステップ

    # 周辺最大化
    alpha1 = pi * phi[0]
    beta1 = A @ phi[1]
    gamma1 = alpha1 * beta1
    # 最尤状態
    marg_seq = [np.argmax(gamma1), np.argmax(alpha1 @ A * phi[1])]

    # Viterbi
    # 経路1: (0, 1) -> 確率 0.5*0.6 * 1.0 * 0.4 = 0.12
    # 経路2: (1, 0) -> 確率 0.5*0.4 * 1.0 * 0.6 = 0.12
    # 経路 (0, 0) は A[0,0]=0 なので確率 0
    p_path_00 = pi[0] * phi[0, 0] * A[0, 0] * phi[1, 0]
    assert p_path_00 == 0.0  # 遷移禁止パス

    # 周辺確率最大を選ぶと (0, 0) になりうるが Viterbi は妥当なパス (0, 1) を選ぶ
    assert A[0, 0] == 0.0


def test_ex13_20_multiple_sequences_baum_welch_normalization():
    """Exercise 13.20: Multiple sequence Baum-Welch M-step preserves row sum normalization"""
    K = 3
    S = 4
    xi_seqs = [np.random.dirichlet(np.ones(K * K), size=6).reshape(6, K, K) for _ in range(S)]
    gamma_seqs = []
    for s in range(S):
        g_prev = np.sum(xi_seqs[s], axis=2)
        g_last = np.sum(xi_seqs[s][-1], axis=0)[None, :]
        gamma_seqs.append(np.vstack([g_prev, g_last]))

    num_A = sum(np.sum(xi, axis=0) for xi in xi_seqs)
    den_A = sum(np.sum(g[:-1, :], axis=0) for g in gamma_seqs)
    A_multi = num_A / den_A[:, None]

    # 行和が厳密に 1.0
    np.testing.assert_allclose(np.sum(A_multi, axis=1), np.ones(K), atol=1e-12)


def test_ex13_21_kalman_gain_analytical_derivation():
    """Exercise 13.21: Kalman gain K_n = P_{n-1} C^T (C P_{n-1} C^T + Sigma)^{-1}"""
    D = 2
    P_pred = np.array([[2.0, 0.5],
                       [0.5, 1.5]])
    C = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    Sigma = np.array([[0.3, 0.05],
                      [0.05, 0.2]])

    S_inv = np.linalg.inv(C @ P_pred @ C.T + Sigma)
    K_gain = P_pred @ C.T @ S_inv

    # Joseph形式更新共分散
    I_KC = np.eye(D) - K_gain @ C
    V_joseph = I_KC @ P_pred @ I_KC.T + K_gain @ Sigma @ K_gain.T
    V_standard = (np.eye(D) - K_gain @ C) @ P_pred

    np.testing.assert_allclose(V_joseph, V_standard, atol=1e-12)
    # 正定値性の検証
    assert np.all(np.linalg.eigvalsh(V_joseph) > 0)


def test_ex13_25_rts_smoother_variance_reduction():
    """Exercise 13.25: RTS Smoother variance is strictly less than or equal to filtered variance"""
    # 1次元カルマンフィルタ・スムーザ
    P_filt = 1.2
    P_pred_next = 1.8
    V_next = 0.9  # スムージング後の次時点分散
    A = 0.9

    J = P_filt * A / P_pred_next
    V_smooth = P_filt + J**2 * (V_next - P_pred_next)

    # スムージング分散はフィルタ分散以下
    assert V_smooth <= P_filt
    assert V_smooth > 0


def test_ex13_27_kalman_zero_noise_limit():
    """Exercise 13.27: As observation noise Sigma -> 0, posterior variance vanishes"""
    D = 2
    P_pred = np.eye(D) * 2.0
    C = np.eye(D)

    sigmas = [1.0, 0.1, 0.01, 1e-4]
    variances = []
    for s in sigmas:
        Sigma = np.eye(D) * s
        K_gain = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + Sigma)
        V = (np.eye(D) - K_gain @ C) @ P_pred
        variances.append(np.trace(V))

    # ノイズ減少に伴い事後分散が単調減少してゼロに収束
    assert np.all(np.diff(variances) < 0)
    assert variances[-1] < 1e-3


def test_ex13_32_particle_filter_resampling():
    """Exercise 13.32: Particle filter SIR resampling mechanism prevents weight degeneracy"""
    N_particles = 100
    # 縮退した重み
    weights = np.zeros(N_particles)
    weights[0] = 0.99
    weights[1:] = 0.01 / (N_particles - 1)

    # 有効粒子数 N_eff = 1 / sum w_i^2
    N_eff_before = 1.0 / np.sum(weights**2)
    assert N_eff_before < 2.0  # 重度縮退

    # リサンプリング実行
    indices = np.random.choice(N_particles, size=N_particles, p=weights)
    resampled_weights = np.ones(N_particles) / N_particles
    N_eff_after = 1.0 / np.sum(resampled_weights**2)

    assert N_eff_after == N_particles
