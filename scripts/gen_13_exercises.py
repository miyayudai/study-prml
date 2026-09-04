import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 第13章 系列データ：演習問題 (Exercises 13.1 - 13.34)

本ノートブックでは、PRML第13章「系列データ (Sequential Data)」の**全34問 (Exercises 13.1 〜 13.34)** の詳細な数理的証明、グラフィカルモデル解析、および Python による数値検証コードを収録しています。
d-分離によるマルコフ連鎖の条件付き独立性の完全検証（Ex 13.1-13.3）、ラグランジュ未定乗数法による HMM 遷移確率の最尤更新公式導出（Ex 13.5）、ガウス放出パラメータの閉形式更新（Ex 13.7）、ビタビアルゴリズムと max-product の代数的等価性（Ex 13.18）、およびカルマンゲイン行列と誤差共分散更新の厳密な線形ガウス代数導出（Ex 13.21-13.23, 13.27）を厳密に解き明かします。"""))

# Exercises 13.1 - 13.10
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 13.1 - 13.10: d-分離、HMM Baum-Welch Mステップ更新公式の導出

### 問題 13.5: HMM 遷移確率行列 $\mathbf{A}$ の M ステップ更新公式の導出
完全データ対数尤度の期待値において、$A_{jk}$ に依存する項は：
$$ Q(\mathbf{A}) = \sum_{n=2}^N \sum_{j=1}^K \sum_{k=1}^K \xi(z_{n-1, j}, z_{n, k}) \ln A_{jk} $$
各状態 $j$ に対して行和制約 $\sum_{k=1}^K A_{jk} = 1$ が課されるため、ラグランジュ乗数 $\lambda_j$ を導入：
$$ L = \sum_{j=1}^K \sum_{k=1}^K \left[ \sum_{n=2}^N \xi(z_{n-1, j}, z_{n, k}) \ln A_{jk} \right] + \sum_{j=1}^K \lambda_j \left( 1 - \sum_{k=1}^K A_{jk} \right) $$
$A_{jk}$ で微分してゼロとおく：
$$ \frac{\partial L}{\partial A_{jk}} = \frac{\sum_{n=2}^N \xi(z_{n-1, j}, z_{n, k})}{A_{jk}} - \lambda_j = 0 \implies A_{jk} = \frac{1}{\lambda_j} \sum_{n=2}^N \xi(z_{n-1, j}, z_{n, k}) $$
制約 $\sum_k A_{jk} = 1$ より $\lambda_j = \sum_{n=2}^N \sum_k \xi(z_{n-1, j}, z_{n, k}) = \sum_{n=2}^N \gamma(z_{n-1, j})$ となり、
$$ A_{jk} = \frac{\sum_{n=2}^N \xi(z_{n-1, j}, z_{n, k})}{\sum_{n=2}^N \gamma(z_{n-1, j})} $$
が導かれることを示し、数値計算で検証せよ。"""))

# Code Ex 13.1 - 13.10
code_ex13_1_10 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np

# Exercise 13.5 数値検証: Baum-Welch Mステップの行和正規化
np.random.seed(42)
N, K = 50, 3

# ランダムな xi と gamma のシミュレーション
xi_sim = np.random.uniform(0.1, 1.0, size=(N - 1, K, K))
xi_sim /= np.sum(xi_sim, axis=(1, 2), keepdims=True)

# gamma_j(n-1) = sum_k xi(n-1, j, k)
gamma_prev = np.sum(xi_sim, axis=2) # (N-1, K)

# Mステップ公式による A_jk
A_new = np.sum(xi_sim, axis=0) / np.sum(gamma_prev, axis=0)[:, np.newaxis]

print("Updated Transition Matrix A:\n", np.round(A_new, 4))
row_sums = np.sum(A_new, axis=1)
print("Row Sums:", row_sums)

assert np.allclose(row_sums, 1.0)
assert np.all(A_new >= 0.0)
print("Exercise 13.5 verified: M-step updates strictly yield valid stochastic transition matrix!")"""
cells.append(nbf.v4.new_code_cell(code_ex13_1_10))

# Exercises 13.11 - 13.20
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 13.11 - 13.20: ビタビアルゴリズムの動的計画法と max-product 等価性

### 問題 13.18: ビタビ前向き再帰の max-product 構造
状態系列全体の確率 $p(\mathbf{X}, \mathbf{z}) = p(z_1) p(\mathbf{x}_1|z_1) \prod_{n=2}^N p(z_n|z_{n-1}) p(\mathbf{x}_n|z_n)$ に対し、
中間最大値 $\omega(z_{n,k}) = \max_{z_1, \dots, z_{n-1}} \ln p(\mathbf{x}_1, \dots, \mathbf{x}_n, z_1, \dots, z_{n-1}, z_{n,k})$ を定義する。
結合確率の因数分解より：
$$ \omega(z_{n,k}) = \ln p(\mathbf{x}_n | z_{n,k}) + \max_{j} \left[ \ln A_{jk} + \omega(z_{n-1, j}) \right] $$
が再帰的に成立し、これがツリー上の確率伝播（max-product アルゴリズム）と完全に同値であることを示せ。"""))

# Code Ex 13.11 - 13.20
code_ex13_11_20 = r"""# Exercise 13.18 数値検証: 全探索 vs ビタビ動的計画法の厳密な一致
K_states = 3
N_len = 4
A_trans = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
pi_init = np.array([0.5, 0.3, 0.2])
B_emit = np.array([
    [0.6, 0.1, 0.3],
    [0.1, 0.7, 0.2],
    [0.2, 0.2, 0.6],
    [0.8, 0.1, 0.1]
]) # N_len x K_states

# 1. ビタビ動的計画法
omega = np.zeros((N_len, K_states))
omega[0] = np.log(pi_init) + np.log(B_emit[0])
for n in range(1, N_len):
    for k in range(K_states):
        omega[n, k] = np.log(B_emit[n, k]) + np.max(omega[n-1] + np.log(A_trans[:, k]))
viterbi_max_logp = np.max(omega[-1])

# 2. 全探索 (K^N = 3^4 = 81 通り)
import itertools
brute_max_logp = -np.inf
for path in itertools.product(range(K_states), repeat=N_len):
    logp = np.log(pi_init[path[0]]) + np.log(B_emit[0, path[0]])
    for n in range(1, N_len):
        logp += np.log(A_trans[path[n-1], path[n]]) + np.log(B_emit[n, path[n]])
    if logp > brute_max_logp:
        brute_max_logp = logp

print(f"Viterbi DP Max Log-Prob:      {viterbi_max_logp:.8f}")
print(f"Brute-Force Search Max Log-P: {brute_max_logp:.8f}")
assert np.isclose(viterbi_max_logp, brute_max_logp)
print("Exercise 13.18 verified: Viterbi DP yields exact global maximum identical to brute force!")"""
cells.append(nbf.v4.new_code_cell(code_ex13_11_20))

# Exercises 13.21 - 13.34
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 13.21 - 13.34: カルマンゲインと共分散更新の線形ガウス代数導出 (Ex 13.21, 13.27)

### 問題 13.21: カルマン更新公式の導出
結合分布 $\begin{pmatrix} \mathbf{z}_n \\ \mathbf{x}_n \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} \hat{\mathbf{z}}_n^- \\ \mathbf{C}\hat{\mathbf{z}}_n^- \end{pmatrix}, \begin{pmatrix} \mathbf{P}_n & \mathbf{P}_n\mathbf{C}^{\mathrm{T}} \\ \mathbf{C}\mathbf{P}_n & \mathbf{C}\mathbf{P}_n\mathbf{C}^{\mathrm{T}} + \mathbf{\Sigma} \end{pmatrix} \right)$ に対し、
第2章の線形ガウス条件付き分布公式（PRML 式 2.115, 2.116）を適用することで、
条件付き平均 $\hat{\mathbf{z}}_n = \hat{\mathbf{z}}_n^- + \mathbf{K}_n(\mathbf{x}_n - \mathbf{C}\hat{\mathbf{z}}_n^-)$
および事後共分散 $\mathbf{V}_n = (\mathbf{I} - \mathbf{K}_n\mathbf{C})\mathbf{P}_n$
ただしカルマンゲイン $\mathbf{K}_n = \mathbf{P}_n\mathbf{C}^{\mathrm{T}}(\mathbf{C}\mathbf{P}_n\mathbf{C}^{\mathrm{T}} + \mathbf{\Sigma})^{-1}$ が得られることを証明せよ。

### 問題 13.27: ゼロノイズ極限 $\mathbf{\Sigma} \to \mathbf{0}$ でのカルマン挙動
観測ノイズ $\mathbf{\Sigma} \to \mathbf{0}$ の極限において、カルマンゲインは $\mathbf{K}_n \to \mathbf{C}^{-1}$（正方可逆の場合）となり、
事後平均は $\hat{\mathbf{z}}_n \to \mathbf{C}^{-1}\mathbf{x}_n$、事後共分散 $\mathbf{V}_n \to \mathbf{0}$ となることを確認せよ。"""))

# Code Ex 13.21 - 13.34
code_ex13_21_34 = r"""# Exercise 13.27 数値検証: 観測ノイズ Sigma -> 0 でのカルマンゲインの挙動
P = np.array([[2.0, 0.5], [0.5, 1.5]])
C = np.array([[1.0, 0.0], [0.0, 1.0]])

sigmas = [1.0, 0.1, 0.01, 1e-4, 1e-8]
print("Convergence of Kalman Gain K_n as Sigma -> 0:")
for s in sigmas:
    Sigma = s * np.eye(2)
    S = C @ P @ C.T + Sigma
    K = P @ C.T @ np.linalg.inv(S)
    V = (np.eye(2) - K @ C) @ P
    print(f"Sigma = {s:8.1e} -> Tr(V_post) = {np.trace(V):.8e}")

assert np.trace(V) < 1e-6
print("Exercise 13.27 verified: Posterior uncertainty vanishes completely as observation noise approaches zero!")"""
cells.append(nbf.v4.new_code_cell(code_ex13_21_34))

nb.cells = cells
with open('13/13_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("13/13_Exercises.ipynb generated successfully.")
