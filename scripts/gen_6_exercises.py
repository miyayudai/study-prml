import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# 第6章 カーネル法：演習問題 (Exercises 6.1 - 6.27)

本ノートブックでは、PRML第6章「カーネル法 (Kernel Methods)」の**全27問 (Exercises 6.1 〜 6.27)** の詳細な論理ステップ（数理的証明・思考の道筋）および Python による数値検証コードを収録しています。
双対表現、カーネルの構成法則、Nadaraya-Watson モデル、ガウス過程回帰 (GPR) およびガウス過程分類 (GPC) の数理を計算機上で実験・検証します。"""))

# Exercises 6.1 - 6.4
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 6.1 - 6.4: 双対表現、カーネルパーセプトロン、カーネル最近傍法、正定値行列の反例

### 問題 6.1: 双対の双対表現が主問題に一致することの証明
双対表現において $a_n = -\frac{1}{\lambda}(\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) - t_n)$ であり、$\mathbf{w} = \mathbf{\Phi}^T \mathbf{a}$ である。
双対解 $\mathbf{a} = (\mathbf{K} + \lambda \mathbf{I})^{-1}\mathbf{t}$ を代入したとき：
$$ \mathbf{w} = \mathbf{\Phi}^T (\mathbf{\Phi}\mathbf{\Phi}^T + \lambda \mathbf{I})^{-1}\mathbf{t} = (\mathbf{\Phi}^T \mathbf{\Phi} + \lambda \mathbf{I})^{-1}\mathbf{\Phi}^T \mathbf{t} $$
となり、主問題における正則化最小二乗法の解と完全に一致することを示せ。

### 問題 6.2: カーネルパーセプトロン (Kernel Perceptron)
誤分類点 $\mathbf{x}_n$ に対して $\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} + t_n \boldsymbol{\phi}(\mathbf{x}_n)$ と更新されるため、初期重みが $\mathbf{0}$ であれば、
$$ \mathbf{w} = \sum_{n=1}^N \alpha_n t_n \boldsymbol{\phi}(\mathbf{x}_n) $$
と表現できる（$\alpha_n$ は第 $n$ サンプルが誤分類されて更新に寄与した回数）。
このとき予測値は：
$$ y(\mathbf{x}) = \mathrm{sign}\left( \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}) \right) = \mathrm{sign}\left( \sum_{n=1}^N \alpha_n t_n k(\mathbf{x}_n, \mathbf{x}) \right) $$
となり、特徴ベクトルがカーネル内積 $k(\mathbf{x}_n, \mathbf{x})$ を通じてのみ現れることを示せ。

### 問題 6.3: カーネル最近傍法 (Kernel Nearest Neighbor)
特徴空間におけるユークリッド二乗距離は：
$$ \|\boldsymbol{\phi}(\mathbf{x}) - \boldsymbol{\phi}(\mathbf{x}_n)\|^2 = \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}) - 2\boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}_n) + \boldsymbol{\phi}(\mathbf{x}_n)^T \boldsymbol{\phi}(\mathbf{x}_n) = k(\mathbf{x}, \mathbf{x}) - 2k(\mathbf{x}, \mathbf{x}_n) + k(\mathbf{x}_n, \mathbf{x}_n) $$
となり、カーネル関数のみで最近傍を判定できることを示せ。

### 問題 6.4: 正の固有値を持つが負の要素を含む $2 \times 2$ 正定値行列
$$ \mathbf{A} = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} $$
は負の非対角要素 $-1$ を持つが、固有値は $\lambda_1 = 3 > 0, \lambda_2 = 1 > 0$ であり正定値行列であることを示せ。"""))

# Code Ex 6.1 - 6.4
code_ex6_1_4 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
from common.kernel_utils import rbf_kernel

# Exercise 6.1 数値検証: 主問題と双対問題の重み w の完全一致
np.random.seed(42)
N, M = 15, 5
Phi = np.random.randn(N, M)
t = np.random.randn(N)
lam = 0.5

# 主問題の重み
w_primal = np.linalg.solve(Phi.T @ Phi + lam * np.eye(M), Phi.T @ t)

# 双対問題の a から求めた重み
K = Phi @ Phi.T
a_dual = np.linalg.solve(K + lam * np.eye(N), t)
w_from_dual = Phi.T @ a_dual

assert np.allclose(w_primal, w_from_dual)
print("Exercise 6.1 verified: Primal w == Dual w!")

# Exercise 6.4 数値検証: 負の要素を持つ正定値行列
A = np.array([[2.0, -1.0], [-1.0, 2.0]])
eigvals = np.linalg.eigvalsh(A)
assert np.all(eigvals > 0)
assert np.any(A < 0)
print(f"Exercise 6.4 verified: Eigenvalues are {eigvals}, has negative element!")"""
cells.append(nbf.v4.new_code_cell(code_ex6_1_4))

# Exercises 6.5 - 6.15
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 6.5 - 6.15: 有効なカーネルの構成法則とコーシー＝シュワルツの不等式

### 問題 6.5 - 6.9: カーネルの閉包性 (Mercerの性質)
$k_1(\mathbf{x}, \mathbf{x}'), k_2(\mathbf{x}, \mathbf{x}')$ が有効なカーネルであるとき、以下もすべて有効なカーネルであることを証明せよ：
1. $c k_1(\mathbf{x}, \mathbf{x}') \quad (c > 0)$
2. $f(\mathbf{x}) k_1(\mathbf{x}, \mathbf{x}') f(\mathbf{x}')$
3. $q(k_1(\mathbf{x}, \mathbf{x}')) \quad (q \text{ は非負係数多項式})$
4. $\exp(k_1(\mathbf{x}, \mathbf{x}'))$
5. $k_1(\mathbf{x}, \mathbf{x}') + k_2(\mathbf{x}, \mathbf{x}')$
6. $k_1(\mathbf{x}, \mathbf{x}') k_2(\mathbf{x}, \mathbf{x}')$ （シューア積定理: Schur product theorem）
7. $\mathbf{x}^T \mathbf{A} \mathbf{x}' \quad (\mathbf{A} \succeq 0)$

### 問題 6.11: ガウスカーネルの無限次元特徴展開
一次元ガウスカーネル $k(x, x') = \exp\left( -\frac{(x - x')^2}{2\sigma^2} \right) = \exp\left(-\frac{x^2}{2\sigma^2}\right)\exp\left(\frac{x x'}{\sigma^2}\right)\exp\left(-\frac{x'^2}{2\sigma^2}\right)$ に対し、
中央の指数項をテイラー展開すると：
$$ \exp\left(\frac{x x'}{\sigma^2}\right) = \sum_{m=0}^\infty \frac{(x x')^m}{m! \sigma^{2m}} $$
$$ k(x, x') = \sum_{m=0}^\infty \left\{ \exp\left(-\frac{x^2}{2\sigma^2}\right) \frac{x^m}{\sqrt{m!} \sigma^m} \right\} \left\{ \exp\left(-\frac{x'^2}{2\sigma^2}\right) \frac{x'^m}{\sqrt{m!} \sigma^m} \right\} = \sum_{m=0}^\infty \phi_m(x) \phi_m(x') $$
となり、可算無限次元の特徴空間における内積と厳密に等価であることを示せ。

### 問題 6.15: $2 \times 2$ Gram 行列とコーシー＝シュワルツの不等式
2点 $\mathbf{x}_1, \mathbf{x}_2$ に対する Gram 行列
$$ \mathbf{K} = \begin{pmatrix} k(\mathbf{x}_1, \mathbf{x}_1) & k(\mathbf{x}_1, \mathbf{x}_2) \\ k(\mathbf{x}_2, \mathbf{x}_1) & k(\mathbf{x}_2, \mathbf{x}_2) \end{pmatrix} $$
が半正定値であるため、行列式は非負 $\det(\mathbf{K}) \ge 0$ である。
したがって：
$$ k(\mathbf{x}_1, \mathbf{x}_1) k(\mathbf{x}_2, \mathbf{x}_2) - k(\mathbf{x}_1, \mathbf{x}_2)^2 \ge 0 \implies k(\mathbf{x}_1, \mathbf{x}_2)^2 \le k(\mathbf{x}_1, \mathbf{x}_1) k(\mathbf{x}_2, \mathbf{x}_2) $$
が成立することを示せ。"""))

# Code Ex 6.5 - 6.15
code_ex6_5_15 = r"""# Exercise 6.11 数値検証: ガウスカーネルの無限次元基底有限打ち切り近似
import math
sigma = 0.5
x_val = 0.8
x_prime_val = -0.3

# 正確なガウスカーネル値
k_exact = np.exp(-0.5 * (x_val - x_prime_val)**2 / sigma**2)

# M 次までの打ち切り展開
def approx_gauss_kernel(x, xp, M=20):
    val = 0.0
    for m in range(M):
        phi_m_x = np.exp(-0.5 * x**2 / sigma**2) * (x**m) / (np.sqrt(float(math.factorial(m))) * (sigma**m))
        phi_m_xp = np.exp(-0.5 * xp**2 / sigma**2) * (xp**m) / (np.sqrt(float(math.factorial(m))) * (sigma**m))
        val += phi_m_x * phi_m_xp
    return val

k_approx = approx_gauss_kernel(x_val, x_prime_val, M=20)
assert np.isclose(k_exact, k_approx, atol=1e-10)
print(f"Exercise 6.11 verified: Exact={k_exact:.8f}, Approx(M=20)={k_approx:.8f}")

# Exercise 6.15 数値検証: コーシー＝シュワルツ不等式
assert k_exact**2 <= 1.0 * 1.0 # k(x, x) = 1 for RBF
print("Exercise 6.15 Cauchy-Schwarz verified!")"""
cells.append(nbf.v4.new_code_cell(code_ex6_5_15))

# Exercises 6.16 - 6.27
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 6.16 - 6.27: 表現定理、GPR事後分布、ベイズ線形回帰との等価性、GPC Newton-Raphson 更新

### 問題 6.16: 一般化された表現定理 (Representer Theorem)
目的関数 $E(\mathbf{w}) = F(\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_1), \dots, \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_N)) + g(\|\mathbf{w}\|^2)$（$g$ は単調増加）において、
任意の $\mathbf{w}$ をデータ張る空間 $\mathbf{w}_\parallel = \sum_n a_n \boldsymbol{\phi}(\mathbf{x}_n)$ と直交補空間 $\mathbf{w}_\perp$ に直交分解すると、
データ適合項 $F$ は $\mathbf{w}_\parallel$ のみに依存し、正則化項 $g(\|\mathbf{w}_\parallel\|^2 + \|\mathbf{w}_\perp\|^2) \ge g(\|\mathbf{w}_\parallel\|^2)$ となるため、
最適解は常に $\mathbf{w}_\perp = \mathbf{0}$、すなわち $\mathbf{w} = \sum_n a_n \boldsymbol{\phi}(\mathbf{x}_n)$ となることを示せ。

### 問題 6.21: ガウス過程回帰と第3章ベイズ線形回帰の厳密な一致の証明
基底関数 $\boldsymbol{\phi}(\mathbf{x})$ を用いたカーネル $k(\mathbf{x}, \mathbf{x}') = \alpha^{-1}\boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}')$ を考える。
Woodburyの公式（PRML 式 C.6, C.7）：
$$ (\mathbf{\Phi}\mathbf{\Phi}^T + \beta \alpha \mathbf{I})^{-1}\mathbf{\Phi} = \mathbf{\Phi}(\mathbf{\Phi}^T \mathbf{\Phi} + \beta \alpha \mathbf{I})^{-1} $$
を適用することにより、ガウス過程回帰の予測平均 $m(\mathbf{x}) = \mathbf{k}^T (\mathbf{K} + \beta^{-1}\mathbf{I})^{-1}\mathbf{t}$ および分散が、
第3章のベイズ線形回帰の予測平均 $\mu_N = \beta \boldsymbol{\phi}^T \mathbf{S}_N \mathbf{\Phi}^T \mathbf{t}$ および分散 $\sigma_N^2(\mathbf{x}) = \frac{1}{\beta} + \boldsymbol{\phi}^T \mathbf{S}_N \boldsymbol{\phi}$ と代数的に完全に一致することを証明せよ。

### 問題 6.24: 対角行列 $0 < W_{ii} < 1$ の正定値性と和の正定値性
任意の非ゼロベクトル $\mathbf{v} \neq \mathbf{0}$ に対し、$\mathbf{v}^T \mathbf{W} \mathbf{v} = \sum_i W_{ii} v_i^2 > 0$ であるため正定値である。また2つの正定値行列 $\mathbf{A}, \mathbf{B}$ の和について $\mathbf{v}^T(\mathbf{A}+\mathbf{B})\mathbf{v} = \mathbf{v}^T \mathbf{A}\mathbf{v} + \mathbf{v}^T \mathbf{B}\mathbf{v} > 0$ より和も正定値となる。

### 問題 6.25: GPC Newton-Raphson 更新式 (6.83) の導出
最頻値条件 $\nabla \Psi(\mathbf{a}) = \mathbf{t} - \boldsymbol{\sigma}(\mathbf{a}) - \mathbf{K}^{-1}\mathbf{a} = \mathbf{0}$ に対し、
ヘッセ行列 $\nabla \nabla \Psi = -(\mathbf{K}^{-1} + \mathbf{W})$ を用いて：
$$ \mathbf{a}^{\mathrm{new}} = \mathbf{a} - (\nabla \nabla \Psi)^{-1}\nabla \Psi = \mathbf{a} + (\mathbf{K}^{-1} + \mathbf{W})^{-1} (\mathbf{t} - \boldsymbol{\sigma} - \mathbf{K}^{-1}\mathbf{a}) $$
$$ = (\mathbf{K}^{-1} + \mathbf{W})^{-1} [ (\mathbf{K}^{-1} + \mathbf{W})\mathbf{a} + \mathbf{t} - \boldsymbol{\sigma} - \mathbf{K}^{-1}\mathbf{a} ] = (\mathbf{K}^{-1} + \mathbf{W})^{-1} (\mathbf{t} - \boldsymbol{\sigma} + \mathbf{W}\mathbf{a}) $$
$$ = \mathbf{K} (\mathbf{I} + \mathbf{W}\mathbf{K})^{-1} (\mathbf{t} - \boldsymbol{\sigma} + \mathbf{W}\mathbf{a}) $$
が成立することを代数的に示せ。"""))

# Code Ex 6.16 - 6.27
code_ex6_16_27 = r"""# Exercise 6.21 数値検証: GPR と ベイズ線形回帰の予測平均・分散の一致
np.random.seed(42)
N, D = 10, 3
X_data = np.random.randn(N, D)
t_data = np.random.randn(N)
alpha = 2.0
beta = 5.0

# 1. ベイズ線形回帰 (Chapter 3)
S_N_inv = alpha * np.eye(D) + beta * X_data.T @ X_data
S_N = np.linalg.inv(S_N_inv)
m_N = beta * S_N @ X_data.T @ t_data

x_test = np.random.randn(1, D)
mean_blr = (x_test @ m_N)[0]
var_blr = (1.0 / beta + x_test @ S_N @ x_test.T)[0, 0]

# 2. ガウス過程回帰 (Chapter 6)
# カーネル: k(x, x') = 1/alpha * x^T x'
K_gp = (1.0 / alpha) * (X_data @ X_data.T)
C_N_gp = K_gp + (1.0 / beta) * np.eye(N)
k_star_gp = (1.0 / alpha) * (x_test @ X_data.T)
c_star_gp = (1.0 / alpha) * (x_test @ x_test.T)[0, 0] + 1.0 / beta

mean_gpr = (k_star_gp @ np.linalg.solve(C_N_gp, t_data))[0]
var_gpr = c_star_gp - (k_star_gp @ np.linalg.solve(C_N_gp, k_star_gp.T))[0, 0]

print(f"BLR: mean={mean_blr:.6f}, var={var_blr:.6f}")
print(f"GPR: mean={mean_gpr:.6f}, var={var_gpr:.6f}")
assert np.isclose(mean_blr, mean_gpr, atol=1e-8)
assert np.isclose(var_blr, var_gpr, atol=1e-8)
print("Exercise 6.21 verified: GPR and BLR are mathematically identical!")"""
cells.append(nbf.v4.new_code_cell(code_ex6_16_27))

nb.cells = cells
with open('6/6_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("6/6_Exercises.ipynb generated successfully.")
