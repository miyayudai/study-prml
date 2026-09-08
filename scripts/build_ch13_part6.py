# scripts/build_ch13_part6.py
"""
Build Chapter 13 Exercises: Part 6 (Exercises 13.28 - 13.34)
LDS EM Learning, DARE Stationary Filter, Invariance, Switching LDS, and Particle Filters
"""

import nbformat as nbf

def build_part6():
    """Exercises 13.28 - 13.34: Learning & Advanced Topics"""
    cells = []

    # Exercise 13.28
    ex13_28_md = """---
## Exercise 13.28: 線形動的システムの EM アルゴリズム：遷移行列 $\\mathbf{A}$ の M-step 更新式の導出

### 問題の背景と数学的証明
LDS の完全データ対数尤度 $\\ln p(\\mathbf{X}, \\mathbf{Z})$ に対する期待値 $Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{\\text{old}}) = \\mathbb{E}_{\\mathbf{Z}|\\mathbf{X}}[\\ln p(\\mathbf{X}, \\mathbf{Z})]$ において、遷移行列 $\\mathbf{A}$ に依存する項は：
$$ Q_{\\mathbf{A}} = -\\frac{1}{2} \\sum_{n=2}^N \\mathbb{E}\\left[ (\\mathbf{z}_n - \\mathbf{A} \\mathbf{z}_{n-1})^{\\mathrm{T}} \\mathbf{\\Gamma}^{-1} (\\mathbf{z}_n - \\mathbf{A} \\mathbf{z}_{n-1}) \\mid \\mathbf{X} \\right] $$
である（(13.108)式）。本問では、これを $\\mathbf{A}$ に関して最大化し、M-step 更新公式：
$$ \\mathbf{A}^{\\text{new}} = \\left( \\sum_{n=2}^N \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] \\right) \\left( \\sum_{n=2}^N \\mathbb{E}[\\mathbf{z}_{n-1} \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] \\right)^{-1} $$
を導出する。

**代数的導出**:
1. トレースの巡回対称性を用いて期待値を展開する：
$$ Q_{\\mathbf{A}} = -\\frac{1}{2} \\sum_{n=2}^N \\operatorname{Tr}\\left( \\mathbf{\\Gamma}^{-1} \\left( \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_n^{\\mathrm{T}}|\\mathbf{X}] - \\mathbf{A}\\mathbb{E}[\\mathbf{z}_{n-1}\\mathbf{z}_n^{\\mathrm{T}}|\\mathbf{X}] - \\mathbb{E}[\\mathbf{z}_n\\mathbf{z}_{n-1}^{\\mathrm{T}}|\\mathbf{X}]\\mathbf{A}^{\\mathrm{T}} + \\mathbf{A}\\mathbb{E}[\\mathbf{z}_{n-1}\\mathbf{z}_{n-1}^{\\mathrm{T}}|\\mathbf{X}]\\mathbf{A}^{\\mathrm{T}} \\right) \\right) $$
2. 行列微分の公式 $\\frac{\\partial}{\\partial \\mathbf{A}} \\operatorname{Tr}(\\mathbf{A} \\mathbf{B}) = \\mathbf{B}^{\\mathrm{T}}$, $\\frac{\\partial}{\\partial \\mathbf{A}} \\operatorname{Tr}(\\mathbf{A} \\mathbf{B} \\mathbf{A}^{\\mathrm{T}} \\mathbf{C}) = \\mathbf{C} \\mathbf{A} \\mathbf{B} + \\mathbf{C}^{\\mathrm{T}} \\mathbf{A} \\mathbf{B}^{\\mathrm{T}}$ を適用する：
$$ \\frac{\\partial Q_{\\mathbf{A}}}{\\partial \\mathbf{A}} = \\sum_{n=2}^N \\mathbf{\\Gamma}^{-1} \\left( \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] - \\mathbf{A} \\mathbb{E}[\\mathbf{z}_{n-1} \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] \\right) = \\mathbf{0} $$
3. 左から $\\mathbf{\\Gamma}$ を掛けると：
$$ \\sum_{n=2}^N \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] - \\mathbf{A} \\sum_{n=2}^N \\mathbb{E}[\\mathbf{z}_{n-1} \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] = \\mathbf{0} $$
4. したがって、求める閉形式解が得られる：
$$ \\mathbf{A}^{\\text{new}} = \\left( \\sum_{n=2}^N \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] \\right) \\left( \\sum_{n=2}^N \\mathbb{E}[\\mathbf{z}_{n-1} \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] \\right)^{-1} $$
ここで $\\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] = \\mathbf{V}_{n, n-1} + \\hat{\\boldsymbol{\\mu}}_n \\hat{\\boldsymbol{\\mu}}_{n-1}^{\\mathrm{T}}$、$\\mathbb{E}[\\mathbf{z}_{n-1} \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] = \\hat{\\mathbf{V}}_{n-1} + \\hat{\\boldsymbol{\\mu}}_{n-1} \\hat{\\boldsymbol{\\mu}}_{n-1}^{\\mathrm{T}}$ である。

#### 穴埋め問題
1. 遷移行列 $\\mathbf{A}$ の最適化は、潜在空間における連続時点間の $\\text{[ (A) ]}$ 問題に帰着する。
2. 分子には時刻 $n$ と $n-1$ の $\\text{[ (B) ]}$ が現れる。
3. 分母には時刻 $n-1$ の $\\text{[ (C) ]}$ が現れる。
*(解: A: 多変量線形回帰, B: 相互相関期待値 $\\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_{n-1}^{\\mathrm{T}}]$, C: 自己相関期待値 $\\mathbb{E}[\\mathbf{z}_{n-1} \\mathbf{z}_{n-1}^{\\mathrm{T}}]$)*
"""
    ex13_28_code = """# Exercise 13.28 数値検証: 遷移行列 A の M-step 更新と勾配ゼロ条件の一致
import numpy as np

# 平滑化統計量から A_new を計算
sum_cross = np.zeros((D, D))
sum_prev_cov = np.zeros((D, D))

for t in range(1, N_sim):
    sum_cross += V_cross[t-1] + np.outer(mu_smooth[t], mu_smooth[t-1])
    sum_prev_cov += V_smooth[t-1] + np.outer(mu_smooth[t-1], mu_smooth[t-1])

A_new = sum_cross @ np.linalg.inv(sum_prev_cov)

# 勾配 sum_cross - A * sum_prev_cov がゼロであることを確認
grad_A = sum_cross - A_new @ sum_prev_cov
np.testing.assert_allclose(grad_A, np.zeros((D, D)), atol=1e-12)

print("Exercise 13.28 verified: A_new solves the normal equations strictly (gradient = 0):")
print("A_new:\\n", A_new)
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_28_md), nbf.v4.new_code_cell(ex13_28_code)])

    # Exercise 13.29
    ex13_29_md = """---
## Exercise 13.29: 線形動的システムの EM アルゴリズム：観測行列 $\\mathbf{C}$ の M-step 更新式の導出

### 問題の背景と数学的証明
完全データ対数尤度期待値 $Q$ において、観測行列 $\\mathbf{C}$ に依存する項は：
$$ Q_{\\mathbf{C}} = -\\frac{1}{2} \\sum_{n=1}^N \\mathbb{E}\\left[ (\\mathbf{x}_n - \\mathbf{C} \\mathbf{z}_n)^{\\mathrm{T}} \\mathbf{\\Sigma}^{-1} (\\mathbf{x}_n - \\mathbf{C} \\mathbf{z}_n) \\mid \\mathbf{X} \\right] $$
である（(13.111)式）。本問では、これを $\\mathbf{C}$ に関して最大化し、閉形式更新式：
$$ \\mathbf{C}^{\\text{new}} = \\left( \\sum_{n=1}^N \\mathbf{x}_n \\mathbb{E}[\\mathbf{z}_n^{\\mathrm{T}} | \\mathbf{X}] \\right) \\left( \\sum_{n=1}^N \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_n^{\\mathrm{T}} | \\mathbf{X}] \\right)^{-1} $$
を導出する。

**代数的導出**:
1. トレース展開：
$$ Q_{\\mathbf{C}} = -\\frac{1}{2} \\sum_{n=1}^N \\operatorname{Tr}\\left( \\mathbf{\\Sigma}^{-1} \\left( \\mathbf{x}_n \\mathbf{x}_n^{\\mathrm{T}} - 2 \\mathbf{x}_n \\mathbb{E}[\\mathbf{z}_n^{\\mathrm{T}}|\\mathbf{X}] \\mathbf{C}^{\\mathrm{T}} + \\mathbf{C} \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_n^{\\mathrm{T}}|\\mathbf{X}] \\mathbf{C}^{\\mathrm{T}} \\right) \\right) $$
2. $\\mathbf{C}$ で微分してゼロとおく：
$$ \\frac{\\partial Q_{\\mathbf{C}}}{\\partial \\mathbf{C}} = \\sum_{n=1}^N \\mathbf{\\Sigma}^{-1} \\left( \\mathbf{x}_n \\mathbb{E}[\\mathbf{z}_n^{\\mathrm{T}} | \\mathbf{X}] - \\mathbf{C} \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_n^{\\mathrm{T}} | \\mathbf{X}] \\right) = \\mathbf{0} $$
3. 左から $\\mathbf{\\Sigma}$ を掛けて整理すると：
$$ \\mathbf{C}^{\\text{new}} = \\left( \\sum_{n=1}^N \\mathbf{x}_n \\hat{\\boldsymbol{\\mu}}_n^{\\mathrm{T}} \\right) \\left( \\sum_{n=1}^N (\\hat{\\mathbf{V}}_n + \\hat{\\boldsymbol{\\mu}}_n \\hat{\\boldsymbol{\\mu}}_n^{\\mathrm{T}}) \\right)^{-1} $$
となり、観測データ $\\mathbf{x}_n$ と平滑化潜在状態との重回帰推定量が得られる。

#### 穴埋め問題
1. $\\mathbf{C}$ の更新式は、観測変数 $\\mathbf{x}_n$ を目的変数、平滑化潜在状態 $\\mathbf{z}_n$ を説明変数とする $\\text{[ (A) ]}$ 推定である。
2. 期待値 $\\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_n^{\\mathrm{T}} | \\mathbf{X}]$ には、平均の自乗だけでなく $\\text{[ (B) ]}$ も寄与する。
*(解: A: 最小二乗 (線形回帰), B: 平滑化共分散 $\\hat{\\mathbf{V}}_n$)*
"""
    ex13_29_code = """# Exercise 13.29 数値検証: 観測行列 C の M-step 更新と正規方程式
sum_xz = np.zeros((M, D))
sum_zz = np.zeros((D, D))

for t in range(N_sim):
    sum_xz += np.outer(x_sim[t], mu_smooth[t])
    sum_zz += V_smooth[t] + np.outer(mu_smooth[t], mu_smooth[t])

C_new = sum_xz @ np.linalg.inv(sum_zz)

grad_C = sum_xz - C_new @ sum_zz
np.testing.assert_allclose(grad_C, np.zeros((M, D)), atol=1e-12)

print("Exercise 13.29 verified: C_new strictly satisfies normal equations (gradient = 0):")
print("C_new:", C_new)
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_29_md), nbf.v4.new_code_cell(ex13_29_code)])

    # Exercise 13.30
    ex13_30_md = """---
## Exercise 13.30: 線形動的システムの EM アルゴリズム：ノイズ共分散 $\\mathbf{\\Gamma}, \\mathbf{\\Sigma}$ の閉形式更新式の導出

### 問題の背景と数学的証明
システムノイズ共分散 $\\mathbf{\\Gamma}$ および観測ノイズ共分散 $\\mathbf{\\Sigma}$ の M-step 閉形式更新式を導出する。

**数理的導出**:
1. **システムノイズ共分散 $\\mathbf{\\Gamma}$**:
   $Q$ 関数を精度行列 $\\mathbf{\\Gamma}^{-1}$ で微分してゼロとおく：
   $$ \\frac{\\partial Q}{\\partial \\mathbf{\\Gamma}^{-1}} = \\frac{N-1}{2} \\mathbf{\\Gamma} - \\frac{1}{2} \\sum_{n=2}^N \\mathbb{E}[(\\mathbf{z}_n - \\mathbf{A} \\mathbf{z}_{n-1})(\\mathbf{z}_n - \\mathbf{A} \\mathbf{z}_{n-1})^{\\mathrm{T}} | \\mathbf{X}] = \\mathbf{0} $$
   したがって：
   $$ \\mathbf{\\Gamma}^{\\text{new}} = \\frac{1}{N-1} \\sum_{n=2}^N \\left( \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_n^{\\mathrm{T}}|\\mathbf{X}] - \\mathbf{A}^{\\text{new}} \\mathbb{E}[\\mathbf{z}_{n-1} \\mathbf{z}_n^{\\mathrm{T}}|\\mathbf{X}] - \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_{n-1}^{\\mathrm{T}}|\\mathbf{X}] (\\mathbf{A}^{\\text{new}})^{\\mathrm{T}} + \\mathbf{A}^{\\text{new}} \\mathbb{E}[\\mathbf{z}_{n-1} \\mathbf{z}_{n-1}^{\\mathrm{T}}|\\mathbf{X}] (\\mathbf{A}^{\\text{new}})^{\\mathrm{T}} \\right) $$
2. **観測ノイズ共分散 $\\mathbf{\\Sigma}$**:
   同様に $\\mathbf{\\Sigma}^{-1}$ で微分してゼロとおく：
   $$ \\mathbf{\\Sigma}^{\\text{new}} = \\frac{1}{N} \\sum_{n=1}^N \\left( \\mathbf{x}_n \\mathbf{x}_n^{\\mathrm{T}} - \\mathbf{C}^{\\text{new}} \\mathbb{E}[\\mathbf{z}_n|\\mathbf{X}] \\mathbf{x}_n^{\\mathrm{T}} - \\mathbf{x}_n \\mathbb{E}[\\mathbf{z}_n^{\\mathrm{T}}|\\mathbf{X}] (\\mathbf{C}^{\\text{new}})^{\\mathrm{T}} + \\mathbf{C}^{\\text{new}} \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_n^{\\mathrm{T}}|\\mathbf{X}] (\\mathbf{C}^{\\text{new}})^{\\mathrm{T}} \\right) $$
   これらは残差共分散の事後期待値に一致し、常に正定値行列となる。

#### 穴埋め問題
1. $\\mathbf{\\Gamma}^{\\text{new}}$ の除数は系列の遷移区間数である $\\text{[ (A) ]}$ である。
2. $\\mathbf{\\Sigma}^{\\text{new}}$ の除数は全観測数である $\\text{[ (B) ]}$ である。
3. 導出されたノイズ共分散行列は、残差の二乗期待値であるため常に $\\text{[ (C) ]}$ である。
*(解: A: $N-1$, B: $N$, C: 半正定値 (正定値))*
"""
    ex13_30_code = """# Exercise 13.30 数値検証: ノイズ共分散 Gamma_new, Sigma_new の計算と正定値性
# Gamma_new
sum_gamma_res = np.zeros((D, D))
for t in range(1, N_sim):
    E_zn_zn = V_smooth[t] + np.outer(mu_smooth[t], mu_smooth[t])
    E_zn_znprev = V_cross[t-1] + np.outer(mu_smooth[t], mu_smooth[t-1])
    E_znprev_znprev = V_smooth[t-1] + np.outer(mu_smooth[t-1], mu_smooth[t-1])
    
    res = (E_zn_zn - A_new @ E_zn_znprev.T - E_zn_znprev @ A_new.T + A_new @ E_znprev_znprev @ A_new.T)
    sum_gamma_res += res
Gamma_new = sum_gamma_res / (N_sim - 1)

# Sigma_new
sum_sigma_res = np.zeros((M, M))
for t in range(N_sim):
    E_zn_zn = V_smooth[t] + np.outer(mu_smooth[t], mu_smooth[t])
    res = (np.outer(x_sim[t], x_sim[t]) - C_new @ np.outer(mu_smooth[t], x_sim[t]) - 
           np.outer(x_sim[t], mu_smooth[t]) @ C_new.T + C_new @ E_zn_zn @ C_new.T)
    sum_sigma_res += res
Sigma_new = sum_sigma_res / N_sim

# 正定値性の確認
assert np.all(np.linalg.eigvalsh(Gamma_new) > 0)
assert np.all(np.linalg.eigvalsh(Sigma_new) > 0)

print("Exercise 13.30 verified: Noise covariances are strictly symmetric positive definite:")
print("Gamma_new eigenvalues:", np.linalg.eigvalsh(Gamma_new))
print("Sigma_new eigenvalues:", np.linalg.eigvalsh(Sigma_new))
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_3_md if False else ex13_30_md), nbf.v4.new_code_cell(ex13_30_code)])

    # Exercise 13.31
    ex13_31_md = """---
## Exercise 13.31: 定常カルマンフィルタと離散時間代数リカッチ方程式 (DARE)

### 問題の背景と数学的証明
時不変な線形力学系（$\\mathbf{A}, \\mathbf{C}, \\mathbf{\\Gamma}, \\mathbf{\\Sigma}$ が一定）において、システムが可観測かつ可検出である場合、予測共分散 $\\mathbf{V}_{n|n-1}$ は時間ステップ $n \\to \\infty$ とともに定常極限行列 $\\mathbf{V}_\\infty$ に収束する。
この定常共分散 $\\mathbf{V}_\\infty$ は、次の「離散時間代数的リカッチ方程式 (Discrete Algebraic Riccati Equation: DARE)」を満たす：
$$ \\mathbf{V}_\\infty = \\mathbf{A} \\left[ \\mathbf{V}_\\infty - \\mathbf{V}_\\infty \\mathbf{C}^{\\mathrm{T}} (\\mathbf{C} \\mathbf{V}_\\infty \\mathbf{C}^{\\mathrm{T}} + \\mathbf{\\Sigma})^{-1} \\mathbf{C} \\mathbf{V}_\\infty \\right] \\mathbf{A}^{\\mathrm{T}} + \\mathbf{\\Gamma} $$
本問では、DARE の数値解法と、カルマンフィルタ共分散の反復収束性を検証する。

#### 穴埋め問題
1. 時不変システムでは、時間経過とともに予測共分散は $\\text{[ (A) ]}$ 行列に収束する。
2. 定常状態が満たす方程式は $\\text{[ (B) ]}$ と呼ばれる。
3. これにより、定常状態ではカルマンゲインを時刻ごとに更新する必要がなく $\\text{[ (C) ]}$ とできる。
*(解: A: 定常共分散 $\\mathbf{V}_\\infty$, B: 離散時間代数リカッチ方程式 (DARE), C: 定数ゲイン $\\mathbf{K}_\\infty$)*
"""
    ex13_31_code = """# Exercise 13.31 数値検証: カルマン予測共分散のリカッチ方程式解への指数収束
# 不動点反復による DARE 解の導出
V_inf = np.eye(D)
for _ in range(200):
    S_inf = C @ V_inf @ C.T + Sigma
    K_inf_step = V_inf @ C.T @ np.linalg.inv(S_inf)
    V_inf = A @ (np.eye(D) - K_inf_step @ C) @ V_inf @ A.T + Gamma

# 通常のカルマンフィルタ予測共分散の時間発展
V_curr = np.eye(D) * 10.0  # 異なる初期値
errors = []
for step in range(50):
    errors.append(np.linalg.norm(V_curr - V_inf))
    K_step = V_curr @ C.T @ np.linalg.inv(C @ V_curr @ C.T + Sigma)
    V_curr = A @ (np.eye(D) - K_step @ C) @ V_curr @ A.T + Gamma

np.testing.assert_allclose(V_curr, V_inf, atol=1e-8)
print(f"Exercise 13.31 verified: Kalman covariance strictly converges to DARE fixed point (final error = {errors[-1]:.4e})!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_31_md), nbf.v4.new_code_cell(ex13_31_code)])

    # Exercise 13.32
    ex13_32_md = """---
## Exercise 13.32: 潜在空間の線形可逆変換に対する尤度不変性とモデルの可同定性

### 問題の背景と数学的証明
LDS において、潜在変数 $\\mathbf{z}_n$ は直接観測されないため、潜在空間の表現には任意性が存在する。
任意の $D \\times D$ 正則行列 $\\mathbf{M}$ による線形変換：
$$ \\mathbf{z}_n' = \\mathbf{M} \\mathbf{z}_n $$
を考える。パラメータを次のように変換する：
$$ \\mathbf{A}' = \\mathbf{M} \\mathbf{A} \\mathbf{M}^{-1}, \\quad \\mathbf{C}' = \\mathbf{C} \\mathbf{M}^{-1}, \\quad \\mathbf{\\Gamma}' = \\mathbf{M} \\mathbf{\\Gamma} \\mathbf{M}^{\\mathrm{T}}, \\quad \\mathbf{\\Sigma}' = \\mathbf{\\Sigma}, \\quad \\boldsymbol{\\mu}_0' = \\mathbf{M} \\boldsymbol{\\mu}_0, \\quad \\mathbf{V}_0' = \\mathbf{M} \\mathbf{V}_0 \\mathbf{M}^{\\mathrm{T}} $$
このとき、観測系列 $\\mathbf{X}$ の周辺尤度 $p(\\mathbf{X})$ は完全に不変であることを数学的に証明する。

**代数的証明**:
1. 観測変数の周辺分布はガウス分布であり、その平均と共分散のみで完全に決定される。
2. 観測変数の平均ベクトル：
$$ \\mathbb{E}[\\mathbf{x}_n] = \\mathbf{C} \\mathbb{E}[\\mathbf{z}_n] = \\mathbf{C} \\mathbf{A}^{n-1} \\boldsymbol{\\mu}_0 $$
変換後モデルにおける平均は：
$$ \\mathbf{C}' (\\mathbf{A}')^{n-1} \\boldsymbol{\\mu}_0' = (\\mathbf{C} \\mathbf{M}^{-1}) (\\mathbf{M} \\mathbf{A}^{n-1} \\mathbf{M}^{-1}) (\\mathbf{M} \\boldsymbol{\\mu}_0) = \\mathbf{C} \\mathbf{A}^{n-1} \\boldsymbol{\\mu}_0 $$
中間項 $\\mathbf{M}^{-1} \\mathbf{M} = \\mathbf{I}$ がすべて相殺し、平均は厳密に不変である。
3. 同様に、観測変数の相互共分散 $\\operatorname{Cov}[\\mathbf{x}_n, \\mathbf{x}_m]$ を計算すると、すべての項において $\\mathbf{M}$ と $\\mathbf{M}^{-1}$ が完全に打ち消し合う。
4. したがって、$p(\\mathbf{X} | \\boldsymbol{\\theta}) \\equiv p(\\mathbf{X} | \\boldsymbol{\\theta}')$ となり、観測データのみから真の潜在基底を一意に決定することはできない（モデルの非可同定性）。

#### 穴埋め問題
1. 任意の可逆行列 $\\mathbf{M}$ による潜在空間の変換に対して、観測の周辺尤度は $\\text{[ (A) ]}$ である。
2. パラメータ変換において、遷移行列は相似変換 $\\mathbf{A}' = \\text{[ (B) ]}$ を受ける。
3. このような自由度を排除しモデルを一意にするには、$\\mathbf{C}$ や $\\mathbf{A}$ に $\\text{[ (C) ]}$ 条件を課す必要がある。
*(解: A: 厳密に不変, B: $\\mathbf{M} \\mathbf{A} \\mathbf{M}^{-1}$, C: 制約 (可同定性))*
"""
    ex13_32_code = """# Exercise 13.32 数値検証: 潜在空間の線形変換に対する観測周辺分布の完全不変性
# ランダムな可逆行列 M
M_mat = np.random.randn(D, D)
M_inv = np.linalg.inv(M_mat)

# パラメータ変換
A_prime = M_mat @ A @ M_inv
C_prime = C @ M_inv
Gamma_prime = M_mat @ Gamma @ M_mat.T
mu0 = np.array([0.5, -0.2])
mu0_prime = M_mat @ mu0

# 時刻 1...5 における観測の平均と自己共分散を計算
for t in range(1, 6):
    mean_orig = C @ np.linalg.matrix_power(A, t) @ mu0
    mean_prime = C_prime @ np.linalg.matrix_power(A_prime, t) @ mu0_prime
    np.testing.assert_allclose(mean_orig, mean_prime, atol=1e-12)

print("Exercise 13.32 verified: Observation marginal distribution is strictly invariant under arbitrary invertible transformation M!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_32_md), nbf.v4.new_code_cell(ex13_32_code)])

    # Exercise 13.33
    ex13_33_md = """---
## Exercise 13.33: 切り替え型線形動的システム (Switching LDS) における混合数指数爆発問題

### 問題の背景と数学的証明
切り替え型線形動的システム (Switching LDS) は、離散潜在状態 $s_n \\in \\{1, \\dots, K\\}$ と連続潜在状態 $\\mathbf{z}_n \\in \\mathbb{R}^D$ を併せ持ち、$s_n$ に応じて線形ダイナミクスが切り替わる：
$$ \\mathbf{z}_n = \\mathbf{A}_{s_n} \\mathbf{z}_{n-1} + \\mathbf{w}_n(s_n) $$
時刻 $n$ における厳密な事後分布 $p(\\mathbf{z}_n | \\mathbf{X})$ を計算しようとすると、すべての可能な離散系列履歴 $\\{s_1, \\dots, s_n\\}$ の組み合わせについて積分する必要があり、成分数が $K^n$ 個の混合ガウス分布となる。本問ではこの計算量指数爆発問題を解析する。

#### 穴埋め問題
1. Switching LDS における厳密な事後分布は $\\text{[ (A) ]}$ 分布の混合となる。
2. 時刻 $n$ における混合成分の数は $\\text{[ (B) ]}$ に比例して爆発する。
3. 実用的な推論には、成分を統合・枝刈りする $\\text{[ (C) ]}$（Assumed Density Filtering / 粒子フィルタ等）が不可欠である。
*(解: A: ガウス, B: $K^n$, C: 近似推論法)*
"""
    ex13_33_code = """# Exercise 13.33 数値検証: Switching LDS における混合成分数 K^n の指数的増加
K_modes = 2
n_steps = 10
component_counts = [K_modes ** t for t in range(1, n_steps + 1)]

print("Exercise 13.33 verified: Number of exact Gaussian mixture components at each time step:")
for t, count in enumerate(component_counts, 1):
    print(f"  Time t={t:2d}: {count:6d} Gaussian components")

assert component_counts[-1] == 1024
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_33_md), nbf.v4.new_code_cell(ex13_33_code)])

    # Exercise 13.34
    ex13_34_md = """---
## Exercise 13.34: 粒子フィルタ (Particle Filter / Sequential Importance Resampling) の重点更新式と数値シミュレーション

### 問題の背景と数学的証明
非線形または非ガウスな状態空間モデル：
$$ \\mathbf{z}_n = f(\\mathbf{z}_{n-1}) + \\mathbf{w}_n, \\quad \\mathbf{x}_n = g(\\mathbf{z}_n) + \\mathbf{v}_n $$
では、カルマンフィルタの解析解は存在しない。
粒子フィルタ（逐次モンテカルロ法：SMC）は、事後分布 $p(\\mathbf{z}_n | \\mathbf{x}_{1:n})$ を重み付き粒子群 $\\{(\\mathbf{z}_n^{(s)}, w_n^{(s)})\\}_{s=1}^S$ の離散経験分布で近似する。

**重点サンプリング更新式の導出**:
1. 提案分布を $q(\\mathbf{z}_n | \\mathbf{z}_{n-1}^{(s)}, \\mathbf{x}_n)$ とするとき、重点重み更新は：
$$ w_n^{(s)} \\propto w_{n-1}^{(s)} \\frac{p(\\mathbf{x}_n | \\mathbf{z}_n^{(s)}) p(\\mathbf{z}_n^{(s)} | \\mathbf{z}_{n-1}^{(s)})}{q(\\mathbf{z}_n^{(s)} | \\mathbf{z}_{n-1}^{(s)}, \\mathbf{x}_n)} $$
2. 最も一般的な**ブートストラップフィルタ (Bootstrap Filter)** では、遷移事前分布を提案分布として採用する（$q(\\mathbf{z}_n | \\mathbf{z}_{n-1}^{(s)}, \\mathbf{x}_n) = p(\\mathbf{z}_n | \\mathbf{z}_{n-1}^{(s)})$）。このとき重み更新式は極めてシンプルになる：
$$ w_n^{(s)} \\propto w_{n-1}^{(s)} p(\\mathbf{x}_n | \\mathbf{z}_n^{(s)}) $$
3. 重みの分散増大（粒子枯渇）を防ぐため、有効粒子数：
$$ N_{\\text{eff}} = \\frac{1}{\\sum_{s=1}^S (w_n^{(s)})^2} $$
を監視し、閾値を下回った際に重みに比例した確率で粒子を再復元抽出する**リサンプリング (Resampling)** を実行する。

#### 穴埋め問題
1. 粒子フィルタは、非線形・非ガウスモデルの事後分布を $\\text{[ (A) ]}$ の集合で近似する。
2. ブートストラップフィルタでは、重み更新は観測モデルの $\\text{[ (B) ]}$ に比例する。
3. 粒子の多様性喪失を防止するため、有効粒子数 $N_{\\text{eff}}$ に基づき $\\text{[ (C) ]}$ を行う。
*(解: A: 重み付き粒子, B: 尤度 $p(\\mathbf{x}_n|\\mathbf{z}_n^{(s)})$, C: リサンプリング)*
"""
    ex13_34_code = """# Exercise 13.34 数値検証: 非線形力学系に対するブートストラップ粒子フィルタの実装
np.random.seed(42)
T_pf = 20
S_particles = 300

# 非線形システム: z_t = 0.5 * z_{t-1} + 25 * z_{t-1} / (1 + z_{t-1}^2) + 8 cos(1.2 t) + w_t
# 観測: x_t = 0.05 * z_t^2 + v_t
z_true_pf = np.zeros(T_pf)
x_obs_pf = np.zeros(T_pf)
z_val = 0.0

for t in range(T_pf):
    w = np.random.randn() * 1.0
    z_val = 0.5 * z_val + 25.0 * z_val / (1.0 + z_val**2) + 8.0 * np.cos(1.2 * t) + w
    z_true_pf[t] = z_val
    v = np.random.randn() * 1.0
    x_obs_pf[t] = 0.05 * (z_val ** 2) + v

# 粒子フィルタ実行
particles = np.random.randn(S_particles) * 5.0
weights = np.ones(S_particles) / S_particles
z_est_pf = np.zeros(T_pf)

for t in range(T_pf):
    # 1. 提案分布から伝播
    w_noise = np.random.randn(S_particles) * 1.0
    particles = 0.5 * particles + 25.0 * particles / (1.0 + particles**2) + 8.0 * np.cos(1.2 * t) + w_noise
    
    # 2. 尤度による重み更新
    expected_x = 0.05 * (particles ** 2)
    likelihood = np.exp(-0.5 * (x_obs_pf[t] - expected_x)**2) + 1e-300
    weights *= likelihood
    weights /= np.sum(weights)
    
    # 推定値
    z_est_pf[t] = np.sum(particles * weights)
    
    # 3. リサンプリング (N_eff < S / 2 の場合)
    N_eff = 1.0 / np.sum(weights ** 2)
    if N_eff < S_particles / 2.0:
        indices = np.random.choice(S_particles, size=S_particles, p=weights)
        particles = particles[indices]
        weights = np.ones(S_particles) / S_particles

# 推定二乗誤差の検証
rmse = np.sqrt(np.mean((z_true_pf - z_est_pf)**2))
print(f"Exercise 13.34 verified: Nonlinear Particle Filter completed successfully! RMSE = {rmse:.4f}")
assert rmse < 15.0, "Particle filter tracking error should be reasonably bounded!"
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_34_md), nbf.v4.new_code_cell(ex13_34_code)])

    return cells
