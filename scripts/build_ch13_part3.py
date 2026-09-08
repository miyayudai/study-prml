# scripts/build_ch13_part3.py
"""
Build Chapter 13 Exercises: Part 3 (Exercises 13.11 - 13.15)
HMM Posterior Marginals, Scaling Factors, Factor Graphs, and Autoregressive HMM
"""

import nbformat as nbf

def build_part3():
    """Exercises 13.11 - 13.15: HMM Advanced Inference & Extensions"""
    cells = []

    # Exercise 13.11
    ex13_11_md = """---
## Exercise 13.11: 潜在事後周辺分布 $\\gamma(z_n)$ と $\\alpha, \\beta$ の積関係および周辺尤度 $p(\\mathbf{X})$ の時間不変性

### 問題の背景と数学的証明
潜在状態 $z_n$ の事後周辺確率 $\\gamma(z_n) \\equiv p(z_n | \\mathbf{X})$ は、前向き変数 $\\alpha(z_n)$ と後ろ向き変数 $\\beta(z_n)$ の積によって表現される（(13.33)式）：
$$ \\gamma(z_n) = \\frac{p(\\mathbf{X}, z_n)}{p(\\mathbf{X})} = \\frac{\\alpha(z_n)\\beta(z_n)}{p(\\mathbf{X})} $$
本問では、全観測系列の周辺尤度 $p(\\mathbf{X})$ が、任意の時刻 $n \\in \\{1, \\dots, N\\}$ において：
$$ p(\\mathbf{X}) = \\sum_{z_n} \\alpha(z_n) \\beta(z_n) $$
として計算でき、時刻 $n$ の選択に依らず一定値（不変）であることを数学的に証明する。

**代数的証明**:
1. 時刻 $n$ における積の和を定義する：
$$ S_n \\equiv \\sum_{z_n} \\alpha(z_n) \\beta(z_n) $$
2. $\\beta(z_n)$ に後ろ向き再帰式 $\\beta(z_n) = \\sum_{z_{n+1}} \\beta(z_{n+1}) p(\\mathbf{x}_{n+1} | z_{n+1}) p(z_{n+1} | z_n)$ を代入する：
$$ S_n = \\sum_{z_n} \\alpha(z_n) \\left[ \\sum_{z_{n+1}} \\beta(z_{n+1}) p(\\mathbf{x}_{n+1} | z_{n+1}) p(z_{n+1} | z_n) \\right] $$
3. 和の順序を交換する：
$$ S_n = \\sum_{z_{n+1}} \\beta(z_{n+1}) p(\\mathbf{x}_{n+1} | z_{n+1}) \\left[ \\sum_{z_n} \\alpha(z_n) p(z_{n+1} | z_n) \\right] $$
4. ここで、大括弧内の項は前向き再帰式 $\\alpha(z_{n+1}) = p(\\mathbf{x}_{n+1} | z_{n+1}) \\sum_{z_n} \\alpha(z_n) p(z_{n+1} | z_n)$ の一部である。両者を掛け合わせると：
$$ p(\\mathbf{x}_{n+1} | z_{n+1}) \\left[ \\sum_{z_n} \\alpha(z_n) p(z_{n+1} | z_n) \\right] = \\alpha(z_{n+1}) $$
5. したがって：
$$ S_n = \\sum_{z_{n+1}} \\beta(z_{n+1}) \\alpha(z_{n+1}) = S_{n+1} $$
6. 任意の $n$ について $S_n = S_{n+1}$ であるため、$S_1 = S_2 = \\dots = S_N = p(\\mathbf{X})$ が証明された。特に $n=N$ では $\\beta(z_N) = 1$ より $p(\\mathbf{X}) = \\sum_{z_N} \\alpha(z_N)$ となる。

#### 穴埋め問題
1. 時刻 $n$ における結合確率 $p(\\mathbf{X}, z_n)$ は $\\text{[ (A) ]}$ の積で与えられる。
2. 後ろ向き再帰式を代入して和を交換すると、前向き再帰式が現れ、$S_n = \\text{[ (B) ]}$ が示される。
3. これにより、系列の任意の「断面」において $\\sum_{z_n} \\alpha(z_n)\\beta(z_n)$ は常に $\\text{[ (C) ]}$ に等しい。
*(解: A: $\\alpha(z_n) \\beta(z_n)$, B: $S_{n+1}$, C: 周辺尤度 $p(\\mathbf{X})$)*
"""
    ex13_11_code = """# Exercise 13.11 数値検証: 各時点 n における sum_k alpha_n(k) * beta_n(k) の完全時間不変性
p_X_list = []
for n in range(N_steps):
    p_X_n = np.sum(alpha[n, :] * beta[n, :])
    p_X_list.append(p_X_n)

# すべての時刻 n で p(X) が浮動小数点精度で完全に同一であることを検証
for n in range(1, N_steps):
    np.testing.assert_allclose(p_X_list[n], p_X_list[0], atol=1e-12)

# 事後周辺確率 gamma の計算
gamma_computed = (alpha * beta) / p_X_list[0]
# 各時刻の確率の和が 1 であることを検証
np.testing.assert_allclose(np.sum(gamma_computed, axis=1), np.ones(N_steps), atol=1e-12)

print(f"Exercise 13.11 verified: p(X) is strictly invariant across all n = 1...{N_steps}:")
for n, val in enumerate(p_X_list):
    print(f"  Time n={n+1}: p(X) = {val:.10e}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_11_md), nbf.v4.new_code_cell(ex13_11_code)])

    # Exercise 13.12
    ex13_12_md = """---
## Exercise 13.12: 2時点同時事後分布 $\\xi(z_{n-1}, z_n)$ の代数導出と周辺化整合性

### 問題の背景と数学的証明
遷移パラメータの学習に必要な2時点連続潜在状態の事後同時分布（(13.43)式）：
$$ \\xi(z_{n-1}, z_n) \\equiv p(z_{n-1}, z_n | \\mathbf{X}) $$
を、前向き変数 $\\alpha$、後ろ向き変数 $\\beta$、遷移確率 $A$、放出確率を用いて表現し、周辺化整合性：
$$ \\sum_{z_{n-1}} \\xi(z_{n-1}, z_n) = \\gamma(z_n), \\quad \\sum_{z_n} \\xi(z_{n-1}, z_n) = \\gamma(z_{n-1}) $$
を代数的に証明する。

**代数的導出**:
1. 条件付き確率の定義と因数分解より：
$$ p(z_{n-1}, z_n | \\mathbf{X}) = \\frac{p(\\mathbf{X}, z_{n-1}, z_n)}{p(\\mathbf{X})} $$
2. 分子の同時確率をグラフィカルモデルの因数分解構造に従って展開する：
$$ p(\\mathbf{X}, z_{n-1}, z_n) = p(\\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_{n-1}) \\cdot p(z_n | z_{n-1}) \\cdot p(\\mathbf{x}_n | z_n) \\cdot p(\\mathbf{x}_{n+1}, \\dots, \\mathbf{x}_N | z_n) $$
3. 定義を当てはめると：
$$ p(\\mathbf{X}, z_{n-1}, z_n) = \\alpha(z_{n-1}) p(z_n | z_{n-1}) p(\\mathbf{x}_n | z_n) \\beta(z_n) $$
4. したがって：
$$ \\xi(z_{n-1}, z_n) = \\frac{\\alpha(z_{n-1}) p(z_n | z_{n-1}) p(\\mathbf{x}_n | z_n) \\beta(z_n)}{p(\\mathbf{X})} $$
5. **周辺化整合性の証明**:
   - $z_{n-1}$ について和をとると：
   $$ \\sum_{z_{n-1}} p(\\mathbf{X}, z_{n-1}, z_n) = p(\\mathbf{x}_n | z_n) \\beta(z_n) \\left[ \\sum_{z_{n-1}} \\alpha(z_{n-1}) p(z_n | z_{n-1}) \\right] = \\alpha(z_n) \\beta(z_n) = p(\\mathbf{X}, z_n) $$
   よって $\\sum_{z_{n-1}} \\xi(z_{n-1}, z_n) = \\gamma(z_n)$。
   - $z_n$ について和をとると：
   $$ \\sum_{z_n} p(\\mathbf{X}, z_{n-1}, z_n) = \\alpha(z_{n-1}) \\left[ \\sum_{z_n} p(z_n | z_{n-1}) p(\\mathbf{x}_n | z_n) \\beta(z_n) \\right] = \\alpha(z_{n-1}) \\beta(z_{n-1}) = p(\\mathbf{X}, z_{n-1}) $$
   よって $\\sum_{z_n} \\xi(z_{n-1}, z_n) = \\gamma(z_{n-1})$。

#### 穴埋め問題
1. $\\xi(z_{n-1}, z_n)$ の分子は、過去 $\\alpha(z_{n-1})$、遷移確率、放出確率、未来 $\\text{[ (A) ]}$ の4つの因子の積となる。
2. $z_{n-1}$ に関して周辺化すると $\\text{[ (B) ]}$ に帰着する。
3. $z_n$ に関して周辺化すると $\\text{[ (C) ]}$ に帰着する。
*(解: A: $\\beta(z_n)$, B: $\\gamma(z_n)$, C: $\\gamma(z_{n-1})$)*
"""
    ex13_12_code = """# Exercise 13.12 数値検証: xi(z_{n-1}, z_n) の周辺化整合性
p_X_total = p_X_list[0]
xi_all = []

for n in range(1, N_steps):
    # xi(z_{n-1}, z_n): shape (K, K)
    xi_n = np.zeros((K_states, K_states))
    for j in range(K_states):
        for k in range(K_states):
            xi_n[j, k] = (alpha[n-1, j] * A_mat[j, k] * B_mat[k, obs_seq[n]] * beta[n, k]) / p_X_total
    xi_all.append(xi_n)
    
    # 整合性検証 1: sum over z_{n-1} (行和) -> gamma(z_n)
    gamma_curr_check = np.sum(xi_n, axis=0)
    np.testing.assert_allclose(gamma_curr_check, gamma_computed[n, :], atol=1e-12)
    
    # 整合性検証 2: sum over z_n (列和) -> gamma(z_{n-1})
    gamma_prev_check = np.sum(xi_n, axis=1)
    np.testing.assert_allclose(gamma_prev_check, gamma_computed[n-1, :], atol=1e-12)

print("Exercise 13.12 verified: Marginals of xi strictly match gamma(z_{n-1}) and gamma(z_n) at all time steps!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_12_md), nbf.v4.new_code_cell(ex13_12_code)])

    # Exercise 13.13
    ex13_13_md = """---
## Exercise 13.13: 数値的アンダーフロー防止のスケーリング係数 $c_n$ と正規化前向き・後ろ向き変数の証明

### 問題の背景と数学的証明
系列長 $N$ が大きくなると、前向き変数 $\\alpha(z_n)$ および後ろ向き変数 $\\beta(z_n)$ は確率の積の累積により指数関数的に小さくなり、浮動小数点のアンダーフローが発生する。
これを防ぐため、各ステップで正規化スケーリング係数（(13.56)式）：
$$ c_n \\equiv p(\\mathbf{x}_n | \\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}) $$
を導入し、正規化された前向き変数 $\\hat{\\alpha}(z_n) = p(z_n | \\mathbf{x}_1, \\dots, \\mathbf{x}_n)$ を計算する。

**数理的証明**:
1. 定義より、正規化前向き変数は：
$$ \\hat{\\alpha}(z_n) = \\frac{\\alpha(z_n)}{\\prod_{m=1}^n c_m} $$
を満たし、各時点で $\\sum_{z_n} \\hat{\\alpha}(z_n) = 1$ と規格化される。
2. スケーリング係数 $c_n$ は未正規化の前向き更新値の和として得られる：
$$ c_n = \\sum_{z_n} p(\\mathbf{x}_n | z_n) \\sum_{z_{n-1}} \\hat{\\alpha}(z_{n-1}) p(z_n | z_{n-1}) $$
3. 全観測データの対数尤度は、スケーリング係数の対数和としてアンダーフローなく極めて安定に計算できる：
$$ \\ln p(\\mathbf{X}) = \\ln \\prod_{n=1}^N c_n = \\sum_{n=1}^N \\ln c_n $$
4. 一方、後ろ向き変数に対しても同様のスケーリングを適用する：
$$ \\hat{\\beta}(z_n) = \\frac{\\beta(z_n)}{\\prod_{m=n+1}^N c_m} $$
再帰式は次のように簡潔になる：
$$ c_{n+1} \\hat{\\beta}(z_n) = \\sum_{z_{n+1}} \\hat{\\beta}(z_{n+1}) p(\\mathbf{x}_{n+1} | z_{n+1}) p(z_{n+1} | z_n) $$
5. 事後周辺分布は、スケーリング因子が完全に相殺して直接の積となる：
$$ \\gamma(z_n) = \\frac{\\alpha(z_n)\\beta(z_n)}{p(\\mathbf{X})} = \\frac{\\left(\\prod_{m=1}^n c_m \\hat{\\alpha}(z_n)\\right) \\left(\\prod_{m=n+1}^N c_m \\hat{\\beta}(z_n)\\right)}{\\prod_{m=1}^N c_m} = \\hat{\\alpha}(z_n) \\hat{\\beta}(z_n) $$

#### 穴埋め問題
1. 正規化前向き変数 $\\hat{\\alpha}(z_n)$ の各時刻での状態の和は常に $\\text{[ (A) ]}$ である。
2. 観測データの全対数尤度は $\\ln p(\\mathbf{X}) = \\text{[ (B) ]}$ により安定して求められる。
3. 事後周辺確率 $\\gamma(z_n)$ は、正規化された変数の単純な積 $\\text{[ (C) ]}$ として表される。
*(解: A: $1$, B: $\\sum_{n=1}^N \\ln c_n$, C: $\\hat{\\alpha}(z_n)\\hat{\\beta}(z_n)$)*
"""
    ex13_13_code = """# Exercise 13.13 数値検証: スケーリング版 Forward-Backward アルゴリズム
# 1. スケーリング版 Forward
alpha_hat = np.zeros((N_steps, K_states))
c = np.zeros(N_steps)

# n = 0
alpha_0_unnorm = pi_vec * B_mat[:, obs_seq[0]]
c[0] = np.sum(alpha_0_unnorm)
alpha_hat[0, :] = alpha_0_unnorm / c[0]

for n in range(1, N_steps):
    alpha_unnorm = B_mat[:, obs_seq[n]] * (alpha_hat[n-1, :] @ A_mat)
    c[n] = np.sum(alpha_unnorm)
    alpha_hat[n, :] = alpha_unnorm / c[n]

# 2. 対数尤度の一致確認
log_p_X_scaled = np.sum(np.log(c))
log_p_X_exact = np.log(p_X_total)
np.testing.assert_allclose(log_p_X_scaled, log_p_X_exact, atol=1e-12)

# 3. スケーリング版 Backward
beta_hat = np.zeros((N_steps, K_states))
beta_hat[-1, :] = 1.0

for n in range(N_steps - 2, -1, -1):
    beta_hat[n, :] = (A_mat @ (beta_hat[n+1, :] * B_mat[:, obs_seq[n+1]])) / c[n+1]

# 4. 事後周辺確率 gamma = alpha_hat * beta_hat の完全一致
gamma_scaled = alpha_hat * beta_hat
np.testing.assert_allclose(gamma_scaled, gamma_computed, atol=1e-12)

print(f"Exercise 13.13 verified: Log-likelihood strictly equals sum(log c_n): {log_p_X_scaled:.8f} == {log_p_X_exact:.8f}")
print("Gamma from scaled Forward-Backward perfectly matches unscaled calculations!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_13_md), nbf.v4.new_code_cell(ex13_13_code)])

    # Exercise 13.14
    ex13_14_md = """---
## Exercise 13.14: ファクターグラフにおける sum-product アルゴリズムと Forward-Backward の等価性

### 問題の背景と数学的証明
第8章で導入された確率的グラフィカルモデルの「sum-product アルゴリズム」を木構造（鎖状グラフ）である HMM に適用すると、従来の Forward-Backward アルゴリズムと完全に一致することを証明する。

**数理的導出**:
1. HMM の因子ノード分解：
$$ p(\\mathbf{X}, Z) = f_1(z_1) \\prod_{n=2}^N f_n(z_{n-1}, z_n) $$
ここで初期因子 $f_1(z_1) = p(z_1) p(\\mathbf{x}_1 | z_1)$、遷移・放出因子 $f_n(z_{n-1}, z_n) = p(z_n | z_{n-1}) p(\\mathbf{x}_n | z_n)$ である。
2. **前向きメッセージパッシング**:
   変数ノードから因子ノードへのメッセージ $\\mu_{z_{n-1} \\to f_n}(z_{n-1})$ および因子ノードから変数ノードへのメッセージ $\\mu_{f_n \\to z_n}(z_n)$ は：
   $$ \\mu_{f_n \\to z_n}(z_n) = \\sum_{z_{n-1}} f_n(z_{n-1}, z_n) \\mu_{z_{n-1} \\to f_n}(z_{n-1}) = p(\\mathbf{x}_n | z_n) \\sum_{z_{n-1}} p(z_n | z_{n-1}) \\mu_{f_{n-1} \\to z_{n-1}}(z_{n-1}) $$
   これは前向き変数 $\\alpha(z_n)$ の再帰漸化式と全く同一である：
   $$ \\mu_{f_n \\to z_n}(z_n) \\equiv \\alpha(z_n) $$
3. **後ろ向きメッセージパッシング**:
   同様に、未来側の因子から変数ノード $z_n$ へ送られるメッセージは：
   $$ \\mu_{f_{n+1} \\to z_n}(z_n) = \\sum_{z_{n+1}} f_{n+1}(z_n, z_{n+1}) \\mu_{f_{n+2} \\to z_{n+1}}(z_{n+1}) = \\sum_{z_{n+1}} p(z_{n+1} | z_n) p(\\mathbf{x}_{n+1} | z_{n+1}) \\mu_{f_{n+2} \\to z_{n+1}}(z_{n+1}) $$
   これは後ろ向き変数 $\\beta(z_n)$ の漸化式と全く同一である：
   $$ \\mu_{f_{n+1} \\to z_n}(z_n) \\equiv \\beta(z_n) $$
4. したがって、Forward-Backward アルゴリズムは、鎖状ファクターグラフに対する sum-product アルゴリズムの厳密な具現化である。

#### 穴埋め問題
1. HMM の鎖状グラフにおけるファクターグラフ表現では、変数ノード $z_n$ の間に $\\text{[ (A) ]}$ ノードが配置される。
2. 過去から未来へ流れるメッセージは $\\text{[ (B) ]}$ に対応する。
3. 未来から過去へ流れるメッセージは $\\text{[ (C) ]}$ に対応する。
*(解: A: 因子 (ファクター), B: 前向き変数 $\\alpha(z_n)$, C: 後ろ向き変数 $\\beta(z_n)$)*
"""
    ex13_14_code = """# Exercise 13.14 数値検証: ファクターグラフのメッセージパッシング実装
# 因子関数 f_1(z_1)
f1 = pi_vec * B_mat[:, obs_seq[0]]

# 因子関数 f_n(z_{n-1}, z_n)
def fn(n_idx):
    return A_mat * B_mat[:, obs_seq[n_idx]][None, :]

# メッセージ mu_f_to_z (前向き)
mu_f_to_z = [None] * N_steps
mu_f_to_z[0] = f1
for n in range(1, N_steps):
    # mu_{f_n -> z_n} = sum_{z_{n-1}} f_n(z_{n-1}, z_n) * mu_{f_{n-1} -> z_{n-1}}
    mu_f_to_z[n] = np.sum(fn(n) * mu_f_to_z[n-1][:, None], axis=0)

# alpha とメッセージの一致検証
np.testing.assert_allclose(np.array(mu_f_to_z), alpha, atol=1e-12)
print("Exercise 13.14 verified: Sum-product factor messages strictly coincide with alpha(z_n)!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_14_md), nbf.v4.new_code_cell(ex13_14_code)])

    # Exercise 13.15
    ex13_15_md = """---
## Exercise 13.15: 自己回帰隠れマルコフモデル (Autoregressive HMM)

### 問題の背景と数学的証明
自己回帰 HMM (AR-HMM) は、各観測変数 $\\mathbf{x}_n$ が潜在状態 $z_n$ だけでなく直前の観測変数 $\\mathbf{x}_{n-1}$ にも直接依存する拡張モデルである：
$$ p(\\mathbf{X}, Z) = p(z_1) p(\\mathbf{x}_1 | z_1) \\prod_{n=2}^N p(z_n | z_{n-1}) p(\\mathbf{x}_n | \\mathbf{x}_{n-1}, z_n) $$
放出分布を条件付き線形ガウスモデル：
$$ \\mathbf{x}_n = \\mathbf{W}_k \\mathbf{x}_{n-1} + \\mathbf{b}_k + \\boldsymbol{\\epsilon}_n, \\quad \\boldsymbol{\\epsilon}_n \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{\\Sigma}_k) $$
とした場合の EM アルゴリズムの M ステップ更新式を導出する。

**M-step 導出**:
1. 放出パラメータ $\\{\\mathbf{W}_k, \\mathbf{b}_k, \\mathbf{\\Sigma}_k\\}$ に依存する期待対数尤度項は：
$$ Q_{\\text{AR}} = -\\frac{1}{2} \\sum_{n=2}^N \\gamma(z_{nk}) \\left( \\ln |\\mathbf{\\Sigma}_k| + (\\mathbf{x}_n - \\mathbf{W}_k \\mathbf{x}_{n-1} - \\mathbf{b}_k)^{\\mathrm{T}} \\mathbf{\\Sigma}_k^{-1} (\\mathbf{x}_n - \\mathbf{W}_k \\mathbf{x}_{n-1} - \\mathbf{b}_k) \\right) $$
2. 拡張入力ベクトル $\\tilde{\\mathbf{x}}_{n-1} = (\\mathbf{x}_{n-1}^{\\mathrm{T}}, 1)^{\\mathrm{T}}$ および拡張回帰行列 $\\tilde{\\mathbf{W}}_k = [\\mathbf{W}_k, \\mathbf{b}_k]$ を定義すると：
$$ Q_{\\text{AR}} = -\\frac{1}{2} \\sum_{n=2}^N \\gamma(z_{nk}) \\left( \\ln |\\mathbf{\\Sigma}_k| + \\operatorname{Tr}\\left( \\mathbf{\\Sigma}_k^{-1} (\\mathbf{x}_n - \\tilde{\\mathbf{W}}_k \\tilde{\\mathbf{x}}_{n-1})(\\mathbf{x}_n - \\tilde{\\mathbf{W}}_k \\tilde{\\mathbf{x}}_{n-1})^{\\mathrm{T}} \\right) \\right) $$
3. $\\tilde{\\mathbf{W}}_k$ に関して微分してゼロとおく：
$$ \\frac{\\partial Q_{\\text{AR}}}{\\partial \\tilde{\\mathbf{W}}_k} = \\sum_{n=2}^N \\gamma(z_{nk}) \\mathbf{\\Sigma}_k^{-1} (\\mathbf{x}_n - \\tilde{\\mathbf{W}}_k \\tilde{\\mathbf{x}}_{n-1}) \\tilde{\\mathbf{x}}_{n-1}^{\\mathrm{T}} = \\mathbf{0} $$
4. したがって、重み付き最小二乗推定量：
$$ \\tilde{\\mathbf{W}}_k = \\left( \\sum_{n=2}^N \\gamma(z_{nk}) \\mathbf{x}_n \\tilde{\\mathbf{x}}_{n-1}^{\\mathrm{T}} \\right) \\left( \\sum_{n=2}^N \\gamma(z_{nk}) \\tilde{\\mathbf{x}}_{n-1} \\tilde{\\mathbf{x}}_{n-1}^{\\mathrm{T}} \\right)^{-1} $$
が得られる。

#### 穴埋め問題
1. 自己回帰 HMM では、放出確率は現在の潜在状態 $z_n$ と $\\text{[ (A) ]}$ の両方に依存する。
2. 回帰パラメータの最適解は、負担率 $\\gamma(z_{nk})$ を重みとする $\\text{[ (B) ]}$ に帰着する。
*(解: A: 直前の観測 $\\mathbf{x}_{n-1}$, B: 重み付き最小二乗推定量)*
"""
    ex13_15_code = """# Exercise 13.15 数値検証: 自己回帰 HMM の重み付き最小二乗更新
# 1次元 AR(1) 放出モデル: x_n = w_k * x_{n-1} + b_k + noise
N_ar = 40
x_ar = np.zeros(N_ar)
for n in range(1, N_ar):
    x_ar[n] = 0.8 * x_ar[n-1] + np.random.randn() * 0.2

gamma_k_ar = np.random.uniform(0.2, 0.8, size=N_ar)

# 拡張入力: X_tilde = [x_{n-1}, 1]
X_tilde = np.column_stack([x_ar[:-1], np.ones(N_ar - 1)])  # (N-1, 2)
Y_target = x_ar[1:]                                       # (N-1,)
weights = gamma_k_ar[1:]

# 解析的更新式
W_analytical = np.linalg.solve(
    X_tilde.T @ np.diag(weights) @ X_tilde,
    X_tilde.T @ np.diag(weights) @ Y_target
)

# 二乗誤差の重み付き和を数値最小化して一致確認
def wls_loss(params):
    pred = X_tilde @ params
    return 0.5 * np.sum(weights * (Y_target - pred)**2)

res_ar = minimize(wls_loss, [0.0, 0.0])
np.testing.assert_allclose(W_analytical, res_ar.x, atol=1e-5)

print(f"Exercise 13.15 verified: AR-HMM analytical WLS parameters {W_analytical} match numerical minimum {res_ar.x}!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_15_md), nbf.v4.new_code_cell(ex13_15_code)])

    return cells
