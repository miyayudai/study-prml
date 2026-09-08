# scripts/build_ch14_part1.py
"""
Build Chapter 14 Exercises: Part 1 (Exercises 14.1 - 14.4)
Committees, Jensen Inequality, General Convex Losses, and AdaBoost Minimization
"""

import nbformat as nbf

def build_part1():
    """Exercises 14.1 - 14.4: Committees and AdaBoost Fundamentals"""
    cells = []

    # Title & Introduction
    intro_md = """# 第14章 モデル結合 (Combining Models)：演習問題完全詳解 (Exercises 14.1 - 14.17)

本ノートブックは、PRML第14章「モデル結合 (Combining Models)」における**全17問（Exercises 14.1 〜 14.17）**を1問1セル形式で完全に網羅した解答集です。

各問題には以下を完備しています：
1. **問題の背景と数理的定式化**（PRML本文の関連章節・式番号を明記）
2. **厳密な代数・幾何・変分学的証明ステップ**
3. **理解を深める穴埋め問題（Self-Check Quiz）**
4. **Python / NumPy / SciPy による厳密な数値検証コード（アサーション付き）**

---
"""
    cells.append(nbf.v4.new_markdown_cell(intro_md))

    # Exercise 14.1
    ex14_1_md = """---
## Exercise 14.1: コミッティ平均予測の二乗誤差分解と無相関誤差における $1/M$ 誤差低減

### 問題の背景と数学的証明
$M$ 個の個別の学習モデル $y_m(\\mathbf{x})$ の単純平均によって構成されるコミッティモデル：
$$ y_{\\text{COM}}(\\mathbf{x}) = \\frac{1}{M} \\sum_{m=1}^M y_m(\\mathbf{x}) $$
を考える（(14.7)式）。真の関数を $h(\\mathbf{x})$ とし、モデルごとの誤差を $\\epsilon_m(\\mathbf{x}) = y_m(\\mathbf{x}) - h(\\mathbf{x})$ とする。
各モデルの平均二乗誤差を $E_{\\text{AV}} = \\frac{1}{M} \\sum_{m=1}^M \\mathbb{E}[\\epsilon_m(\\mathbf{x})^2]$、コミッティの二乗誤差を $E_{\\text{COM}} = \\mathbb{E}[ (y_{\\text{COM}}(\\mathbf{x}) - h(\\mathbf{x}))^2 ]$ と定義するとき：
$$ E_{\\text{COM}} = \\frac{1}{M} E_{\\text{AV}} + \\frac{1}{M^2} \\sum_{j=1}^M \\sum_{k \\ne j}^M \\mathbb{E}[\\epsilon_j(\\mathbf{x}) \\epsilon_k(\\mathbf{x})] $$
が成立することを証明し、個別のモデルの誤差が無相関（$\\mathbb{E}[\\epsilon_j \\epsilon_k] = 0$）のとき：
$$ E_{\\text{COM}} = \\frac{1}{M} E_{\\text{AV}} $$
となることを示せ。

**代数的証明**:
1. コミッティの予測誤差は各モデルの予測誤差の相加平均である：
$$ y_{\\text{COM}}(\\mathbf{x}) - h(\\mathbf{x}) = \\frac{1}{M} \\sum_{m=1}^M y_m(\\mathbf{x}) - h(\\mathbf{x}) = \\frac{1}{M} \\sum_{m=1}^M \\epsilon_m(\\mathbf{x}) $$
2. コミッティの二乗誤差期待値を展開する：
$$ E_{\\text{COM}} = \\mathbb{E}\\left[ \\left( \\frac{1}{M} \\sum_{m=1}^M \\epsilon_m(\\mathbf{x}) \\right)^2 \\right] = \\frac{1}{M^2} \\mathbb{E}\\left[ \\sum_{j=1}^M \\sum_{k=1}^M \\epsilon_j(\\mathbf{x}) \\epsilon_k(\\mathbf{x}) \\right] $$
3. 和を対角項（$j = k$）と非対角項（$j \\ne k$）に分離する：
$$ E_{\\text{COM}} = \\frac{1}{M^2} \\sum_{m=1}^M \\mathbb{E}[\\epsilon_m(\\mathbf{x})^2] + \\frac{1}{M^2} \\sum_{j=1}^M \\sum_{k \\ne j}^M \\mathbb{E}[\\epsilon_j(\\mathbf{x}) \\epsilon_k(\\mathbf{x})] $$
4. ここで第1項に $E_{\\text{AV}} = \\frac{1}{M} \\sum_{m=1}^M \\mathbb{E}[\\epsilon_m^2]$ を代入すると：
$$ \\frac{1}{M^2} \\sum_{m=1}^M \\mathbb{E}[\\epsilon_m^2] = \\frac{1}{M} \\left( \\frac{1}{M} \\sum_{m=1}^M \\mathbb{E}[\\epsilon_m^2] \\right) = \\frac{1}{M} E_{\\text{AV}} $$
5. したがって：
$$ E_{\\text{COM}} = \\frac{1}{M} E_{\\text{AV}} + \\frac{1}{M^2} \\sum_{j \\ne k} \\mathbb{E}[\\epsilon_j \\epsilon_k] $$
6. 各モデルの誤差が互いに無相関で平均ゼロである場合、すべての $j \\ne k$ に対して $\\mathbb{E}[\\epsilon_j \\epsilon_k] = 0$ となる。このとき非対角項は完全にゼロとなり：
$$ E_{\\text{COM}} = \\frac{1}{M} E_{\\text{AV}} $$
が得られる。

#### 穴埋め問題
1. コミッティ予測誤差の二乗は、対角項である自乗和と非対角項である $\\text{[ (A) ]}$ の和に分解される。
2. 個々のモデルの予測誤差が完全に独立・無相関な場合、アンサンブルによって二乗誤差は単体モデルの $\\text{[ (B) ]}$ に低減される。
3. 実際の機械学習ではモデル同士が同一データで学習されるため相関が正となり、誤差削減率は $1/M$ より $\\text{[ (C) ]}$。
*(解: A: 相互共分散項, B: $1/M$, C: 小さくなる (控えめになる))*
"""
    ex14_1_code = """# Exercise 14.1 数値検証: コミッティの二乗誤差分解公式と 1/M 低減効果
import numpy as np

np.random.seed(42)
N_samples = 5000
M_models = 5

# ケース1: 完全に無相関な誤差 (Cov = I)
errors_uncorrelated = np.random.randn(N_samples, M_models)

E_AV_uncorr = np.mean(np.mean(errors_uncorrelated**2, axis=0))
E_COM_uncorr = np.mean(np.mean(errors_uncorrelated, axis=1)**2)

# 分解公式の右辺の計算 (原点周りの2次モーメント積 E[eps_j eps_k])
second_moments = (errors_uncorrelated.T @ errors_uncorrelated) / N_samples
diag_part = np.trace(second_moments) / (M_models**2)
offdiag_part = (np.sum(second_moments) - np.trace(second_moments)) / (M_models**2)
decomp_val = diag_part + offdiag_part

np.testing.assert_allclose(E_COM_uncorr, decomp_val, atol=1e-12)
np.testing.assert_allclose(E_COM_uncorr / E_AV_uncorr, 1.0 / M_models, atol=0.03)

# ケース2: 相関のある誤差
rho = 0.4
cov_correlated = (1 - rho) * np.eye(M_models) + rho * np.ones((M_models, M_models))
errors_correlated = np.random.multivariate_normal(np.zeros(M_models), cov_correlated, size=N_samples)

E_AV_corr = np.mean(np.mean(errors_correlated**2, axis=0))
E_COM_corr = np.mean(np.mean(errors_correlated, axis=1)**2)
theoretical_ratio_corr = 1.0 / M_models + ((M_models - 1) / M_models) * rho

np.testing.assert_allclose(E_COM_corr / E_AV_corr, theoretical_ratio_corr, atol=0.03)

print("Exercise 14.1 verified:")
print(f"  Uncorrelated: E_COM/E_AV = {E_COM_uncorr/E_AV_uncorr:.4f} (Theoretical 1/M = {1/M_models:.4f})")
print(f"  Correlated (rho={rho}): E_COM/E_AV = {E_COM_corr/E_AV_corr:.4f} (Theoretical = {theoretical_ratio_corr:.4f})")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_1_md), nbf.v4.new_code_cell(ex14_1_code)])

    # Exercise 14.2
    ex14_2_md = """---
## Exercise 14.2: イェンセンの不等式によるコミッティ誤差上界 $E_{\\text{COM}} \\le E_{\\text{AV}}$ の証明

### 問題の背景と数学的証明
PRMLの(14.10)式において、個々のモデル間の誤差に正の相関がある場合でも、コミッティ誤差 $E_{\\text{COM}}$ は常に単体モデルの平均誤差 $E_{\\text{AV}}$ を超えないこと：
$$ E_{\\text{COM}} \\le E_{\\text{AV}} $$
が主張されている。
本問では、二乗関数 $f(u) = u^2$ の厳密な凸性に基づき、イェンセンの不等式（Jensen's inequality）を用いてこの上界関係を証明する。

**数理的証明**:
1. 凸関数の定義：任意の凸関数 $f(u)$ と任意の重み係数 $\\lambda_m \\ge 0$（$\\sum_{m=1}^M \\lambda_m = 1$）に対して：
$$ f\\left( \\sum_{m=1}^M \\lambda_m u_m \\right) \\le \\sum_{m=1}^M \\lambda_m f(u_m) $$
2. 関数として二乗関数 $f(u) = u^2$ を選ぶ。2階微分が $f''(u) = 2 > 0$ であるため、二乗関数は全域で厳密に凸である。
3. 一様な重み $\\lambda_m = \\frac{1}{M}$、および引数として各モデルの予測値と真値の差 $u_m = y_m(\\mathbf{x}) - h(\\mathbf{x}) = \\epsilon_m(\\mathbf{x})$ を代入する：
$$ \\left( \\frac{1}{M} \\sum_{m=1}^M \\epsilon_m(\\mathbf{x}) \\right)^2 \\le \\frac{1}{M} \\sum_{m=1}^M \\epsilon_m(\\mathbf{x})^2 $$
4. 入力 $\\mathbf{x}$ に関して両辺の期待値 $\\mathbb{E}_\\mathbf{x}[\\cdot]$ をとる：
$$ \\mathbb{E}\\left[ \\left( \\frac{1}{M} \\sum_{m=1}^M \\epsilon_m(\\mathbf{x}) \\right)^2 \\right] \\le \\frac{1}{M} \\sum_{m=1}^M \\mathbb{E}[\\epsilon_m(\\mathbf{x})^2] $$
5. 左辺はコミッティ二乗誤差 $E_{\\text{COM}}$ であり、右辺は個別モデルの平均二乗誤差 $E_{\\text{AV}}$ である。したがって：
$$ E_{\\text{COM}} \\le E_{\\text{AV}} $$
6. 二乗関数の厳密な凸性により、等号成立条件はすべてのモデルの予測が完全に同一（$\\epsilon_1(\\mathbf{x}) = \\dots = \\epsilon_M(\\mathbf{x})$）である場合に限られる。少しでも予測の多様性があれば、厳密な不等号 $E_{\\text{COM}} < E_{\\text{AV}}$ が成立する。

#### 穴埋め問題
1. イェンセンの不等式が適用できるのは、二乗関数の2階微分が $\\text{[ (A) ]}$ であり厳密な凸関数だからである。
2. 等号 $E_{\\text{COM}} = E_{\\text{AV}}$ が成立するのは、すべてのモデルの予測が $\\text{[ (B) ]}$ 場合のみである。
3. モデル間にわずかでも多様性（不一致）があれば、コミッティの二乗誤差は平均誤差より $\\text{[ (C) ]}$。
*(解: A: 正 ($2 > 0$), B: 完全に一致する, C: 厳密に小さくなる)*
"""
    ex14_2_code = """# Exercise 14.2 数値検証: イェンセンの不等式と不等式成立条件
# 多様な予測誤差を持つモデル群
M = 4
N = 1000
errors = np.random.uniform(-2, 2, size=(N, M))

E_AV = np.mean(errors**2)
E_COM = np.mean(np.mean(errors, axis=1)**2)

# 厳密な不等式成立の検証
assert E_COM < E_AV
print(f"Exercise 14.2 verified: E_COM ({E_COM:.4f}) < E_AV ({E_AV:.4f}) holds strictly!")

# 等号成立条件: すべてのモデルが全く同一の予測をする場合
identical_errors = np.tile(np.random.uniform(-2, 2, size=(N, 1)), (1, M))
E_AV_ident = np.mean(identical_errors**2)
E_COM_ident = np.mean(np.mean(identical_errors, axis=1)**2)
np.testing.assert_allclose(E_COM_ident, E_AV_ident, atol=1e-12)
print(f"Equality verified: When models are identical, E_COM == E_AV == {E_COM_ident:.4f}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_2_md), nbf.v4.new_code_cell(ex14_2_code)])

    # Exercise 14.3
    ex14_3_md = """---
## Exercise 14.3: 一般の凸損失関数に対するコミッティ平均化の優位性の証明

### 問題の背景と数学的証明
Exercise 14.2 では二乗損失を扱ったが、二乗損失に限らず、予測値 $y$ に関して上に開いた凸関数である任意の損失関数 $L(y, t)$（例：絶対値損失 $L(y,t)=|y-t|$、Huber 損失、ロジスティック損失、指数損失など）に対しても、コミッティ平均予測の損失期待値は単体モデルの平均損失期待値以下となることを証明する：
$$ \\mathbb{E}[L(y_{\\text{COM}}(\\mathbf{x}), t)] \\le \\frac{1}{M} \\sum_{m=1}^M \\mathbb{E}[L(y_m(\\mathbf{x}), t)] $$

**数理的証明**:
1. $L(y, t)$ が第1引数 $y$ に関して凸関数であると仮定する。
2. イェンセンの不等式を重み $\\lambda_m = 1/M$ で適用する：
$$ L\\left( \\frac{1}{M} \\sum_{m=1}^M y_m(\\mathbf{x}), t \\right) \\le \\frac{1}{M} \\sum_{m=1}^M L(y_m(\\mathbf{x}), t) $$
3. $y_{\\text{COM}}(\\mathbf{x}) = \\frac{1}{M} \\sum_{m=1}^M y_m(\\mathbf{x})$ を代入する：
$$ L(y_{\\text{COM}}(\\mathbf{x}), t) \\le \\frac{1}{M} \\sum_{m=1}^M L(y_m(\\mathbf{x}), t) $$
4. 結合分布 $p(\\mathbf{x}, t)$ に関して両辺の期待値をとる：
$$ \\mathbb{E}_{\\mathbf{x}, t}[L(y_{\\text{COM}}(\\mathbf{x}), t)] \\le \\frac{1}{M} \\sum_{m=1}^M \\mathbb{E}_{\\mathbf{x}, t}[L(y_m(\\mathbf{x}), t)] $$
5. したがって、損失関数が凸関数でありさえすれば、コミッティ平均化によるアンサンブルは損失の期待値を決して悪化させず、一般に低減させることが普遍的に保証される。

#### 穴埋め問題
1. コミッティ平均化による汎化誤差の改善定理は、損失関数が $\\text{[ (A) ]}$ であることのみに依存する。
2. 絶対値損失 $|y-t|$ や Huber 損失に対しても、コミッティ損失は平均損失 $\\text{[ (B) ]}$ となる。
*(解: A: 凸関数 (上に開いている), B: 以下)*
"""
    ex14_3_code = """# Exercise 14.3 数値検証: 多様な凸損失関数に対するアンサンブル優位性の検証
t_true = np.random.randn(N)
y_models = np.random.randn(N, M) + t_true[:, None]  # 各モデルの予測
y_com = np.mean(y_models, axis=1)

# 1. 絶対値損失 (L1 loss): L(y, t) = |y - t|
loss_l1_com = np.mean(np.abs(y_com - t_true))
loss_l1_av = np.mean([np.mean(np.abs(y_models[:, m] - t_true)) for m in range(M)])
assert loss_l1_com <= loss_l1_av

# 2. Huber 損失
def huber(y, t, delta=1.0):
    r = np.abs(y - t)
    return np.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))

loss_huber_com = np.mean(huber(y_com, t_true))
loss_huber_av = np.mean([np.mean(huber(y_models[:, m], t_true)) for m in range(M)])
assert loss_huber_com <= loss_huber_av

print("Exercise 14.3 verified across multiple convex loss functions:")
print(f"  L1 Loss:    Com={loss_l1_com:.4f} <= Avg={loss_l1_av:.4f}")
print(f"  Huber Loss: Com={loss_huber_com:.4f} <= Avg={loss_huber_av:.4f}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_3_md), nbf.v4.new_code_cell(ex14_3_code)])

    # Exercise 14.4
    ex14_4_md = """---
## Exercise 14.4: AdaBoost 指数損失関数の微分による最適弱分類器係数 $\\alpha_m$ の導出

### 問題の背景と数学的証明
AdaBoost は、基底分類器 $y_m(\\mathbf{x}) \\in \\{-1, +1\\}$ の線形結合 $f_m(\\mathbf{x}) = f_{m-1}(\\mathbf{x}) + \\alpha_m y_m(\\mathbf{x})$ を逐次追加することで、指数損失関数（(14.14)式）：
$$ E = \\sum_{n=1}^N \\exp(-t_n f_m(\\mathbf{x}_n)) = \\sum_{n=1}^N w_n^{(m)} \\exp(-\\alpha_m t_n y_m(\\mathbf{x}_n)) $$
を最小化する貪欲な加法モデル（Greedy Additive Model）である。
基底分類器 $y_m(\\mathbf{x})$ が与えられたとき、この指数損失関数 $E$ を係数 $\\alpha_m$ について偏微分し、極値条件 $\\frac{\\partial E}{\\partial \\alpha_m} = 0$ から AdaBoost の最適重み更新公式：
$$ \\alpha_m = \\frac{1}{2} \\ln \\left( \\frac{1 - \\epsilon_m}{\\epsilon_m} \\right) $$
（ただし $\\epsilon_m = \\frac{\\sum_{n=1}^N w_n^{(m)} I(y_m(\\mathbf{x}_n) \\ne t_n)}{\\sum_{n=1}^N w_n^{(m)}}$ は重み付き誤分類率）を厳密に導出する。

**代数的導出**:
1. 各データ点 $n$ において $t_n, y_m(\\mathbf{x}_n) \\in \\{-1, +1\\}$ であるため、積 $t_n y_m(\\mathbf{x}_n)$ は正解の場合 $+1$、誤分類の場合 $-1$ のいずれかの値をとる。
2. 指数損失 $E$ を正解データ集合 $\\mathcal{T}_m = \\{n \\mid t_n y_m(\\mathbf{x}_n) = 1\\}$ と誤分類データ集合 $\\mathcal{M}_m = \\{n \\mid t_n y_m(\\mathbf{x}_n) = -1\\}$ に分割する：
$$ E = \\sum_{n \\in \\mathcal{T}_m} w_n^{(m)} e^{-\\alpha_m} + \\sum_{n \\in \\mathcal{M}_m} w_n^{(m)} e^{\\alpha_m} $$
3. 全データ重みの和を $W_m = \\sum_{n=1}^N w_n^{(m)}$、誤分類重みの和を $E_m = \\sum_{n \\in \\mathcal{M}_m} w_n^{(m)}$ とする。すると正解重みの和は $W_m - E_m$ であり、誤分類率は $\\epsilon_m = \\frac{E_m}{W_m}$ である。
4. これらを代入すると：
$$ E = e^{-\\alpha_m} (W_m - E_m) + e^{\\alpha_m} E_m = W_m \\left[ (1 - \\epsilon_m) e^{-\\alpha_m} + \\epsilon_m e^{\\alpha_m} \\right] $$
5. $E$ を $\\alpha_m$ で偏微分してゼロとおく：
$$ \\frac{\\partial E}{\\partial \\alpha_m} = W_m \\left[ -(1 - \\epsilon_m) e^{-\\alpha_m} + \\epsilon_m e^{\\alpha_m} \\right] = 0 $$
6. $W_m > 0$ であるから：
$$ \\epsilon_m e^{\\alpha_m} = (1 - \\epsilon_m) e^{-\\alpha_m} $$
7. 両辺に $e^{\\alpha_m}$ を掛け、$\\epsilon_m$ で割る：
$$ e^{2\\alpha_m} = \\frac{1 - \\epsilon_m}{\\epsilon_m} $$
8. 自然対数をとり $2$ で割ると、求める公式が導出される：
$$ \\alpha_m = \\frac{1}{2} \\ln \\left( \\frac{1 - \\epsilon_m}{\\epsilon_m} \\right) $$
9. また、2階微分は $\\frac{\\partial^2 E}{\\partial \\alpha_m^2} = W_m [(1-\\epsilon_m)e^{-\\alpha_m} + \\epsilon_m e^{\\alpha_m}] = E > 0$ であるため、この解は厳密な大域的最小値を与える。

#### 穴埋め問題
1. AdaBoost は、各反復で $\\text{[ (A) ]}$ 損失関数を最小化する。
2. 弱分類器の誤分類率が $\\epsilon_m < 0.5$ のとき、係数 $\\alpha_m$ は $\\text{[ (B) ]}$ となる。
3. 2階微分が常に正であるため、求めた $\\alpha_m$ は関数 $E$ の $\\text{[ (C) ]}$ 値であることが保証される。
*(解: A: 指数, B: 正の値, C: 大域的最小)*
"""
    ex14_4_code = """# Exercise 14.4 数値検証: 最適 alpha_m 公式と数値最小化の完全一致
from scipy.optimize import minimize_scalar

# 任意の重み付き誤分類率
epsilon_vals = [0.1, 0.25, 0.4, 0.49]

for eps in epsilon_vals:
    # 解析解
    alpha_analytical = 0.5 * np.log((1.0 - eps) / eps)
    
    # 指数損失関数を直接スカラー最小化
    def exp_loss(alpha):
        return (1.0 - eps) * np.exp(-alpha) + eps * np.exp(alpha)
    
    res = minimize_scalar(exp_loss, bounds=(-5, 5), method='bounded', options={'xatol': 1e-10})
    alpha_numerical = res.x
    
    np.testing.assert_allclose(alpha_analytical, alpha_numerical, atol=1e-5)
    print(f"eps={eps:.2f} -> alpha_analytical={alpha_analytical:.6f} == alpha_numerical={alpha_numerical:.6f}")

print("Exercise 14.4 verified: Analytical alpha_m formula strictly minimizes exponential loss!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_4_md), nbf.v4.new_code_cell(ex14_4_code)])

    return cells
