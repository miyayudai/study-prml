# scripts/build_ch14_part2.py
"""
Build Chapter 14 Exercises: Part 2 (Exercises 14.5 - 14.8)
AdaBoost Weight Updates, Exponential Loss Convergence, Population Log-Odds, and Loss Functions
"""

import nbformat as nbf

def build_part2():
    """Exercises 14.5 - 14.8: AdaBoost Theory & Loss Analysis"""
    cells = []

    # Exercise 14.5
    ex14_5_md = """---
## Exercise 14.5: AdaBoost データ重み更新式と全重み和の縮小率 $2\\sqrt{\\epsilon_m(1-\\epsilon_m)}$ の証明

### 問題の背景と数学的証明
AdaBoost では、$m$ ステップ目で弱分類器 $y_m(\\mathbf{x})$ とその係数 $\\alpha_m$ が決定された後、次ステップのデータ重みは以下のように更新される（(14.16)式）：
$$ w_n^{(m+1)} = w_n^{(m)} \\exp(-\\alpha_m t_n y_m(\\mathbf{x}_n)) $$
本問では、最適係数 $\\alpha_m = \\frac{1}{2}\\ln\\left(\\frac{1-\\epsilon_m}{\\epsilon_m}\\right)$ の下で、ステップ $m+1$ における全データ重みの総和 $W_{m+1} = \\sum_{n=1}^N w_n^{(m+1)}$ が、直前の総和 $W_m = \\sum_{n=1}^N w_n^{(m)}$ に対して：
$$ W_{m+1} = 2 \\sqrt{\\epsilon_m (1 - \\epsilon_m)} \\, W_m $$
を満たすことを数学的に証明する。

**代数的証明**:
1. 正解データ集合 $\\mathcal{T}_m$（$t_n y_m = 1$）および誤分類データ集合 $\\mathcal{M}_m$（$t_n y_m = -1$）における重みの和をそれぞれ $W_m - E_m$、$E_m$ とする（$\\epsilon_m = E_m / W_m$）。
2. 更新後の全重みの総和を展開する：
$$ W_{m+1} = \\sum_{n=1}^N w_n^{(m+1)} = \\sum_{n \\in \\mathcal{T}_m} w_n^{(m)} e^{-\\alpha_m} + \\sum_{n \\in \\mathcal{M}_m} w_n^{(m)} e^{\\alpha_m} $$
$$ W_{m+1} = (W_m - E_m) e^{-\\alpha_m} + E_m e^{\\alpha_m} = W_m \\left[ (1 - \\epsilon_m) e^{-\\alpha_m} + \\epsilon_m e^{\\alpha_m} \\right] $$
3. 最適係数 $\\alpha_m = \\frac{1}{2}\\ln\\left(\\frac{1-\\epsilon_m}{\\epsilon_m}\\right)$ より：
$$ e^{\\alpha_m} = \\sqrt{\\frac{1 - \\epsilon_m}{\\epsilon_m}}, \\quad e^{-\\alpha_m} = \\sqrt{\\frac{\\epsilon_m}{1 - \\epsilon_m}} $$
4. これを代入する：
$$ (1 - \\epsilon_m) e^{-\\alpha_m} = (1 - \\epsilon_m) \\sqrt{\\frac{\\epsilon_m}{1 - \\epsilon_m}} = \\sqrt{\\epsilon_m (1 - \\epsilon_m)} $$
$$ \\epsilon_m e^{\\alpha_m} = \\epsilon_m \\sqrt{\\frac{1 - \\epsilon_m}{\\epsilon_m}} = \\sqrt{\\epsilon_m (1 - \\epsilon_m)} $$
5. したがって、大括弧内は：
$$ (1 - \\epsilon_m) e^{-\\alpha_m} + \\epsilon_m e^{\\alpha_m} = \\sqrt{\\epsilon_m (1 - \\epsilon_m)} + \\sqrt{\\epsilon_m (1 - \\epsilon_m)} = 2 \\sqrt{\\epsilon_m (1 - \\epsilon_m)} $$
となり、
$$ W_{m+1} = 2 \\sqrt{\\epsilon_m (1 - \\epsilon_m)} \\, W_m $$
が厳密に証明された。
6. $0 \\le \\epsilon_m < 0.5$ のとき、$2\\sqrt{\\epsilon_m(1-\\epsilon_m)} < 1$ であり、ステップが進むごとに全重みの総和（＝指数損失）は厳密に減少する。

#### 穴埋め問題
1. 正解したサンプルの重みは $e^{-\\alpha_m}$ 倍され、誤分類されたサンプルの重みは $\\text{[ (A) ]}$ 倍される。
2. 最適な $\\alpha_m$ のもとで、正解グループの重み総和と誤分類グループの重み総和は $\\text{[ (B) ]}$ になる。
3. 全重みの縮小率は $\\text{[ (C) ]}$ であり、ランダム予測（$\\epsilon_m=0.5$）で最大値 $1$ をとる。
*(解: A: $e^{\\alpha_m}$, B: 完全に等しく, C: $2\\sqrt{\\epsilon_m(1-\\epsilon_m)}$)*
"""
    ex14_5_code = """# Exercise 14.5 数値検証: 重み更新式と全重み和の縮小率の一致
N_samples = 200
w_m = np.random.uniform(0.5, 2.0, size=N_samples)
W_m = np.sum(w_m)

# 弱分類器の予測
t = np.random.choice([-1, 1], size=N_samples)
y = t.copy()
# 誤分類率約 0.3 となるように反転
flip_idx = np.random.choice(N_samples, size=int(0.3 * N_samples), replace=False)
y[flip_idx] *= -1

# 重み付き誤分類率
E_m = np.sum(w_m[t != y])
eps_m = E_m / W_m

alpha_m = 0.5 * np.log((1.0 - eps_m) / eps_m)

# 重み更新
w_next = w_m * np.exp(-alpha_m * t * y)
W_next = np.sum(w_next)

# 理論縮小率
factor_theory = 2.0 * np.sqrt(eps_m * (1.0 - eps_m))
W_next_theory = factor_theory * W_m

np.testing.assert_allclose(W_next, W_next_theory, atol=1e-12)
print(f"Exercise 14.5 verified: W_{{m+1}} = {W_next:.6f} matches theory {W_next_theory:.6f} (factor = {factor_theory:.4f} < 1)!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_5_md), nbf.v4.new_code_cell(ex14_5_code)])

    # Exercise 14.6
    ex14_6_md = """---
## Exercise 14.6: AdaBoost 指数損失の累積減少率と訓練誤分類率ゼロへの指数収束

### 問題の背景と数学的証明
初期重みを $w_n^{(1)} = 1$ と正規化すると、初期指数損失は $E_0 = N$ である。
Exercise 14.5 より、各ステップ $m$ で全重み和（指数損失）は $2\\sqrt{\\epsilon_m(1-\\epsilon_m)}$ 倍されるため、$M$ ステップ後の指数損失は：
$$ E_M = N \\prod_{m=1}^M 2 \\sqrt{\\epsilon_m (1 - \\epsilon_m)} $$
と表される（(14.19)式）。
本問では、各ステップの弱分類器がランダム推測よりわずかに優れている（弱学習性仮説：ある $\\gamma > 0$ に対して $\\epsilon_m \\le \\frac{1}{2} - \\gamma$）とき、アンサンブル全体の訓練誤分類率が $M$ の増加に伴って指数関数的にゼロへ収束することを証明する。

**数理的証明**:
1. 0-1 損失と指数損失の関係：指示関数 $I(f_M(\\mathbf{x}_n) \\ne t_n) = I(t_n f_M(\\mathbf{x}_n) \\le 0)$ について、常に不等式：
$$ I(t_n f_M(\\mathbf{x}_n) \\le 0) \\le \\exp(-t_n f_M(\\mathbf{x}_n)) $$
が成立する。したがって、全訓練誤分類数 $N_{\\text{err}}$ は指数損失 $E_M$ で上から抑えられる：
$$ N_{\\text{err}} = \\sum_{n=1}^N I(t_n f_M(\\mathbf{x}_n) \\le 0) \\le \\sum_{n=1}^N \\exp(-t_n f_M(\\mathbf{x}_n)) = E_M $$
2. 誤分類率の上界は：
$$ \\frac{N_{\\text{err}}}{N} \\le \\frac{E_M}{N} = \\prod_{m=1}^M 2 \\sqrt{\\epsilon_m (1 - \\epsilon_m)} $$
3. 各ステップで $\\epsilon_m \\le \\frac{1}{2} - \\gamma$（$\\gamma > 0$）であるとする。
$$ 4 \\epsilon_m (1 - \\epsilon_m) = 4 \\left( \\frac{1}{2} - \\gamma \\right) \\left( \\frac{1}{2} + \\gamma \\right) = 4 \\left( \\frac{1}{4} - \\gamma^2 \\right) = 1 - 4\\gamma^2 $$
4. したがって：
$$ 2 \\sqrt{\\epsilon_m (1 - \\epsilon_m)} = \\sqrt{1 - 4\\gamma^2} $$
不等式 $1 - u \\le e^{-u}$（特に $u = 4\\gamma^2$）を用いると：
$$ \\sqrt{1 - 4\\gamma^2} \\le \\sqrt{e^{-4\\gamma^2}} = e^{-2\\gamma^2} $$
5. これを $M$ 個のステップにわたって累積すると：
$$ \\frac{N_{\\text{err}}}{N} \\le \\prod_{m=1}^M e^{-2\\gamma^2} = \\exp(-2 \\gamma^2 M) $$
6. $M \\to \\infty$ において $\\exp(-2\\gamma^2 M) \\to 0$ であるため、訓練誤分類率は指数関数的なレートで確実にゼロに収束する。

#### 穴埋め問題
1. 0-1 損失は常に指数損失によって $\\text{[ (A) ]}$ から抑えられる。
2. 弱学習器のマージンが $\\gamma$ であるとき、訓練誤分類率の上界は $\\text{[ (B) ]}$ である。
3. この定理は、弱分類器がコイン投げよりほんの少しでも良ければ、アンサンブルによって訓練誤差を $\\text{[ (C) ]}$ にできることを意味する。
*(解: A: 上, B: $\\exp(-2\\gamma^2 M)$, C: ゼロ)*
"""
    ex14_6_code = """# Exercise 14.6 数値検証: AdaBoost 指数損失の累積減少と訓練誤差の上界
M_steps = 15
gamma_edge = 0.1  # 各ステップで eps = 0.4 (0.5 - gamma)
eps_seq = np.full(M_steps, 0.5 - gamma_edge)

# 累積減少係数
factors = 2.0 * np.sqrt(eps_seq * (1.0 - eps_seq))
cum_error_bound = np.cumprod(factors)
exp_bound = np.exp(-2.0 * (gamma_edge**2) * np.arange(1, M_steps + 1))

# 厳密な不等式検証
for m in range(M_steps):
    assert cum_error_bound[m] <= exp_bound[m] + 1e-12

print("Exercise 14.6 verified: Training error bound decays exponentially:")
for m in [1, 5, 10, 15]:
    print(f"  Step {m:2d}: Product bound = {cum_error_bound[m-1]:.6f} <= exp(-2*gamma^2*M) = {exp_bound[m-1]:.6f}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_6_md), nbf.v4.new_code_cell(ex14_6_code)])

    # Exercise 14.7
    ex14_7_md = """---
## Exercise 14.7: 指数損失を母集団レベルで最小化する関数 $y(\\mathbf{x})$ と対数オッズ比（ロジット）の一致証明

### 問題の背景と数学的証明
AdaBoost の最適化対象である期待指数損失（母集団損失）：
$$ E[y] = \\mathbb{E}_{\\mathbf{x}, t} [\\exp(-t y(\\mathbf{x}))] = \\int \\sum_{t \\in \\{-1, +1\\}} \\exp(-t y(\\mathbf{x})) p(t | \\mathbf{x}) p(\\mathbf{x}) d\\mathbf{x} $$
を考える（(14.21)式）。
本問では、変分法（各 $\\mathbf{x}$ ごとの独立な最適化）を用いて、任意の関数 $y(\\mathbf{x})$ の空間において期待損失 $E[y]$ を大域的に最小化する最適関数 $y^*(\\mathbf{x})$ が、事後クラス確率の**真の対数オッズ比の半分**：
$$ y^*(\\mathbf{x}) = \\frac{1}{2} \\ln \\left( \\frac{p(t = 1 | \\mathbf{x})}{p(t = -1 | \\mathbf{x})} \\right) $$
に厳密に一致することを証明する。

**数理的証明**:
1. $p(\\mathbf{x}) \\ge 0$ であるため、期待損失を最小化することは、各 $\\mathbf{x}$ において条件付き期待値：
$$ J(y(\\mathbf{x})) \\equiv \\sum_{t \\in \\{-1, +1\\}} \\exp(-t y(\\mathbf{x})) p(t | \\mathbf{x}) $$
を最小化することと等価である。
2. 和を展開する：
$$ J(y(\\mathbf{x})) = e^{-y(\\mathbf{x})} p(t = 1 | \\mathbf{x}) + e^{y(\\mathbf{x})} p(t = -1 | \\mathbf{x}) $$
3. $y(\\mathbf{x})$ に関して偏微分してゼロとおく：
$$ \\frac{\\partial J}{\\partial y(\\mathbf{x})} = -e^{-y(\\mathbf{x})} p(t = 1 | \\mathbf{x}) + e^{y(\\mathbf{x})} p(t = -1 | \\mathbf{x}) = 0 $$
4. 項を整理する：
$$ e^{y(\\mathbf{x})} p(t = -1 | \\mathbf{x}) = e^{-y(\\mathbf{x})} p(t = 1 | \\mathbf{x}) $$
5. 両辺に $e^{y(\\mathbf{x})}$ を掛け、$p(t = -1 | \\mathbf{x})$ で割る：
$$ e^{2y(\\mathbf{x})} = \\frac{p(t = 1 | \\mathbf{x})}{p(t = -1 | \\mathbf{x})} $$
6. 自然対数をとり $2$ で割ると：
$$ y^*(\\mathbf{x}) = \\frac{1}{2} \\ln \\left( \\frac{p(t = 1 | \\mathbf{x})}{p(t = -1 | \\mathbf{x})} \\right) $$
が得られる。
7. 2階微分は $\\frac{\\partial^2 J}{\\partial y^2} = e^{-y} p(1|\\mathbf{x}) + e^y p(-1|\\mathbf{x}) > 0$ であるため、これは大域的最小値を与える。
8. したがって、AdaBoost の出力 $f_M(\\mathbf{x})$ の符号 $\\operatorname{sign}(f_M(\\mathbf{x}))$ はベイズ最適決定境界（$p(t=1|\\mathbf{x}) > p(t=-1|\\mathbf{x})$）に漸近的に一致する。

#### 穴埋め問題
1. 母集団レベルで指数損失を最小化する関数 $y^*(\\mathbf{x})$ は、真のクラス事後確率の $\\text{[ (A) ]}$ の半分となる。
2. これより、AdaBoost の出力は単なる分類境界だけでなく、クラス事後確率の $\\text{[ (B) ]}$ にも対応する。
3. 2階微分が常に $\\text{[ (C) ]}$ であるため、この極値は厳密な大域的最小点である。
*(解: A: 対数オッズ比 (ロジット), B: 推定値, C: 正)*
"""
    ex14_7_code = """# Exercise 14.7 数値検証: 条件付き指数損失の最小解と対数オッズ比の完全一致
# 任意のクラス事後確率 p(t=1|x)
p1_vals = np.linspace(0.05, 0.95, 10)

for p1 in p1_vals:
    p_minus1 = 1.0 - p1
    # 理論最適解
    y_star_theory = 0.5 * np.log(p1 / p_minus1)
    
    # 条件付き損失 J(y) の数値最小化
    def loss_func(y):
        return np.exp(-y) * p1 + np.exp(y) * p_minus1
    
    res = minimize_scalar(loss_func, bounds=(-10, 10), method='bounded')
    y_star_numerical = res.x
    
    np.testing.assert_allclose(y_star_theory, y_star_numerical, atol=1e-6)

print("Exercise 14.7 verified: Minimum of population exponential loss strictly equals half log-odds ratio across all p(t=1|x)!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_7_md), nbf.v4.new_code_cell(ex14_7_code)])

    # Exercise 14.8
    ex14_8_md = """---
## Exercise 14.8: 指数損失とクロスエントロピー損失の比較、および外れ値に対する頑健性解析

### 問題の背景と数学的証明
マージン $z = t y(\\mathbf{x})$ に対する各種損失関数を比較する：
1. **0-1 損失**: $L_{01}(z) = I(z \\le 0)$
2. **指数損失 (AdaBoost)**: $L_{\\text{exp}}(z) = \\exp(-z)$
3. **ロジスティック損失 (Cross-Entropy / LogitBoost)**: $L_{\\text{log}}(z) = \\ln(1 + \\exp(-z))$

**外れ値に対する頑健性（Robustness）の比較**:
- 誤分類度が大きいサンプル（$z \\ll 0$、すなわち重度の外れ値やラベルノイズ）において：
  - 指数損失の勾配（ペナルティ）は $\\frac{d L_{\\text{exp}}}{dz} = -e^{-z} \\to -\\infty$ と指数関数的に爆発する。そのため、外れ値1点に引っ張られて決定境界が激しく歪む（過学習・脆弱性）。
  - 一方、ロジスティック損失の勾配は $\\frac{d L_{\\text{log}}}{dz} = -\\frac{e^{-z}}{1 + e^{-z}} = -\\frac{1}{1 + e^z} \\to -1$ と一定値に飽和する（線形増加）。
- したがって、ロジスティック損失や GentleBoost は外れ値に対して格段に頑健である。

#### 穴埋め問題
1. 誤分類マージン $z < 0$ が負に大きいとき、指数損失のペナルティは $\\text{[ (A) ]}$ 的に増加する。
2. ロジスティック損失の勾配は、大きな誤分類に対して最大でも $\\text{[ (B) ]}$ に抑えられる。
3. このため、ノイズ混入データに対しては指数損失よりもロジスティック損失を採用した $\\text{[ (C) ]}$ の方が頑健である。
*(解: A: 指数関数, B: 1 (一定値に飽和), C: LogitBoost (クロスエントロピー))*
"""
    ex14_8_code = """# Exercise 14.8 数値検証: マージンに対する損失関数とその微分の振る舞い
z_margin = np.linspace(-3, 3, 100)

loss_exp = np.exp(-z_margin)
loss_log = np.log(1.0 + np.exp(-z_margin))

# 勾配の絶対値 (ペナルティの強さ)
grad_exp = np.exp(-z_margin)
grad_log = 1.0 / (1.0 + np.exp(z_margin))

# 外れ値 (z = -3) における勾配比較
assert grad_exp[0] > 15.0  # e^3 = 20.08
assert grad_log[0] < 1.0   # 1 / (1 + e^-3) = 0.952

print("Exercise 14.8 verified: Exponential loss gradient explodes on outliers while logistic gradient saturates:")
print(f"  At z = -3.0 (strong outlier): |grad_exp| = {grad_exp[0]:.4f}, |grad_log| = {grad_log[0]:.4f}")
print(f"  At z = +3.0 (well classified): |grad_exp| = {grad_exp[-1]:.4f}, |grad_log| = {grad_log[-1]:.4f}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_8_md), nbf.v4.new_code_cell(ex14_8_code)])

    return cells
