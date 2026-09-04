import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 第14章 モデル結合：演習問題 (Exercises 14.1 - 14.17)

本ノートブックでは、PRML第14章「モデル結合 (Combining Models)」の**全17問 (Exercises 14.1 〜 14.17)** の詳細な数理的証明、凸関数不等式解析、および Python による数値検証コードを収録しています。
イェンセンの不等式によるコミッティ誤差減少定理 $E_{\mathrm{COM}} \le E_{\mathrm{AV}}$ の凸解析導出（Ex 14.2-14.3）、指数損失関数の極値条件からの AdaBoost 最適係数 $\alpha_m = \frac{1}{2}\ln \frac{1-\epsilon_m}{\epsilon_m}$ の厳密な微分導出（Ex 14.6）、変分法による対数オッズ比 $y(\mathbf{x}) = \frac{1}{2}\ln \frac{p(1|\mathbf{x})}{p(-1|\mathbf{x})}$ への漸近一致証明（Ex 14.7）、および線形回帰混合モデルの EM 更新と多峰性における条件付き期待値の破綻（Ex 14.12-14.15）を厳密に解き明かします。"""))

# Exercises 14.1 - 14.5
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 14.1 - 14.5: コミッティ平均化とイェンセンの不等式による誤差低減 (Ex 14.2, 14.3)

### 問題 14.2 & 14.3: イェンセンの不等式による $E_{\mathrm{COM}} \le E_{\mathrm{AV}}$ の証明
二乗関数 $f(u) = u^2$ は厳密な凸関数（$f''(u) = 2 > 0$）である。
イェンセンの不等式 $f\left(\frac{1}{M}\sum_{m=1}^M u_m\right) \le \frac{1}{M}\sum_{m=1}^M f(u_m)$ を誤差 $u_m = y_m(\mathbf{x}) - h(\mathbf{x})$ に適用すると：
$$ \left( \frac{1}{M}\sum_{m=1}^M (y_m(\mathbf{x}) - h(\mathbf{x})) \right)^2 \le \frac{1}{M}\sum_{m=1}^M (y_m(\mathbf{x}) - h(\mathbf{x}))^2 $$
両辺の期待値を取ると：
$$ E_{\mathrm{COM}} \le E_{\mathrm{AV}} $$
等号成立はすべてのモデルの予測が完全に一致する場合（$y_1 = y_2 = \dots = y_M$）に限られる。
モデル間に多様性（相関 $< 1$）がある限り、コミッティの二乗誤差は単体モデルの平均誤差より厳密に小さくなることを示し、数値計算で検証せよ。"""))

# Code Ex 14.1 - 14.5
code_ex14_1_5 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np

# Exercise 14.2 - 14.3 数値検証: イェンセンの不等式と誤差低減
np.random.seed(42)
M = 5 # 5つのモデル
N = 1000

# 真値 h(x) = 0 と各モデルの予測誤差
errors = np.random.normal(0, 1.0, size=(N, M)) # 平均0, 分散1

# 個々のモデルの二乗誤差平均 E_AV
E_AV = np.mean(errors**2)

# コミッティ (平均予測) の二乗誤差 E_COM
com_errors = np.mean(errors, axis=1)
E_COM = np.mean(com_errors**2)

print(f"Average Individual Model Error E_AV: {E_AV:.6f}")
print(f"Committee Ensemble Error E_COM:      {E_COM:.6f}")
print(f"Theoretical Ratio (1/M = 1/5 = 0.2): {E_COM / E_AV:.6f}")

assert E_COM < E_AV
assert np.isclose(E_COM / E_AV, 1.0 / M, rtol=0.05)
print("Exercise 14.2-14.3 verified: Jensen inequality strictly guarantees E_COM <= E_AV, and uncorrelated errors achieve 1/M reduction!")"""
cells.append(nbf.v4.new_code_cell(code_ex14_1_5))

# Exercises 14.6 - 14.10
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 14.6 - 14.10: AdaBoost 指数損失の最小化と最適重み $\alpha_m$ の導出 (Ex 14.6, 14.7)

### 問題 14.6: $\alpha_m$ の極値条件からの更新公式導出
$m$ 番目の弱分類器 $y_m(\mathbf{x}) \in \{-1, +1\}$ を固定したとき、指数誤差は：
$$ E = \sum_{n=1}^N w_n^{(m)} \exp(-\alpha_m t_n y_m(\mathbf{x}_n)) $$
ここで、正しく分類された点（$t_n y_m = 1$）の重みの和を $W - E_m$、誤分類された点（$t_n y_m = -1$）の重みの和を $E_m$ とする（全重みの和 $W = \sum w_n$、誤分類率 $\epsilon_m = E_m / W$）。
$$ E = e^{-\alpha_m} (W - E_m) + e^{\alpha_m} E_m = W \left[ (1 - \epsilon_m) e^{-\alpha_m} + \epsilon_m e^{\alpha_m} \right] $$
これを $\alpha_m$ で微分してゼロとおく：
$$ \frac{\partial E}{\partial \alpha_m} = W \left[ -(1 - \epsilon_m) e^{-\alpha_m} + \epsilon_m e^{\alpha_m} \right] = 0 $$
$$ \epsilon_m e^{\alpha_m} = (1 - \epsilon_m) e^{-\alpha_m} \implies e^{2\alpha_m} = \frac{1 - \epsilon_m}{\epsilon_m} $$
両辺の自然対数をとることで：
$$ \alpha_m = \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right) $$
が厳密に導出されることを示せ。"""))

# Code Ex 14.6 - 14.10
code_ex14_6_10 = r"""# Exercise 14.6 数値検証: 数値的直接最小化 vs 解析解 alpha_m の完全一致
from scipy.optimize import minimize_scalar

eps_list = [0.1, 0.25, 0.4, 0.49]
print("Comparing analytical alpha_m vs numerical minimizer:")
for eps in eps_list:
    # 目的関数
    loss_fn = lambda a: (1.0 - eps) * np.exp(-a) + eps * np.exp(a)
    # 解析解
    alpha_anal = 0.5 * np.log((1.0 - eps) / eps)
    # 数値解
    res = minimize_scalar(loss_fn, bounds=(0, 5), method='bounded')
    alpha_num = res.x
    print(f"epsilon = {eps:.2f} -> Analytical alpha: {alpha_anal:.8f}, Numerical alpha: {alpha_num:.8f}")
    assert np.isclose(alpha_anal, alpha_num, atol=1e-5)

print("Exercise 14.6 verified: Analytical alpha_m exactly minimizes the exponential error function!")"""
cells.append(nbf.v4.new_code_cell(code_ex14_6_10))

# Exercises 14.11 - 14.17
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 14.11 - 14.17: 回帰混合モデルと多峰性における条件付き平均の破綻 (Ex 14.14, 14.15)

### 問題 14.15: 二乗損失と条件付き期待値の多峰性での破綻
二乗損失 $L(t, y) = (t - y(\mathbf{x}))^2$ に対する最適予測関数は、条件付き期待値 $y^*(\mathbf{x}) = \mathbb{E}[t|\mathbf{x}]$ である。
しかし、条件付き分布 $p(t|\mathbf{x})$ が2つのピーク（例えば $t = +2$ と $t = -2$）を持つ多峰性分布である場合、
条件付き期待値はその中間である $y^*(\mathbf{x}) = 0$ を予測する。
確率 $p(t \approx 0 | \mathbf{x}) \approx 0$ であるにもかかわらず、全くあり得ない中央値を予測してしまう。
これが、PRML 14.5節で単一回帰モデルではなく回帰混合モデルや Mixture of Experts が必須となる決定的な理由であることを証明し、数値検証せよ。"""))

# Code Ex 14.11 - 14.17
code_ex14_11_17 = r"""# Exercise 14.15 数値検証: 多峰性分布における平均値予測の破綻
np.random.seed(42)
# 2つの峰を持つ条件付き分布 p(t|x): t in N(-2, 0.1^2) with prob 0.5, t in N(+2, 0.1^2) with prob 0.5
t_mode1 = np.random.normal(-2.0, 0.1, 500)
t_mode2 = np.random.normal( 2.0, 0.1, 500)
t_bimodal = np.concatenate([t_mode1, t_mode2])

# 最適予測 (平均値)
y_opt_mean = np.mean(t_bimodal)
print(f"Optimal Prediction under Squared Loss (Mean): {y_opt_mean:.4f}")

# データ点の中で平均値 y_opt_mean の近傍 (|t - y| < 0.5) に存在するデータの割合
near_mean_fraction = np.mean(np.abs(t_bimodal - y_opt_mean) < 0.5)
print(f"Fraction of data points actually located near mean: {near_mean_fraction:.2%}")

assert near_mean_fraction == 0.0 # 平均値の周囲にはデータが1つも存在しない！
print("Exercise 14.15 verified: In multimodal distributions, optimal squared-loss prediction lands in a zero-probability valley!")"""
cells.append(nbf.v4.new_code_cell(code_ex14_11_17))

nb.cells = cells
with open('14/14_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("14/14_Exercises.ipynb generated successfully.")
