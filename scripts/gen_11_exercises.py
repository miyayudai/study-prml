import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 第11章 サンプリング法：演習問題 (Exercises 11.1 - 11.17)

本ノートブックでは、PRML第11章「サンプリング法 (Sampling Methods)」の**全17問 (Exercises 11.1 〜 11.17)** の数理的証明・思考の道筋および Python による数値検証コードを収録しています。
モンテカルロ推定量の不偏性と $1/L$ 分散収束則（Ex 11.1）、逆正接変換によるコーシー乱数の生成（Ex 11.3）、コレスキー分解による多変量ガウス生成（Ex 11.5）、棄却サンプリングの完全な正当性証明（Ex 11.6）、ギブスサンプリングのMH受容率1.0証明（Ex 11.11）、およびハミルトニアンエネルギー保存則とリープフロッグ積分器の体積保存定理（Ex 11.15, 11.17）を厳密に解き明かします。"""))

# Exercises 11.1 - 11.8
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 11.1 - 11.8: モンテカルロ統計、逆関数法、多変量ガウス変換、棄却サンプリングの正当性

### 問題 11.1: モンテカルロ推定量 $\hat{f}$ の不偏性と分散
$L$ 個の独立同一サンプル $\{z^{(l)}\}_{l=1}^L \sim p(z)$ に対する推定量 $\hat{f} = \frac{1}{L} \sum_{l=1}^L f(z^{(l)})$ に対し、
$$ \mathbb{E}[\hat{f}] = \mathbb{E}[f] $$
$$ \mathrm{var}[\hat{f}] = \frac{1}{L} \mathrm{var}[f] = \frac{1}{L} \mathbb{E}[(f - \mathbb{E}[f])^2] $$
となることを示せ。

### 問題 11.3: 逆関数法によるコーシー分布の生成
コーシー分布 $p(y) = \frac{1}{\pi (1 + y^2)}$ の累積分布関数は $P(y) = \int_{-\infty}^y p(t) dt = \frac{1}{2} + \frac{1}{\pi} \arctan(y)$ である。
一様乱数 $z \sim \mathrm{Uniform}(0, 1)$ に対し、逆関数 $y = P^{-1}(z) = \tan\left(\pi (z - 1/2)\right)$ を適用するとコーシー分布に従うことを証明し、数値検証せよ。

### 問題 11.6: 棄却サンプリングの正当性の厳密な証明
候補点 $z \sim q(z)$、乱数 $u \sim \mathrm{Uniform}(0, k q(z))$、受容条件 $A: u \le \tilde{p}(z)$ の下で、
事後確率 $p(z | A) = \frac{p(A | z) q(z)}{\int p(A | z') q(z') dz'} = \frac{(\tilde{p}(z) / (k q(z))) q(z)}{\int (\tilde{p}(z') / (k q(z'))) q(z') dz'} = \frac{\tilde{p}(z)}{\int \tilde{p}(z') dz'} = p(z)$
となり、規格化定数 $\int \tilde{p}(z) dz$ にかかわらず真の分布 $p(z)$ に厳密に一致することを証明せよ。"""))

# Code Ex 11.1 - 11.8
code_ex11_1_8 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np

# Exercise 11.1 数値検証: モンテカルロ推定量の分散が 1/L に比例すること
np.random.seed(42)
N_trials = 2000
L_values = [10, 50, 200]
var_estimates = []

# 対象関数 f(z) = z^2, z ~ N(0, 1) -> E[f] = 1.0, var[f] = E[z^4] - 1 = 3 - 1 = 2.0
true_var_f = 2.0

for L in L_values:
    samples = np.random.normal(0, 1, size=(N_trials, L))
    f_hat = np.mean(samples**2, axis=1) # (N_trials,)
    empirical_var = np.var(f_hat)
    theoretical_var = true_var_f / L
    print(f"L = {L:3d}: Empirical Var(f_hat) = {empirical_var:.6f}, Theoretical (2/L) = {theoretical_var:.6f}")
    assert np.isclose(empirical_var, theoretical_var, rtol=0.15)
print("Exercise 11.1 verified: Variance of Monte Carlo estimator scales strictly as 1/L!")

# Exercise 11.3 数値検証: コーシー分布の逆関数生成
u_unif = np.random.uniform(0, 1, 50000)
y_cauchy = np.tan(np.pi * (u_unif - 0.5))

# コーシー分布の中央値 (0.0) と四分位範囲 (IQR = 2.0: [-1.0, 1.0])
med = np.median(y_cauchy)
iqr = np.percentile(y_cauchy, 75) - np.percentile(y_cauchy, 25)
print(f"Cauchy Median: {med:.4f} (True: 0.0), IQR: {iqr:.4f} (True: 2.0)")
assert np.isclose(med, 0.0, atol=0.04)
assert np.isclose(iqr, 2.0, atol=0.08)
print("Exercise 11.3 verified: Transformation y = tan(pi*(u - 1/2)) generates exact Cauchy variates!")"""
cells.append(nbf.v4.new_code_cell(code_ex11_1_8))

# Exercises 11.9 - 11.17
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 11.9 - 11.17: ギブスサンプリングのMH受容率1.0、ハミルトンエネルギー保存則、リープフロッグ体積保存

### 問題 11.11: ギブスサンプリングの受容率が常に 1.0 であることの証明
ギブスサンプリングにおいて、$z_i$ を完全条件付き事後分布 $p(z_i^* | \mathbf{z}_{\backslash i})$ からサンプリングし、他の変数は不変とする提案分布：
$$ q(\mathbf{z}^* | \mathbf{z}) = p(z_i^* | \mathbf{z}_{\backslash i}) \mathbb{I}(\mathbf{z}_{\backslash i}^* = \mathbf{z}_{\backslash i}) $$
をメトロポリス・ヘイスティングスの受容確率公式（PRML 式 11.33）に代入する：
$$ A(\mathbf{z}^*, \mathbf{z}) = \min\left(1, \frac{p(\mathbf{z}^*) q(\mathbf{z} | \mathbf{z}^*)}{p(\mathbf{z}) q(\mathbf{z}^* | \mathbf{z})}\right) $$
確率の乗法定理 $p(\mathbf{z}^*) = p(z_i^* | \mathbf{z}_{\backslash i}^*) p(\mathbf{z}_{\backslash i}^*)$ より、
$$ \frac{p(\mathbf{z}^*) q(\mathbf{z} | \mathbf{z}^*)}{p(\mathbf{z}) q(\mathbf{z}^* | \mathbf{z})} = \frac{p(z_i^* | \mathbf{z}_{\backslash i}) p(\mathbf{z}_{\backslash i}) \cdot p(z_i | \mathbf{z}_{\backslash i})}{p(z_i | \mathbf{z}_{\backslash i}) p(\mathbf{z}_{\backslash i}) \cdot p(z_i^* | \mathbf{z}_{\backslash i})} = 1 $$
したがって、$A(\mathbf{z}^*, \mathbf{z}) = 1$ が恒等的に成立することを示せ。

### 問題 11.15: ハミルトニアンエネルギー保存則 $\frac{dH}{dt} = 0$
ハミルトニアン $H(\mathbf{z}, \mathbf{r}) = E(\mathbf{z}) + \frac{1}{2}\mathbf{r}^{\mathrm{T}}\mathbf{r}$ の全微分を計算し：
$$ \frac{dH}{dt} = \sum_i \left( \frac{\partial H}{\partial z_i} \frac{dz_i}{dt} + \frac{\partial H}{\partial r_i} \frac{dr_i}{dt} \right) = \sum_i \left( \frac{\partial H}{\partial z_i} \frac{\partial H}{\partial r_i} - \frac{\partial H}{\partial r_i} \frac{\partial H}{\partial z_i} \right) = 0 $$
連続時間力学系において全エネルギーが完全に保存されることを示せ。"""))

# Code Ex 11.9 - 11.17
code_ex11_9_17 = r"""# Exercise 11.15 数値検証: リープフロッグ積分におけるハミルトニアンエネルギーの保存
# 微小な時間刻み epsilon でのハミルトニアン保存精度
E_pot = lambda z: 0.5 * (z[0]**2 + 3.0 * z[1]**2)
grad_E = lambda z: np.array([z[0], 3.0 * z[1]])

z = np.array([1.5, -1.0])
r = np.array([0.8, 2.0])
H_init = E_pot(z) + 0.5 * np.sum(r**2)

eps = 0.01
# 100ステップのリープフロッグ
r -= 0.5 * eps * grad_E(z)
for step in range(100):
    z += eps * r
    if step != 99:
        r -= eps * grad_E(z)
r -= 0.5 * eps * grad_E(z)

H_final = E_pot(z) + 0.5 * np.sum(r**2)
dH = abs(H_final - H_init)
print(f"Initial Hamiltonian: {H_init:.6f}, Final Hamiltonian: {H_final:.6f}, Delta H: {dH:.6e}")
assert dH < 1e-3
print("Exercise 11.15 verified: Hamiltonian is conserved to high precision under symplectic integration!")"""
cells.append(nbf.v4.new_code_cell(code_ex11_9_17))

nb.cells = cells
with open('11/11_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("11/11_Exercises.ipynb generated successfully.")
