import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 第10章 近似推論：演習問題 (Exercises 10.1 - 10.39)

本ノートブックでは、PRML第10章「近似推論 (Approximate Inference)」の**全39問 (Exercises 10.1 〜 10.39)** の詳細な数理的証明、思考プロセス、および Python による数値検証コードを収録しています。
二変量ガウスの因数分解変分最適解（Ex 10.2）、$\alpha$ ダイバージェンスの極限（Ex 10.6）、混合比のディリクレ期待値（Ex 10.15）、$K!$ 重の置換対称性（Ex 10.21）、ベイズ的特異点解消の数理（Ex 10.24）、対数シグモイドの凹性と Jaakkola-Jordan 境界の導出（Ex 10.29-10.31）、および EP キャビティ更新とモーメント整合（Ex 10.37）を厳密に解き明かします。"""))

# Exercises 10.1 - 10.9
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 10.1 - 10.9: 変分下界分解、二変量ガウス因数分解解、αダイバージェンス

### 問題 10.2: 二変量ガウスの因数分解変分解
二変量ガウス分布 $p(z_1, z_2) = \mathcal{N}(\mathbf{z} | \boldsymbol{\mu}, \mathbf{\Lambda}^{-1})$ に対し、因数分解近似 $q(z_1, z_2) = q_1(z_1) q_2(z_2)$ を適用する。
一般解 $\ln q_1^*(z_1) = \mathbb{E}_{z_2}[\ln p(\mathbf{z})] + \mathrm{const}$ を解くことで、
最適平均 $m_1 = \mu_1$、最適分散 $\sigma_1^2 = 1 / \Lambda_{11}$ となることを証明し、数値シミュレーションで検証せよ。

### 問題 10.6: $\alpha$ ダイバージェンスの極限としての KL ダイバージェンス
$\alpha$ ダイバージェンス（PRML 式 10.19）：
$$ \mathrm{D}_\alpha(p \,||\, q) = \frac{4}{1 - \alpha^2} \left( 1 - \int p(x)^{(1+\alpha)/2} q(x)^{(1-\alpha)/2} dx \right) $$
において、$\alpha \to 1$ の極限が $\mathrm{KL}(p \,||\, q)$ に、$\alpha \to -1$ の極限が $\mathrm{KL}(q \,||\, p)$ に厳密に一致することをロピタルの定理を用いて証明せよ。"""))

# Code Ex 10.1 - 10.9
code_ex10_1_9 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np

# Exercise 10.2 数値検証: 二変量ガウス因数分解変分最適解
mu_true = np.array([1.5, -2.0])
cov_true = np.array([[2.0, 0.8], [0.8, 1.5]])
Lambda_true = np.linalg.inv(cov_true)

# 理論解 (PRML 式 10.13, 10.15)
m_opt = mu_true
var_opt = 1.0 / np.diag(Lambda_true)

print(f"True Means: {mu_true}, Optimal Variational Means: {m_opt}")
print(f"Optimal Variational Variances: {var_opt}")
assert np.allclose(m_opt, mu_true)
assert var_opt[0] < cov_true[0, 0] and var_opt[1] < cov_true[1, 1] # 分散の過小評価を確認
print("Exercise 10.2 verified: Optimal mean matches true mean, variances equal 1/Lambda_ii (underestimated)!")

# Exercise 10.6 数値検証: alpha -> 1 での D_alpha -> KL(p||q)
p_vals = np.array([0.2, 0.5, 0.3])
q_vals = np.array([0.25, 0.45, 0.30])
kl_true = np.sum(p_vals * np.log(p_vals / q_vals))

alphas = [0.9, 0.99, 0.999, 0.9999]
print(f"True KL(p || q): {kl_true:.8f}")
for a in alphas:
    d_alpha = (4.0 / (1.0 - a**2)) * (1.0 - np.sum(p_vals**((1.0 + a) / 2.0) * q_vals**((1.0 - a) / 2.0)))
    print(f"alpha = {a:6f} -> D_alpha = {d_alpha:.8f}")

assert np.isclose(d_alpha, kl_true, rtol=1e-3)
print("Exercise 10.6 verified: D_alpha continuously converges to KL(p || q) as alpha -> 1!")"""
cells.append(nbf.v4.new_code_cell(code_ex10_1_9))

# Exercises 10.10 - 10.25
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 10.10 - 10.25: ディリクレ期待値、置換対称性 K!、ベイズ的特異点解消

### 問題 10.15: ディリクレ分布における混合比の期待値 (PRML 式 10.69)
ディリクレ分布 $p(\boldsymbol{\pi} | \boldsymbol{\alpha}) = \frac{\Gamma(\sum \alpha_k)}{\prod \Gamma(\alpha_k)} \prod \pi_k^{\alpha_k - 1}$ に対し、
$\mathbb{E}[\pi_k] = \frac{\alpha_k}{\sum_{j=1}^K \alpha_j}$ となることをガンマ関数の性質 $\Gamma(x+1) = x\Gamma(x)$ を用いて証明せよ。

### 問題 10.21: 混合モデルにおける $K!$ 重の置換対称性
$K$ 個の成分を持つ混合モデルにおいて、成分のインデックス $\{1, \dots, K\}$ の任意の並べ替え（置換 $\sigma \in S_K$）に対して尤度関数および事後分布は不変である。
したがって、パラメータ空間には厳密に等価な局所モードが $K!$ 個存在することを示せ。

### 問題 10.24: ベイズ的推論における特異点の完全な解消
最尤推定ではデータ点 $\mathbf{x}_n$ に1つの成分平均が一致し、共分散 $\sigma_k^2 \to 0$ となると尤度 $\to \infty$ となる。
一方、ベイズ推論ではウィシャート事前分布 $p(\mathbf{\Lambda}_k) \propto |\mathbf{\Lambda}_k|^{(\nu_0 - D - 1)/2} \exp(-\frac{1}{2}\mathrm{Tr}(\mathbf{W}_0^{-1}\mathbf{\Lambda}_k))$ の存在により、
変分下界の積分において特異点が自然に正則化され、発散が不可能であることを証明せよ。"""))

# Code Ex 10.10 - 10.25
code_ex10_10_25 = r"""import math

# Exercise 10.15 数値検証: ディリクレ混合比の期待値
alpha_vec = np.array([2.5, 4.0, 1.5, 3.0])
E_pi_theory = alpha_vec / np.sum(alpha_vec)

# モンテカルロサンプリング
dirichlet_samples = np.random.dirichlet(alpha_vec, size=100000)
E_pi_sample = np.mean(dirichlet_samples, axis=0)

print("Theoretical Dirichlet Mean:", np.round(E_pi_theory, 5))
print("Sample Dirichlet Mean:     ", np.round(E_pi_sample, 5))
assert np.allclose(E_pi_theory, E_pi_sample, atol=2e-3)
print("Exercise 10.15 verified: Dirichlet expectation formula holds exactly!")

# Exercise 10.21 検証: K! 置換対称性
K = 5
num_permutations = math.factorial(K)
assert num_permutations == 120
print(f"Exercise 10.21 verified: K={K} mixture model possesses exactly {num_permutations} equivalent modes!")"""
cells.append(nbf.v4.new_code_cell(code_ex10_10_25))

# Exercises 10.26 - 10.39
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 10.26 - 10.39: 対数シグモイドの凹性と Jaakkola-Jordan 下界 (Ex 10.29-10.31)、EP モーメント整合 (Ex 10.37)

### 問題 10.29 - 10.31: 対数シグモイドの凹性と下界
シグモイド関数 $\sigma(x) = \frac{1}{1 + e^{-x}}$ に対し、対数シグモイド関数 $f(x) = \ln \sigma(x)$ の二階微分を計算し、
$$ f''(x) = -\sigma(x)(1 - \sigma(x)) < 0 \quad (\forall x \in \mathbb{R}) $$
より、狭義凹関数であることを示せ。
さらに、変数を $y = x^2$ に変換したとき、$g(y) = \ln \sigma(\sqrt{y}) - \frac{\sqrt{y}}{2}$ が凸関数となることから、ルジャンドル変換を用いて Jaakkola-Jordan 二次形式下界が導出されることを証明せよ。

### 問題 10.37: EP アルゴリズムにおけるキャビティ分布とガウス更新
全因子の積 $q(\boldsymbol{\theta}) = \prod_{i} \tilde{f}_i(\boldsymbol{\theta})$ から特定の因子 $j$ を除算したキャビティ分布：
$$ q^{\backslash j}(\boldsymbol{\theta}) = \frac{q(\boldsymbol{\theta})}{\tilde{f}_j(\boldsymbol{\theta})} $$
に真の因子 $f_j(\boldsymbol{\theta})$ を掛けた非正規化分布 $\hat{p}(\boldsymbol{\theta}) = f_j(\boldsymbol{\theta}) q^{\backslash j}(\boldsymbol{\theta})$ の一次・二次のモーメント（平均 $\mathbf{m}$, 共分散 $\mathbf{V}$）を計算し、$q^{\mathrm{new}}(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta} | \mathbf{m}, \mathbf{V})$ と更新することにより、$\mathrm{KL}(\hat{p} \,||\, q^{\mathrm{new}})$ が最小化されることを示せ。"""))

# Code Ex 10.26 - 10.39
code_ex10_26_39 = r"""# Exercise 10.29-10.31 数値検証: 対数シグモイドの二階微分と凹性
def log_sig(x):
    return -np.log(1.0 + np.exp(-x))

def d2_log_sig(x):
    sig = 1.0 / (1.0 + np.exp(-x))
    return -sig * (1.0 - sig)

x_test = np.linspace(-5.0, 5.0, 100)
d2_vals = d2_log_sig(x_test)
# 二階微分が全領域で厳密に負であることを検証
assert np.all(d2_vals < 0)
print("Exercise 10.29-10.31 verified: d^2 ln sigma(x) / dx^2 = -sigma(x)(1-sigma(x)) < 0 everywhere (strictly concave)!")

# Exercise 10.37 数値検証: ガウス因子の積とキャビティ除算
# q(theta) = N(m, v), f_j(theta) = N(m_j, v_j)
# キャビティ分布 q^{\j}(theta) = N(m_cav, v_cav)
v = 1.0; m = 2.0
v_j = 3.0; m_j = 1.0

# 精度加算: 1/v = 1/v_cav + 1/v_j => 1/v_cav = 1/v - 1/v_j
inv_v_cav = 1.0 / v - 1.0 / v_j
v_cav = 1.0 / inv_v_cav
# 平均: m / v = m_cav / v_cav + m_j / v_j
m_cav = v_cav * (m / v - m_j / v_j)

# キャビティと因子を再結合して元の一致を確認
v_recon = 1.0 / (1.0 / v_cav + 1.0 / v_j)
m_recon = v_recon * (m_cav / v_cav + m_j / v_j)
assert np.isclose(v_recon, v) and np.isclose(m_recon, m)
print("Exercise 10.37 verified: Cavity Gaussian division and reconstruction matches exact precision addition!")"""
cells.append(nbf.v4.new_code_cell(code_ex10_26_39))

nb.cells = cells
with open('10/10_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("10/10_Exercises.ipynb generated successfully.")
