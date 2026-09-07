"""
Master script to construct, enhance, and execute all PRML Chapter 2 notebooks:
- 2/2.1_Binary_Variables.ipynb
- 2/2.2_Multinomial_Variables.ipynb
- 2/2.3_The_Gaussian_Distribution.ipynb
- 2/2.4_The_Exponential_Family.ipynb
- 2/2.5_Nonparametric_Methods.ipynb
- 2/2_Exercises.ipynb
"""

import os
import json
import subprocess
import nbformat as nbf

os.makedirs("2/result", exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Build 2/2.1_Binary_Variables.ipynb
# ------------------------------------------------------------------------------
def build_nb_2_1():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell("""# 2.1 二値変数 (Binary Variables)
本ノートブックでは、二値（バイナリ）確率変数のモデル化である**ベルヌーイ分布 (Bernoulli distribution)**、**二項分布 (Binomial distribution)**、そしてベイズ推論のための共役事前分布である**ベータ分布 (Beta distribution)** について理論と実装の両面から詳しく学びます。

### 本ノートブックの構成:
1. **ベルヌーイ分布と最尤推定 (MLE)**: ゼロ頻度問題と過学習の数理
2. **二項分布 (Binomial distribution)**: 試行回数 $N$ と成功回数 $m$ の分布
3. **ベータ分布 (Beta distribution)**: 共役事前分布とハイパーパラメータの有効観測数解釈
4. **逐次ベイズ更新 (Sequential Learning)**: データの逐次到着に伴う事後分布の進化
5. **事後予測分布とラプラスの継起則 (Laplace's Rule of Succession)**"""),

        nbf.v4.new_markdown_cell("""## 2.1.1 ベルヌーイ分布と最尤推定 (MLE)

1回の試行で2つの結果（表/裏、成功/失敗、0/1）のいずれかをとる確率変数 $x \\in \\{0, 1\\}$ を考えます。
パラメータ $\\mu \\in [0, 1]$ を $x=1$ となる確率とすると、$p(x=1|\\mu) = \\mu$, $p(x=0|\\mu) = 1 - \\mu$ です。
これを1つの数式で表現したものが**ベルヌーイ分布 (Bernoulli distribution)** です。

$$ \\mathrm{Bern}(x | \\mu) = \\mu^x (1 - \\mu)^{1 - x} $$

### 期待値と分散
- 期待値: $\\mathbb{E}[x] = 1 \\cdot \\mu + 0 \\cdot (1 - \\mu) = \\mu$
- 分散: $\\mathrm{var}[x] = \\mathbb{E}[x^2] - \\mathbb{E}[x]^2 = \\mu - \\mu^2 = \\mu(1 - \\mu)$

### 最尤推定 (Maximum Likelihood Estimation)
観測データセット $\\mathcal{D} = \\{x_1, x_2, \\dots, x_N\\}$ が独立同分布 (i.i.d.) で得られたと仮定すると、尤度関数は以下のようになります。

$$ p(\\mathcal{D} | \\mu) = \\prod_{n=1}^N p(x_n | \\mu) = \\prod_{n=1}^N \\mu^{x_n} (1 - \\mu)^{1 - x_n} $$

表が出た回数を $m = \\sum_{n=1}^N x_n$ と置くと、最尤推定量 $\\mu_{\\mathrm{ML}}$ は以下のように求まります。

$$ \\mu_{\\mathrm{ML}} = \\frac{m}{N} $$

### 最尤推定の過学習（ゼロ頻度問題）
データ数が少ない場合、例えばコインを3回投げて3回とも表が出た場合（$N=3, m=3$）、最尤推定は $\\mu_{\\mathrm{ML}} = 1.0$ と予測します。
これにより、次に裏が出る確率は厳密に 0 と予測されてしまい、これは過学習の典型例です。この問題を解決するのが**ベイズ的アプローチ**です。"""),

        nbf.v4.new_code_cell("""# ゼロ頻度問題とラプラスの継起則 (Laplace's Rule of Succession)
import numpy as np
import matplotlib.pyplot as plt

small_N = np.array([1, 2, 3, 5, 8, 12])
all_heads_mle = np.ones_like(small_N, dtype=float)
# 一様事前分布 Beta(1, 1) を用いた場合の事後予測確率: (m + 1) / (N + 2)
all_heads_laplace = (small_N + 1.0) / (small_N + 2.0)

plt.figure(figsize=(8, 4.5))
plt.plot(small_N, all_heads_mle, 'ro--', lw=2.5, markersize=8, label='MLE $\\mu_{\\mathrm{ML}} = 1.0$ (ゼロ頻度問題)')
plt.plot(small_N, all_heads_laplace, 'bs-', lw=2.5, markersize=8, label=r'Laplace Smoothing $\\frac{N+1}{N+2}$')
plt.axhline(1.0, color='gray', linestyle=':', alpha=0.7)
plt.title('PRML 2.1.1 Zero-Frequency Catastrophe & Laplace Smoothing', fontsize=12, fontweight='bold')
plt.xlabel('連続して表が出た回数 $N$', fontsize=11)
plt.ylabel('次の試行で表が出る予測確率 $p(x=1)$', fontsize=11)
plt.ylim(0.5, 1.08)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('2/result/fig2_laplace_smoothing.png', dpi=150)
plt.show()
print("Saved to 2/result/fig2_laplace_smoothing.png")"""),

        nbf.v4.new_markdown_cell("""## 2.1.2 二項分布 (Binomial Distribution)

サイズ $N$ のデータセットにおいて、$x=1$ となる回数 $m$ の確率分布は**二項分布 (Binomial distribution)** で与えられます。

$$ \\mathrm{Bin}(m | N, \\mu) = \\binom{N}{m} \\mu^m (1 - \\mu)^{N - m} $$

ここで $\\binom{N}{m} = \\frac{N!}{m!(N-m)!}$ は二項係数です。
二項分布の平均と分散は、独立な $N$ 個のベルヌーイ変数の和であることから直ちに求まります。
$$ \\mathbb{E}[m] = N\\mu, \\quad \\mathrm{var}[m] = N\\mu(1 - \\mu) $$"""),

        nbf.v4.new_markdown_cell("""## 2.1.3 ベータ分布 (The Beta Distribution)

ベイズ推論を行うために、パラメータ $\\mu$ に対する事前分布 $p(\\mu)$ を導入します。
二項尤度 $\\mu^m (1-\\mu)^{N-m}$ に対して、事後分布が事前分布と同じ関数形となる**共役事前分布 (conjugate prior)** として**ベータ分布**が選ばれます。

$$ \\mathrm{Beta}(\\mu | a, b) = \\frac{\\Gamma(a + b)}{\\Gamma(a)\\Gamma(b)} \\mu^{a - 1} (1 - \\mu)^{b - 1} $$

ここで $\\Gamma(x)$ はガンマ関数です。$a, b$ は事前分布の形状を制御する**ハイパーパラメータ (hyperparameters)** です。

### 平均と分散
$$ \\mathbb{E}[\\mu] = \\frac{a}{a + b} $$
$$ \\mathrm{var}[\\mu] = \\frac{ab}{(a + b)^2 (a + b + 1)} $$

### 事後分布の計算 (ベイズの更新)
観測データ（表が $m$ 回、裏が $l = N - m$ 回）が得られたときの事後分布 $p(\\mu | m, l, a, b)$ は、尤度と事前分布の積に比例します。

$$ p(\\mu | m, l, a, b) \\propto p(m | N, \\mu) p(\\mu | a, b) \\propto \\mu^{m + a - 1} (1 - \\mu)^{l + b - 1} $$

規格化定数を考慮すると、事後分布も再びベータ分布となります！
$$ p(\\mu | m, l, a, b) = \\mathrm{Beta}(\\mu | a + m, b + l) $$

この式から、ハイパーパラメータ $a, b$ は、あたかも「事前に行った仮想的な試行において、$x=1$ が $a-1$ 回、$x=0$ が $b-1$ 回観測された」かのような**有効観測数 (effective number of observations)** として解釈できます。"""),

        nbf.v4.new_code_cell("""# 様々なハイパーパラメータに対するベータ分布の形状 (PRML Fig 2.2)
from prml.distributions import BetaDistribution

mu_vals = np.linspace(0.001, 0.999, 500)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
params = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (2.0, 8.0)]
titles = ["Beta(0.5, 0.5) [Jeffreys Prior]", "Beta(1, 1) [Uniform Prior]",
          "Beta(2, 2) [Informative Bell]", "Beta(2, 8) [Skewed to Tails]"]

for ax, (a, b), title in zip(axes.ravel(), params, titles):
    beta = BetaDistribution(a=a, b=b)
    pdf = beta.pdf(mu_vals)
    ax.plot(mu_vals, pdf, 'r-', lw=2.5, label=f'a={a}, b={b}')
    ax.fill_between(mu_vals, 0, pdf, color='red', alpha=0.15)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\\mu$', fontsize=11)
    ax.set_ylabel(r'$p(\\mu)$', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig('2/result/fig2_2_beta_priors.png', dpi=150)
plt.show()
print("Saved to 2/result/fig2_2_beta_priors.png")"""),

        nbf.v4.new_markdown_cell("""## 2.1.4 逐次ベイズ更新 (Sequential Learning) と MLE との比較

ベイズ推論では、1つの観測データが得られるたびに事後分布を逐次更新できます。
前のステップの事後分布が、次のステップの事前分布として自然に機能します。
以下では、真のパラメータ $\\mu^* = 0.7$ からのコイン投げ観測に伴い、事後ベータ分布がどのように真値に集中していくかを検証します。"""),

        nbf.v4.new_code_cell("""# 逐次ベイズ更新のシミュレーション (PRML Fig 2.3)
np.random.seed(42)
true_mu = 0.7
flips = np.random.binomial(1, true_mu, size=30)
prior = BetaDistribution(a=2.0, b=2.0)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
checkpoints = [0, 1, 5, 20]

for ax, n in zip(axes.ravel(), checkpoints):
    m = int(np.sum(flips[:n])) if n > 0 else 0
    l = n - m
    post = prior.bayesian_update(n_heads=m, n_tails=l)
    pdf = post.pdf(mu_vals)
    
    ax.plot(mu_vals, pdf, 'b-', lw=2.5, label=f'Beta({post.a:.0f}, {post.b:.0f})')
    ax.fill_between(mu_vals, 0, pdf, color='blue', alpha=0.15)
    ax.axvline(true_mu, color='darkgreen', linestyle='--', lw=2, label=f'True $\\mu={true_mu}$')
    if n > 0:
        ax.axvline(m / n, color='magenta', linestyle=':', lw=2, label=f'MLE $\\mu_{{ML}}={m/n:.2f}$')
    ax.set_title(f'N = {n} (Heads={m}, Tails={l})', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\\mu$', fontsize=11)
    ax.set_ylabel(r'$p(\\mu|\\mathcal{D})$', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('2/result/fig2_3_bayesian_sequential_learning.png', dpi=150)
plt.show()
print("Saved to 2/result/fig2_3_bayesian_sequential_learning.png")"""),

        nbf.v4.new_code_cell("""# MLE とベイズ推論のサンプルサイズ N に対する収束挙動 (PRML Fig 2.1)
N_steps = np.arange(1, 101)
flips_100 = np.random.binomial(1, true_mu, size=100)
heads_cum = np.cumsum(flips_100)
mle_estimates = heads_cum / N_steps
bayes_estimates = (2.0 + heads_cum) / (2.0 + 2.0 + N_steps)

plt.figure(figsize=(9, 5))
plt.plot(N_steps, mle_estimates, 'm-', lw=2, label=r'MLE $\mu_{\mathrm{ML}} = m/N$')
plt.plot(N_steps, bayes_estimates, 'b-', lw=2.5, label=r'Bayes Posterior Mean $\mathbb{E}[\mu] = \frac{a+m}{a+b+N}$')
plt.axhline(true_mu, color='darkgreen', linestyle='--', lw=2, label=f'True $\\mu = {true_mu}$')
plt.fill_between(N_steps, true_mu - 0.05, true_mu + 0.05, color='green', alpha=0.1, label=r'$\\pm 5\\%$ 許容帯')
plt.title(r'PRML Fig 2.1: MLE vs. ベイズ推定量 の収束比較 ($\mu^*=0.7$)', fontsize=13, fontweight='bold')
plt.xlabel('サンプル数 $N$', fontsize=12)
plt.ylabel(r'推定値 $\mu$', fontsize=12)
plt.ylim(0.3, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('2/result/fig2_1_mle_vs_bayes_convergence.png', dpi=150)
plt.show()
print("Saved to 2/result/fig2_1_mle_vs_bayes_convergence.png")"""),

        nbf.v4.new_markdown_cell("""## まとめ
1. **最尤推定量** $\\mu_{\\mathrm{ML}} = m/N$ はデータ数が少ない場合に深刻な過学習（ゼロ頻度問題）を引き起こす。
2. **共役事前分布** としてベータ分布を用いることで、事後分布が閉じた形で得られ、解析的な逐次更新が可能になる。
3. 事前分布のパラメータ $a, b$ は**有効観測数**として機能し、データが少ない領域で自然な正則化（ラプラス平滑化など）を提供する。
4. サンプル数 $N \\to \\infty$ において、ベイズ事後平均は最尤推定量 $\\mu_{\\mathrm{ML}}$ に漸近し、客観的な真のパラメータに収束する。""")
    ]
    with open("2/2.1_Binary_Variables.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Built 2/2.1_Binary_Variables.ipynb")

build_nb_2_1()
