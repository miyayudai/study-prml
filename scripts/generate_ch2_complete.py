"""
Master builder and executor for PRML Chapter 2 notebooks:
2.1 Binary Variables
2.2 Multinomial Variables
2.3 The Gaussian Distribution
2.4 The Exponential Family
2.5 Nonparametric Methods
"""

import os
import nbformat as nbf

os.makedirs("2/result", exist_ok=True)

# ------------------------------------------------------------------------------
# 2.1 Binary Variables
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
パラメータ $\mu \\in [0, 1]$ を $x=1$ となる確率とすると、$p(x=1|\mu) = \mu$, $p(x=0|\mu) = 1 - \mu$ です。
これを1つの数式で表現したものが**ベルヌーイ分布 (Bernoulli distribution)** です。

$$ \mathrm{Bern}(x | \mu) = \mu^x (1 - \mu)^{1 - x} $$

### 期待値と分散
- 期待値: $\mathbb{E}[x] = 1 \\cdot \mu + 0 \\cdot (1 - \mu) = \mu$
- 分散: $\mathrm{var}[x] = \mathbb{E}[x^2] - \mathbb{E}[x]^2 = \mu - \mu^2 = \mu(1 - \mu)$

### 最尤推定 (Maximum Likelihood Estimation)
観測データセット $\mathcal{D} = \\{x_1, x_2, \\dots, x_N\\}$ が独立同分布 (i.i.d.) で得られたと仮定すると、最尤推定量 $\mu_{\mathrm{ML}}$ は以下のように求まります。

$$ \mu_{\mathrm{ML}} = \\frac{m}{N} $$
ここで $m = \\sum_{n=1}^N x_n$ は表が出た回数です。

データ数が少ない場合、例えばコインを3回投げて3回とも表が出た場合（$N=3, m=3$）、最尤推定は $\mu_{\mathrm{ML}} = 1.0$ となり、次に裏が出る確率を厳密に 0 と予測してしまいます（過学習・ゼロ頻度問題）。"""),

        nbf.v4.new_code_cell(r"""# ゼロ頻度問題とラプラスの継起則 (Laplace's Rule of Succession)
import sys, os
sys.path.append(os.path.abspath('../'))
os.makedirs("result", exist_ok=True)
from common.plot_utils import save_plot, setup_style
setup_style()

import numpy as np
import matplotlib.pyplot as plt

small_N = np.array([1, 2, 3, 5, 8, 12])
all_heads_mle = np.ones_like(small_N, dtype=float)
all_heads_laplace = (small_N + 1.0) / (small_N + 2.0)

fig = plt.figure(figsize=(8, 4.5))
plt.plot(small_N, all_heads_mle, 'ro--', lw=2.5, markersize=8, label='MLE mu=1.0 (Overfitting)')
plt.plot(small_N, all_heads_laplace, 'bs-', lw=2.5, markersize=8, label='Laplace Smoothing (N+1)/(N+2)')
plt.axhline(1.0, color='gray', linestyle=':', alpha=0.7)
plt.title('PRML 2.1.1 Zero-Frequency Catastrophe & Laplace Smoothing', fontsize=12, fontweight='bold')
plt.xlabel('Number of Consecutive Heads $N$', fontsize=11)
plt.ylabel('Predictive Probability $p(x=1)$', fontsize=11)
plt.ylim(0.5, 1.08)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
save_plot(fig, "result", "fig2_laplace_smoothing.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 2.1.2 二項分布 (Binomial Distribution)

サイズ $N$ のデータセットにおいて、$x=1$ となる回数 $m$ の確率分布は**二項分布**で与えられます。

$$ \mathrm{Bin}(m | N, \mu) = \\binom{N}{m} \mu^m (1 - \mu)^{N - m} $$

平均と分散:
$$ \mathbb{E}[m] = N\mu, \\quad \mathrm{var}[m] = N\mu(1 - \mu) $$"""),

        nbf.v4.new_markdown_cell("""## 2.1.3 ベータ分布 (The Beta Distribution)

二項尤度に対する共役事前分布として**ベータ分布**を導入します。

$$ \mathrm{Beta}(\mu | a, b) = \\frac{\\Gamma(a + b)}{\\Gamma(a)\\Gamma(b)} \mu^{a - 1} (1 - \mu)^{b - 1} $$

- 期待値: $\mathbb{E}[\mu] = \\frac{a}{a + b}$
- 分散: $\mathrm{var}[\mu] = \\frac{ab}{(a + b)^2 (a + b + 1)}$
- 最頻値 ($a, b > 1$): $\mu^* = \\frac{a - 1}{a + b - 2}$

### ベイズ更新
表が $m$ 回、裏が $l = N - m$ 回観測されたときの事後分布:
$$ p(\mu | m, l, a, b) = \mathrm{Beta}(\mu | a + m, b + l) $$
ハイパーパラメータ $a, b$ は**有効観測数**（事前擬似観測カウント）として機能します。"""),

        nbf.v4.new_code_cell(r"""# 様々なハイパーパラメータに対するベータ分布の形状 (PRML Fig 2.2)
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
    ax.set_xlabel(r'$\mu$', fontsize=11)
    ax.set_ylabel(r'$p(\mu)$', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

plt.tight_layout()
save_plot(fig, "result", "fig2_2_beta_priors.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 2.1.4 逐次ベイズ更新 (Sequential Learning) と MLE 収束比較

真のパラメータ $\mu^* = 0.7$ からのコイン投げ観測に伴い、事後ベータ分布が真値に集中していく過程を検証します。"""),

        nbf.v4.new_code_cell(r"""# 逐次ベイズ更新のシミュレーション (PRML Fig 2.3)
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
    ax.axvline(true_mu, color='darkgreen', linestyle='--', lw=2, label=f'True $\mu={true_mu}$')
    if n > 0:
        ax.axvline(m / n, color='magenta', linestyle=':', lw=2, label=f'MLE $\mu_{{ML}}={m/n:.2f}$')
    ax.set_title(f'N = {n} (Heads={m}, Tails={l})', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\mu$', fontsize=11)
    ax.set_ylabel(r'$p(\mu|\mathcal{D})$', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
save_plot(fig, "result", "fig2_3_bayesian_sequential_learning.png")
plt.show()"""),

        nbf.v4.new_code_cell(r"""# MLE とベイズ推論のサンプルサイズ N に対する収束挙動 (PRML Fig 2.1)
N_steps = np.arange(1, 101)
flips_100 = np.random.binomial(1, true_mu, size=100)
heads_cum = np.cumsum(flips_100)
mle_estimates = heads_cum / N_steps
bayes_estimates = (2.0 + heads_cum) / (2.0 + 2.0 + N_steps)

fig = plt.figure(figsize=(9, 5))
plt.plot(N_steps, mle_estimates, 'm-', lw=2, label=r'MLE $\mu = m/N$')
plt.plot(N_steps, bayes_estimates, 'b-', lw=2.5, label='Bayes Posterior Mean (a+m)/(a+b+N)')
plt.axhline(true_mu, color='darkgreen', linestyle='--', lw=2, label=f'True $\mu = {true_mu}$')
plt.fill_between(N_steps, true_mu - 0.05, true_mu + 0.05, color='green', alpha=0.1, label=r'$\pm 5\%$ Tolerance Band')
plt.title(r'PRML Fig 2.1: Convergence of MLE vs. Bayesian Estimator ($\mu^*=0.7$)', fontsize=13, fontweight='bold')
plt.xlabel('Sample Count $N$', fontsize=12)
plt.ylabel(r'Estimated $\mu$', fontsize=12)
plt.ylim(0.3, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
save_plot(fig, "result", "fig2_1_mle_vs_bayes_convergence.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## まとめ
1. 最尤推定量 $\mu_{\mathrm{ML}} = m/N$ はデータ数が少ない場合に深刻な過学習（ゼロ頻度問題）を引き起こす。
2. 共役事前分布としてベータ分布を用いることで、事後分布が閉じた形で得られ、解析的な逐次更新が可能になる。
3. 事前分布のパラメータ $a, b$ は有効観測数として機能し、データが少ない領域で自然な正則化（ラプラス平滑化など）を提供する。
4. サンプル数 $N \\to \\infty$ において、ベイズ事後平均は最尤推定量 $\mu_{\mathrm{ML}}$ に漸近し、客観的な真のパラメータに収束する。""")
    ]
    with open("2/2.1_Binary_Variables.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Saved 2/2.1_Binary_Variables.ipynb")


# ------------------------------------------------------------------------------
# 2.2 Multinomial Variables
# ------------------------------------------------------------------------------
def build_nb_2_2():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell("""# 2.2 多項変数 (Multinomial Variables)
本ノートブックでは、$K$ 個の離散的な状態のいずれかをとる多項確率変数のモデル化である**多項分布 (Multinomial distribution)**、およびその共役事前分布である**ディリクレ分布 (Dirichlet distribution)** について学びます。

### 本ノートブックの構成:
1. **多項分布と最尤推定 (MLE)**: 1-of-$K$ 符号化とラグランジュ未定乗数法による最尤解の導出
2. **ディリクレ分布 (The Dirichlet Distribution)**: 多項尤度に対する共役事前分布
3. **単体 (Simplex) 上のディリクレ密度の可視化**: ハイパーパラメータ $\\boldsymbol{\alpha}$ による形状の変化
4. **ベイズ更新と事後予測分布**: 有効観測数としての解釈とディリクレ事後分布の進化"""),

        nbf.v4.new_markdown_cell("""## 2.2.1 多項分布 (The Multinomial Distribution)

$K$ 個の相互排他的な状態のうち1つをとる変数を考えます。これを $K$ 次元の二値ベクトル $\\mathbf{x} = (x_1, x_2, \\dots, x_K)^T$ で表します（1-of-$K$ 表現）。
ここで $x_k \\in \\{0, 1\\}$ かつ $\\sum_{k=1}^K x_k = 1$ です。

パラメータ $\mu_k = p(x_k = 1)$ とすると、制約 $\mu_k \\ge 0$ かつ $\\sum_{k=1}^K \mu_k = 1$ の下で、分布は次式で与えられます。

$$ p(\\mathbf{x}|\\boldsymbol{\mu}) = \\prod_{k=1}^K \mu_k^{x_k} $$

### 多項分布
サイズ $N$ の独立同分布データセット $\mathcal{D} = \\{\\mathbf{x}_1, \\dots, \\mathbf{x}_N\\}$ において、各カテゴリ $k$ が観測された回数を $m_k = \\sum_{n=1}^N x_{nk}$ と置くと、観測回数ベクトル $\\mathbf{m} = (m_1, \\dots, m_K)^T$ の確率分布は**多項分布**となります。

$$ \mathrm{Mult}(\\mathbf{m}|N, \\boldsymbol{\mu}) = \\frac{N!}{m_1! m_2! \\dots m_K!} \\prod_{k=1}^K \mu_k^{m_k} $$

### 最尤推定 (MLE)
制約 $\\sum_{k=1}^K \mu_k = 1$ の下でラグランジュ未定乗数法を解くと、最尤推定量は以下になります。
$$ \mu_k^{\mathrm{ML}} = \\frac{m_k}{N} $$"""),

        nbf.v4.new_markdown_cell("""## 2.2.2 ディリクレ分布 (The Dirichlet Distribution)

多項尤度 に対する共役事前分布として**ディリクレ分布 (Dirichlet distribution)** を導入します。

$$ \mathrm{Dir}(\\boldsymbol{\mu}|\\boldsymbol{\alpha}) = \\frac{\\Gamma(\alpha_0)}{\\Gamma(\alpha_1)\\dots\\Gamma(\alpha_K)} \\prod_{k=1}^K \mu_k^{\alpha_k - 1} $$

ここで $\alpha_0 = \\sum_{k=1}^K \alpha_k$ です。
事後分布は以下のように更新されます。
$$ p(\\boldsymbol{\mu}|\mathcal{D}, \\boldsymbol{\alpha}) = \mathrm{Dir}(\\boldsymbol{\mu}|\\boldsymbol{\alpha} + \\mathbf{m}) $$"""),

        nbf.v4.new_code_cell(r"""# 2-単体（正三角形）上でのディリクレ密度の等高線プロット (PRML Fig 2.4)
import sys, os
sys.path.append(os.path.abspath('../'))
os.makedirs("result", exist_ok=True)
from common.plot_utils import save_plot, setup_style
setup_style()

import numpy as np
import matplotlib.pyplot as plt
from prml.distributions import DirichletDistribution, plot_dirichlet_contour

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
alphas = [
    np.array([0.5, 0.5, 0.5]),  # スパース（頂点に集中）
    np.array([1.0, 1.0, 1.0]),  # 一様分布
    np.array([5.0, 5.0, 5.0]),  # 中心に集中する対称分布
]
titles = [
    r'$\alpha = (0.5, 0.5, 0.5)$ [Sparse]',
    r'$\alpha = (1.0, 1.0, 1.0)$ [Uniform]',
    r'$\alpha = (5.0, 5.0, 5.0)$ [Concentrated]'
]

for ax, alpha, title in zip(axes, alphas, titles):
    plot_dirichlet_contour(alpha, ax=ax, n_grid=200, levels=25, cmap='plasma')
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.tight_layout()
save_plot(fig, "result", "fig2_4_dirichlet_distributions.png")
plt.show()"""),

        nbf.v4.new_code_cell(r"""# 多項データの逐次観測に伴うディリクレ事後分布の更新過程
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 初期事前分布 (一様)
alpha_prior = np.array([1.0, 1.0, 1.0])
plot_dirichlet_contour(alpha_prior, ax=axes[0], levels=20, cmap='viridis')
axes[0].set_title(r'Prior $\mathrm{Dir}(1, 1, 1)$ ($N=0$)', fontsize=12, fontweight='bold')

# 観測データ 1: m = (3, 1, 0)
m1 = np.array([3, 1, 0])
post1 = alpha_prior + m1
plot_dirichlet_contour(post1, ax=axes[1], levels=20, cmap='viridis')
axes[1].set_title(r'Post $\mathrm{Dir}(4, 2, 1)$ ($N=4, m=(3,1,0)$)', fontsize=12, fontweight='bold')

# 観測データ 2: m = (12, 6, 2)
m2 = np.array([12, 6, 2])
post2 = alpha_prior + m2
plot_dirichlet_contour(post2, ax=axes[2], levels=20, cmap='viridis')
axes[2].set_title(r'Post $\mathrm{Dir}(13, 7, 3)$ ($N=20$)', fontsize=12, fontweight='bold')

plt.tight_layout()
save_plot(fig, "result", "fig2_dirichlet_posterior_update.png")
plt.show()"""),

        nbf.v4.new_code_cell(r"""# 多項MLE vs ディリクレベイズ事後期待値のサンプルサイズに対する収束
np.random.seed(42)
true_mu = np.array([0.5, 0.3, 0.2])
N_total = 200
samples = np.random.multinomial(1, true_mu, size=N_total)
counts_cum = np.cumsum(samples, axis=0)
N_range = np.arange(1, N_total + 1)

mle_p0 = counts_cum[:, 0] / N_range
bayes_p0 = (1.0 + counts_cum[:, 0]) / (3.0 + N_range)

fig = plt.figure(figsize=(9, 4.5))
plt.plot(N_range, mle_p0, 'r--', lw=1.8, label=r'MLE $\mu_{1,\mathrm{ML}} = m_1 / N$')
plt.plot(N_range, bayes_p0, 'b-', lw=2.2, label=r'Bayes Posterior Mean $\mathbb{E}[\mu_1]$')
plt.axhline(true_mu[0], color='darkgreen', linestyle='-', lw=2, label=f'True $\mu_1 = {true_mu[0]}$')
plt.title('Convergence of Multinomial MLE and Dirichlet Bayesian Estimator', fontsize=12, fontweight='bold')
plt.xlabel('Sample Count $N$', fontsize=11)
plt.ylabel('Estimated Probability $\mu_1$', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
save_plot(fig, "result", "fig2_multinomial_convergence.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## まとめ
1. 多項分布の最尤推定量 $\mu_k^{\mathrm{ML}} = m_k / N$ は、観測されないカテゴリに対してゼロ確率を割り当てるゼロ頻度問題を持つ。
2. ディリクレ分布は多項分布に対する自然な共役事前分布であり、事後分布の更新はパラメータの加算 $\\boldsymbol{\alpha} + \\mathbf{m}$ のみで実行できる。
3. ディリクレハイパーパラメータ $\alpha_k$ はカテゴリ $k$ の事前有効観測数として機能し、平滑化（スムージング）をもたらす。""")
    ]
    with open("2/2.2_Multinomial_Variables.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Saved 2/2.2_Multinomial_Variables.ipynb")


# ------------------------------------------------------------------------------
# 2.3 The Gaussian Distribution
# ------------------------------------------------------------------------------
def build_nb_2_3():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell("""# 2.3 ガウス分布 (The Gaussian Distribution)
本ノートブックでは、連続確率変数のモデル化において中心的な役割を果たす**多変量ガウス分布 (Multivariate Gaussian distribution)** の全貌を深く学びます。

### 本ノートブックの構成:
1. **幾何学的性質とマハラノビス距離**: 固有値分解と等確率密度楕円
2. **条件付きガウス分布と周辺ガウス分布**: 分割精度行列・共分散行列の解析解
3. **線形ガウスモデル (Linear Gaussian Systems)**: ベイズの定理による同時・周辺・事後分布の解析的導出
4. **最尤推定と標本分散のバイアス**: 有限サンプルにおける過小評価の理論とシミュレーション
5. **逐次推定とロビンス・モンロー (Robbins-Monro) アルゴリズム**: 確率的近似法によるオンライン推定
6. **ガウス分布のベイズ推論**: ガウス事前分布（平均）、ガンマ事前分布（精度）、正規ガンマ事前分布（同時）
7. **スチューデントのt分布 (Student's t-distribution)**: 外れ値に対する頑健性 (Robustness)
8. **周期変数とフォン・ミーゼス分布 (The von Mises Distribution)**: 円周上の正規分布"""),

        nbf.v4.new_markdown_cell("""## 2.3.1 ガウス分布の幾何学的性質とマハラノビス距離

$D$ 次元ベクトル $\\mathbf{x}$ に対する多変量ガウス分布の確率密度関数は以下で与えられます。

$$ \mathcal{N}(\\mathbf{x} | \\boldsymbol{\mu}, \\boldsymbol{\\Sigma}) = \\frac{1}{(2\\pi)^{D/2} |\\boldsymbol{\\Sigma}|^{1/2}} \\exp \\left( -\\frac{1}{2} (\\mathbf{x} - \\boldsymbol{\mu})^T \\boldsymbol{\\Sigma}^{-1} (\\mathbf{x} - \\boldsymbol{\mu}) \\right) $$

ここで $\\boldsymbol{\mu}$ は $D$ 次元平均ベクトル、$\\boldsymbol{\\Sigma}$ は $D \\times D$ の対称正定値共分散行列です。
マハラノビス距離の二乗 $\\Delta^2 = (\\mathbf{x} - \\boldsymbol{\mu})^T \\boldsymbol{\\Sigma}^{-1} (\\mathbf{x} - \\boldsymbol{\mu})$ の等高線は超楕円体を形成します。"""),

        nbf.v4.new_markdown_cell("""## 2.3.2 条件付きガウス分布と周辺ガウス分布

確率変数ベクトルを $\\mathbf{x} = (\\mathbf{x}_a^T, \\mathbf{x}_b^T)^T$ と2分割します。
条件付き分布 $p(\\mathbf{x}_a | \\mathbf{x}_b)$ および周辺分布 $p(\\mathbf{x}_a)$ は以下で与えられます。

$$ \\boldsymbol{\mu}_{a|b} = \\boldsymbol{\mu}_a + \\boldsymbol{\\Sigma}_{ab}\\boldsymbol{\\Sigma}_{bb}^{-1}(\\mathbf{x}_b - \\boldsymbol{\mu}_b) $$
$$ \\boldsymbol{\\Sigma}_{a|b} = \\boldsymbol{\\Sigma}_{aa} - \\boldsymbol{\\Sigma}_{ab}\\boldsymbol{\\Sigma}_{bb}^{-1}\\boldsymbol{\\Sigma}_{ba} $$
$$ \mathbb{E}[\\mathbf{x}_a] = \\boldsymbol{\mu}_a, \\quad \mathrm{cov}[\\mathbf{x}_a] = \\boldsymbol{\\Sigma}_{aa} $$"""),

        nbf.v4.new_code_cell(r"""# 条件付き分布と周辺分布の可視化 (PRML Section 2.3.3 - 2.3.4)
import sys, os
sys.path.append(os.path.abspath('../'))
os.makedirs("result", exist_ok=True)
from common.plot_utils import save_plot, setup_style
setup_style()

import numpy as np
import matplotlib.pyplot as plt
from prml.distributions import MultivariateGaussian

mu = np.array([1.0, 1.5])
cov = np.array([[1.5, 0.9], [0.9, 1.2]])
mg = MultivariateGaussian(mean=mu, cov=cov)

# グリッド上の2D同時ガウス密度
x1_grid = np.linspace(-2.5, 4.5, 200)
x2_grid = np.linspace(-2.5, 5.5, 200)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
pos = np.dstack((X1, X2))
Z = mg.pdf(pos.reshape(-1, 2)).reshape(X1.shape)

# 条件付け: x2 = 2.5
x2_fixed = 2.5
cond = mg.condition(x_b=[x2_fixed], idx_a=[0], idx_b=[1])
marg = mg.marginalize(idx_a=[0])

fig, ax = plt.subplots(figsize=(8, 6))
cs = ax.contour(X1, X2, Z, levels=12, cmap='Blues', alpha=0.8)
ax.clabel(cs, inline=True, fontsize=8)

# 条件付き切断線
ax.axhline(x2_fixed, color='crimson', linestyle='--', lw=2, label=f'Condition line $x_2 = {x2_fixed}$')
ax.plot(cond.mean[0], x2_fixed, 'ro', markersize=8, label=f'Cond Mean $\mu_{{1|2}} = {cond.mean[0]:.2f}$')
ax.plot(mu[0], mu[1], 'k*', markersize=10, label=r'Joint Mean $\mu = (1.0, 1.5)$')

ax.set_title('Joint, Conditional, and Marginal Gaussians', fontsize=13, fontweight='bold')
ax.set_xlabel('$x_1$', fontsize=11)
ax.set_ylabel('$x_2$', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=10)
plt.tight_layout()
save_plot(fig, "result", "fig2_gaussian_condition_marginal.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 2.3.3 線形ガウスモデル (Linear Gaussian Systems)

PRML 式 (2.113) - (2.115) で与えられる線形ガウスモデルを扱います。
$$ p(\\mathbf{x}) = \mathcal{N}(\\mathbf{x} | \\boldsymbol{\mu}_x, \\boldsymbol{\\Sigma}_x) $$
$$ p(\\mathbf{y} | \\mathbf{x}) = \mathcal{N}(\\mathbf{y} | \\mathbf{A}\\mathbf{x} + \\mathbf{b}, \\mathbf{L}^{-1}) $$

### 周辺分布 $p(\\mathbf{y})$
$$ p(\\mathbf{y}) = \mathcal{N}(\\mathbf{y} | \\mathbf{A}\\boldsymbol{\mu}_x + \\mathbf{b}, \\mathbf{L}^{-1} + \\mathbf{A}\\boldsymbol{\\Sigma}_x\\mathbf{A}^T) $$

### 事後分布 $p(\\mathbf{x} | \\mathbf{y})$
$$ p(\\mathbf{x} | \\mathbf{y}) = \mathcal{N}(\\mathbf{x} | \\boldsymbol{\\Sigma}_{x|y} \\{ \\mathbf{A}^T\\mathbf{L}(\\mathbf{y} - \\mathbf{b}) + \\boldsymbol{\\Sigma}_x^{-1}\\boldsymbol{\mu}_x \\}, \\boldsymbol{\\Sigma}_{x|y}) $$
ここで $\\boldsymbol{\\Sigma}_{x|y} = (\\boldsymbol{\\Sigma}_x^{-1} + \\mathbf{A}^T\\mathbf{L}\\mathbf{A})^{-1}$ です。"""),

        nbf.v4.new_code_cell(r"""# 線形ガウスモデルの数値検証 (PRML Section 2.3.5)
mu_x = np.array([1.0])
sigma_x = np.array([[2.0]])
A = np.array([[1.5]])
b = np.array([0.5])
L_cov = np.array([[0.8]])

marginal_y, post_solver = MultivariateGaussian.linear_gaussian_system(
    mu_x=mu_x, sigma_x=sigma_x, A=A, b=b, L_cov=L_cov
)

print(f"Marginal p(y) Mean: {marginal_y.mean[0]:.3f} (Theoretical: {1.5*1.0 + 0.5:.3f})")
print(f"Marginal p(y) Var:  {marginal_y.cov[0, 0]:.3f} (Theoretical: {0.8 + 1.5**2 * 2.0:.3f})")

post_x = post_solver([3.0])
print(f"Posterior p(x|y=3.0) Mean: {post_x.mean[0]:.3f}")
print(f"Posterior p(x|y=3.0) Var:  {post_x.cov[0, 0]:.3f}")

# プロット
x_axis = np.linspace(-3, 5, 300)
from prml.distributions import Gaussian1D
prior_pdf = Gaussian1D(mu=mu_x[0], var=sigma_x[0, 0]).pdf(x_axis)
post_pdf = Gaussian1D(mu=post_x.mean[0], var=post_x.cov[0, 0]).pdf(x_axis)

fig = plt.figure(figsize=(8, 4.5))
plt.plot(x_axis, prior_pdf, 'k--', lw=2, label=f'Prior $p(x) = \mathcal{{N}}({mu_x[0]}, {sigma_x[0,0]})$')
plt.plot(x_axis, post_pdf, 'b-', lw=2.5, label=f'Posterior $p(x|y=3) = \mathcal{{N}}({post_x.mean[0]:.2f}, {post_x.cov[0,0]:.2f})$')
plt.axvline(post_x.mean[0], color='blue', linestyle=':', alpha=0.7)
plt.title('Linear Gaussian System: Prior vs. Posterior Inversion', fontsize=12, fontweight='bold')
plt.xlabel('$x$', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
save_plot(fig, "result", "fig2_linear_gaussian_system.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 2.3.4 最尤推定における標本分散のバイアス (Bias of Sample Variance)

$N$ 個の独立観測データに対するガウス最尤推定量は以下です。
$$ \mu_{\mathrm{ML}} = \\frac{1}{N}\\sum_{n=1}^N x_n, \\quad \sigma_{\mathrm{ML}}^2 = \\frac{1}{N}\\sum_{n=1}^N (x_n - \mu_{\mathrm{ML}})^2 $$

期待値:
$$ \mathbb{E}[\mu_{\mathrm{ML}}] = \mu, \\quad \mathbb{E}[\sigma_{\mathrm{ML}}^2] = \\left( \\frac{N - 1}{N} \\right) \sigma^2 $$"""),

        nbf.v4.new_code_cell(r"""# 標本分散のバイアスのモンテカルロ検証 (PRML Section 2.3.4)
np.random.seed(42)
true_var = 4.0
sample_sizes = np.arange(2, 21)
n_trials = 5000

avg_mle_var = []
avg_unbiased_var = []

for n in sample_sizes:
    samples = np.random.normal(loc=0.0, scale=np.sqrt(true_var), size=(n_trials, n))
    mle_v = np.var(samples, axis=1, ddof=0)
    unb_v = np.var(samples, axis=1, ddof=1)
    avg_mle_var.append(np.mean(mle_v))
    avg_unbiased_var.append(np.mean(unb_v))

theory_mle = [(n - 1) / n * true_var for n in sample_sizes]

fig = plt.figure(figsize=(8, 4.5))
plt.plot(sample_sizes, avg_mle_var, 'ro', label='Empirical MLE Var')
plt.plot(sample_sizes, theory_mle, 'r-', lw=2, label=r'Theoretical ((N-1)/N) $\sigma^2$')
plt.plot(sample_sizes, avg_unbiased_var, 'bs--', lw=1.8, label='Unbiased s^2 (ddof=1)')
plt.axhline(true_var, color='darkgreen', linestyle='-', lw=2, label=f'True $\sigma^2 = {true_var}$')
plt.title('PRML 2.3.4 Bias of Maximum Likelihood Variance', fontsize=12, fontweight='bold')
plt.xlabel('Sample Size $N$', fontsize=11)
plt.ylabel('Variance Estimate', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
save_plot(fig, "result", "fig2_sample_var_bias.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 2.3.5 逐次推定とロビンス・モンロー (Robbins-Monro) アルゴリズム

根探索問題 $f(\theta) = \mathbb{E}[z|\theta] = 0$ に対し、**ロビンス・モンローアルゴリズム**は以下の逐次更新を行います。
$$ \theta^{(N)} = \theta^{(N-1)} - a_{N-1} z(x_N, \theta^{(N-1)}) $$
$\\sum_{N=1}^\\infty a_N = \\infty$ かつ $\\sum_{N=1}^\\infty a_N^2 < \\infty$ を満たすステップ幅列 $a_N$（例: $a/N$）により確率収束が保証されます。"""),

        nbf.v4.new_code_cell(r"""# ロビンス・モンローによる逐次平均推定のシミュレーション (PRML Section 2.3.5)
from prml.distributions import RobbinsMonro

np.random.seed(42)
mu_true = 3.0
X_seq = np.random.normal(loc=mu_true, scale=1.5, size=500)

rm = RobbinsMonro(a_coeff=1.0)
final_mu, history = rm.estimate_mean(X_seq, init_mu=-2.0)

fig = plt.figure(figsize=(9, 4.5))
plt.plot(history, 'b-', lw=2, label=r'Robbins-Monro Estimate $\theta^{(N)}$')
plt.axhline(mu_true, color='red', linestyle='--', lw=2, label=f'True $\mu = {mu_true}$')
plt.title('PRML 2.3.5 Robbins-Monro Sequential Estimation of Gaussian Mean', fontsize=12, fontweight='bold')
plt.xlabel('Step $N$', fontsize=11)
plt.ylabel(r'Estimate $\theta$', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
save_plot(fig, "result", "fig2_robbins_monro.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 2.3.6 スチューデントのt分布と外れ値に対する頑健性

スチューデントのt分布は、ガウス分布の精度をガンマ分布で周辺化した混合分布です。
自由度 $\nu$ が小さいとき裾が重く (heavy tails)、外れ値（outliers）が存在しても中心パラメータの推定が引きずられない頑健性 (Robustness) を持ちます。"""),

        nbf.v4.new_code_cell(r"""# 外れ値に対するガウス分布 vs スチューデントt分布の頑健性比較 (PRML Fig 2.16)
from prml.distributions import Gaussian1D, StudentsTDistribution

x_axis = np.linspace(-6, 6, 500)
g = Gaussian1D(mu=0.0, var=1.0)
st1 = StudentsTDistribution(mu=0.0, lam=1.0, nu=1.0)
st4 = StudentsTDistribution(mu=0.0, lam=1.0, nu=4.0)

fig = plt.figure(figsize=(9, 5))
plt.plot(x_axis, g.pdf(x_axis), 'k-', lw=2.2, label=r'Gaussian $\mathcal{N}(0, 1)$')
plt.plot(x_axis, [st4.pdf(x) for x in x_axis], 'b--', lw=2, label=r"Student's $t$ ($\nu=4$)")
plt.plot(x_axis, [st1.pdf(x) for x in x_axis], 'r:', lw=2.5, label=r"Student's $t$ ($\nu=1$, Cauchy)")
plt.title("PRML Fig 2.16: Robustness of Student's t-Distribution to Outliers", fontsize=12, fontweight='bold')
plt.xlabel('$x$', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.ylim(0, 0.45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
save_plot(fig, "result", "fig2_robust_student_t.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 2.3.7 周期変数とフォン・ミーゼス分布 (The von Mises Distribution)

円周上の正規分布として**フォン・ミーゼス分布**が定義されます。
$$ p(\theta | \mu, \kappa) = \\frac{1}{2\\pi I_0(\kappa)} \\exp (\kappa \\cos(\theta - \mu)) $$
ここで $I_0(\kappa)$ は0次の第1種変形ベッセル関数です。"""),

        nbf.v4.new_code_cell(r"""# フォン・ミーゼス分布の極座標プロット (PRML Fig 2.18)
from prml.distributions import VonMisesDistribution

thetas = np.linspace(0, 2 * np.pi, 500)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))

kappas = [0.5, 1.0, 4.0]
colors = ['green', 'blue', 'crimson']

for kappa, c in zip(kappas, colors):
    vm = VonMisesDistribution(mu=np.pi / 4.0, kappa=kappa)
    r_vals = [vm.pdf(th) for th in thetas]
    ax.plot(thetas, r_vals, color=c, lw=2.2, label=f'$\kappa = {kappa}$')

ax.set_title(r'PRML Fig 2.18: Von Mises Distribution ($\mu = \pi/4$)', fontsize=12, fontweight='bold', pad=15)
ax.legend(loc='upper right')
plt.tight_layout()
save_plot(fig, "result", "fig2_von_mises.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## まとめ
1. 多変量ガウス分布は条件付き分布および周辺分布のいずれもが閉じたガウス分布となる極めて有用な数学的性質を持つ。
2. 線形ガウスモデルにより、観測の反転問題（ベイズ推論）が解析的な行列演算のみで高速に実行できる。
3. 最尤推定の標本分散には $\\frac{N-1}{N}$ のバイアスが存在し、ベイズ的アプローチや不偏推定量が不可欠となる。
4. 外れ値に対してはスチューデントt分布、周期データに対してはフォン・ミーゼス分布を用いることで、ガウス分布の弱点を克服できる。""")
    ]
    with open("2/2.3_The_Gaussian_Distribution.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Saved 2/2.3_The_Gaussian_Distribution.ipynb")


# ------------------------------------------------------------------------------
# 2.4 The Exponential Family
# ------------------------------------------------------------------------------
def build_nb_2_4():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell("""# 2.4 指数型分布族 (The Exponential Family)
本ノートブックでは、ベルヌーイ分布、多項分布、ガウス分布、ポアソン分布、ガンマ分布などを共通の統一的な枠組みで包括する**指数型分布族 (The Exponential Family)** の理論的構造と性質を学びます。

### 本ノートブックの構成:
1. **指数型分布族の標準形 (Canonical Form)**: 自然母数、十分統計量、対数分配関数
2. **対数分配関数の微分とモーメントの導出**: 期待値と共分散の美しい双対関係
3. **共役事前分布の統一的導出**: 指数型分布族に対する普遍的な事後分布更新
4. **無情報事前分布とジェフリーズの事前分布 (Jeffreys Prior)**: 尺度不変性と変数変換不変性"""),

        nbf.v4.new_markdown_cell("""## 2.4.1 指数型分布族の標準形

パラメータ $\\boldsymbol{\eta}$ を持つ確率変数 $\\mathbf{x}$ の分布が以下の形で表されるとき、**指数型分布族**に属すると言います。

$$ p(\\mathbf{x} | \\boldsymbol{\eta}) = h(\\mathbf{x}) g(\\boldsymbol{\eta}) \\exp \\{ \\boldsymbol{\eta}^T \\mathbf{u}(\\mathbf{x}) \\} $$

- $\\boldsymbol{\eta}$: 自然母数 (natural parameters)
- $\\mathbf{u}(\\mathbf{x})$: 十分統計量 (sufficient statistics)
- $g(\\boldsymbol{\eta})$: 分配関数の逆数"""),

        nbf.v4.new_markdown_cell("""## 2.4.2 対数分配関数によるモーメントの導出

規格化条件の両辺を $\\boldsymbol{\eta}$ について微分することでモーメント関係式が得られます。
$$ -\\nabla_{\\boldsymbol{\eta}} \\ln g(\\boldsymbol{\eta}) = \mathbb{E}[\\mathbf{u}(\\mathbf{x})] $$
$$ -\\nabla_{\\boldsymbol{\eta}}\\nabla_{\\boldsymbol{\eta}} \\ln g(\\boldsymbol{\eta}) = \mathrm{cov}[\\mathbf{u}(\\mathbf{x})] $$"""),

        nbf.v4.new_code_cell(r"""# ガウス分布における対数分配関数の微分によるモーメント導出の数値検証
import sys, os
sys.path.append(os.path.abspath('../'))
os.makedirs("result", exist_ok=True)
from common.plot_utils import save_plot, setup_style
setup_style()

import numpy as np
import matplotlib.pyplot as plt

mu_true = 2.5
sigma_true = 1.2
var_true = sigma_true ** 2

eta1 = mu_true / var_true
eta2 = -1.0 / (2.0 * var_true)
eta = np.array([eta1, eta2])

def neg_log_g(e):
    return - (e[0]**2) / (4.0 * e[1]) - 0.5 * np.log(-2.0 * e[1]) + 0.5 * np.log(2.0 * np.pi)

# 数値微分
eps = 1e-5
grad = np.zeros(2)
for i in range(2):
    e_plus = eta.copy(); e_plus[i] += eps
    e_minus = eta.copy(); e_minus[i] -= eps
    grad[i] = (neg_log_g(e_plus) - neg_log_g(e_minus)) / (2.0 * eps)

print(f"Gradient -d ln g / d eta1: {grad[0]:.4f} (E[x] = {mu_true})")
print(f"Gradient -d ln g / d eta2: {grad[1]:.4f} (E[x^2] = {mu_true**2 + var_true})")

eta1_vals = np.linspace(1.0, 3.5, 100)
energies = [neg_log_g([e1, eta2]) for e1 in eta1_vals]
fig = plt.figure(figsize=(7, 4.5))
plt.plot(eta1_vals, energies, 'b-', lw=2.2, label=r'$-\ln g(\eta)$ vs $\eta_1$')
plt.title(r'Cumulant Generating Function $-\ln g(\eta)$ Convexity', fontsize=12, fontweight='bold')
plt.xlabel(r'$\eta_1 = \mu / \sigma^2$', fontsize=11)
plt.ylabel(r'$-\ln g(\eta)$', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
save_plot(fig, "result", "fig2_exponential_family_moments.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 2.4.3 無情報事前分布とジェフリーズの事前分布 (Jeffreys Prior)

フィッシャー情報量 $I(\theta) = -\mathbb{E}\\left[\\frac{\\partial^2 \\ln p}{\\partial \theta^2}\\right]$ を用いて、変数変換に対して不変な客観的事前分布を定めます。
$$ p(\theta) \\propto \\sqrt{\\det I(\theta)} $$"""),

        nbf.v4.new_code_cell(r"""# ジェフリーズ事前分布の変数変換不変性の可視化
sigma_vals = np.linspace(0.1, 4.0, 300)
prior_sigma = 1.0 / sigma_vals

lam_vals = 1.0 / (sigma_vals ** 2)
prior_lam = 1.0 / lam_vals

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(sigma_vals, prior_sigma, 'r-', lw=2.2)
ax1.set_title(r'Jeffreys Prior for Scale: $p(\sigma) \propto 1/\sigma$', fontsize=11, fontweight='bold')
ax1.set_xlabel(r'$\sigma$', fontsize=11)
ax1.set_ylabel(r'$p(\sigma)$', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.plot(lam_vals, prior_lam, 'b-', lw=2.2)
ax2.set_title(r'Jeffreys Prior for Precision: $p(\lambda) \propto 1/\lambda$', fontsize=11, fontweight='bold')
ax2.set_xlabel(r'$\lambda = 1/\sigma^2$', fontsize=11)
ax2.set_ylabel(r'$p(\lambda)$', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
save_plot(fig, "result", "fig2_jeffreys_prior.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## まとめ
1. 指数型分布族は十分統計量を通じたデータ要約と、分配関数の微分によるモーメント導出という普遍的な枠組みを提供する。
2. 指数型分布族に対する共役事前分布も普遍的に定義でき、ベイズ更新はハイパーパラメータの加算として定式化される。
3. ジェフリーズの事前分布は幾何学的な不変性を満たす客観的ベイズ事前分布の標準である。""")
    ]
    with open("2/2.4_The_Exponential_Family.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Saved 2/2.4_The_Exponential_Family.ipynb")


# ------------------------------------------------------------------------------
# 2.5 Nonparametric Methods
# ------------------------------------------------------------------------------
def build_nb_2_5():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell("""# 2.5 ノンパラメトリック法 (Nonparametric Methods)
本ノートブックでは、特定の関数形を仮定せず、データそのものから柔軟に確率密度を推定する**ノンパラメトリック密度推定法**を学びます。

### 本ノートブックの構成:
1. **カーネル密度推定量 (Kernel Density Estimator / Parzen Window)**: ガウス・箱型・エパネチニコフカーネル
2. **バンド幅 (Bandwidth) の選択**: アンダーフィッティングとオーバーフィッティング、シルバマンの経験則
3. **K近傍法 (K-Nearest Neighbours Density Estimation)**: データ密度に応じた適応的解像度
4. **KNN分類器と決定境界**: パラメータ $K$ による決定領域の平滑化と汎化性能"""),

        nbf.v4.new_markdown_cell("""## 2.5.1 カーネル密度推定 (Parzen Window)

データ点 $\\mathbf{x}_n$ を中心とする局所領域の確率密度をカーネル関数 $k(\\cdot)$ で重ね合わせます。
$$ p(\\mathbf{x}) = \\frac{1}{N} \\sum_{n=1}^N \\frac{1}{h^D} k\\left( \\frac{\\mathbf{x} - \\mathbf{x}_n}{h} \\right) $$
ここで $h$ はバンド幅 (bandwidth) です。"""),

        nbf.v4.new_code_cell(r"""# 様々なカーネル関数による混合ガウスデータの密度推定比較
import sys, os
sys.path.append(os.path.abspath('../'))
os.makedirs("result", exist_ok=True)
from common.plot_utils import save_plot, setup_style
setup_style()

import numpy as np
import matplotlib.pyplot as plt
from prml.distributions import KernelDensityEstimator

np.random.seed(42)
data = np.concatenate([np.random.normal(-1.5, 0.6, 120), np.random.normal(1.8, 0.8, 180)])
x_eval = np.linspace(-4, 5, 400)

kde_gauss = KernelDensityEstimator(bandwidth=0.45, kernel='gaussian').fit(data)
kde_box = KernelDensityEstimator(bandwidth=0.65, kernel='box').fit(data)
kde_epan = KernelDensityEstimator(bandwidth=0.65, kernel='epanechnikov').fit(data)

fig = plt.figure(figsize=(10, 5))
plt.hist(data, bins=35, density=True, alpha=0.25, color='gray', label='Empirical Histogram')
plt.plot(x_eval, kde_gauss.score_samples(x_eval), 'b-', lw=2.2, label='Gaussian Kernel ($h=0.45$)')
plt.plot(x_eval, kde_epan.score_samples(x_eval), 'r-', lw=2.2, label='Epanechnikov Kernel ($h=0.65$)')
plt.plot(x_eval, kde_box.score_samples(x_eval), 'g--', lw=1.8, label='Box Kernel ($h=0.65$)')

plt.title('PRML 2.5.1 Kernel Density Estimators on Bimodal Mixture Data', fontsize=12, fontweight='bold')
plt.xlabel('$x$', fontsize=11)
plt.ylabel('Density $p(x)$', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
save_plot(fig, "result", "fig2_parzen_kernel_comparison.png")
plt.show()"""),

        nbf.v4.new_code_cell(r"""# バンド幅 h の影響: アンダー平滑化 vs 最適 vs オーバー平滑化 (PRML Fig 2.24)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
hs = [0.1, 0.45, 1.5]
titles = [r'Undersmoothed ($h=0.1$)', r'Optimal ($h=0.45$, Silverman)', r'Oversmoothed ($h=1.5$)']

for ax, h, title in zip(axes, hs, titles):
    kde = KernelDensityEstimator(bandwidth=h, kernel='gaussian').fit(data)
    ax.hist(data, bins=35, density=True, alpha=0.2, color='gray')
    ax.plot(x_eval, kde.score_samples(x_eval), 'r-', lw=2.2)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
save_plot(fig, "result", "fig2_kde_bandwidth.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 2.5.2 K近傍法 (K-Nearest Neighbours Density Estimation)

各点 $\\mathbf{x}$ を中心に $K$ 個の訓練データを含む最小超球の体積 $V(\\mathbf{x})$ を求め、密度を推定します。
$$ p(\\mathbf{x}) = \\frac{K}{N V(\\mathbf{x})} $$"""),

        nbf.v4.new_code_cell(r"""# KDE (固定バンド幅) と KNN (適応的体積) の密度推定比較
from prml.distributions import KNearestNeighborsDensity

knn_dense = KNearestNeighborsDensity(k=20).fit(data)

fig = plt.figure(figsize=(9, 4.5))
plt.hist(data, bins=35, density=True, alpha=0.2, color='gray', label='Empirical Histogram')
plt.plot(x_eval, kde_gauss.score_samples(x_eval), 'b-', lw=2.2, label='KDE (Gaussian, $h=0.45$)')
plt.plot(x_eval, knn_dense.score_samples(x_eval), 'm-', lw=2.2, label='KNN Density ($K=20$)')
plt.title('PRML 2.5.2 Comparison: Fixed-Bandwidth KDE vs. Adaptive KNN', fontsize=12, fontweight='bold')
plt.xlabel('$x$', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
save_plot(fig, "result", "fig2_knn_density_vs_kde.png")
plt.show()"""),

        nbf.v4.new_code_cell(r"""# 2次元非線形分類問題における KNN 決定境界の可視化 (PRML Fig 2.26)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons

np.random.seed(42)
X_moons, y_moons = make_moons(n_samples=250, noise=0.25, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
Ks = [1, 7, 35]

gx = np.linspace(-1.5, 2.5, 200)
gy = np.linspace(-1.0, 1.5, 200)
GX, GY = np.meshgrid(gx, gy)
grid_pts = np.c_[GX.ravel(), GY.ravel()]

for ax, k in zip(axes, Ks):
    clf = KNeighborsClassifier(n_neighbors=k).fit(X_moons, y_moons)
    Z_cls = clf.predict(grid_pts).reshape(GX.shape)
    
    ax.contourf(GX, GY, Z_cls, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_moons[y_moons==0, 0], X_moons[y_moons==0, 1], c='blue', s=20, edgecolors='k', label='Class 0')
    ax.scatter(X_moons[y_moons==1, 0], X_moons[y_moons==1, 1], c='red', s=20, edgecolors='k', label='Class 1')
    ax.set_title(f'K-NN Classifier (K = {k})', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
save_plot(fig, "result", "fig2_knn_decision_boundary.png")
plt.show()"""),

        nbf.v4.new_markdown_cell("""## まとめ
1. カーネル密度推定は各データ点をカーネル関数で滑らかに重ね合わせる手法であり、バンド幅 $h$ の適切な調整が本質的である。
2. K近傍法は近傍点の探索半径をデータ密度に応じて伸縮させる適応的解像度を持ち、次元の呪いへの耐性も高い。
3. ノンパラメトリック法は訓練データをすべて保持・参照する必要があるため、計算量やメモリ消費量は $O(N)$ となるトレードオフがある。""")
    ]
    with open("2/2.5_Nonparametric_Methods.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Saved 2/2.5_Nonparametric_Methods.ipynb")

build_nb_2_1()
build_nb_2_2()
build_nb_2_3()
build_nb_2_4()
build_nb_2_5()
print("All notebooks built successfully!")
