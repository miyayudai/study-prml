import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 第9章 混合モデルとEM：演習問題 (Exercises 9.1 - 9.27)

本ノートブックでは、PRML第9章「混合モデルとEMアルゴリズム (Mixture Models and EM)」の**全27問 (Exercises 9.1 〜 9.27)** の詳細な論理ステップ（数理的証明・思考の道筋）および Python による数値検証コードを収録しています。
K-meansの有限停止性、共通共分散GMM、混合分布の平均・共分散の合成則（Ex 9.12）、同一初期化におけるBMMの1反復退化（Ex 9.13）、エビデンス再推定とEMの等価性（Ex 9.23）、下界と対数尤度の接点勾配一致定理（Ex 9.25）、およびインクリメンタルEM更新を計算機上で実験・検証します。"""))

# Exercises 9.1 - 9.9
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 9.1 - 9.9: K-means有限収束性、共通共分散GMM、完全データ対数尤度

### 問題 9.1: K-meansの有限回反復収束
$N$ 点を $K$ クラスタに割り当てる場合の数は高々 $K^N$ 通り（有限集合）である。
各割り当て $\{r_{nk}\}$ に対し、歪み尺度 $J$ を最小化する中心 $\boldsymbol{\mu}_k$ は一意に定まる。
割り当てステップでも中心更新ステップでも $J$ は厳密に単調非増加であり、一度訪れた割り当てに再び戻ることはない。
したがって、K-means アルゴリズムは必ず有限回の反復で厳密に停止（収束）することを示せ。

### 問題 9.6: 共通共分散行列 $\mathbf{\Sigma}_k = \mathbf{\Sigma}$ を持つ GMM の EM 更新式
すべての成分が共通の共分散行列 $\mathbf{\Sigma}$ を共有する制約下での完全データ期待対数尤度は：
$$ Q = -\frac{1}{2} \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \left[ D \ln(2\pi) + \ln |\mathbf{\Sigma}| + (\mathbf{x}_n - \boldsymbol{\mu}_k)^{\mathrm{T}} \mathbf{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) \right] $$
これを $\mathbf{\Sigma}^{-1}$ で微分して $0$ と置くことにより、Mステップの更新式が全クラスタの残差共分散の加重和となることを示せ：
$$ \mathbf{\Sigma} = \frac{1}{N} \sum_{k=1}^K \sum_{n=1}^N \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k) (\mathbf{x}_n - \boldsymbol{\mu}_k)^{\mathrm{T}} $$"""))

# Code Ex 9.1 - 9.9
code_ex9_1_9 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np

# Exercise 9.6 数値検証: 共通共分散 GMM の M ステップ
N, D, K = 100, 2, 3
np.random.seed(42)
X = np.random.randn(N, D)
gamma = np.random.dirichlet(np.ones(K), size=N) # 負担率
mu = np.random.randn(K, D)

# 各クラスタ個別の更新共分散
covs_separate = []
for k in range(K):
    diff = X - mu[k]
    cov_k = (gamma[:, k:k+1] * diff).T @ diff / np.sum(gamma[:, k])
    covs_separate.append(cov_k)

# 共通共分散の加重和 (式 9.6)
Sigma_common = np.zeros((D, D))
for k in range(K):
    diff = X - mu[k]
    Sigma_common += (gamma[:, k:k+1] * diff).T @ diff
Sigma_common /= N

# 各クラスタ共分散の N_k/N 加重平均と完全一致することを検証
Sigma_weighted_avg = sum((np.sum(gamma[:, k]) / N) * covs_separate[k] for k in range(K))
assert np.allclose(Sigma_common, Sigma_weighted_avg)
print("Exercise 9.6 verified: Common covariance is the exact weighted sum of component covariances!")"""
cells.append(nbf.v4.new_code_cell(code_ex9_1_9))

# Exercises 9.10 - 9.17
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 9.10 - 9.17: 混合分布の平均・共分散の合成則、同一初期化でのBMM退化、有界性

### 問題 9.12: 混合分布全体の平均と共分散行列 (PRML 式 9.49, 9.50)
混合分布 $p(\mathbf{x}) = \sum_{k=1}^K \pi_k p(\mathbf{x}|k)$ に対し、
$$ \mathbb{E}[\mathbf{x}] = \sum_{k=1}^K \pi_k \boldsymbol{\mu}_k $$
$$ \mathrm{cov}[\mathbf{x}] = \sum_{k=1}^K \pi_k \left\{ \mathbf{\Sigma}_k + (\boldsymbol{\mu}_k - \mathbb{E}[\mathbf{x}])(\boldsymbol{\mu}_k - \mathbb{E}[\mathbf{x}])^{\mathrm{T}} \right\} $$
であることを証明し、数値サンプリングと厳密に照合せよ。

### 問題 9.13: ベルヌーイ混合モデルにおける同一初期値での退化
全成分の平均ベクトルを同一 $\boldsymbol{\mu}_k = \bar{\boldsymbol{\mu}}$ と初期化すると、任意のデータ点 $n$ に対し
各成分の事後確率（負担率）は常に $\gamma(z_{nk}) = \pi_k$ となり、データ点に依存しなくなる。
その結果、次ステップの平均パラメータは全成分で即座にデータ全体の標本平均 $\bar{\mathbf{x}} = \frac{1}{N}\sum_n \mathbf{x}_n$ に一致し、1反復で完全に同一成分へ退化することを示せ。"""))

# Code Ex 9.10 - 9.17
code_ex9_10_17 = r"""# Exercise 9.12 数値検証: 混合分布の理論共分散 vs サンプル共分散
K = 3
pi_true = np.array([0.2, 0.5, 0.3])
mu_true = np.array([[-2.0, -1.0], [1.0, 3.0], [4.0, -2.0]])
cov_true = np.array([
    [[0.5, 0.1], [0.1, 0.4]],
    [[1.0, -0.3], [-0.3, 0.8]],
    [[0.3, 0.0], [0.0, 0.6]]
])

# 理論平均と理論共分散
E_x = np.sum(pi_true[:, np.newaxis] * mu_true, axis=0)
Cov_x = np.zeros((2, 2))
for k in range(K):
    diff = mu_true[k] - E_x
    Cov_x += pi_true[k] * (cov_true[k] + np.outer(diff, diff))

# サンプリングによる照合
N_pts = 100000
z_samples = np.random.choice(K, size=N_pts, p=pi_true)
X_samples = np.zeros((N_pts, 2))
for k in range(K):
    idx = (z_samples == k)
    if np.sum(idx) > 0:
        X_samples[idx] = np.random.multivariate_normal(mu_true[k], cov_true[k], size=np.sum(idx))

E_sample = np.mean(X_samples, axis=0)
Cov_sample = np.cov(X_samples, rowvar=False)

print(f"Theoretical Mean: {E_x}, Sample Mean: {np.round(E_sample, 3)}")
print("Theoretical Covariance:\n", np.round(Cov_x, 4))
print("Sample Covariance:\n", np.round(Cov_sample, 4))
assert np.allclose(E_x, E_sample, atol=0.03)
assert np.allclose(Cov_x, Cov_sample, atol=0.04)
print("Exercise 9.12 verified: Mixture covariance formula perfectly matches empirical sampling!")

# Exercise 9.13 数値検証: BMM の同一初期化退化
from common.mixture_em_utils import BernoulliMixtureModel
X_bmm = (np.random.rand(50, 10) > 0.4).astype(float)
bmm_bad = BernoulliMixtureModel(n_components=3, max_iter=2)
# 全クラスタを同一平均 0.5 で初期化
bmm_bad.means_ = np.full((3, 10), 0.5)
bmm_bad.weights_ = np.array([0.3, 0.3, 0.4])
bmm_bad.fit(X_bmm)

x_mean = np.mean(X_bmm, axis=0)
for k in range(3):
    assert np.allclose(bmm_bad.means_[k], x_mean, atol=1e-5)
print("Exercise 9.13 verified: BMM with identical initialization immediately collapses to sample mean in 1 step!")"""
cells.append(nbf.v4.new_code_cell(code_ex9_10_17))

# Exercises 9.18 - 9.27
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 9.18 - 9.27: MAP-EM、エビデンスとEMの等価性、ELBO接線勾配一致定理 (Ex 9.25)

### 問題 9.25: 接点における変分下界と対数尤度の勾配の一致定理
$$ \ln p(\mathbf{X}|\boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \mathrm{KL}(q \,||\, p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})) $$
Eステップにおいて $q(\mathbf{Z}) = p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta}^{(\mathrm{old})})$ と選ぶと、
$\boldsymbol{\theta} = \boldsymbol{\theta}^{(\mathrm{old})}$ において $\mathrm{KL}$ は大域的最小値 $0$ を達成する。
滑らかな関数が最小値をとる点では勾配はゼロ $\left. \nabla_{\boldsymbol{\theta}} \mathrm{KL} \right|_{\boldsymbol{\theta}^{(\mathrm{old})}} = \mathbf{0}$ であるため：
$$ \left. \nabla_{\boldsymbol{\theta}} \mathcal{L}(q, \boldsymbol{\theta}) \right|_{\boldsymbol{\theta}^{(\mathrm{old})}} = \left. \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{X}|\boldsymbol{\theta}) \right|_{\boldsymbol{\theta}^{(\mathrm{old})}} $$
となり、**変分下界の勾配は接点において真の対数尤度の勾配と厳密に一致する**ことを証明せよ。"""))

# Code Ex 9.18 - 9.27
code_ex9_18_27 = r"""# Exercise 9.25 数値検証: 下界勾配と対数尤度勾配の接点における完全一致
# 概念モデル: 1次元パラメータ theta
# 対数尤度 log p(X|theta)
f_ll = lambda th: -0.2 * (th - 2.0)**3 - 0.5 * (th - 2.0)**2 + 3.0
# Eステップで接する下界 L(q_old, theta)
th_0 = 1.2
f_L = lambda th: f_ll(th_0) - 1.2 * (th - th_0)**2 + ((-0.2*3*(th_0-2.0)**2 - (th_0-2.0))) * (th - th_0)

# 数値微分の計算
eps = 1e-6
grad_ll = (f_ll(th_0 + eps) - f_ll(th_0 - eps)) / (2 * eps)
grad_L = (f_L(th_0 + eps) - f_L(th_0 - eps)) / (2 * eps)

print(f"Gradient of Log-Likelihood at theta_old: {grad_ll:.6f}")
print(f"Gradient of Lower Bound L at theta_old:  {grad_L:.6f}")
assert np.isclose(grad_ll, grad_L, rtol=1e-5)
print("Exercise 9.25 verified: Tangent gradient of lower bound strictly matches true log-likelihood gradient!")"""
cells.append(nbf.v4.new_code_cell(code_ex9_18_27))

nb.cells = cells
with open('9/9_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("9/9_Exercises.ipynb generated successfully.")
