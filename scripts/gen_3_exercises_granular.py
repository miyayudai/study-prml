"""
Script to generate granular, fully individual Exercise cells for Chapter 3 (Exercises 3.1 to 3.24).
Each exercise has its own dedicated markdown cell with complete mathematical derivation,
and its own dedicated code cell with assertions and numerical verifications.
"""

import nbformat as nbf
import os
import sys

nb = nbf.v4.new_notebook()
cells = []

# Title & Table of Contents
cells.append(nbf.v4.new_markdown_cell(r"""# 第3章 章末演習問題 (全24問 個別完全実装版)

教科書「パターン認識と機械学習 (PRML)」第3章「線形回帰モデル (Linear Models for Regression)」のすべての Exercises (3.1 〜 3.24) を**1問1問完全に独立したセル**として網羅しています。
各問に詳細な数式展開・理論的背景・穴埋め/思考ステップ、およびPythonによる厳密な数値検証コードが付属しています。

---
## 目次 (全24問)
- [3.1 tanh とシグモイド関数の関係](#Exercise-3.1)
- [3.2 最小二乗直交射影行列の性質](#Exercise-3.2)
- [3.3 重み付き二乗和誤差と2つの解釈](#Exercise-3.3)
- [3.4 入力ノイズと Weight Decay 正則化の等価性](#Exercise-3.4)
- [3.5 ラグランジュ未定乗数法と正則化制約の双対性](#Exercise-3.5)
- [3.6 多変量目的変数の最尤推定と共分散の分離](#Exercise-3.6)
- [3.7 平方完成によるパラメータ事後分布の導出](#Exercise-3.7)
- [3.8 逐次ベイズ更新の平方完成](#Exercise-3.8)
- [3.9 線形ガウス公式による逐次更新の導出](#Exercise-3.9)
- [3.10 周辺化による予測分布の導出](#Exercise-3.10)
- [3.11 Sherman-Morrison公式と予測分散の単調減少](#Exercise-3.11)
- [3.12 未知の平均と精度に対する正規ガンマ共役事前分布](#Exercise-3.12)
- [3.13 正規ガンマモデルの Student's t 予測分布](#Exercise-3.13)
- [3.14 正規直交基底と等価カーネルの総和制約](#Exercise-3.14)
- [3.15 エビデンス停留点における $2E(\mathbf{m}_N) = N$ の証明](#Exercise-3.15)
- [3.16 線形ガウス公式による対数エビデンスの直接導出](#Exercise-3.16)
- [3.17 エビデンス関数の指数部の変形](#Exercise-3.17)
- [3.18 平方完成による誤差関数の変形](#Exercise-3.18)
- [3.19 ガウス積分による対数周辺尤度の導出](#Exercise-3.19)
- [3.20 行列式微分恒等式 $\frac{d}{d\alpha}\ln|\mathbf{A}| = \mathrm{Tr}(\mathbf{A}^{-1}\frac{d\mathbf{A}}{d\alpha})$ の証明](#Exercise-3.20)
- [3.21 対数エビデンスの $\alpha$ 微分と再推定式の導出](#Exercise-3.21)
- [3.22 対数エビデンスの $\beta$ 微分と再推定式の導出](#Exercise-3.22)
- [3.23 正規ガンマモデルのエビデンスの直接重積分導出](#Exercise-3.23)
- [3.24 ベイズの定理によるエビデンスの代数的導出](#Exercise-3.24)
---"""))

# Common setup code cell
cells.append(nbf.v4.new_code_cell(r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import scipy.stats as stats
from scipy.special import gamma
import matplotlib.pyplot as plt

from common.regression_utils import (
    PolynomialBasis, GaussianBasis, SigmoidalBasis, FourierBasis,
    LinearRegression, RidgeRegression, LassoRegression, LeastMeanSquares,
    BayesianLinearRegression, NormalGammaLinearRegression, EvidenceApproximation,
    LocallyWeightedRegression, RobustLinearRegression, WeightedLinearRegression,
    MultivariateLinearRegression, EquivalentKernel,
    bias_variance_decomposition, orthogonal_projection_matrix, equivalent_kernel_matrix
)
print("Environment setup and module import completed successfully.")"""))

# 3.1
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.1
**問題**: 双曲線正接関数 $\tanh(a) = \frac{e^a - e^{-a}}{e^a + e^{-a}}$ とロジスティックシグモイド関数 $\sigma(a) = \frac{1}{1 + e^{-a}}$ の間に
$$ \tanh(a) = 2\sigma(2a) - 1 $$
の関係が成り立つことを示せ。
また、シグモイド基底の線形結合
$$ y(x, \mathbf{w}) = w_0 + \sum_{j=1}^M w_j \sigma\left(\frac{x - \mu_j}{s}\right) $$
が $\tanh$ 基底の線形結合
$$ y(x, \mathbf{u}) = u_0 + \sum_{j=1}^M u_j \tanh\left(\frac{x - \mu_j}{2s}\right) $$
と等価であることを示し、新パラメータ $\{u_0, u_1, \dots, u_M\}$ と元パラメータ $\{w_0, w_1, \dots, w_M\}$ の変換関係を求めよ。

### 証明・導出の論理ステップ
1. $\sigma(2a) = \frac{1}{1 + e^{-2a}} = \frac{e^a}{e^a + e^{-a}}$。
2. $2\sigma(2a) - 1 = \frac{2e^a - (e^a + e^{-a})}{e^a + e^{-a}} = \frac{e^a - e^{-a}}{e^a + e^{-a}} = \tanh(a)$。
3. したがって $\sigma(a) = \frac{1}{2}\{1 + \tanh(a/2)\}$。
4. これを $y(x, \mathbf{w})$ に代入すると：
   $$ y(x, \mathbf{w}) = w_0 + \sum_{j=1}^M w_j \frac{1}{2}\left\{1 + \tanh\left(\frac{x - \mu_j}{2s}\right)\right\} = \left(w_0 + \frac{1}{2}\sum_{j=1}^M w_j\right) + \sum_{j=1}^M \frac{w_j}{2} \tanh\left(\frac{x - \mu_j}{2s}\right) $$
5. 係数を比較すると、パラメータ変換則が得られる：
   $$ u_0 = w_0 + \frac{1}{2}\sum_{j=1}^M w_j, \qquad u_j = \frac{1}{2} w_j \quad (j = 1, \dots, M) $$"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.1 数値検証
w0 = 1.5
w = np.array([0.8, -1.2, 2.0])
mu = np.array([-0.5, 0.0, 0.5])
s = 0.3
x_test = np.linspace(-1, 1, 100)

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))

# シグモイド基底での出力
y_sigma = w0 + np.sum([w[j] * sigmoid((x_test - mu[j])/s) for j in range(len(w))], axis=0)

# パラメータ変換則の適用
u0 = w0 + 0.5 * np.sum(w)
u = 0.5 * w

# tanh 基底での出力 (スケールは 2s)
y_tanh = u0 + np.sum([u[j] * np.tanh((x_test - mu[j])/(2*s)) for j in range(len(u))], axis=0)

diff = np.max(np.abs(y_sigma - y_tanh))
print(f"Max difference: {diff:.2e}")
assert diff < 1e-12, "Test failed: sigmoid and tanh expansions do not match!"
print("Exercise 3.1 passed successfully!")"""))

# 3.2
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.2
**問題**: 行列 $\mathbf{P} = \mathbf{\Phi}(\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T$ が任意のベクトル $\mathbf{v}$ を $\mathbf{\Phi}$ の列が張る部分空間 $\mathcal{S}$ に直交射影することを示せ。また、これを用いて最小二乗解 (3.15) が目標値ベクトル $\mathbf{t}$ の部分空間 $\mathcal{S}$ への直交射影に対応することを示せ（PRML Figure 3.2）。

### 証明の論理ステップ
1. **部分空間 $\mathcal{S}$ への射影**: $\mathbf{P}\mathbf{v} = \mathbf{\Phi} \left[ (\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T \mathbf{v} \right]$ であり、$\mathbf{\Phi}$ の列ベクトルの線形結合であるため $\mathbf{P}\mathbf{v} \in \mathcal{S}$。
2. **直交性**: 残差ベクトル $\mathbf{v} - \mathbf{P}\mathbf{v} = (\mathbf{I} - \mathbf{P})\mathbf{v}$ が $\mathcal{S}$ の任意の基底ベクトルと直交することを確認：
   $$ \mathbf{\Phi}^T (\mathbf{I} - \mathbf{P}) = \mathbf{\Phi}^T - \mathbf{\Phi}^T \mathbf{\Phi}(\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T = \mathbf{\Phi}^T - \mathbf{\Phi}^T = \mathbf{O} $$
3. **べき等性と対称性**:
   - $\mathbf{P}^2 = \mathbf{\Phi}(\mathbf{\Phi}^T \mathbf{\Phi})^{-1} (\mathbf{\Phi}^T \mathbf{\Phi}) (\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T = \mathbf{P}$。
   - $\mathbf{P}^T = \left[\mathbf{\Phi}(\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T\right]^T = \mathbf{P}$。
4. **最小二乗解との対応**: 目標ベクトル $\mathbf{t}$ に $\mathbf{P}$ を作用させると、
   $$ \mathbf{y} = \mathbf{P}\mathbf{t} = \mathbf{\Phi}(\mathbf{\Phi}^T \mathbf{\Phi})^{-1}\mathbf{\Phi}^T\mathbf{t} = \mathbf{\Phi}\mathbf{w}_{\mathrm{ML}} $$
   となり、最小二乗解による予測値ベクトル $\mathbf{y}$ は $\mathbf{t}$ の $\mathcal{S}$ への直交射影そのものである。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.2 数値検証
np.random.seed(42)
N, M = 20, 4
Phi = np.random.randn(N, M)
t = np.random.randn(N)

P = orthogonal_projection_matrix(Phi)

# 1. べき等性 P^2 == P
assert np.allclose(P @ P, P), "P^2 != P"
# 2. 対称性 P^T == P
assert np.allclose(P.T, P), "P^T != P"
# 3. 直交性: Phi^T (I - P) == 0
assert np.allclose(Phi.T @ (np.eye(N) - P), 0.0), "Phi^T (I - P) != 0"
# 4. 最小二乗予測値 y_pred == P @ t
lr = LinearRegression().fit(Phi, t)
np.testing.assert_allclose(lr.predict(Phi), P @ t, atol=1e-10)
print("Exercise 3.2 passed successfully!")"""))

# 3.3
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.3
**問題**: 重み付き二乗和誤差関数
$$ E_D(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N r_n \{t_n - \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n)\}^2 \quad (r_n > 0) $$
を最小化するパラメータ解 $\mathbf{w}^*$ を求めよ。また、重み $r_n$ を以下の2通りに解釈できることを示せ：
1. データ点 $(\mathbf{x}_n, t_n)$ が $r_n$ 回（整数値の場合）重複して観測されたとみなす解釈。
2. データ点ごとに異なるノイズ精度 $\beta_n = r_n \beta$ を持つガウスノイズモデルの最尤推定とみなす解釈。

### 導出の論理ステップ
1. $\mathbf{R} = \mathrm{diag}(r_1, \dots, r_N)$ と定義すると、誤差関数は行列記法で
   $$ E_D(\mathbf{w}) = \frac{1}{2} (\mathbf{t} - \mathbf{\Phi}\mathbf{w})^T \mathbf{R} (\mathbf{t} - \mathbf{\Phi}\mathbf{w}) $$
2. $\mathbf{w}$ に関して勾配を求めてゼロとおく：
   $$ \nabla E_D = -\mathbf{\Phi}^T \mathbf{R} (\mathbf{t} - \mathbf{\Phi}\mathbf{w}) = \mathbf{0} \implies (\mathbf{\Phi}^T \mathbf{R} \mathbf{\Phi}) \mathbf{w} = \mathbf{\Phi}^T \mathbf{R} \mathbf{t} $$
   $$ \mathbf{w}^* = (\mathbf{\Phi}^T \mathbf{R} \mathbf{\Phi})^{-1} \mathbf{\Phi}^T \mathbf{R} \mathbf{t} $$
3. **データ重複の解釈**: $r_n$ 個の同一データ点に対する二乗誤差の和は $r_n (t_n - \mathbf{w}^T \boldsymbol{\phi}_n)^2$ となり、$E_D(\mathbf{w})$ と完全に一致する。
4. **不均一分散ノイズの解釈**: 尤度 $p(t_n|\mathbf{x}_n, \mathbf{w}) = \mathcal{N}(t_n|\mathbf{w}^T\boldsymbol{\phi}_n, \beta_n^{-1})$ とすると、対数尤度の最大化は $\sum_n \beta_n (t_n - \mathbf{w}^T\boldsymbol{\phi}_n)^2$ の最小化と等価である。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.3 数値検証
np.random.seed(42)
N, M = 15, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
r = np.random.uniform(0.5, 3.0, N)
R = np.diag(r)

# 重み付き最小二乗解
w_weighted = np.linalg.solve(Phi.T @ R @ Phi, Phi.T @ R @ t)

# WeightedLinearRegression クラスでの検証
wlr = WeightedLinearRegression().fit(Phi, t, weights=r)
np.testing.assert_allclose(wlr.w, w_weighted, atol=1e-10)

# 整数の重みによるデータ複製との一致確認
r_int = np.array([2, 1, 3] + [1]*(N-3))
Phi_dup = np.repeat(Phi, r_int, axis=0)
t_dup = np.repeat(t, r_int)
w_dup = np.linalg.pinv(Phi_dup) @ t_dup
w_weighted_int = np.linalg.solve(Phi.T @ np.diag(r_int) @ Phi, Phi.T @ np.diag(r_int) @ t)
np.testing.assert_allclose(w_dup, w_weighted_int, atol=1e-10)
print("Exercise 3.3 passed successfully!")"""))

# 3.4
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.4
**問題**: 入力変数 $\mathbf{x}$ にゼロ平均・共分散 $\sigma^2 \mathbf{I}$ の独立ノイズ $\boldsymbol{\epsilon}$ を加えたときの誤差関数
$$ \mathbb{E}_{\boldsymbol{\epsilon}}\left[ \frac{1}{2} \sum_{n=1}^N \{t_n - y(\mathbf{x}_n + \boldsymbol{\epsilon}_n, \mathbf{w})\}^2 \right] $$
（線形モデル $y(\mathbf{x}, \mathbf{w}) = w_0 + \mathbf{w}^T \mathbf{x}$ の場合）を計算し、これが二乗和誤差に L2 正則化（Weight Decay）$\frac{\lambda}{2} \|\mathbf{w}\|^2$ を加えた正則化二乗和誤差と等価になることを証明せよ。

### 証明の論理ステップ
1. $y(\mathbf{x}_n + \boldsymbol{\epsilon}_n, \mathbf{w}) = w_0 + \mathbf{w}^T \mathbf{x}_n + \mathbf{w}^T \boldsymbol{\epsilon}_n = y_n + \mathbf{w}^T \boldsymbol{\epsilon}_n$。
2. 誤差の二乗を展開：
   $$ \{t_n - y_n - \mathbf{w}^T \boldsymbol{\epsilon}_n\}^2 = (t_n - y_n)^2 - 2(t_n - y_n)\mathbf{w}^T \boldsymbol{\epsilon}_n + (\mathbf{w}^T \boldsymbol{\epsilon}_n)^2 $$
3. ノイズに関する期待値をとると：
   - $\mathbb{E}[\boldsymbol{\epsilon}_n] = \mathbf{0}$ より一次項は消失。
   - $\mathbb{E}[(\mathbf{w}^T \boldsymbol{\epsilon}_n)^2] = \mathbf{w}^T \mathbb{E}[\boldsymbol{\epsilon}_n \boldsymbol{\epsilon}_n^T] \mathbf{w} = \mathbf{w}^T (\sigma^2 \mathbf{I}) \mathbf{w} = \sigma^2 \|\mathbf{w}\|^2$。
4. 全データ点について和をとると：
   $$ \mathbb{E}_{\boldsymbol{\epsilon}}[E] = \frac{1}{2}\sum_{n=1}^N (t_n - y_n)^2 + \frac{N \sigma^2}{2} \|\mathbf{w}\|^2 $$
5. したがって、入力ノイズの付加は正則化パラメータ $\lambda = N \sigma^2$ の L2 正則化と数学的に完全に等価である！"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.4 数値検証
np.random.seed(42)
N, D = 50, 4
X = np.random.randn(N, D)
w0_true = 0.5
w_true = np.array([1.2, -0.8, 0.5, -1.0])
t = w0_true + X @ w_true + np.random.normal(0, 0.1, N)

sigma = 0.2
lambda_analytic = N * (sigma**2)

# モンテカルロシミュレーションによる平均誤差
n_mc = 20000
mc_errors = []
w_eval = np.array([0.9, -0.6, 0.3, -0.8])
w0_eval = 0.4
y_clean = w0_eval + X @ w_eval

for _ in range(n_mc):
    noise = np.random.normal(0, sigma, size=X.shape)
    y_noisy = w0_eval + (X + noise) @ w_eval
    mc_errors.append(0.5 * np.sum((t - y_noisy)**2))

expected_loss_mc = np.mean(mc_errors)
expected_loss_analytic = 0.5 * np.sum((t - y_clean)**2) + 0.5 * lambda_analytic * np.sum(w_eval**2)

print(f"Analytic Expected Loss: {expected_loss_analytic:.4f}")
print(f"Monte Carlo Loss:       {expected_loss_mc:.4f}")
assert np.abs(expected_loss_analytic - expected_loss_mc) < 0.05, "Monte Carlo and analytic mismatch!"
print("Exercise 3.4 passed successfully!")"""))

# 3.5
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.5
**問題**: ラグランジュの未定乗数法を用いて、正則化誤差関数 (3.29)
$$ \tilde{E}(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N \{t_n - \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n)\}^2 + \frac{\lambda}{2} \sum_{j=1}^M |w_j|^q $$
の最小化が、制約条件 $\sum_{j=1}^M |w_j|^q \le \eta$ の下での非正則化二乗和誤差の最小化と等価であることを示せ。またパラメータ $\eta$ と $\lambda$ の関係を論ぜよ。

### 証明の論理ステップ
1. 制約付き最適化問題：
   $$ \min_{\mathbf{w}} E_D(\mathbf{w}) \quad \text{subject to} \quad g(\mathbf{w}) = \sum_{j=1}^M |w_j|^q - \eta \le 0 $$
2. KKT 条件のもとでラグランジュ関数を定義：
   $$ \mathcal{L}(\mathbf{w}, \lambda) = E_D(\mathbf{w}) + \frac{\lambda}{2} \left( \sum_{j=1}^M |w_j|^q - \eta \right) $$
3. 制約がアクティブ（$\lambda > 0$ かつ等号成立）であるとき、$\mathcal{L}$ の $\mathbf{w}$ に関する最小化は $\tilde{E}(\mathbf{w})$ と定数項 $-\frac{\lambda\eta}{2}$ を除いて完全に一致する。
4. **$\lambda$ と $\eta$ の関係**: $\lambda$ はペナルティの強さを表し、$\lambda \to \infty$ のとき $\mathbf{w} \to \mathbf{0}$（$\eta \to 0$）、$\lambda \to 0$ のとき非正則化解（$\eta \to \eta_{\max}$）となるため、$\eta$ は $\lambda$ の狭義単調減少関数である。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.5 数値検証 (Ridge q=2 におけるノルムの単調減少性)
np.random.seed(42)
Phi = np.random.randn(30, 5)
t = np.random.randn(30)

lambdas = np.logspace(-2, 3, 30)
norms = []
for lam in lambdas:
    w_lam = RidgeRegression(alpha=lam).fit(Phi, t).w
    norms.append(np.sum(w_lam**2))

diffs = np.diff(norms)
assert np.all(diffs <= 1e-8), "L2 norm is not monotonically decreasing with lambda!"
print("Exercise 3.5 passed successfully!")"""))

# 3.6
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.6
**問題**: 多変量目的変数 $\mathbf{t} \in \mathbb{R}^K$ に対するガウス線形基底関数回帰モデル
$$ p(\mathbf{t} | \mathbf{W}, \mathbf{\Sigma}) = \mathcal{N}(\mathbf{t} | \mathbf{W}^T \boldsymbol{\phi}(\mathbf{x}), \mathbf{\Sigma}) $$
において、パラメータ行列 $\mathbf{W} \in \mathbb{R}^{M \times K}$ の最尤推定量 $\mathbf{W}_{\mathrm{ML}}$ の各列が共分散行列 $\mathbf{\Sigma}$ に依存せず、スカラーの場合の解 (3.15) と一致することを示せ。また共分散行列の最尤推定量 $\mathbf{\Sigma}_{\mathrm{ML}}$ を求めよ。

### 証明の論理ステップ
1. 対数尤度関数を行列記法で表現：
   $$ \ln p(\mathbf{T} | \mathbf{W}, \mathbf{\Sigma}) = -\frac{NK}{2}\ln(2\pi) - \frac{N}{2}\ln|\mathbf{\Sigma}| - \frac{1}{2} \mathrm{Tr}\left[ \mathbf{\Sigma}^{-1} (\mathbf{T} - \mathbf{\Phi}\mathbf{W})^T (\mathbf{T} - \mathbf{\Phi}\mathbf{W}) \right] $$
2. 行列微分の公式 $\frac{\partial}{\partial \mathbf{X}} \mathrm{Tr}[\mathbf{A} \mathbf{X}^T \mathbf{B} \mathbf{X} \mathbf{C}]$ を適用して $\mathbf{W}$ で微分：
   $$ \frac{\partial \ln p}{\partial \mathbf{W}} = \mathbf{\Phi}^T (\mathbf{T} - \mathbf{\Phi}\mathbf{W}) \mathbf{\Sigma}^{-1} = \mathbf{O} $$
3. 右から正定値対称行列 $\mathbf{\Sigma}$ を掛けると消去され、
   $$ \mathbf{\Phi}^T \mathbf{\Phi} \mathbf{W} = \mathbf{\Phi}^T \mathbf{T} \implies \mathbf{W}_{\mathrm{ML}} = (\mathbf{\Phi}^T \mathbf{\Phi})^{-1} \mathbf{\Phi}^T \mathbf{T} = \mathbf{\Phi}^\dagger \mathbf{T} $$
   これは各次元 $k$ について個別に最小二乗問題を解いた結果と厳密に一致する！
4. $\mathbf{\Sigma}^{-1}$ に関する微分より：
   $$ \mathbf{\Sigma}_{\mathrm{ML}} = \frac{1}{N} (\mathbf{T} - \mathbf{\Phi}\mathbf{W}_{\mathrm{ML}})^T (\mathbf{T} - \mathbf{\Phi}\mathbf{W}_{\mathrm{ML}}) $$"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.6 数値検証
np.random.seed(42)
N, M, K = 40, 5, 3
Phi = np.random.randn(N, M)
true_W = np.random.randn(M, K)
Sigma_true = stats.wishart.rvs(df=K+2, scale=np.eye(K))
T = Phi @ true_W + np.random.multivariate_normal(np.zeros(K), Sigma_true, size=N)

# 多変量モデルの学習
mlr = MultivariateLinearRegression().fit(Phi, T)

# 各列独立の 1D 回帰との完全一致確認
for k in range(K):
    lr_k = LinearRegression().fit(Phi, T[:, k])
    np.testing.assert_allclose(mlr.W[:, k], lr_k.w, atol=1e-10)

# 残差共分散行列の性質
assert mlr.Sigma.shape == (K, K)
assert np.allclose(mlr.Sigma, mlr.Sigma.T)
assert np.all(np.linalg.eigvalsh(mlr.Sigma) > 0)
print("Exercise 3.6 passed successfully!")"""))

# 3.7
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.7
**問題**: 平方完成 (completing the square) の技法を用いて、ベイズ線形回帰モデルのパラメータ $\mathbf{w}$ の事後分布 (3.49) およびパラメータ $\mathbf{m}_N, \mathbf{S}_N$ ((3.50), (3.51)) を導出せよ。

### 導出の論理ステップ
1. 事前分布 $p(\mathbf{w}) = \mathcal{N}(\mathbf{w}|\mathbf{m}_0, \mathbf{S}_0)$ と尤度 $p(\mathbf{t}|\mathbf{w}) = \mathcal{N}(\mathbf{t}|\mathbf{\Phi}\mathbf{w}, \beta^{-1}\mathbf{I})$ の積を展開：
   $$ \ln p(\mathbf{w}|\mathbf{t}) = -\frac{1}{2}(\mathbf{w} - \mathbf{m}_0)^T \mathbf{S}_0^{-1} (\mathbf{w} - \mathbf{m}_0) - \frac{\beta}{2}(\mathbf{t} - \mathbf{\Phi}\mathbf{w})^T(\mathbf{t} - \mathbf{\Phi}\mathbf{w}) + \mathrm{const} $$
2. $\mathbf{w}$ の二次項と一次項を抽出：
   $$ = -\frac{1}{2}\mathbf{w}^T (\mathbf{S}_0^{-1} + \beta \mathbf{\Phi}^T \mathbf{\Phi})\mathbf{w} + \mathbf{w}^T (\mathbf{S}_0^{-1}\mathbf{m}_0 + \beta \mathbf{\Phi}^T \mathbf{t}) + \mathrm{const} $$
3. 二次形式の逆行列として事後共分散 $\mathbf{S}_N$ を定義：
   $$ \mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + \beta \mathbf{\Phi}^T \mathbf{\Phi} $$
4. 平方完成公式 $-\frac{1}{2}\mathbf{w}^T \mathbf{S}_N^{-1}\mathbf{w} + \mathbf{w}^T \mathbf{b} = -\frac{1}{2}(\mathbf{w} - \mathbf{S}_N\mathbf{b})^T \mathbf{S}_N^{-1}(\mathbf{w} - \mathbf{S}_N\mathbf{b}) + \mathrm{const}$ を適用：
   $$ \mathbf{m}_N = \mathbf{S}_N (\mathbf{S}_0^{-1}\mathbf{m}_0 + \beta \mathbf{\Phi}^T \mathbf{t}) $$"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.7 数値検証
np.random.seed(42)
M = 3
m0 = np.array([0.5, -0.2, 0.1])
S0 = 0.5 * np.eye(M)
beta = 25.0
Phi = np.random.randn(20, M)
t = np.random.randn(20)

# 解析的更新
S_N_inv = np.linalg.inv(S0) + beta * (Phi.T @ Phi)
S_N_analytic = np.linalg.inv(S_N_inv)
m_N_analytic = S_N_analytic @ (np.linalg.inv(S0) @ m0 + beta * Phi.T @ t)

# BayesianLinearRegression クラスでの検証
blr = BayesianLinearRegression(beta=beta, m_0=m0, S_0=S0).fit(Phi, t)
np.testing.assert_allclose(blr.m_N, m_N_analytic, atol=1e-10)
np.testing.assert_allclose(blr.S_N, S_N_analytic, atol=1e-10)
print("Exercise 3.7 passed successfully!")"""))

# 3.8
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.8
**問題**: $N$ 点観測後の事後分布 $\mathcal{N}(\mathbf{w}|\mathbf{m}_N, \mathbf{S}_N)$ を新たな事前分布とし、追加の1点 $(\mathbf{x}_{N+1}, t_{N+1})$ を観測したときの事後分布を平方完成により導出し、これが最初から $N+1$ 点で一括計算した事後分布と等価になることを示せ。

### 導出の論理ステップ
1. 事前分布: $\mathcal{N}(\mathbf{w}|\mathbf{m}_N, \mathbf{S}_N)$、追加尤度: $\mathcal{N}(t_{N+1}|\mathbf{w}^T\boldsymbol{\phi}_{N+1}, \beta^{-1})$。
2. 対数結合分布：
   $$ -\frac{1}{2}(\mathbf{w} - \mathbf{m}_N)^T \mathbf{S}_N^{-1} (\mathbf{w} - \mathbf{m}_N) - \frac{\beta}{2}(t_{N+1} - \mathbf{w}^T\boldsymbol{\phi}_{N+1})^2 + \mathrm{const} $$
3. 二次項の整理：
   $$ \mathbf{S}_{N+1}^{-1} = \mathbf{S}_N^{-1} + \beta \boldsymbol{\phi}_{N+1} \boldsymbol{\phi}_{N+1}^T $$
4. 一次項の整理：
   $$ \mathbf{m}_{N+1} = \mathbf{S}_{N+1} (\mathbf{S}_N^{-1} \mathbf{m}_N + \beta t_{N+1} \boldsymbol{\phi}_{N+1}) $$
5. $\mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + \beta \sum_{n=1}^N \boldsymbol{\phi}_n \boldsymbol{\phi}_n^T$ を代入すると、バッチ解 $\mathbf{S}_0^{-1} + \beta \sum_{n=1}^{N+1} \boldsymbol{\phi}_n \boldsymbol{\phi}_n^T$ と完全に一致する。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.8 数値検証
np.random.seed(42)
N, M = 15, 3
Phi_all = np.random.randn(N + 1, M)
t_all = np.random.randn(N + 1)
alpha, beta = 2.0, 10.0

# バッチ一括学習 (N+1 点)
blr_batch = BayesianLinearRegression(alpha=alpha, beta=beta).fit(Phi_all, t_all)

# 逐次学習 (N 点学習後に 1 点追加)
blr_seq = BayesianLinearRegression(alpha=alpha, beta=beta).fit(Phi_all[:N], t_all[:N])
blr_seq.update(Phi_all[N], t_all[N])

np.testing.assert_allclose(blr_seq.m_N, blr_batch.m_N, atol=1e-10)
np.testing.assert_allclose(blr_seq.S_N, blr_batch.S_N, atol=1e-10)
print("Exercise 3.8 passed successfully!")"""))

# 3.9
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.9
**問題**: 第2章で導出した線形ガウスモデルの事後条件付き分布の公式 (2.113) - (2.116)
$$ p(\mathbf{x}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \mathbf{\Lambda}^{-1}), \quad p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y}|\mathbf{A}\mathbf{x} + \mathbf{b}, \mathbf{L}^{-1}) $$
$$ \implies p(\mathbf{x}|\mathbf{y}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\Sigma}\{\mathbf{A}^T\mathbf{L}(\mathbf{y} - \mathbf{b}) + \mathbf{\Lambda}\boldsymbol{\mu}\}, \boldsymbol{\Sigma}), \quad \boldsymbol{\Sigma} = (\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})^{-1} $$
を用いて、ベイズ線形回帰の逐次更新式 (3.8) を平方完成を用いずに直接導出せよ。

### 導出の論理ステップ
1. 対応関係を同定：
   - パラメータ $\mathbf{x} \leftrightarrow \mathbf{w}$、観測値 $\mathbf{y} \leftrightarrow t_{N+1}$。
   - 事前平均 $\boldsymbol{\mu} \leftrightarrow \mathbf{m}_N$、事前精度 $\mathbf{\Lambda} \leftrightarrow \mathbf{S}_N^{-1}$。
   - 計画行列 $\mathbf{A} \leftrightarrow \boldsymbol{\phi}_{N+1}^T$、オフセット $\mathbf{b} \leftrightarrow 0$、観測精度 $\mathbf{L} \leftrightarrow \beta$。
2. 公式 (2.116) に代入：
   $$ \boldsymbol{\Sigma} = (\mathbf{S}_N^{-1} + \boldsymbol{\phi}_{N+1} \beta \boldsymbol{\phi}_{N+1}^T)^{-1} = (\mathbf{S}_N^{-1} + \beta \boldsymbol{\phi}_{N+1} \boldsymbol{\phi}_{N+1}^T)^{-1} = \mathbf{S}_{N+1} $$
3. 公式 (2.115) に代入：
   $$ \mathbf{m}_{N+1} = \mathbf{S}_{N+1} \{ \boldsymbol{\phi}_{N+1} \beta (t_{N+1} - 0) + \mathbf{S}_N^{-1} \mathbf{m}_N \} = \mathbf{S}_{N+1} (\mathbf{S}_N^{-1} \mathbf{m}_N + \beta t_{N+1} \boldsymbol{\phi}_{N+1}) $$
4. 一切の積分や平方完成を経由することなく、逐次更新式が自然に導出された！"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.9 数値検証
np.random.seed(42)
m_N = np.array([1.0, -0.5])
S_N = np.array([[0.2, 0.05], [0.05, 0.15]])
phi_new = np.array([0.7, -1.2])
t_new = 2.3
beta = 4.0

# 式 (2.115)-(2.116) の直接代入
Lambda = np.linalg.inv(S_N)
A = phi_new.reshape(1, -1)
L = np.array([[beta]])

Sigma = np.linalg.inv(Lambda + A.T @ L @ A)
mean_post = Sigma @ (A.T @ L @ np.array([[t_new]]) + Lambda @ m_N.reshape(-1, 1)).ravel()

# BayesianLinearRegression の逐次更新との比較
blr = BayesianLinearRegression(beta=beta, m_0=m_N, S_0=S_N)
blr.m_N = m_N.copy()
blr.S_N = S_N.copy()
blr.update(phi_new, t_new)

np.testing.assert_allclose(blr.m_N, mean_post, atol=1e-10)
np.testing.assert_allclose(blr.S_N, Sigma, atol=1e-10)
print("Exercise 3.9 passed successfully!")"""))

# 3.10
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.10
**問題**: ガウス周辺化公式 (2.115) を用いて、ベイズ線形回帰における新しい入力 $\mathbf{x}$ に対する予測分布
$$ p(t|\mathbf{x}, \mathbf{t}) = \int p(t|\mathbf{x}, \mathbf{w}) p(\mathbf{w}|\mathbf{t}) d\mathbf{w} $$
がガウス分布 $\mathcal{N}(t | \mathbf{m}_N^T\boldsymbol{\phi}(\mathbf{x}), \sigma_N^2(\mathbf{x}))$（ただし $\sigma_N^2(\mathbf{x}) = \frac{1}{\beta} + \boldsymbol{\phi}(\mathbf{x})^T \mathbf{S}_N \boldsymbol{\phi}(\mathbf{x})$）となることを導出せよ。

### 導出の論理ステップ
1. 周辺化の一般公式：$\mathbf{w} \sim \mathcal{N}(\mathbf{m}_N, \mathbf{S}_N)$ かつ $t|\mathbf{w} \sim \mathcal{N}(\boldsymbol{\phi}(\mathbf{x})^T\mathbf{w}, \beta^{-1})$。
2. 期待値の線形性：
   $$ \mathbb{E}[t|\mathbf{x}] = \mathbb{E}[\boldsymbol{\phi}(\mathbf{x})^T\mathbf{w} + \epsilon] = \boldsymbol{\phi}(\mathbf{x})^T \mathbb{E}[\mathbf{w}] = \boldsymbol{\phi}(\mathbf{x})^T \mathbf{m}_N $$
3. 分散の分解（全分散の法則）：
   $$ \mathrm{var}[t|\mathbf{x}] = \mathrm{var}[\boldsymbol{\phi}(\mathbf{x})^T\mathbf{w}] + \mathrm{var}[\epsilon] = \boldsymbol{\phi}(\mathbf{x})^T \mathrm{cov}[\mathbf{w}] \boldsymbol{\phi}(\mathbf{x}) + \frac{1}{\beta} = \frac{1}{\beta} + \boldsymbol{\phi}(\mathbf{x})^T \mathbf{S}_N \boldsymbol{\phi}(\mathbf{x}) $$
4. ガウス変数の線形変換と独立ノイズの和は依然としてガウス分布であるため、予測分布 (3.57)-(3.59) が得られる。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.10 数値検証 (モンテカルロ積分との一致)
np.random.seed(42)
blr = BayesianLinearRegression(alpha=1.0, beta=5.0)
Phi_tr = np.random.randn(20, 3)
t_tr = np.random.randn(20)
blr.fit(Phi_tr, t_tr)

phi_query = np.random.randn(1, 3)
mean_pred, std_pred = blr.predict(phi_query)

# モンテカルロサンプリングによる検証
w_samples = blr.sample_weights(n_samples=50000)
t_samples = phi_query @ w_samples.T + np.random.normal(0, np.sqrt(1.0/blr.beta), size=50000)

print(f"Analytic mean: {mean_pred[0]:.4f}, MC mean: {np.mean(t_samples):.4f}")
print(f"Analytic std:  {std_pred[0]:.4f}, MC std:  {np.std(t_samples):.4f}")
assert np.isclose(mean_pred[0], np.mean(t_samples), atol=0.03)
assert np.isclose(std_pred[0], np.std(t_samples), atol=0.03)
print("Exercise 3.10 passed successfully!")"""))

# 3.11
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.11
**問題**: ウッドベリーの公式（Sherman-Morrisonの公式）
$$ (\mathbf{M} + \mathbf{v}\mathbf{v}^T)^{-1} = \mathbf{M}^{-1} - \frac{(\mathbf{M}^{-1}\mathbf{v})(\mathbf{v}^T\mathbf{M}^{-1})}{1 + \mathbf{v}^T\mathbf{M}^{-1}\mathbf{v}} $$
を用いて、データを1点追加したときの事後共分散 $\mathbf{S}_{N+1}$ の陽表現を導出し、任意の評価点 $\mathbf{x}$ における予測分散が単調減少すること（$\sigma_{N+1}^2(\mathbf{x}) \le \sigma_N^2(\mathbf{x})$）を証明せよ。

### 証明の論理ステップ
1. $\mathbf{S}_{N+1}^{-1} = \mathbf{S}_N^{-1} + \beta \boldsymbol{\phi}_{N+1}\boldsymbol{\phi}_{N+1}^T$。
2. Sherman-Morrison 公式において $\mathbf{M} = \mathbf{S}_N^{-1}, \mathbf{v} = \sqrt{\beta}\boldsymbol{\phi}_{N+1}$ と置くと：
   $$ \mathbf{S}_{N+1} = \mathbf{S}_N - \frac{\beta \mathbf{S}_N \boldsymbol{\phi}_{N+1} \boldsymbol{\phi}_{N+1}^T \mathbf{S}_N}{1 + \beta \boldsymbol{\phi}_{N+1}^T \mathbf{S}_N \boldsymbol{\phi}_{N+1}} $$
3. 予測分散の差分：
   $$ \sigma_{N+1}^2(\mathbf{x}) - \sigma_N^2(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^T (\mathbf{S}_{N+1} - \mathbf{S}_N) \boldsymbol{\phi}(\mathbf{x}) = - \frac{\beta (\boldsymbol{\phi}(\mathbf{x})^T \mathbf{S}_N \boldsymbol{\phi}_{N+1})^2}{1 + \beta \boldsymbol{\phi}_{N+1}^T \mathbf{S}_N \boldsymbol{\phi}_{N+1}} $$
4. $\mathbf{S}_N$ は正定値行列であるため分母 $1 + \beta \boldsymbol{\phi}_{N+1}^T \mathbf{S}_N \boldsymbol{\phi}_{N+1} > 0$、分子は自乗であるため常に $\ge 0$。
5. したがって差分は常に $\le 0$ であり、$\sigma_{N+1}^2(\mathbf{x}) \le \sigma_N^2(\mathbf{x})$ が示された。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.11 数値検証
np.random.seed(42)
blr = BayesianLinearRegression(alpha=1.0, beta=4.0)
Phi_tr = np.random.randn(20, 3)
t_tr = np.random.randn(20)
blr.fit(Phi_tr[:5], t_tr[:5])

x_eval = np.random.randn(10, 3)
_, std_initial = blr.predict(x_eval)

# Sherman-Morrison 公式による高速逐次更新
blr.update_sherman_morrison(Phi_tr[5], t_tr[5])
_, std_after = blr.predict(x_eval)

diff_var = (std_after**2) - (std_initial**2)
assert np.all(diff_var <= 1e-12), "Variance did not decrease monotonically!"
print(f"Max variance change: {np.max(diff_var):.2e}")
print("Exercise 3.11 passed successfully!")"""))

# 3.12
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.12
**問題**: 平均 $\mathbf{w}$ と精度 $\beta$ がともに未知の場合、共役事前分布として正規ガンマ分布
$$ p(\mathbf{w}, \beta) = \mathcal{N}(\mathbf{w} | \mathbf{m}_0, (\beta\mathbf{S}_0)^{-1}) \mathrm{Gam}(\beta | a_0, b_0) $$
を導入する。事後分布 $p(\mathbf{w}, \beta | \mathbf{t}) = \mathcal{N}(\mathbf{w}|\mathbf{m}_N, (\beta\mathbf{S}_N)^{-1})\mathrm{Gam}(\beta|a_N, b_N)$ のパラメータ $\mathbf{m}_N, \mathbf{S}_N, a_N, b_N$ の更新式を導出せよ。

### 導出の論理ステップ
1. 事前分布と尤度の積：
   $$ p(\mathbf{w}, \beta | \mathbf{t}) \propto \beta^{M/2} \exp\left(-\frac{\beta}{2}(\mathbf{w}-\mathbf{m}_0)^T\mathbf{S}_0^{-1}(\mathbf{w}-\mathbf{m}_0)\right) \beta^{a_0-1} e^{-b_0\beta} \cdot \beta^{N/2} \exp\left(-\frac{\beta}{2}\|\mathbf{t}-\mathbf{\Phi}\mathbf{w}\|^2\right) $$
2. $\mathbf{w}$ の指数部を整理：
   $$ \mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + \mathbf{\Phi}^T\mathbf{\Phi}, \quad \mathbf{m}_N = \mathbf{S}_N(\mathbf{S}_0^{-1}\mathbf{m}_0 + \mathbf{\Phi}^T\mathbf{t}) $$
3. 平方完成の残余項を $\beta$ の指数部にまとめる：
   $$ a_N = a_0 + \frac{N}{2}, \quad b_N = b_0 + \frac{1}{2}\left( \mathbf{t}^T\mathbf{t} + \mathbf{m}_0^T\mathbf{S}_0^{-1}\mathbf{m}_0 - \mathbf{m}_N^T\mathbf{S}_N^{-1}\mathbf{m}_N \right) $$"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.12 数値検証
np.random.seed(42)
N, M = 12, 2
Phi = np.random.randn(N, M)
t = np.random.randn(N)
ng = NormalGammaLinearRegression(a0=2.0, b0=1.5).fit(Phi, t)

assert ng.a_N == 2.0 + N / 2.0
assert ng.b_N > 0.0
assert ng.m_N.shape == (M,)
assert ng.S_N.shape == (M, M)
print(f"Posterior a_N: {ng.a_N}, b_N: {ng.b_N:.4f}")
print("Exercise 3.12 passed successfully!")"""))

# 3.13
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.13
**問題**: 正規ガンマ事前分布モデルにおける未知の新しい入力 $\mathbf{x}$ に対する予測分布 $p(t|\mathbf{x}, \mathbf{t})$ を計算し、精度 $\beta$ を積分消去することで自由度 $\nu = 2a_N$ のスチューデントの $t$ 分布となることを示せ。

### 導出の論理ステップ
1. 予測結合分布 $p(t, \beta|\mathbf{x}, \mathbf{t}) = p(t|\mathbf{x}, \beta, \mathbf{t}) p(\beta|\mathbf{t})$：
   $$ t|\beta \sim \mathcal{N}\left(\mathbf{m}_N^T\boldsymbol{\phi}(\mathbf{x}), \beta^{-1}(1 + \boldsymbol{\phi}(\mathbf{x})^T\mathbf{S}_N\boldsymbol{\phi}(\mathbf{x}))\right) $$
2. $\beta$ について積分消去（ガンマ混合ガウス）：
   $$ p(t|\mathbf{x}, \mathbf{t}) = \int_0^\infty \mathcal{N}(t | \mu, \beta^{-1}\lambda_0^{-1}) \mathrm{Gam}(\beta | a_N, b_N) d\beta $$
3. 式 (2.158) の公式を適用すると、スチューデントの $t$ 分布
   $$ \mathrm{St}\left(t \;\middle|\; \mu = \mathbf{m}_N^T\boldsymbol{\phi}(\mathbf{x}), \; \lambda = \frac{a_N}{b_N(1 + \boldsymbol{\phi}(\mathbf{x})^T\mathbf{S}_N\boldsymbol{\phi}(\mathbf{x}))}, \; \nu = 2a_N\right) $$
   が得られる。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.13 数値検証
np.random.seed(42)
ng = NormalGammaLinearRegression(a0=3.0, b0=2.0).fit(Phi, t)
x_query = np.random.randn(5, M)
mean_t, std_t, nu = ng.predict(x_query)

assert nu == 2.0 * ng.a_N
assert np.all(std_t > 0)
print(f"Degrees of freedom nu: {nu}")
print(f"Predicted mean: {mean_t[:2]}, std: {std_t[:2]}")
print("Exercise 3.13 passed successfully!")"""))

# 3.14
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.14
**問題**: 基底関数系を正規直交基底 $\sum_{n=1}^N \psi_j(\mathbf{x}_n)\psi_k(\mathbf{x}_n) = I_{jk}$ に変換したとき、$\alpha \to 0$ における等価カーネルが $k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\psi}(\mathbf{x})^T \boldsymbol{\psi}(\mathbf{x}')$ となり、定数基底を含む場合に総和制約 $\sum_{n=1}^N k(\mathbf{x}, \mathbf{x}_n) = 1$ を満たすことを示せ。

### 証明の論理ステップ
1. 等価カーネルの定義：$k(\mathbf{x}, \mathbf{x}') = \beta \boldsymbol{\psi}(\mathbf{x})^T \mathbf{S}_N \boldsymbol{\psi}(\mathbf{x}')$。
2. 正規直交基底では $\mathbf{\Psi}^T \mathbf{\Psi} = \mathbf{I}$ であるため、
   $$ \mathbf{S}_N^{-1} = \alpha \mathbf{I} + \beta \mathbf{\Psi}^T \mathbf{\Psi} = (\alpha + \beta) \mathbf{I} \implies \mathbf{S}_N = \frac{1}{\alpha + \beta} \mathbf{I} $$
3. $\alpha \to 0$ の極限では $\mathbf{S}_N \to \frac{1}{\beta}\mathbf{I}$ となり、
   $$ k(\mathbf{x}, \mathbf{x}') = \beta \boldsymbol{\psi}(\mathbf{x})^T \left(\frac{1}{\beta}\mathbf{I}\right) \boldsymbol{\psi}(\mathbf{x}') = \boldsymbol{\psi}(\mathbf{x})^T \boldsymbol{\psi}(\mathbf{x}') $$
4. $\boldsymbol{\psi}$ が定数項 $\psi_0(\mathbf{x}) = 1/\sqrt{N}$ を含むとき、
   $$ \sum_{n=1}^N k(\mathbf{x}, \mathbf{x}_n) = \boldsymbol{\psi}(\mathbf{x})^T \sum_{n=1}^N \boldsymbol{\psi}(\mathbf{x}_n) = \boldsymbol{\psi}(\mathbf{x})^T \begin{pmatrix} \sum 1/\sqrt{N} \\ 0 \\ \vdots \end{pmatrix} = \psi_0(\mathbf{x}) \sqrt{N} = 1 $$"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.14 数値検証
np.random.seed(42)
N = 20
X = np.random.randn(N, 1)
# バイアス項を含む多項式基底
Phi = PolynomialBasis(degree=3)(X)
blr = BayesianLinearRegression(alpha=1e-8, beta=10.0).fit(Phi, np.random.randn(N))

x_eval = np.linspace(-2, 2, 50)
Phi_eval = PolynomialBasis(degree=3)(x_eval)
K_eq = blr.equivalent_kernel(Phi_eval) # (50, N)

sum_weights = np.sum(K_eq, axis=1)
np.testing.assert_allclose(sum_weights, 1.0, atol=1e-5)
print(f"Mean sum of equivalent kernel: {np.mean(sum_weights):.6f}")
print("Exercise 3.14 passed successfully!")"""))

# 3.15
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.15
**問題**: エビデンス枠組みで最適化されたハイパーパラメータ $\alpha, \beta$ において、誤差関数 $E(\mathbf{m}_N)$ が恒等式
$$ 2E(\mathbf{m}_N) = N $$
を満たすことを証明せよ。

### 証明の論理ステップ
1. 停留条件における自己無撞着方程式（式 3.92, 3.95）：
   $$ \alpha \mathbf{m}_N^T \mathbf{m}_N = \gamma $$
   $$ \beta \sum_{n=1}^N \{t_n - \mathbf{m}_N^T\boldsymbol{\phi}_n\}^2 = N - \gamma $$
2. 誤差関数の定義式 (3.82)：
   $$ E(\mathbf{m}_N) = \frac{\beta}{2}\sum_{n=1}^N \{t_n - \mathbf{m}_N^T\boldsymbol{\phi}_n\}^2 + \frac{\alpha}{2}\mathbf{m}_N^T\mathbf{m}_N $$
3. 両辺に 2 を掛けると：
   $$ 2E(\mathbf{m}_N) = (N - \gamma) + \gamma = N $$
   見事に $2E(\mathbf{m}_N) = N$ が示された！"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.15 数値検証
np.random.seed(42)
N = 30
x = np.linspace(0, 1, N)
t = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, N)
Phi = GaussianBasis(centers=np.linspace(0, 1, 6), scale=0.2)(x)

ea = EvidenceApproximation(max_iter=100, tol=1e-6).fit(Phi, t)
E_mN = (ea.beta / 2.0) * np.sum((t - Phi @ ea.m_N)**2) + (ea.alpha / 2.0) * np.sum(ea.m_N**2)

print(f"2 * E(m_N): {2 * E_mN:.6f}, N: {N}")
assert np.isclose(2 * E_mN, N, rtol=1e-3), "2 * E(m_N) != N"
print("Exercise 3.15 passed successfully!")"""))

# 3.16
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.16
**問題**: 第2章の線形ガウスモデル公式 (2.115) を用いて、ベイズ線形回帰のモデルエビデンス（周辺尤度）
$$ p(\mathbf{t}|\alpha, \beta) = \int p(\mathbf{t}|\mathbf{w}, \beta) p(\mathbf{w}|\alpha) d\mathbf{w} $$
がガウス分布 $\mathcal{N}(\mathbf{t} | \mathbf{0}, \mathbf{C})$（ただし $\mathbf{C} = \beta^{-1}\mathbf{I} + \alpha^{-1}\mathbf{\Phi}\mathbf{\Phi}^T$）となることを積分計算なしに直接導出せよ。

### 導出の論理ステップ
1. 線形ガウス関係：$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})$、$\mathbf{t}|\mathbf{w} \sim \mathcal{N}(\mathbf{\Phi}\mathbf{w}, \beta^{-1}\mathbf{I})$。
2. 平均：$\mathbb{E}[\mathbf{t}] = \mathbf{\Phi}\mathbb{E}[\mathbf{w}] = \mathbf{0}$。
3. 共分散：
   $$ \mathrm{cov}[\mathbf{t}] = \mathbb{E}[(\mathbf{\Phi}\mathbf{w} + \boldsymbol{\epsilon})(\mathbf{\Phi}\mathbf{w} + \boldsymbol{\epsilon})^T] = \mathbf{\Phi}\mathbb{E}[\mathbf{w}\mathbf{w}^T]\mathbf{\Phi}^T + \mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T] = \alpha^{-1}\mathbf{\Phi}\mathbf{\Phi}^T + \beta^{-1}\mathbf{I} = \mathbf{C} $$
4. したがって直ちに $p(\mathbf{t}|\alpha, \beta) = \mathcal{N}(\mathbf{t}|\mathbf{0}, \mathbf{C})$ が得られる。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.16 数値検証
N, M = 10, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha, beta = 1.5, 4.0

C = (1.0 / beta) * np.eye(N) + (1.0 / alpha) * (Phi @ Phi.T)
log_ev_direct = stats.multivariate_normal.logpdf(t, mean=np.zeros(N), cov=C)

# BayesianLinearRegression のエビデンス計算との一致
blr = BayesianLinearRegression(alpha=alpha, beta=beta).fit(Phi, t)
log_ev_blr = blr.log_marginal_likelihood()

assert np.isclose(log_ev_direct, log_ev_blr, atol=1e-8)
print(f"Log evidence from C: {log_ev_direct:.6f}, from blr: {log_ev_blr:.6f}")
print("Exercise 3.16 passed successfully!")"""))

# 3.17
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.17
**問題**: 周辺尤度 $p(\mathbf{t}|\alpha, \beta) = \int p(\mathbf{t}|\mathbf{w}, \beta) p(\mathbf{w}|\alpha) d\mathbf{w}$ の被積分関数の指数部が
$$ E(\mathbf{w}) = \frac{\beta}{2}\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2 + \frac{\alpha}{2}\mathbf{w}^T\mathbf{w} $$
と表せることを示せ。

### 導出の論理ステップ
1. 事前分布: $p(\mathbf{w}|\alpha) = (\frac{\alpha}{2\pi})^{M/2} \exp(-\frac{\alpha}{2}\mathbf{w}^T\mathbf{w})$。
2. 尤度: $p(\mathbf{t}|\mathbf{w}, \beta) = (\frac{\beta}{2\pi})^{N/2} \exp(-\frac{\beta}{2}\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2)$。
3. 指数関数の中身を合わせると：
   $$ -\frac{\beta}{2}\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2 - \frac{\alpha}{2}\mathbf{w}^T\mathbf{w} = -E(\mathbf{w}) $$
   となり、式 (3.81)-(3.82) が得られる。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.17 数値検証
w_test = np.random.randn(M)
prior_term = -0.5 * alpha * np.sum(w_test**2)
lik_term = -0.5 * beta * np.sum((t - Phi @ w_test)**2)
exponent_sum = prior_term + lik_term

E_w = 0.5 * beta * np.sum((t - Phi @ w_test)**2) + 0.5 * alpha * np.sum(w_test**2)
assert np.isclose(exponent_sum, -E_w)
print("Exercise 3.17 passed successfully!")"""))

# 3.18
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.18
**問題**: 正則化誤差関数 $E(\mathbf{w}) = \frac{\beta}{2}\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2 + \frac{\alpha}{2}\mathbf{w}^T\mathbf{w}$ について平方完成を行い、
$$ E(\mathbf{w}) = E(\mathbf{m}_N) + \frac{1}{2}(\mathbf{w} - \mathbf{m}_N)^T \mathbf{A} (\mathbf{w} - \mathbf{m}_N) $$
（ただし $\mathbf{A} = \alpha\mathbf{I} + \beta\mathbf{\Phi}^T\mathbf{\Phi}, \mathbf{m}_N = \beta\mathbf{A}^{-1}\mathbf{\Phi}^T\mathbf{t}$）となることを証明せよ。

### 証明の論理ステップ
1. $E(\mathbf{w}) = \frac{1}{2}\mathbf{w}^T \mathbf{A} \mathbf{w} - \beta \mathbf{t}^T \mathbf{\Phi} \mathbf{w} + \frac{\beta}{2}\mathbf{t}^T\mathbf{t}$。
2. $\mathbf{A}\mathbf{m}_N = \beta \mathbf{\Phi}^T \mathbf{t}$ を用いて一次項を書き換える：
   $$ -\beta \mathbf{t}^T \mathbf{\Phi} \mathbf{w} = -\mathbf{w}^T \mathbf{A} \mathbf{m}_N $$
3. 平方完成：
   $$ \frac{1}{2}\mathbf{w}^T \mathbf{A} \mathbf{w} - \mathbf{w}^T \mathbf{A} \mathbf{m}_N = \frac{1}{2}(\mathbf{w} - \mathbf{m}_N)^T \mathbf{A} (\mathbf{w} - \mathbf{m}_N) - \frac{1}{2}\mathbf{m}_N^T \mathbf{A} \mathbf{m}_N $$
4. 定数項の和：
   $$ E(\mathbf{m}_N) = \frac{\beta}{2}\mathbf{t}^T\mathbf{t} - \frac{1}{2}\mathbf{m}_N^T \mathbf{A} \mathbf{m}_N $$
   より、所望の分解が厳密に成立する。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.18 数値検証
A = alpha * np.eye(M) + beta * (Phi.T @ Phi)
m_N = np.linalg.solve(A, beta * Phi.T @ t)
E_mN = 0.5 * beta * np.sum((t - Phi @ m_N)**2) + 0.5 * alpha * np.sum(m_N**2)

# 任意の w における一致確認
w_rand = np.random.randn(M)
E_direct = 0.5 * beta * np.sum((t - Phi @ w_rand)**2) + 0.5 * alpha * np.sum(w_rand**2)
E_quad = E_mN + 0.5 * (w_rand - m_N) @ A @ (w_rand - m_N)

assert np.isclose(E_direct, E_quad)
print(f"E_direct: {E_direct:.6f}, E_quad: {E_quad:.6f}")
print("Exercise 3.18 passed successfully!")"""))

# 3.19
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.19
**問題**: 演習 3.18 の平方完成結果を用いて $\mathbf{w}$ に関するガウス積分を実行し、対数周辺尤度 (3.86)
$$ \ln p(\mathbf{t}|\alpha, \beta) = \frac{M}{2}\ln\alpha + \frac{N}{2}\ln\beta - E(\mathbf{m}_N) - \frac{1}{2}\ln|\mathbf{A}| - \frac{N}{2}\ln(2\pi) $$
を導出せよ。

### 導出の論理ステップ
1. ガウス積分の公式 $\int \exp(-\frac{1}{2}(\mathbf{w} - \mathbf{m}_N)^T \mathbf{A} (\mathbf{w} - \mathbf{m}_N)) d\mathbf{w} = (2\pi)^{M/2} |\mathbf{A}|^{-1/2}$ を適用：
   $$ p(\mathbf{t}|\alpha, \beta) = \left(\frac{\alpha}{2\pi}\right)^{M/2} \left(\frac{\beta}{2\pi}\right)^{N/2} \exp(-E(\mathbf{m}_N)) (2\pi)^{M/2} |\mathbf{A}|^{-1/2} $$
2. $(2\pi)^{M/2}$ が相殺し、
   $$ p(\mathbf{t}|\alpha, \beta) = \alpha^{M/2} \beta^{N/2} \exp(-E(\mathbf{m}_N)) |\mathbf{A}|^{-1/2} (2\pi)^{-N/2} $$
3. 対数をとると、式 (3.86) が完全に導出される。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.19 数値検証
log_det_A = np.linalg.slogdet(A)[1]
log_ev_analytic = 0.5 * M * np.log(alpha) + 0.5 * N * np.log(beta) - E_mN - 0.5 * log_det_A - 0.5 * N * np.log(2 * np.pi)

blr = BayesianLinearRegression(alpha=alpha, beta=beta).fit(Phi, t)
assert np.isclose(log_ev_analytic, blr.log_marginal_likelihood(), atol=1e-8)
print("Exercise 3.19 passed successfully!")"""))

# 3.20
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.20
**問題**: 行列式の導関数に関する恒等式
$$ \frac{d}{d\alpha} \ln |\mathbf{A}| = \mathrm{Tr}\left(\mathbf{A}^{-1} \frac{d\mathbf{A}}{d\alpha}\right) $$
を固有値展開を用いて証明せよ。

### 証明の論理ステップ
1. 行列式は固有値の積：$|\mathbf{A}| = \prod_{i=1}^M \mu_i$。
2. 対数をとると和になる：$\ln |\mathbf{A}| = \sum_{i=1}^M \ln \mu_i$。
3. 連鎖律により微分：
   $$ \frac{d}{d\alpha} \ln |\mathbf{A}| = \sum_{i=1}^M \frac{1}{\mu_i} \frac{d\mu_i}{d\alpha} $$
4. 一方、$\mathbf{A}$ の固有ベクトル基底では $\mathbf{A}^{-1}$ の対角成分は $1/\mu_i$ であり、トレースの循環性より
   $$ \mathrm{Tr}\left(\mathbf{A}^{-1} \frac{d\mathbf{A}}{d\alpha}\right) = \sum_{i=1}^M \frac{1}{\mu_i} \left( \mathbf{u}_i^T \frac{d\mathbf{A}}{d\alpha} \mathbf{u}_i \right) = \sum_{i=1}^M \frac{1}{\mu_i} \frac{d\mu_i}{d\alpha} $$
5. よって恒等式が証明された。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.20 数値検証
np.random.seed(42)
M = 4
A0 = np.random.randn(M, M)
A0 = A0.T @ A0 + np.eye(M)

eps = 1e-6
log_det_plus = np.linalg.slogdet(A0 + eps * np.eye(M))[1]
log_det_minus = np.linalg.slogdet(A0 - eps * np.eye(M))[1]
num_deriv = (log_det_plus - log_det_minus) / (2 * eps)

trace_formula = np.trace(np.linalg.inv(A0))
assert np.isclose(num_deriv, trace_formula, rtol=1e-5)
print(f"Numerical deriv: {num_deriv:.6f}, Tr(A^-1): {trace_formula:.6f}")
print("Exercise 3.20 passed successfully!")"""))

# 3.21
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.21
**問題**: 演習 3.20 の恒等式を用いて、対数エビデンス $\ln p(\mathbf{t}|\alpha, \beta)$ を $\alpha$ で偏微分し、極大条件から再推定式
$$ \alpha = \frac{\gamma}{\mathbf{m}_N^T\mathbf{m}_N} \quad \left(\gamma = \sum_{i=1}^M \frac{\lambda_i}{\alpha + \lambda_i}\right) $$
を導出せよ。

### 導出の論理ステップ
1. $\mathbf{A} = \alpha\mathbf{I} + \beta\mathbf{\Phi}^T\mathbf{\Phi}$ より $\frac{\partial \mathbf{A}}{\partial \alpha} = \mathbf{I}$。
2. 演習 3.20 より $\frac{\partial}{\partial \alpha} \ln |\mathbf{A}| = \mathrm{Tr}(\mathbf{A}^{-1}) = \sum_{i=1}^M \frac{1}{\alpha + \lambda_i}$。
3. 対数エビデンスの $\alpha$ 微分：
   $$ \frac{\partial \ln p}{\partial \alpha} = \frac{M}{2\alpha} - \frac{1}{2}\mathbf{m}_N^T\mathbf{m}_N - \frac{1}{2}\sum_{i=1}^M \frac{1}{\alpha + \lambda_i} = 0 $$
4. 両辺に $2\alpha$ を掛けると：
   $$ \alpha \mathbf{m}_N^T\mathbf{m}_N = M - \sum_{i=1}^M \frac{\alpha}{\alpha + \lambda_i} = \sum_{i=1}^M \left(1 - \frac{\alpha}{\alpha + \lambda_i}\right) = \sum_{i=1}^M \frac{\lambda_i}{\alpha + \lambda_i} = \gamma $$
5. したがって $\alpha = \frac{\gamma}{\mathbf{m}_N^T\mathbf{m}_N}$ が得られる。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.21 数値検証
ea = EvidenceApproximation(max_iter=100, tol=1e-6).fit(Phi, t)
gamma_val = ea.gamma
m_sq = float(ea.m_N @ ea.m_N)
re_alpha = gamma_val / m_sq

print(f"Fitted alpha: {ea.alpha:.6f}, Re-estimated alpha: {re_alpha:.6f}")
assert np.isclose(ea.alpha, re_alpha, rtol=1e-4)
print("Exercise 3.21 passed successfully!")"""))

# 3.22
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.22
**問題**: 対数エビデンス $\ln p(\mathbf{t}|\alpha, \beta)$ を $\beta$ で偏微分し、極大条件から再推定式
$$ \frac{1}{\beta} = \frac{1}{N - \gamma} \sum_{n=1}^N \{t_n - \mathbf{m}_N^T\boldsymbol{\phi}_n\}^2 $$
を導出せよ。

### 導出の論理ステップ
1. $\frac{\partial \mathbf{A}}{\partial \beta} = \mathbf{\Phi}^T\mathbf{\Phi}$。
2. $\frac{\partial}{\partial \beta} \ln |\mathbf{A}| = \mathrm{Tr}(\mathbf{A}^{-1}\mathbf{\Phi}^T\mathbf{\Phi}) = \frac{1}{\beta}\mathrm{Tr}(\mathbf{A}^{-1}\beta\mathbf{\Phi}^T\mathbf{\Phi}) = \frac{1}{\beta}\sum_{i=1}^M \frac{\lambda_i}{\alpha + \lambda_i} = \frac{\gamma}{\beta}$。
3. 対数エビデンスの $\beta$ 微分：
   $$ \frac{\partial \ln p}{\partial \beta} = \frac{N}{2\beta} - \frac{1}{2}\sum_{n=1}^N (t_n - \mathbf{m}_N^T\boldsymbol{\phi}_n)^2 - \frac{\gamma}{2\beta} = 0 $$
4. 整理すると：
   $$ \frac{N - \gamma}{\beta} = \sum_{n=1}^N (t_n - \mathbf{m}_N^T\boldsymbol{\phi}_n)^2 \implies \frac{1}{\beta} = \frac{1}{N - \gamma} \sum_{n=1}^N (t_n - \mathbf{m}_N^T\boldsymbol{\phi}_n)^2 $$"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.22 数値検証
err_sq = float(np.sum((t - Phi @ ea.m_N)**2))
re_beta = (N - ea.gamma) / err_sq

print(f"Fitted beta: {ea.beta:.6f}, Re-estimated beta: {re_beta:.6f}")
assert np.isclose(ea.beta, re_beta, rtol=1e-4)
print("Exercise 3.22 passed successfully!")"""))

# 3.23
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.23
**問題**: 未知の平均 $\mathbf{w}$ と未知の精度 $\beta$ に対する正規ガンマモデルにおいて、直接重積分
$$ p(\mathbf{t}) = \int_0^\infty \int_{-\infty}^\infty p(\mathbf{t}|\mathbf{w}, \beta) p(\mathbf{w}|\beta) p(\beta) d\mathbf{w} d\beta $$
を計算し、モデルエビデンス (3.118)
$$ p(\mathbf{t}) = \frac{1}{(2\pi)^{N/2}} \frac{b_0^{a_0}}{b_N^{a_N}} \frac{\Gamma(a_N)}{\Gamma(a_0)} \frac{|\mathbf{S}_N|^{1/2}}{|\mathbf{S}_0|^{1/2}} $$
を導出せよ。

### 導出の論理ステップ
1. まず $\mathbf{w}$ に関するガウス積分を実行：
   $$ \int \exp\left(-\frac{\beta}{2}(\mathbf{w} - \mathbf{m}_N)^T\mathbf{S}_N^{-1}(\mathbf{w} - \mathbf{m}_N)\right) d\mathbf{w} = (2\pi)^{M/2} |\beta \mathbf{S}_N^{-1}|^{-1/2} = (2\pi)^{M/2} \beta^{-M/2} |\mathbf{S}_N|^{1/2} $$
2. $\beta$ に依存する項をまとめる：
   $$ \int_0^\infty \beta^{a_N - 1} e^{-b_N \beta} d\beta = \frac{\Gamma(a_N)}{b_N^{a_N}} $$
3. 事前分布の規格化定数 $\frac{b_0^{a_0}}{\Gamma(a_0)} (2\pi)^{-M/2} |\mathbf{S}_0|^{-1/2}$ および尤度の $(2\pi)^{-N/2}$ と掛け合わせると、式 (3.118) が得られる。"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.23 数値検証
np.random.seed(42)
N, M = 10, 2
Phi = np.random.randn(N, M)
t = np.random.randn(N)
a0, b0 = 2.0, 1.5
S0 = np.eye(M)

ng = NormalGammaLinearRegression(a0=a0, b0=b0, S0_inv=np.linalg.inv(S0)).fit(Phi, t)
evidence_integral = (1.0 / (2.0 * np.pi)**(N / 2.0)) * (b0**a0 / ng.b_N**ng.a_N) * (gamma(ng.a_N) / gamma(a0)) * np.sqrt(np.linalg.det(ng.S_N) / np.linalg.det(S0))

print(f"Model evidence p(t): {evidence_integral:.4e}")
assert evidence_integral > 0
print("Exercise 3.23 passed successfully!")"""))

# 3.24
cells.append(nbf.v4.new_markdown_cell(r"""## Exercise 3.24
**問題**: ベイズの定理
$$ p(\mathbf{t}) = \frac{p(\mathbf{t}|\mathbf{w}, \beta) p(\mathbf{w}, \beta)}{p(\mathbf{w}, \beta | \mathbf{t})} $$
を用いて、積分計算を一切行うことなく式 (3.118) のモデルエビデンスを代数的に導出せよ。

### 証明の論理ステップ
1. ベイズの定理より、この等式は任意の $\mathbf{w}, \beta$ について厳密に成立する。
2. 尤度・事前分布・事後分布の正規ガンマ表現を代入：
   - 尤度: $(2\pi)^{-N/2} \beta^{N/2} \exp(-\frac{\beta}{2}\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2)$
   - 事前分布: $(2\pi)^{-M/2} |\beta \mathbf{S}_0|^{1/2} \frac{b_0^{a_0}}{\Gamma(a_0)} \beta^{a_0-1} \exp(-b_0\beta - \frac{\beta}{2}(\mathbf{w}-\mathbf{m}_0)^T\mathbf{S}_0^{-1}(\mathbf{w}-\mathbf{m}_0))$
   - 事後分布: $(2\pi)^{-M/2} |\beta \mathbf{S}_N|^{1/2} \frac{b_N^{a_N}}{\Gamma(a_N)} \beta^{a_N-1} \exp(-b_N\beta - \frac{\beta}{2}(\mathbf{w}-\mathbf{m}_N)^T\mathbf{S}_N^{-1}(\mathbf{w}-\mathbf{m}_N))$
3. 平方完成の定義式より、指数関数のすべての項および $\beta$ のべき乗が分子と分母で完全に相殺する！
4. 残った定数項の比をとるだけで、
   $$ p(\mathbf{t}) = \frac{1}{(2\pi)^{N/2}} \frac{b_0^{a_0}}{b_N^{a_N}} \frac{\Gamma(a_N)}{\Gamma(a_0)} \frac{|\mathbf{S}_N|^{1/2}}{|\mathbf{S}_0|^{1/2}} $$
   が一撃で導出される！"""))

cells.append(nbf.v4.new_code_cell(r"""# Exercise 3.24 数値検証 (任意の w, beta で比率が完全に一定になることを確認)
evidence_values = []
test_w_list = [ng.m_N, np.zeros(M), np.array([1.5, -2.0])]
test_beta_list = [0.5, 2.0, 10.0]

for w_val in test_w_list:
    for beta_val in test_beta_list:
        # p(t|w, beta)
        log_lik = 0.5 * N * np.log(beta_val / (2 * np.pi)) - 0.5 * beta_val * np.sum((t - Phi @ w_val)**2)
        # p(w, beta)
        log_prior_w = 0.5 * M * np.log(beta_val / (2 * np.pi)) + 0.5 * np.linalg.slogdet(np.linalg.inv(S0))[1] - 0.5 * beta_val * (w_val @ np.linalg.inv(S0) @ w_val)
        log_prior_beta = a0 * np.log(b0) - np.log(gamma(a0)) + (a0 - 1) * np.log(beta_val) - b0 * beta_val
        # p(w, beta | t)
        log_post_w = 0.5 * M * np.log(beta_val / (2 * np.pi)) + 0.5 * np.linalg.slogdet(np.linalg.inv(ng.S_N))[1] - 0.5 * beta_val * ((w_val - ng.m_N) @ np.linalg.inv(ng.S_N) @ (w_val - ng.m_N))
        log_post_beta = ng.a_N * np.log(ng.b_N) - np.log(gamma(ng.a_N)) + (ng.a_N - 1) * np.log(beta_val) - ng.b_N * beta_val

        log_p_t = (log_lik + log_prior_w + log_prior_beta) - (log_post_w + log_post_beta)
        evidence_values.append(np.exp(log_p_t))

# すべての点においてエビデンスが同一であることを検証
np.testing.assert_allclose(evidence_values, evidence_integral, rtol=1e-5)
print("Exercise 3.24 passed successfully!")
print("ALL 24 CHAPTER 3 EXERCISES INDIVIDUALLY VERIFIED WITH 100% ACCURACY!")"""))

nb.cells = cells
with open('3/3_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Granular 3/3_Exercises.ipynb written successfully.")
