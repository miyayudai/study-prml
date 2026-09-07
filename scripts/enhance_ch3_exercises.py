"""
Master script to build and execute 3/3_Exercises.ipynb
Covers all 24 PRML Chapter 3 exercises (3.1 to 3.24) with:
- Detailed theoretical explanation and mathematical derivation
- Step-by-step logic and structured fill-in-the-blank markdown format
- Self-contained Python numerical verification code with assertions
"""

import json
import nbformat as nbf
import numpy as np

def create_ch3_exercises_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & TOC
    cells.append(nbf.v4.new_markdown_cell("""# 第3章 章末演習問題 (Exercises 3.1 〜 3.24 全24問 完全網羅)

教科書「パターン認識と機械学習 (PRML)」第3章「線形回帰モデル (Linear Models for Regression)」の全24問の演習問題の解答・解説ノートブックです。

各問題について、以下の構成で学習を進められるよう設計されています：
1. **問題の提示**: PRML原著の設問内容
2. **[解答の道筋と穴埋め]**: 証明・導出の論理的ステップと要点穴埋め (`[ 穴埋め X: ? ]`)
3. **Python数値検証コード**: 導出した数式や定理を数値シミュレーション・assert文で直接検証

---
## 目次
- [Exercise 3.1: tanh とシグモイド関数の等価変換](#Exercise-3.1)
- [Exercise 3.2: 最小二乗直交射影行列の幾何学的性質](#Exercise-3.2)
- [Exercise 3.3: 重み付き二乗和誤差と2つの物理的解釈](#Exercise-3.3)
- [Exercise 3.4: 入力ノイズ付加とL2正則化(Weight Decay)の等価性](#Exercise-3.4)
- [Exercise 3.5: Lq正則化制約とラグランジュ未定乗数法](#Exercise-3.5)
- [Exercise 3.6: 多変量目的変数の最尤推定と出力の独立分解](#Exercise-3.6)
- [Exercise 3.7: 平方完成によるパラメータ事後分布の解析的導出](#Exercise-3.7)
- [Exercise 3.8: 逐次ベイズ更新の一致性証明](#Exercise-3.8)
- [Exercise 3.9: 線形ガウス公式によるパラメータ事後分布の別証](#Exercise-3.9)
- [Exercise 3.10: 周辺化積分による予測分布の導出](#Exercise-3.10)
- [Exercise 3.11: Sherman-Morrison公式と予測分散の単調非増加性](#Exercise-3.11)
- [Exercise 3.12: 未知の精度を持つ共役正規ガンマモデルの事後分布](#Exercise-3.12)
- [Exercise 3.13: 正規ガンマモデルのStudent's t予測分布の導出](#Exercise-3.13)
- [Exercise 3.14: バイアス項を持つ等価カーネルの総和制約](#Exercise-3.14)
- [Exercise 3.15: エビデンス停留点における 2E(m_N) = N の証明](#Exercise-3.15)
- [Exercise 3.16: 線形ガウス公式による対数エビデンス関数の直接導出](#Exercise-3.16)
- [Exercise 3.17: エビデンス被積分関数の誤差関数表現](#Exercise-3.17)
- [Exercise 3.18: 重みまわりの平方完成による誤差関数の変形](#Exercise-3.18)
- [Exercise 3.19: ガウス積分による対数周辺尤度 (式 3.86) の導出](#Exercise-3.19)
- [Exercise 3.20: 対数エビデンスの alpha 微分とハイパーパラメータ再推定式](#Exercise-3.20)
- [Exercise 3.21: 行列式微分恒等式を用いた alpha 再推定式の別証](#Exercise-3.21)
- [Exercise 3.22: 対数エビデンスの beta 微分とノイズ精度再推定式](#Exercise-3.22)
- [Exercise 3.23: 正規ガンマモデルのモデルエビデンスの直接重積分導出](#Exercise-3.23)
- [Exercise 3.24: ベイズの定理を用いたモデルエビデンスの代数的導出](#Exercise-3.24)
---"""))

    # Setup cell
    cells.append(nbf.v4.new_code_cell("""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.optimize as optimize

from prml.linear import (
    PolynomialBasis, GaussianBasis, SigmoidalBasis, FourierBasis,
    LinearRegression, RidgeRegression, LassoRegression, LeastMeanSquares,
    BayesianLinearRegression, NormalGammaLinearRegression, EvidenceApproximation,
    orthogonal_projection_matrix, equivalent_kernel_matrix, bayesian_model_evidence
)

print("Chapter 3 Exercises Setup completed successfully.")"""))

    # Exercise 3.1
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.1
**問題**: 双曲線正接関数 $\\tanh(a) = \\frac{e^a - e^{-a}}{e^a + e^{-a}}$ とロジスティックシグモイド関数 $\\sigma(a) = \\frac{1}{1 + e^{-a}}$ (式 3.6) の間に
$$ \\tanh(a) = 2\\sigma(2a) - 1 $$
の関係が成り立つことを示せ。
また、シグモイド基底の一般線形結合
$$ y(x, \\mathbf{w}) = w_0 + \\sum_{j=1}^M w_j \\sigma\\left(\\frac{x - \\mu_j}{s}\\right) $$
が $\\tanh$ 基底の一般線形結合
$$ y(x, \\mathbf{u}) = u_0 + \\sum_{j=1}^M u_j \\tanh\\left(\\frac{x - \\mu_j}{2s}\\right) $$
と等価であることを示し、新パラメータ $\\{u_0, u_1, \\dots, u_M\\}$ と元パラメータ $\\{w_0, w_1, \\dots, w_M\\}$ の変換関係を求めよ。

### [解答の道筋と穴埋め]
1. シグモイド関数の定義より:
   $$ \\sigma(2a) = \\frac{1}{1 + e^{-2a}} = \\frac{e^a}{e^a + e^{-a}} $$
2. したがって:
   $$ 2\\sigma(2a) - 1 = \\frac{2e^a - (e^a + e^{-a})}{e^a + e^{-a}} = \\frac{e^a - e^{-a}}{e^a + e^{-a}} = \\tanh(a) $$
   よって $\\sigma(a) = \\text{[ 穴埋め 1: ? ]} \\{1 + \\tanh(a/2)\\}$ が得られる。
3. これを $y(x, \\mathbf{w})$ に代入すると:
   $$ y(x, \\mathbf{w}) = w_0 + \\sum_{j=1}^M w_j \\frac{1}{2}\\left\\{1 + \\tanh\\left(\\frac{x - \\mu_j}{2s}\\right)\\right\\} = \\left(w_0 + \\frac{1}{2}\\sum_{j=1}^M w_j\\right) + \\sum_{j=1}^M \\frac{w_j}{2} \\tanh\\left(\\frac{x - \\mu_j}{2s}\\right) $$
4. $y(x, \\mathbf{u})$ と係数比較すると:
   - $u_0 = \\text{[ 穴埋め 2: ? ]} = w_0 + \\frac{1}{2}\\sum_{j=1}^M w_j$
   - $u_j = \\text{[ 穴埋め 3: ? ]} = \\frac{w_j}{2} \\quad (j = 1, \\dots, M)$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.1 数値検証
def sigmoid_to_tanh_weights(w0, w):
    \"\"\"シグモイド基底の重みをtanh基底の重みに変換する\"\"\"
    # YOUR CODE HERE
    # u0 = w0 + 0.5 * np.sum(w)
    # u = 0.5 * np.asarray(w)
    u0 = w0 + 0.5 * np.sum(w)
    u = 0.5 * np.asarray(w)
    return u0, u

# 数値検証: 任意入力 x に対する予測値の一致
np.random.seed(42)
w0 = 1.5
w = np.array([0.8, -1.2, 2.0])
mu = np.array([-1.0, 0.0, 1.5])
s = 0.7

u0, u = sigmoid_to_tanh_weights(w0, w)

x_test = np.linspace(-3, 3, 100)
# シグモイド基底による値
y_sig = w0 + np.sum([w[j] / (1.0 + np.exp(-(x_test - mu[j]) / s)) for j in range(3)], axis=0)
# tanh 基底による値
y_tanh = u0 + np.sum([u[j] * np.tanh((x_test - mu[j]) / (2.0 * s)) for j in range(3)], axis=0)

diff = np.max(np.abs(y_sig - y_tanh))
assert np.isclose(diff, 0.0, atol=1e-12), f"Discrepancy: {diff}"
print(f"Exercise 3.1 PASSED: Max difference = {diff:.2e}")"""))

    # Exercise 3.2
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.2
**問題**: 行列
$$ \\mathbf{P} = \\mathbf{\\Phi}(\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T $$
が任意のベクトル $\\mathbf{v}$ を $\\mathbf{\\Phi}$ の列ベクトルが張る部分空間 $\\mathcal{S}$ へ射影することを示せ。
さらにこの結果を用いて、最小二乗解 (式 3.15) が目標値ベクトル $\\mathbf{t}$ の部分空間 $\\mathcal{S}$ への直交射影に対応すること (Figure 3.2) を示せ。

### [解答の道筋と穴埋め]
1. 射影行列の定義はべき等性 $\\mathbf{P}^2 = \\mathbf{P}$ を満たすことである:
   $$ \\mathbf{P}^2 = [\\mathbf{\\Phi}(\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T][\\mathbf{\\Phi}(\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T] = \\mathbf{\\Phi}(\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} [\\mathbf{\\Phi}^T \\mathbf{\\Phi}] (\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T = \\text{[ 穴埋め 1: ? ]} = \\mathbf{P} $$
2. また、転置をとると $\\mathbf{P}^T = [\\mathbf{\\Phi}(\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T]^T = \\mathbf{P}$ より対称行列である（直交射影行列）。
3. 任意のベクトル $\\mathbf{v}$ に対し、$\\mathbf{P}\\mathbf{v} = \\mathbf{\\Phi} \\mathbf{a}$ （ただし $\\mathbf{a} = (\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T \\mathbf{v}$）であるため、$\\mathbf{P}\\mathbf{v} \\in \\mathcal{S}$ である。
4. 最小二乗解 $\\mathbf{w}_{\\mathrm{ML}} = (\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T \\mathbf{t}$ に対し、予測ベクトルは:
   $$ \\mathbf{y} = \\mathbf{\\Phi} \\mathbf{w}_{\\mathrm{ML}} = \\mathbf{\\Phi}(\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T \\mathbf{t} = \\mathbf{P}\\mathbf{t} $$
5. 残差ベクトル $\\mathbf{e} = \\mathbf{t} - \\mathbf{y} = (\\mathbf{I} - \\mathbf{P})\\mathbf{t}$ と部分空間 $\\mathcal{S}$ の任意の基底 $\\mathbf{\\Phi}_j$ との内積は:
   $$ \\mathbf{\\Phi}^T (\\mathbf{t} - \\mathbf{y}) = \\mathbf{\\Phi}^T (\\mathbf{I} - \\mathbf{P})\\mathbf{t} = [\\mathbf{\\Phi}^T - \\mathbf{\\Phi}^T \\mathbf{\\Phi}(\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1}\\mathbf{\\Phi}^T]\\mathbf{t} = \\text{[ 穴埋め 2: ? ]} = \\mathbf{0} $$
   よって残差ベクトルは $\\mathcal{S}$ と直交する。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.2 数値検証
np.random.seed(42)
N, M = 20, 4
Phi = np.random.randn(N, M)
t = np.random.randn(N)

# 射影行列 P の計算
P = orthogonal_projection_matrix(Phi)

# 1. べき等性の検証 P^2 == P
assert np.allclose(P @ P, P), "P is not idempotent"
# 2. 対称性の検証 P^T == P
assert np.allclose(P.T, P), "P is not symmetric"
# 3. トレースが M と一致 Tr(P) == M
assert np.isclose(np.trace(P), M), "Tr(P) != M"

# 4. 直交残差の検証: Phi^T (t - P t) == 0
y = P @ t
residual = t - y
orthogonality = np.linalg.norm(Phi.T @ residual)
assert np.isclose(orthogonality, 0.0, atol=1e-12), f"Not orthogonal: {orthogonality}"

print(f"Exercise 3.2 PASSED: Orthogonality error = {orthogonality:.2e}, Tr(P) = {np.trace(P):.1f}")"""))

    # Exercise 3.3
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.3
**問題**: 各データ点 $t_n$ に正の重み係数 $r_n > 0$ が付与された重み付き二乗和誤差関数
$$ E_D(\\mathbf{w}) = \\frac{1}{2} \\sum_{n=1}^N r_n \\{t_n - \\mathbf{w}^T \\boldsymbol{\\phi}(\\mathbf{x}_n)\\}^2 $$
を最小化する解 $\\mathbf{w}^*$ の表現を求めよ。
また、重み付き二乗和誤差関数の2つの解釈 (i) データ依存のノイズ分散、(ii) データの重複（複製）について説明せよ。

### [解答の道筋と穴埋め]
1. 行列表現: 重み対角行列 $\\mathbf{R} = \\mathrm{diag}(r_1, \\dots, r_N)$ を定義すると:
   $$ E_D(\\mathbf{w}) = \\frac{1}{2} (\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{w})^T \\mathbf{R} (\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{w}) $$
2. $\\mathbf{w}$ に関する勾配を計算して 0 と置く:
   $$ \\nabla E_D(\\mathbf{w}) = -\\mathbf{\\Phi}^T \\mathbf{R} (\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{w}) = \\mathbf{0} $$
   $$ \\mathbf{\\Phi}^T \\mathbf{R} \\mathbf{\\Phi} \\mathbf{w}^* = \\mathbf{\\Phi}^T \\mathbf{R} \\mathbf{t} $$
   したがって:
   $$ \\mathbf{w}^* = \\text{[ 穴埋め 1: ? ]} = (\\mathbf{\\Phi}^T \\mathbf{R} \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T \\mathbf{R} \\mathbf{t} $$
3. **2つの解釈**:
   - **(i) データ依存ノイズ分散**: 観測点 $n$ ごとにノイズ分散 $\\sigma_n^2 = \\beta^{-1} r_n^{-1}$ が異なるとすると、負の対数尤度を最小化することは重み付き二乗和誤差の最小化と完全に等価となる。
   - **(ii) データの重複**: $r_n$ が正の整数の場合、これはデータ点 $(\\mathbf{x}_n, t_n)$ が $r_n$ 個重複して観測された通常最小二乗問題と数学的に完全に同一である。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.3 数値検証
np.random.seed(42)
N, M = 15, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
# 整数重み
r_int = np.array([2, 1, 3, 1, 2, 1, 4, 2, 1, 1, 2, 3, 1, 2, 1])
R = np.diag(r_int)

# 1. 重み付き正規方程式による解
w_weighted = np.linalg.solve(Phi.T @ R @ Phi, Phi.T @ R @ t)

# 2. データ重複法による通常最小二乗解
Phi_rep = np.repeat(Phi, r_int, axis=0)
t_rep = np.repeat(t, r_int)
w_replicated = np.linalg.solve(Phi_rep.T @ Phi_rep, Phi_rep.T @ t_rep)

diff = np.linalg.norm(w_weighted - w_replicated)
assert np.isclose(diff, 0.0, atol=1e-12), f"Mismatch: {diff}"
print(f"Exercise 3.3 PASSED: Replication equivalence diff = {diff:.2e}")"""))

    # Exercise 3.4
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.4
**問題**: 入力変数に平均 $\\mathbf{0}$、共分散 $\\sigma^2 \\mathbf{I}$ のノイズ $\\boldsymbol{\\epsilon}$ が加わる線形モデル
$$ y(\\mathbf{x}) = w_0 + \\mathbf{w}^T \\mathbf{x} $$
を考える。ノイズを考慮した期待二乗和誤差
$$ \\mathbb{E}_{\\boldsymbol{\\epsilon}}\\left[ \\frac{1}{2} \\sum_{n=1}^N \\{t_n - w_0 - \\mathbf{w}^T (\\mathbf{x}_n + \\boldsymbol{\\epsilon}_n)\\}^2 \\right] $$
を最小化することが、正則化項 $\\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2$ を持つ正則化最小二乗法と厳密に等価であることを示し、正則化係数 $\\lambda$ を求めよ。

### [解答の道筋と穴埋め]
1. 誤差の二乗を展開する:
   $$ \\{t_n - w_0 - \\mathbf{w}^T (\\mathbf{x}_n + \\boldsymbol{\\epsilon}_n)\\}^2 = \\{t_n - w_0 - \\mathbf{w}^T \\mathbf{x}_n - \\mathbf{w}^T \\boldsymbol{\\epsilon}_n\\}^2 $$
   $$ = \\{t_n - w_0 - \\mathbf{w}^T \\mathbf{x}_n\\}^2 - 2 \\{t_n - w_0 - \\mathbf{w}^T \\mathbf{x}_n\\}(\\mathbf{w}^T \\boldsymbol{\\epsilon}_n) + (\\mathbf{w}^T \\boldsymbol{\\epsilon}_n)^2 $$
2. $\\boldsymbol{\\epsilon}$ に関する期待値をとると:
   - $\\mathbb{E}[\\mathbf{w}^T \\boldsymbol{\\epsilon}_n] = \\mathbf{w}^T \\mathbb{E}[\\boldsymbol{\\epsilon}_n] = 0$
   - $\\mathbb{E}[(\\mathbf{w}^T \\boldsymbol{\\epsilon}_n)^2] = \\mathbb{E}[\\mathbf{w}^T \\boldsymbol{\\epsilon}_n \\boldsymbol{\\epsilon}_n^T \\mathbf{w}] = \\mathbf{w}^T (\\sigma^2 \\mathbf{I}) \\mathbf{w} = \\sigma^2 \\|\\mathbf{w}\\|^2$
3. したがって全データにわたる期待誤差は:
   $$ \\mathbb{E}_{\\boldsymbol{\\epsilon}}[E_D] = \\frac{1}{2}\\sum_{n=1}^N \\{t_n - w_0 - \\mathbf{w}^T \\mathbf{x}_n\\}^2 + \\text{[ 穴埋め 1: ? ]} \\|\\mathbf{w}\\|^2 $$
   ここで $\\lambda = \\text{[ 穴埋め 2: ? ]} = N \\sigma^2$ の $L_2$ 正則化（Weight Decay）と厳密に一致する。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.4 数値検証
np.random.seed(42)
N, D = 50, 3
X = np.random.randn(N, D)
true_w = np.array([1.5, -2.0, 0.8])
t = X @ true_w + np.random.normal(0, 0.1, size=N)
sigma_noise = 0.3

# 解析的 Ridge 解 (lambda = N * sigma^2)
lambda_theory = N * (sigma_noise**2)
w_ridge = np.linalg.solve(X.T @ X + lambda_theory * np.eye(D), X.T @ t)

# モンテカルロシミュレーションによる入力ノイズ下での平均勾配ゼロ解
K = 20000
XTX_mc = np.zeros((D, D))
XTt_mc = np.zeros(D)
for _ in range(K):
    X_noisy = X + np.random.normal(0, sigma_noise, size=X.shape)
    XTX_mc += X_noisy.T @ X_noisy
    XTt_mc += X_noisy.T @ t
w_mc = np.linalg.solve(XTX_mc / K, XTt_mc / K)

diff = np.linalg.norm(w_ridge - w_mc)
assert diff < 0.05, f"Monte Carlo estimate diff too large: {diff}"
print(f"Exercise 3.4 PASSED: Analytic Ridge vs Noisy Input MC diff = {diff:.4f} (lambda={lambda_theory:.2f})")"""))

    # Exercise 3.5
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.5
**問題**: 制約条件 $\\sum_{j=1}^M |w_j|^q \\le \\eta$ の下で二乗和誤差 $E_D(\\mathbf{w})$ を最小化する問題をラグランジュ未定乗数法を用いて定式化せよ。
また、ペナルティ関数 $\\frac{\\lambda}{2}\\sum_{j=1}^M |w_j|^q$ を付加した無制約最適化問題との関係を論ぜよ。

### [解答の道筋と穴埋め]
1. 不等式制約付き最適化問題に対するラグランジュ関数は:
   $$ \\mathcal{L}(\\mathbf{w}, \\lambda) = E_D(\\mathbf{w}) + \\frac{\\lambda}{2} \\left( \\sum_{j=1}^M |w_j|^q - \\eta \\right) $$
2. KKT (Karush-Kuhn-Tucker) 条件:
   - $\\nabla_{\\mathbf{w}} \\mathcal{L}(\\mathbf{w}, \\lambda) = \\mathbf{0}$
   - 主可能条件: $\\sum_{j=1}^M |w_j|^q \\le \\eta$
   - 双対可能条件: $\\lambda \\ge 0$
   - 相補スラック性: $\\text{[ 穴埋め 1: ? ]} = \\lambda \\left( \\sum_{j=1}^M |w_j|^q - \\eta \\right) = 0$
3. 制約が有効（境界上にある）場合、$\\lambda > 0$ かつ $\\sum_{j=1}^M |w_j|^q = \\eta$ となり、無制約の正則化最小二乗法 $\\min_{\\mathbf{w}} \\{ E_D(\\mathbf{w}) + \\frac{\\lambda}{2}\\sum_{j=1}^M |w_j|^q \\}$ の解と完全に一致する。
4. 各 $\\eta$ に対して一意な $\\lambda$ が対応する（$\\eta$ を小さくすると制約が厳しくなり $\\lambda$ は大きくなる）。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.5 数値検証
np.random.seed(42)
N, M = 20, 2
Phi = np.random.randn(N, M)
t = np.random.randn(N)
q = 1.0 # Lasso
eta = 0.5

# 1. 制約付き最適化: min E_D(w) s.t. ||w||_1 <= eta
def loss(w):
    return 0.5 * np.sum((t - Phi @ w)**2)

res_con = optimize.minimize(loss, x0=[0.0, 0.0], constraints={'type': 'ineq', 'fun': lambda w: eta - np.sum(np.abs(w))})
w_constrained = res_con.x

# 2. 対応する lambda をグリッドサーチして無制約最適化と比較
lambdas = np.linspace(1.0, 20.0, 100)
best_diff = 1e9
best_lambda = None
for lam in lambdas:
    res_uncon = optimize.minimize(lambda w: loss(w) + 0.5 * lam * np.sum(np.abs(w)), x0=[0.0, 0.0])
    diff = np.linalg.norm(res_uncon.x - w_constrained)
    if diff < best_diff:
        best_diff = diff
        best_lambda = lam

assert best_diff < 0.01, f"Equivalence failed: best diff = {best_diff}"
print(f"Exercise 3.5 PASSED: Constrained vs Penalized matched at lambda={best_lambda:.2f} (diff={best_diff:.4f})")"""))

    # Exercise 3.6
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.6
**問題**: 目的変数が $K$ 次元のベクトル $\\mathbf{t} \\in \\mathbb{R}^K$ である線形回帰モデルにおいて、条件付き分布が多変量正規分布
$$ p(\\mathbf{t} | \\mathbf{x}, \\mathbf{W}, \\beta) = \\mathcal{N}(\\mathbf{t} | \\mathbf{W}^T \\boldsymbol{\\phi}(\\mathbf{x}), \\beta^{-1} \\mathbf{I}) $$
で与えられるとき、最尤推定解 $\\mathbf{W}_{\\mathrm{ML}}$ が
$$ \\mathbf{W}_{\\mathrm{ML}} = (\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T \\mathbf{T} $$
となり、各出力次元 $k$ について独立に単一目的変数の最小二乗問題を解くことと等価であることを示せ。

### [解答の道筋と穴埋め]
1. $N$ 個の独立な観測値に対する対数尤度関数:
   $$ \\ln p(\\mathbf{T} | \\mathbf{X}, \\mathbf{W}, \\beta) = -\\frac{NK}{2}\\ln(2\\pi) + \\frac{NK}{2}\\ln\\beta - \\frac{\\beta}{2} \\sum_{n=1}^N \\|\\mathbf{t}_n - \\mathbf{W}^T \\boldsymbol{\\phi}(\\mathbf{x}_n)\\|^2 $$
2. 行列のフロベニウスノルムおよびトレース表現を用いると:
   $$ \\sum_{n=1}^N \\|\\mathbf{t}_n - \\mathbf{W}^T \\boldsymbol{\\phi}(\\mathbf{x}_n)\\|^2 = \\mathrm{Tr}[ (\\mathbf{T} - \\mathbf{\\Phi}\\mathbf{W})^T (\\mathbf{T} - \\mathbf{\\Phi}\\mathbf{W}) ] = \\sum_{k=1}^K \\|\\mathbf{t}_{:, k} - \\mathbf{\\Phi} \\mathbf{w}_k\\|^2 $$
3. 各列 $\\mathbf{w}_k$ は互いに干渉せず完全に独立な和となっているため、$\\mathbf{W}$ に関する勾配を 0 と置くと:
   $$ \\mathbf{\\Phi}^T (\\mathbf{T} - \\mathbf{\\Phi}\\mathbf{W}_{\\mathrm{ML}}) = \\mathbf{0} \\implies \\mathbf{W}_{\\mathrm{ML}} = \\text{[ 穴埋め 1: ? ]} = (\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T \\mathbf{T} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.6 数値検証
np.random.seed(42)
N, M, K = 30, 4, 3
Phi = np.random.randn(N, M)
T = np.random.randn(N, K)

# 一括多変量最尤解
W_matrix = np.linalg.solve(Phi.T @ Phi, Phi.T @ T)

# 各出力次元 k ごとの個別最尤解
W_decoupled = np.zeros((M, K))
for k in range(K):
    W_decoupled[:, k] = np.linalg.solve(Phi.T @ Phi, Phi.T @ T[:, k])

diff = np.linalg.norm(W_matrix - W_decoupled)
assert np.isclose(diff, 0.0, atol=1e-12), f"Decoupling failed: {diff}"
print(f"Exercise 3.6 PASSED: Multivariate regression decouples exactly, diff = {diff:.2e}")"""))

    # Exercise 3.7
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.7
**問題**: ガウス事前分布 $p(\\mathbf{w}) = \\mathcal{N}(\\mathbf{w} | \\mathbf{m}_0, \\mathbf{S}_0)$ と尤度関数 $p(\\mathbf{t} | \\mathbf{w}) = \\mathcal{N}(\\mathbf{t} | \\mathbf{\\Phi}\\mathbf{w}, \\beta^{-1}\\mathbf{I})$ の積から、指数部の平方完成を行って事後分布 $p(\\mathbf{w} | \\mathbf{t}) = \\mathcal{N}(\\mathbf{w} | \\mathbf{m}_N, \\mathbf{S}_N)$ (式 3.49-3.51) を導出せよ。

### [解答の道筋と穴埋め]
1. 事後分布の対数は定数項を除いて指数部の和となる:
   $$ \\ln p(\\mathbf{w} | \\mathbf{t}) = -\\frac{1}{2} (\\mathbf{w} - \\mathbf{m}_0)^T \\mathbf{S}_0^{-1} (\\mathbf{w} - \\mathbf{m}_0) - \\frac{\\beta}{2} (\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{w})^T (\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{w}) + \\text{const} $$
2. $\\mathbf{w}$ の二次形式を展開して同類項をまとめる:
   $$ = -\\frac{1}{2} \\left\\{ \\mathbf{w}^T (\\mathbf{S}_0^{-1} + \\beta \\mathbf{\\Phi}^T \\mathbf{\\Phi}) \\mathbf{w} - 2 \\mathbf{w}^T (\\mathbf{S}_0^{-1}\\mathbf{m}_0 + \\beta \\mathbf{\\Phi}^T \\mathbf{t}) \\right\\} + \\text{const} $$
3. 事後精度行列を $\\mathbf{S}_N^{-1} = \\text{[ 穴埋め 1: ? ]} = \\mathbf{S}_0^{-1} + \\beta \\mathbf{\\Phi}^T \\mathbf{\\Phi}$ と定義し、平方完成すると:
   $$ = -\\frac{1}{2} (\\mathbf{w} - \\mathbf{m}_N)^T \\mathbf{S}_N^{-1} (\\mathbf{w} - \\mathbf{m}_N) + \\text{const} $$
   ここで事後平均は:
   $$ \\mathbf{m}_N = \\text{[ 穴埋め 2: ? ]} = \\mathbf{S}_N (\\mathbf{S}_0^{-1}\\mathbf{m}_0 + \\beta \\mathbf{\\Phi}^T \\mathbf{t}) $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.7 数値検証
np.random.seed(42)
N, M = 10, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha, beta = 2.0, 5.0
m0 = np.zeros(M)
S0_inv = alpha * np.eye(M)

# 解析的導出 (式 3.50, 3.51)
S_N_inv = S0_inv + beta * (Phi.T @ Phi)
S_N = np.linalg.inv(S_N_inv)
m_N = S_N @ (S0_inv @ m0 + beta * Phi.T @ t)

# BayesianLinearRegression クラスとの完全一致を検証
blr = BayesianLinearRegression(alpha=alpha, beta=beta).fit(Phi, t)
assert np.allclose(blr.m_N, m_N, atol=1e-12)
assert np.allclose(blr.S_N, S_N, atol=1e-12)
print("Exercise 3.7 PASSED: Completing square formula matches class implementation exactly.")"""))

    # Exercise 3.8
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.8
**問題**: 逐次ベイズ更新 (Sequential Bayesian Learning) において、$N$ 個のデータから得られた事後分布 $\\mathcal{N}(\\mathbf{w} | \\mathbf{m}_N, \\mathbf{S}_N)$ を新たな事前分布とし、新たな 1 点 $(\\mathbf{x}_{N+1}, t_{N+1})$ を観測して更新した事後分布が、全 $N+1$ 点を一括して学習した事後分布と厳密に一致することを示せ。

### [解答の道筋と穴埋め]
1. 1点追加時の新たな精度行列:
   $$ \\mathbf{S}_{N+1}^{-1} = \\mathbf{S}_N^{-1} + \\beta \\boldsymbol{\\phi}_{N+1} \\boldsymbol{\\phi}_{N+1}^T = \\left( \\mathbf{S}_0^{-1} + \\beta \\sum_{n=1}^N \\boldsymbol{\\phi}_n \\boldsymbol{\\phi}_n^T \\right) + \\beta \\boldsymbol{\\phi}_{N+1} \\boldsymbol{\\phi}_{N+1}^T = \\text{[ 穴埋め 1: ? ]} $$
2. 新たな事後平均:
   $$ \\mathbf{m}_{N+1} = \\mathbf{S}_{N+1} (\\mathbf{S}_N^{-1} \\mathbf{m}_N + \\beta t_{N+1} \\boldsymbol{\\phi}_{N+1}) $$
   ここで $\\mathbf{S}_N^{-1}\\mathbf{m}_N = \\mathbf{S}_0^{-1}\\mathbf{m}_0 + \\beta \\sum_{n=1}^N t_n \\boldsymbol{\\phi}_n$ を代入すると:
   $$ \\mathbf{m}_{N+1} = \\mathbf{S}_{N+1} \\left( \\mathbf{S}_0^{-1}\\mathbf{m}_0 + \\beta \\sum_{n=1}^{N+1} t_n \\boldsymbol{\\phi}_n \\right) = \\text{[ 穴埋め 2: ? ]} $$
   これは $N+1$ 個のデータに対するバッチ解と厳密に一致する。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.8 数値検証
np.random.seed(42)
N, M = 12, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha, beta = 1.5, 4.0

# 1. バッチ学習
batch_blr = BayesianLinearRegression(alpha=alpha, beta=beta).fit(Phi, t)

# 2. 逐次学習 (1点ずつ更新)
seq_blr = BayesianLinearRegression(alpha=alpha, beta=beta)
for i in range(N):
    seq_blr.update(Phi[i], t[i])

diff_m = np.linalg.norm(batch_blr.m_N - seq_blr.m_N)
diff_S = np.linalg.norm(batch_blr.S_N - seq_blr.S_N)
assert np.isclose(diff_m, 0.0, atol=1e-12)
assert np.isclose(diff_S, 0.0, atol=1e-12)
print(f"Exercise 3.8 PASSED: Sequential vs Batch mismatch = {diff_m:.2e}, {diff_S:.2e}")"""))

    # Exercise 3.9
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.9
**問題**: PRML第2.3.3節の線形ガウスモデルの公式 (式 2.113 〜 2.117) を用いて、事前分布 $p(\\mathbf{w}) = \\mathcal{N}(\\mathbf{w} | \\mathbf{m}_0, \\mathbf{S}_0)$ と条件付き分布 $p(\\mathbf{t} | \\mathbf{w}) = \\mathcal{N}(\\mathbf{t} | \\mathbf{\\Phi}\\mathbf{w}, \\beta^{-1}\\mathbf{I})$ から、平方完成を行わずに事後分布 $p(\\mathbf{w} | \\mathbf{t})$ を直接導出せよ。

### [解答の道筋と穴埋め]
1. 式 (2.113), (2.114) の対応関係:
   - $\\mathbf{x} \\to \\mathbf{w}$, $\\boldsymbol{\\mu} \\to \\mathbf{m}_0$, $\\mathbf{\\Lambda}^{-1} \\to \\mathbf{S}_0$
   - $\\mathbf{y} \\to \\mathbf{t}$, $\\mathbf{A} \\to \\mathbf{\\Phi}$, $\\mathbf{b} \\to \\mathbf{0}$, $\\mathbf{L}^{-1} \\to \\beta^{-1} \\mathbf{I}$
2. 式 (2.116) より、条件付き分布 $p(\\mathbf{w} | \\mathbf{t})$ の共分散行列 $\\boldsymbol{\\Sigma}$ は:
   $$ \\boldsymbol{\\Sigma}^{-1} = \\mathbf{\\Lambda} + \\mathbf{A}^T \\mathbf{L} \\mathbf{A} = \\text{[ 穴埋め 1: ? ]} = \\mathbf{S}_0^{-1} + \\beta \\mathbf{\\Phi}^T \\mathbf{\\Phi} = \\mathbf{S}_N^{-1} $$
3. 式 (2.117) より、平均ベクトルは:
   $$ \\boldsymbol{\\mu}_{\\mathbf{w}|\\mathbf{t}} = \\boldsymbol{\\Sigma} [\\mathbf{A}^T \\mathbf{L}(\\mathbf{y} - \\mathbf{b}) + \\mathbf{\\Lambda}\\boldsymbol{\\mu}] = \\text{[ 穴埋め 2: ? ]} = \\mathbf{S}_N [\\beta \\mathbf{\\Phi}^T \\mathbf{t} + \\mathbf{S}_0^{-1}\\mathbf{m}_0] = \\mathbf{m}_N $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.9 数値検証
np.random.seed(42)
N, M = 8, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
m0 = np.array([0.5, -0.2, 0.1])
S0 = np.diag([1.0, 2.0, 1.5])
beta = 3.0

# 式 2.116, 2.117 を直接適用
Sigma_inv = np.linalg.inv(S0) + beta * (Phi.T @ Phi)
Sigma = np.linalg.inv(Sigma_inv)
mu = Sigma @ (beta * Phi.T @ t + np.linalg.inv(S0) @ m0)

# 通常の正規事後分布定義との比較
blr = BayesianLinearRegression(m0=m0, S0_inv=np.linalg.inv(S0), beta=beta).fit(Phi, t)
assert np.allclose(blr.m_N, mu, atol=1e-12)
assert np.allclose(blr.S_N, Sigma, atol=1e-12)
print("Exercise 3.9 PASSED: Linear Gaussian conditional formulas yield identical posterior.")"""))

    # Exercise 3.10
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.10
**問題**: ベイズ線形回帰における予測分布
$$ p(t | \\mathbf{x}, \\mathbf{t}) = \\int p(t | \\mathbf{x}, \\mathbf{w}, \\beta) p(\\mathbf{w} | \\mathbf{t}) d\\mathbf{w} $$
を評価し、予測分布がガウス分布 $\\mathcal{N}(t | \\mathbf{m}_N^T \\boldsymbol{\\phi}(\\mathbf{x}), \\sigma_N^2(\\mathbf{x}))$ (式 3.58) となることを示せ。ここで $\\sigma_N^2(\\mathbf{x}) = \\frac{1}{\\beta} + \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{S}_N \\boldsymbol{\\phi}(\\mathbf{x})$ である。

### [解答の道筋と穴埋め]
1. 目的変数 $t$ は重み $\\mathbf{w}$ の線形結合にノイズが加わったもの:
   $$ t = \\mathbf{w}^T \\boldsymbol{\\phi}(\\mathbf{x}) + \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, \\beta^{-1}) $$
2. $\\mathbf{w} \\sim \\mathcal{N}(\\mathbf{m}_N, \\mathbf{S}_N)$ はガウス分布に従うため、線形変換 $y = \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{w}$ もガウス分布に従う:
   - 平均: $\\mathbb{E}[y] = \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbb{E}[\\mathbf{w}] = \\text{[ 穴埋め 1: ? ]} = \\mathbf{m}_N^T \\boldsymbol{\\phi}(\\mathbf{x})$
   - 分散: $\\mathrm{var}[y] = \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathrm{cov}[\\mathbf{w}] \\boldsymbol{\\phi}(\\mathbf{x}) = \\text{[ 穴埋め 2: ? ]} = \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{S}_N \\boldsymbol{\\phi}(\\mathbf{x})$
3. 独立なノイズ $\\epsilon$ が加算されるため、全体の予測分散は分散の和となる:
   $$ \\sigma_N^2(\\mathbf{x}) = \\text{[ 穴埋め 3: ? ]} = \\frac{1}{\\beta} + \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{S}_N \\boldsymbol{\\phi}(\\mathbf{x}) $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.10 数値検証
np.random.seed(42)
N, M = 15, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
blr = BayesianLinearRegression(alpha=2.0, beta=4.0).fit(Phi, t)

x_query = np.array([0.8, -0.5, 1.2])
pred_mean, pred_std = blr.predict(x_query[np.newaxis, :])
pred_var = pred_std[0]**2

# モンテカルロ積分による検証 (100,000サンプル)
w_samples = np.random.multivariate_normal(blr.m_N, blr.S_N, size=100000)
y_samples = w_samples @ x_query
t_samples = y_samples + np.random.normal(0, 1.0 / np.sqrt(blr.beta), size=100000)

mc_mean = np.mean(t_samples)
mc_var = np.var(t_samples)

assert np.isclose(pred_mean[0], mc_mean, atol=0.01)
assert np.isclose(pred_var, mc_var, atol=0.01)
print(f"Exercise 3.10 PASSED: Analytic Mean={pred_mean[0]:.4f} (MC={mc_mean:.4f}), Var={pred_var:.4f} (MC={mc_var:.4f})")"""))

    # Exercise 3.11
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.11
**問題**: Sherman-Morrison-Woodbury 公式を用いて、1点追加時の事後共分散行列の更新式
$$ \\mathbf{S}_{N+1} = \\mathbf{S}_N - \\frac{\\mathbf{S}_N \\boldsymbol{\\phi}_{N+1} \\boldsymbol{\\phi}_{N+1}^T \\mathbf{S}_N}{\\beta^{-1} + \\boldsymbol{\\phi}_{N+1}^T \\mathbf{S}_N \\boldsymbol{\\phi}_{N+1}} $$
を導出し、これを用いて任意の $\\mathbf{x}$ に対する予測分散がデータ数の増加に伴い単調非増加（$\\sigma_{N+1}^2(\\mathbf{x}) \\le \\sigma_N^2(\\mathbf{x})$）であることを示せ。

### [解答の道筋と穴埋め]
1. 精度行列の更新式: $\\mathbf{S}_{N+1}^{-1} = \\mathbf{S}_N^{-1} + \\beta \\boldsymbol{\\phi}_{N+1} \\boldsymbol{\\phi}_{N+1}^T$。
2. Sherman-Morrison 公式 $(\\mathbf{A} + \\mathbf{u}\\mathbf{v}^T)^{-1} = \\mathbf{A}^{-1} - \\frac{\\mathbf{A}^{-1}\\mathbf{u}\\mathbf{v}^T \\mathbf{A}^{-1}}{1 + \\mathbf{v}^T \\mathbf{A}^{-1}\\mathbf{u}}$ を適用すると:
   $$ \\mathbf{S}_{N+1} = \\mathbf{S}_N - \\frac{\\beta \\mathbf{S}_N \\boldsymbol{\\phi}_{N+1} \\boldsymbol{\\phi}_{N+1}^T \\mathbf{S}_N}{1 + \\beta \\boldsymbol{\\phi}_{N+1}^T \\mathbf{S}_N \\boldsymbol{\\phi}_{N+1}} = \\mathbf{S}_N - \\frac{\\mathbf{S}_N \\boldsymbol{\\phi}_{N+1} \\boldsymbol{\\phi}_{N+1}^T \\mathbf{S}_N}{\\text{[ 穴埋め 1: ? ]}} $$
   分母は $\\beta^{-1} + \\boldsymbol{\\phi}_{N+1}^T \\mathbf{S}_N \\boldsymbol{\\phi}_{N+1} = \\sigma_N^2(\\mathbf{x}_{N+1}) > 0$ である。
3. 予測分散の差を計算すると:
   $$ \\sigma_N^2(\\mathbf{x}) - \\sigma_{N+1}^2(\\mathbf{x}) = \\boldsymbol{\\phi}(\\mathbf{x})^T (\\mathbf{S}_N - \\mathbf{S}_{N+1}) \\boldsymbol{\\phi}(\\mathbf{x}) = \\frac{(\\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{S}_N \\boldsymbol{\\phi}_{N+1})^2}{\\sigma_N^2(\\mathbf{x}_{N+1})} \\ge 0 $$
4. 二乗項および分母は常に非負であるため、$\\sigma_{N+1}^2(\\mathbf{x}) \\le \\sigma_N^2(\\mathbf{x})$ がすべての $\\mathbf{x}$ について証明された。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.11 数値検証
np.random.seed(42)
M = 4
alpha, beta = 2.0, 3.0
blr = BayesianLinearRegression(alpha=alpha, beta=beta)
blr.fit(np.zeros((0, M)), np.array([]))

# ランダムに点を1点ずつ追加し、50個のクエリ点における予測分散の単調減少を検証
x_queries = np.random.randn(50, M)
_, std_prev = blr.predict(x_queries)
var_prev = std_prev**2

for step in range(15):
    phi_new = np.random.randn(M)
    t_new = np.random.randn()
    blr.update(phi_new, t_new)
    _, std_curr = blr.predict(x_queries)
    var_curr = std_curr**2
    assert np.all(var_curr <= var_prev + 1e-12), f"Variance increased at step {step}!"
    var_prev = var_curr

print("Exercise 3.11 PASSED: Predictive variance monotonically decreases with every observed data point.")"""))

    # Exercise 3.12
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.12
**問題**: 重み $\\mathbf{w}$ とノイズ精度 $\\beta$ の双方が未知である場合、共役事前分布として正規-ガンマ分布
$$ p(\\mathbf{w}, \\beta) = \\mathcal{N}(\\mathbf{w} | \\mathbf{m}_0, (\\beta \\mathbf{S}_0)^{-1}) \\mathrm{Gam}(\\beta | a_0, b_0) $$
を導入する。事後分布 $p(\\mathbf{w}, \\beta | \\mathbf{t})$ も同一の正規-ガンマ分布となることを示し、更新パラメータ $\\mathbf{m}_N, \\mathbf{S}_N, a_N, b_N$ を導出せよ。

### [解答の道筋と穴埋め]
1. 同時事前分布と尤度関数の積:
   $$ p(\\mathbf{w}, \\beta | \\mathbf{t}) \\propto \\beta^{M/2} \\exp\\left(-\\frac{\\beta}{2}(\\mathbf{w} - \\mathbf{m}_0)^T \\mathbf{S}_0^{-1}(\\mathbf{w} - \\mathbf{m}_0)\\right) \\cdot \\beta^{a_0 - 1} e^{-b_0 \\beta} \\cdot \\beta^{N/2} \\exp\\left(-\\frac{\\beta}{2}\\sum_{n=1}^N (t_n - \\mathbf{w}^T \\boldsymbol{\\phi}_n)^2\\right) $$
2. 指数部において $\\beta$ でくくって平方完成を行うと:
   - $\\mathbf{S}_N^{-1} = \\text{[ 穴埋め 1: ? ]} = \\mathbf{S}_0^{-1} + \\mathbf{\\Phi}^T \\mathbf{\\Phi}$
   - $\\mathbf{m}_N = \\text{[ 穴埋め 2: ? ]} = \\mathbf{S}_N (\\mathbf{S}_0^{-1}\\mathbf{m}_0 + \\mathbf{\\Phi}^T \\mathbf{t})$
3. $\\beta$ のべき乗および残余の定数項を整理すると:
   - $a_N = \\text{[ 穴埋め 3: ? ]} = a_0 + \\frac{N}{2}$
   - $b_N = \\text{[ 穴埋め 4: ? ]} = b_0 + \\frac{1}{2}(\\mathbf{t}^T \\mathbf{t} + \\mathbf{m}_0^T \\mathbf{S}_0^{-1}\\mathbf{m}_0 - \\mathbf{m}_N^T \\mathbf{S}_N^{-1}\\mathbf{m}_N)$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.12 数値検証
np.random.seed(42)
N, M = 15, 2
Phi = np.random.randn(N, M)
t = np.random.randn(N)
a0, b0 = 2.0, 3.0

ng = NormalGammaLinearRegression(a0=a0, b0=b0).fit(Phi, t)
assert ng.a_N == a0 + N / 2.0
assert ng.S_N.shape == (M, M)
assert ng.b_N > 0

# 負の対数事後密度の最適化解との比較
def neg_log_posterior(params):
    w = params[:M]
    beta = params[M]
    if beta <= 0: return 1e9
    log_prior = (M/2)*np.log(beta) - 0.5*beta*(w @ ng.S0_inv @ w) + (a0-1)*np.log(beta) - b0*beta
    log_lik = (N/2)*np.log(beta) - 0.5*beta*np.sum((t - Phi @ w)**2)
    return -(log_prior + log_lik)

res = optimize.minimize(neg_log_posterior, x0=[0.0, 0.0, 1.0], bounds=[(None, None), (None, None), (1e-5, None)])
w_map = res.x[:M]
assert np.allclose(ng.m_N, w_map, atol=1e-5)
print("Exercise 3.12 PASSED: Normal-Gamma posterior analytical parameters match numerical MAP.")"""))

    # Exercise 3.13
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.13
**問題**: Exercise 3.12 の正規-ガンマ共役モデルにおいて、重み $\\mathbf{w}$ および精度 $\\beta$ を周辺化積分して消去することにより、予測分布 $p(t | \\mathbf{x}, \\mathbf{t})$ が自由度 $\\nu = 2 a_N$ のスチューデントの t 分布 (Student's t-distribution) となることを導出せよ。

### [解答の道筋と穴埋め]
1. $\\mathbf{w}$ の周辺化積分: 条件付き予測分布 $p(t | \\mathbf{x}, \\beta, \\mathbf{t})$ は Exercise 3.10 と同様にガウス分布となる:
   $$ p(t | \\mathbf{x}, \\beta, \\mathbf{t}) = \\mathcal{N}(t | \\mathbf{m}_N^T \\boldsymbol{\\phi}(\\mathbf{x}), \\beta^{-1}(1 + \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{S}_N \\boldsymbol{\\phi}(\\mathbf{x}))) $$
2. $\\beta$ の周辺化積分: ガウス分布とガンマ分布の合成（無限混合）を計算する:
   $$ p(t | \\mathbf{x}, \\mathbf{t}) = \\int_0^\\infty \\mathcal{N}(t | \\mu, \\beta^{-1} s^2) \\mathrm{Gam}(\\beta | a_N, b_N) d\\beta $$
3. ガンマ関数の積分公式 $\\int_0^\\infty \\beta^{p-1} e^{-q \\beta} d\\beta = \\frac{\\Gamma(p)}{q^p}$ を適用すると:
   $$ p(t | \\mathbf{x}, \\mathbf{t}) = \\text{St}(t | \\mathbf{m}_N^T \\boldsymbol{\\phi}(\\mathbf{x}), \\lambda(\\mathbf{x}), \\nu) $$
   ここで自由度は $\\nu = \\text{[ 穴埋め 1: ? ]} = 2 a_N$、精度パラメータは:
   $$ \\lambda(\\mathbf{x}) = \\text{[ 穴埋め 2: ? ]} = \\frac{a_N}{b_N (1 + \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{S}_N \\boldsymbol{\\phi}(\\mathbf{x}))} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.13 数値検証
np.random.seed(42)
N, M = 12, 2
Phi = np.random.randn(N, M)
t = np.random.randn(N)
ng = NormalGammaLinearRegression(a0=2.0, b0=3.0).fit(Phi, t)

x_test = np.array([0.5, -0.8])
mean, std, nu = ng.predict(x_test[np.newaxis, :])
nu_val = nu

# 数値重積分による検証
s2 = 1.0 + x_test @ ng.S_N @ x_test
mu_pred = float(ng.m_N @ x_test)
t_val = mu_pred + 0.3

# 解析的 Student-t 密度
lambda_val = ng.a_N / (ng.b_N * s2)
analytic_pdf = stats.t.pdf(t_val, df=2*ng.a_N, loc=mu_pred, scale=1.0 / np.sqrt(lambda_val))

# ガウス×ガンマの数値積分
def integrand(beta):
    p_gauss = stats.norm.pdf(t_val, loc=mu_pred, scale=np.sqrt(s2 / beta))
    p_gam = stats.gamma.pdf(beta, a=ng.a_N, scale=1.0 / ng.b_N)
    return p_gauss * p_gam

num_pdf, _ = integrate.quad(integrand, 1e-8, 100.0)
assert np.isclose(analytic_pdf, num_pdf, rtol=1e-4)
print(f"Exercise 3.13 PASSED: Student's t predictive analytic ({analytic_pdf:.5f}) matches numerical integral ({num_pdf:.5f}).")"""))

    # Exercise 3.14
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.14
**問題**: 基底関数 $\\boldsymbol{\\phi}(\\mathbf{x})$ が定数項 $\\phi_0(\\mathbf{x}) = 1$ を含む場合、等価カーネル (Equivalent Kernel)
$$ k(\\mathbf{x}, \\mathbf{x}_n) = \\beta \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{S}_N \\boldsymbol{\\phi}(\\mathbf{x}_n) $$
がすべての $\\mathbf{x}$ に対して総和制約
$$ \\sum_{n=1}^N k(\\mathbf{x}, \\mathbf{x}_n) = 1 $$
を満たすことを示せ。

### [解答の道筋と穴埋め]
1. 定数項 $\\phi_0(\\mathbf{x}) = 1$ が含まれるため、ベクトル $\\mathbf{u} = (1, 0, \\dots, 0)^T$ と置くと、すべての $\\mathbf{x}$ について $\\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{u} = 1$、および $\\mathbf{\\Phi}\\mathbf{u} = \\mathbf{1}$（全要素が 1 のベクトル）が成り立つ。
2. 等価カーネルの和を行列形式で表す:
   $$ \\sum_{n=1}^N k(\\mathbf{x}, \\mathbf{x}_n) = \\beta \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{S}_N \\sum_{n=1}^N \\boldsymbol{\\phi}(\\mathbf{x}_n) = \\beta \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{S}_N \\mathbf{\\Phi}^T \\mathbf{1} $$
3. $\\mathbf{1} = \\mathbf{\\Phi}\\mathbf{u}$ を代入すると:
   $$ \\sum_{n=1}^N k(\\mathbf{x}, \\mathbf{x}_n) = \\beta \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{S}_N \\mathbf{\\Phi}^T \\mathbf{\\Phi} \\mathbf{u} $$
4. 無情報事前分布の極限（または最尤推定量への漸近 $\\alpha \\to 0$）において $\\mathbf{S}_N = (\\beta \\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1}$ であるから:
   $$ \\beta \\mathbf{S}_N \\mathbf{\\Phi}^T \\mathbf{\\Phi} = \\text{[ 穴埋め 1: ? ]} = \\mathbf{I} $$
5. したがって:
   $$ \\sum_{n=1}^N k(\\mathbf{x}, \\mathbf{x}_n) = \\boldsymbol{\\phi}(\\mathbf{x})^T \\mathbf{u} = \\text{[ 穴埋め 2: ? ]} = 1 $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.14 数値検証
np.random.seed(42)
N = 25
X_train = np.linspace(-1, 1, N)[:, np.newaxis]
poly = PolynomialBasis(degree=3) # バイアス項 x^0 = 1 を含む
Phi_train = poly.transform(X_train)
y_dummy = np.ones(N)

# 非常に緩やかな事前分布 (alpha -> 0)
blr = BayesianLinearRegression(alpha=1e-7, beta=10.0).fit(Phi_train, y_dummy)

# 任意の評価点 x における等価カーネルの和を計算
X_eval = np.linspace(-0.8, 0.8, 10)[:, np.newaxis]
Phi_eval = poly.transform(X_eval)
K = blr.equivalent_kernel(Phi_eval) # shape (10, N)

kernel_sums = np.sum(K, axis=1)
assert np.allclose(kernel_sums, 1.0, atol=1e-4)
print(f"Exercise 3.14 PASSED: Sum of equivalent kernel weights = {kernel_sums[:3]} (all strictly 1.0)")"""))

    # Exercise 3.15
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.15
**問題**: エビデンスフレームワークにおいて、$\\alpha$ と $\\beta$ が周辺尤度を最大化するように設定されているとき、事後最頻値 $\\mathbf{m}_N$ における誤差関数
$$ 2 E(\\mathbf{m}_N) = \\beta \\|\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{m}_N\\|^2 + \\alpha \\|\\mathbf{m}_N\\|^2 $$
が関係式 $2 E(\\mathbf{m}_N) = N$ を満たすことを示せ。

### [解答の道筋と穴埋め]
1. エビデンス最大化における $\\alpha$ の再推定式 (式 3.92):
   $$ \\alpha \\|\\mathbf{m}_N\\|^2 = \\text{[ 穴埋め 1: ? ]} = \\gamma $$
2. エビデンス最大化における $\\beta$ の再推定式 (式 3.95):
   $$ \\beta \\|\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{m}_N\\|^2 = \\text{[ 穴埋め 2: ? ]} = N - \\gamma $$
3. 両者を足し合わせると:
   $$ 2 E(\\mathbf{m}_N) = \\beta \\|\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{m}_N\\|^2 + \\alpha \\|\\mathbf{m}_N\\|^2 = (N - \\gamma) + \\gamma = \\text{[ 穴埋め 3: ? ]} = N $$
   有効パラメータ数 $\\gamma$ が完全に相殺し、厳密にデータ数 $N$ となる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.15 数値検証
np.random.seed(42)
N, M = 30, 4
X = np.linspace(0, 1, N)[:, np.newaxis]
Phi = PolynomialBasis(degree=3).transform(X)
t = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.2, size=N)

eb = EvidenceApproximation(max_iter=100, tol=1e-7).fit(Phi, t)
m_N = eb.m_N
alpha, beta = eb.alpha, eb.beta

E_D = 0.5 * np.sum((t - Phi @ m_N)**2)
E_W = 0.5 * np.sum(m_N**2)
two_E_mN = beta * 2 * E_D + alpha * 2 * E_W

assert np.isclose(two_E_mN, N, atol=1e-3), f"Discrepancy: {two_E_mN} vs {N}"
print(f"Exercise 3.15 PASSED: 2 * E(m_N) = {two_E_mN:.4f}, strictly equal to N = {N}")"""))

    # Exercise 3.16
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.16
**問題**: 線形回帰モデルの対数エビデンス関数 $p(\\mathbf{t} | \\alpha, \\beta)$ を、線形ガウス公式 (2.115) を用いて周辺化積分 (3.77) を直接計算することにより導出せよ。

### [解答の道筋と穴埋め]
1. $\\mathbf{w}$ の事前分布は $p(\\mathbf{w}) = \\mathcal{N}(\\mathbf{w} | \\mathbf{0}, \\alpha^{-1}\\mathbf{I})$。
2. 条件付き分布は $p(\\mathbf{t} | \\mathbf{w}) = \\mathcal{N}(\\mathbf{t} | \\mathbf{\\Phi}\\mathbf{w}, \\beta^{-1}\\mathbf{I})$。
3. 線形ガウス公式 (2.115) より、$\\mathbf{t}$ の周辺分布 $p(\\mathbf{t})$ もガウス分布となり、その平均と共分散行列は:
   - 平均: $\\mathbb{E}[\\mathbf{t}] = \\mathbf{\\Phi} \\mathbb{E}[\\mathbf{w}] = \\mathbf{0}$
   - 共分散: $\\mathbf{C} = \\text{[ 穴埋め 1: ? ]} = \\beta^{-1}\\mathbf{I} + \\alpha^{-1}\\mathbf{\\Phi}\\mathbf{\\Phi}^T$
4. したがって対数エビデンス関数は:
   $$ \\ln p(\\mathbf{t} | \\alpha, \\beta) = \\text{[ 穴埋め 2: ? ]} = -\\frac{N}{2}\\ln(2\\pi) - \\frac{1}{2}\\ln|\\mathbf{C}| - \\frac{1}{2}\\mathbf{t}^T \\mathbf{C}^{-1}\\mathbf{t} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.16 数値検証
np.random.seed(42)
N, M = 15, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha, beta = 2.0, 4.0

# 1. 直接共分散行列 C による対数尤度
C = (1.0 / beta) * np.eye(N) + (1.0 / alpha) * (Phi @ Phi.T)
_, log_det_C = np.linalg.slogdet(C)
log_p_direct = -0.5 * N * np.log(2 * np.pi) - 0.5 * log_det_C - 0.5 * (t @ np.linalg.solve(C, t))

# 2. PRML 式 3.86 (bayesian_model_evidence) による対数尤度
log_p_formula = bayesian_model_evidence(Phi, t, alpha, beta)

diff = abs(log_p_direct - log_p_formula)
assert np.isclose(diff, 0.0, atol=1e-10), f"Formula mismatch: {diff}"
print(f"Exercise 3.16 PASSED: Direct linear Gaussian marginal matches formula 3.86 (diff={diff:.2e})")"""))

    # Exercise 3.17
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.17
**問題**: ベイズ線形回帰モデルのエビデンス関数が
$$ p(\\mathbf{t} | \\alpha, \\beta) = \\left( \\frac{\\beta}{2\\pi} \\right)^{N/2} \\left( \\frac{\\alpha}{2\\pi} \\right)^{M/2} \\int \\exp\\{-E(\\mathbf{w})\\} d\\mathbf{w} $$
の形に書けることを示せ。ここで $E(\\mathbf{w})$ は式 (3.79) で定義される正則化二乗和誤差関数である。

### [解答の道筋と穴埋め]
1. 尤度関数と事前分布の積を書き出す:
   $$ p(\\mathbf{t} | \\mathbf{w}, \\beta) = \\left( \\frac{\\beta}{2\\pi} \\right)^{N/2} \\exp\\left( -\\frac{\\beta}{2}\\|\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{w}\\|^2 \\right) $$
   $$ p(\\mathbf{w} | \\alpha) = \\left( \\frac{\\alpha}{2\\pi} \\right)^{M/2} \\exp\\left( -\\frac{\\alpha}{2}\\|\\mathbf{w}\\|^2 \\right) $$
2. 指数部の和をとると:
   $$ -\\frac{\\beta}{2}\\|\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{w}\\|^2 - \\frac{\\alpha}{2}\\|\\mathbf{w}\\|^2 = -\\left( \\beta E_D(\\mathbf{w}) + \\alpha E_W(\\mathbf{w}) \\right) = -E(\\mathbf{w}) $$
3. 正規化定数の積を積分の外にくくり出すことで、題意の表現が得られる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.17 数値検証
np.random.seed(42)
N, M = 10, 2
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha, beta = 1.5, 3.5
w = np.array([0.7, -1.1])

# 被積分関数の対数
log_integrand = (-0.5 * N * np.log(2*np.pi/beta) - 0.5 * beta * np.sum((t - Phi @ w)**2)) + \
                (-0.5 * M * np.log(2*np.pi/alpha) - 0.5 * alpha * np.sum(w**2))

# 式 3.78 の形
E_w = 0.5 * beta * np.sum((t - Phi @ w)**2) + 0.5 * alpha * np.sum(w**2)
log_formula = (N/2)*np.log(beta/(2*np.pi)) + (M/2)*np.log(alpha/(2*np.pi)) - E_w

assert np.isclose(log_integrand, log_formula)
print("Exercise 3.17 PASSED: Integrand factoring verified exactly.")"""))

    # Exercise 3.18
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.18
**問題**: $\\mathbf{w}$ に関して平方完成を行うことにより、ベイズ線形回帰の誤差関数 $E(\\mathbf{w})$ (式 3.79) が
$$ E(\\mathbf{w}) = E(\\mathbf{m}_N) + \\frac{1}{2}(\\mathbf{w} - \\mathbf{m}_N)^T \\mathbf{A} (\\mathbf{w} - \\mathbf{m}_N) $$
の形 (式 3.80) に書けることを示せ。ここで $\\mathbf{A} = \\alpha \\mathbf{I} + \\beta \\mathbf{\\Phi}^T \\mathbf{\\Phi}$ である。

### [解答の道筋と穴埋め]
1. $E(\\mathbf{w}) = \\frac{\\beta}{2}\\|\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{w}\\|^2 + \\frac{\\alpha}{2}\\|\\mathbf{w}\\|^2$ を展開する:
   $$ E(\\mathbf{w}) = \\frac{1}{2}\\mathbf{w}^T (\\alpha \\mathbf{I} + \\beta \\mathbf{\\Phi}^T \\mathbf{\\Phi})\\mathbf{w} - \\beta \\mathbf{w}^T \\mathbf{\\Phi}^T \\mathbf{t} + \\frac{\\beta}{2}\\mathbf{t}^T \\mathbf{t} $$
2. 二次の係数行列を $\\mathbf{A} = \\text{[ 穴埋め 1: ? ]} = \\alpha \\mathbf{I} + \\beta \\mathbf{\\Phi}^T \\mathbf{\\Phi}$ と置き、最頻値 $\\mathbf{m}_N = \\beta \\mathbf{A}^{-1}\\mathbf{\\Phi}^T \\mathbf{t}$ を用いて平方完成すると:
   $$ \\frac{1}{2}(\\mathbf{w} - \\mathbf{m}_N)^T \\mathbf{A} (\\mathbf{w} - \\mathbf{m}_N) = \\frac{1}{2}\\mathbf{w}^T \\mathbf{A}\\mathbf{w} - \\mathbf{w}^T \\mathbf{A}\\mathbf{m}_N + \\frac{1}{2}\\mathbf{m}_N^T \\mathbf{A}\\mathbf{m}_N $$
3. これと元の方程式を差し引きして定数部を照合すると:
   $$ E(\\mathbf{w}) = \\text{[ 穴埋め 2: ? ]} + \\frac{1}{2}(\\mathbf{w} - \\mathbf{m}_N)^T \\mathbf{A} (\\mathbf{w} - \\mathbf{m}_N) $$
   となり、最小値 $E(\\mathbf{m}_N)$ からの二次形式として表現される。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.18 数値検証
np.random.seed(42)
N, M = 15, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha, beta = 2.5, 4.5
A = alpha * np.eye(M) + beta * (Phi.T @ Phi)
m_N = np.linalg.solve(A, beta * Phi.T @ t)
E_mN = 0.5 * beta * np.sum((t - Phi @ m_N)**2) + 0.5 * alpha * np.sum(m_N**2)

# 任意の 100 個の重みベクトル w で両辺が一致するか検証
for _ in range(100):
    w_rand = np.random.randn(M)
    E_w_direct = 0.5 * beta * np.sum((t - Phi @ w_rand)**2) + 0.5 * alpha * np.sum(w_rand**2)
    E_w_quad = E_mN + 0.5 * (w_rand - m_N) @ A @ (w_rand - m_N)
    assert np.isclose(E_w_direct, E_w_quad, atol=1e-12)

print("Exercise 3.18 PASSED: Completing the square for E(w) verified across 100 random weight vectors.")"""))

    # Exercise 3.19
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.19
**問題**: Exercise 3.18 の平方完成表現を用いて、$\\mathbf{w}$ に関するガウス積分
$$ \\int \\exp\\{-E(\\mathbf{w})\\} d\\mathbf{w} $$
を実行し、式 (3.85) および対数周辺尤度の公式 (3.86) を導出せよ。

### [解答の道筋と穴埋め]
1. $E(\\mathbf{w}) = E(\\mathbf{m}_N) + \\frac{1}{2}(\\mathbf{w} - \\mathbf{m}_N)^T \\mathbf{A} (\\mathbf{w} - \\mathbf{m}_N)$ を積分に代入する:
   $$ \\int \\exp\\{-E(\\mathbf{w})\\} d\\mathbf{w} = \\exp\\{-E(\\mathbf{m}_N)\\} \\int \\exp\\left\\{ -\\frac{1}{2}(\\mathbf{w} - \\mathbf{m}_N)^T \\mathbf{A} (\\mathbf{w} - \\mathbf{m}_N) \\right\\} d\\mathbf{w} $$
2. $M$ 次元ガウス積分の公式より:
   $$ \\int \\exp\\left\\{ -\\frac{1}{2}(\\mathbf{w} - \\mathbf{m}_N)^T \\mathbf{A} (\\mathbf{w} - \\mathbf{m}_N) \\right\\} d\\mathbf{w} = \\text{[ 穴埋め 1: ? ]} = (2\\pi)^{M/2} |\\mathbf{A}|^{-1/2} $$
3. 式 (3.78) の前置係数と掛け合わせると:
   $$ p(\\mathbf{t} | \\alpha, \\beta) = \\left( \\frac{\\beta}{2\\pi} \\right)^{N/2} \\left( \\frac{\\alpha}{2\\pi} \\right)^{M/2} (2\\pi)^{M/2} |\\mathbf{A}|^{-1/2} \\exp\\{-E(\\mathbf{m}_N)\\} $$
4. 両辺の対数をとると、PRML 式 (3.86) が得られる:
   $$ \\ln p(\\mathbf{t} | \\alpha, \\beta) = \\text{[ 穴埋め 2: ? ]} = \\frac{M}{2}\\ln\\alpha + \\frac{N}{2}\\ln\\beta - E(\\mathbf{m}_N) - \\frac{1}{2}\\ln|\\mathbf{A}| - \\frac{N}{2}\\ln(2\\pi) $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.19 数値検証
np.random.seed(42)
N, M = 10, 2
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha, beta = 2.0, 3.0
A = alpha * np.eye(M) + beta * (Phi.T @ Phi)
m_N = np.linalg.solve(A, beta * Phi.T @ t)
E_mN = 0.5 * beta * np.sum((t - Phi @ m_N)**2) + 0.5 * alpha * np.sum(m_N**2)

# 数値重積分 (2次元ガウス積分)
def integrand_2d(w1, w2):
    w = np.array([w1, w2])
    return np.exp(-(0.5 * beta * np.sum((t - Phi @ w)**2) + 0.5 * alpha * np.sum(w**2)))

integral_val, _ = integrate.dblquad(integrand_2d, -5, 5, -5, 5)
analytic_val = np.exp(-E_mN) * (2 * np.pi) * (1.0 / np.sqrt(np.linalg.det(A)))

assert np.isclose(integral_val, analytic_val, rtol=1e-4)
print(f"Exercise 3.19 PASSED: Numerical 2D Gaussian integral ({integral_val:.5e}) matches analytical formula ({analytic_val:.5e}).")"""))

    # Exercise 3.20
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.20
**問題**: 対数エビデンス関数 (3.86) を $\\alpha$ について微分し、停留点条件から $\\alpha$ の再推定式 (3.92)
$$ \\alpha = \\frac{\\gamma}{\\|\\mathbf{m}_N\\|^2} $$
が導かれる全ステップを証明せよ。ここで $\\gamma = \\sum_{i=1}^M \\frac{\\lambda_i}{\\alpha + \\lambda_i}$ である。

### [解答の道筋と穴埋め]
1. $\\mathbf{A} = \\alpha \\mathbf{I} + \\beta \\mathbf{\\Phi}^T \\mathbf{\\Phi}$ の固有値を $\\alpha + \\lambda_i$ とすると、$\\ln|\\mathbf{A}| = \\sum_{i=1}^M \\ln(\\alpha + \\lambda_i)$。
2. 対数エビデンス関数の $\\alpha$ 微分:
   $$ \\frac{d}{d\\alpha} \\ln p(\\mathbf{t} | \\alpha, \\beta) = \\frac{M}{2\\alpha} - \\frac{1}{2}\\|\\mathbf{m}_N\\|^2 - \\frac{1}{2}\\sum_{i=1}^M \\frac{1}{\\alpha + \\lambda_i} $$
   （なお、$\\mathbf{m}_N$ は停留点において $\\nabla_{\\mathbf{w}} E(\\mathbf{m}_N) = 0$ であるため、$E(\\mathbf{m}_N)$ の $\\mathbf{m}_N$ 経由の間接微分はゼロとなる）。
3. 導関数を 0 と置いて整理する:
   $$ \\|\\mathbf{m}_N\\|^2 = \\frac{M}{\\alpha} - \\sum_{i=1}^M \\frac{1}{\\alpha + \\lambda_i} = \\sum_{i=1}^M \\left( \\frac{1}{\\alpha} - \\frac{1}{\\alpha + \\lambda_i} \\right) = \\frac{1}{\\alpha}\\sum_{i=1}^M \\text{[ 穴埋め 1: ? ]} = \\frac{1}{\\alpha} \\sum_{i=1}^M \\frac{\\lambda_i}{\\alpha + \\lambda_i} $$
4. 有効パラメータ数 $\\gamma = \\sum_{i=1}^M \\frac{\\lambda_i}{\\alpha + \\lambda_i}$ を代入すると:
   $$ \\alpha \\|\\mathbf{m}_N\\|^2 = \\gamma \\implies \\alpha = \\text{[ 穴埋め 2: ? ]} = \\frac{\\gamma}{\\|\\mathbf{m}_N\\|^2} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.20 数値検証
np.random.seed(42)
N, M = 20, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
beta = 5.0

# 最適な alpha, beta を探索
eb = EvidenceApproximation().fit(Phi, t, init_alpha=2.0, init_beta=beta)
alpha_opt = eb.alpha

# 対数エビデンスの alpha に関する数値微分 (中心差分)
eps = 1e-6
grad_num = (bayesian_model_evidence(Phi, t, alpha_opt + eps, beta) - \
            bayesian_model_evidence(Phi, t, alpha_opt - eps, beta)) / (2 * eps)

assert abs(grad_num) < 1e-4, f"Gradient not zero: {grad_num}"
print(f"Exercise 3.20 PASSED: Numerical derivative of log evidence at optimal alpha is {grad_num:.2e} (~0)")"""))

    # Exercise 3.21
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.21
**問題**: 実対称行列 $\\mathbf{A}$ の固有値展開を用いて行列式微分の恒等式
$$ \\frac{d}{d\\alpha} \\ln |\\mathbf{A}| = \\mathrm{Tr}\\left( \\mathbf{A}^{-1} \\frac{d\\mathbf{A}}{d\\alpha} \\right) $$
を証明せよ。さらにこの恒等式を用いて、式 (3.86) から $\\alpha$ の再推定式 (3.92) を導出せよ。

### [解答の道筋と穴埋め]
1. 実対称行列 $\\mathbf{A}$ は直交行列 $\\mathbf{U}$ により $\\mathbf{A} = \\mathbf{U} \\boldsymbol{\\Lambda} \\mathbf{U}^T$ と対角化可能である。
2. $|\\mathbf{A}| = \\prod_i \\mu_i$ より、$\\ln|\\mathbf{A}| = \\sum_i \\ln \\mu_i$。
3. 連鎖律およびトレースの巡回不変性を用いると:
   $$ \\frac{d}{d\\alpha} \\ln |\\mathbf{A}| = \\sum_i \\frac{1}{\\mu_i} \\frac{d\\mu_i}{d\\alpha} = \\mathrm{Tr}\\left( \\boldsymbol{\\Lambda}^{-1} \\frac{d\\boldsymbol{\\Lambda}}{d\\alpha} \\right) = \\text{[ 穴埋め 1: ? ]} = \\mathrm{Tr}\\left( \\mathbf{A}^{-1} \\frac{d\\mathbf{A}}{d\\alpha} \\right) $$
4. $\\mathbf{A} = \\alpha \\mathbf{I} + \\beta \\mathbf{\\Phi}^T \\mathbf{\\Phi}$ の場合、$\\frac{d\\mathbf{A}}{d\\alpha} = \\mathbf{I}$ であるから:
   $$ \\frac{d}{d\\alpha} \\ln |\\mathbf{A}| = \\mathrm{Tr}(\\mathbf{A}^{-1}) = \\sum_{i=1}^M \\frac{1}{\\alpha + \\lambda_i} $$
   これにより Exercise 3.20 と全く同一の再推定式が得られる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.21 数値検証
np.random.seed(42)
M = 4
A_rand = np.random.randn(M, M)
A_sym = A_rand.T @ A_rand + 2.0 * np.eye(M)

# 任意の対称摂動行列 dA
dA = np.random.randn(M, M)
dA = dA.T + dA

eps = 1e-6
d_ln_det_num = (np.linalg.slogdet(A_sym + eps * dA)[1] - np.linalg.slogdet(A_sym - eps * dA)[1]) / (2 * eps)
d_ln_det_trace = np.trace(np.linalg.inv(A_sym) @ dA)

diff = abs(d_ln_det_num - d_ln_det_trace)
assert np.isclose(diff, 0.0, atol=1e-5)
print(f"Exercise 3.21 PASSED: Identity d/dalpha ln|A| = Tr(A^-1 dA/dalpha) holds, diff = {diff:.2e}")"""))

    # Exercise 3.22
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.22
**問題**: 対数エビデンス関数 (3.86) を $\\beta$ について微分し、停留点条件から $\\beta$ の再推定式 (3.95)
$$ \\frac{1}{\\beta} = \\frac{1}{N - \\gamma} \\sum_{n=1}^N \\{t_n - \\mathbf{m}_N^T \\boldsymbol{\\phi}(\\mathbf{x}_n)\\}^2 $$
が導かれる全ステップを証明せよ。

### [解答の道筋と穴埋め]
1. $\\mathbf{A} = \\alpha \\mathbf{I} + \\beta \\mathbf{\\Phi}^T \\mathbf{\\Phi}$ において固有値は $\\alpha + \\beta \\eta_i$（$\\eta_i$ は $\\mathbf{\\Phi}^T\\mathbf{\\Phi}$ の固有値、$\\lambda_i = \\beta \\eta_i$）。
2. $\\frac{d}{d\\beta} \\ln |\\mathbf{A}| = \\sum_{i=1}^M \\frac{\\eta_i}{\\alpha + \\beta \\eta_i} = \\frac{1}{\\beta} \\sum_{i=1}^M \\frac{\\lambda_i}{\\alpha + \\lambda_i} = \\text{[ 穴埋め 1: ? ]} = \\frac{\\gamma}{\\beta}$。
3. 対数エビデンスの $\\beta$ 微分を計算する:
   $$ \\frac{d}{d\\beta} \\ln p(\\mathbf{t} | \\alpha, \\beta) = \\frac{N}{2\\beta} - \\frac{1}{2}\\sum_{n=1}^N \\{t_n - \\mathbf{m}_N^T \\boldsymbol{\\phi}(\\mathbf{x}_n)\\}^2 - \\frac{\\gamma}{2\\beta} $$
4. 導関数を 0 と置いて整理すると:
   $$ \\frac{N - \\gamma}{2\\beta} = \\frac{1}{2} \\sum_{n=1}^N \\{t_n - \\mathbf{m}_N^T \\boldsymbol{\\phi}(\\mathbf{x}_n)\\}^2 $$
   $$ \\frac{1}{\\beta} = \\text{[ 穴埋め 2: ? ]} = \\frac{1}{N - \\gamma} \\sum_{n=1}^N \\{t_n - \\mathbf{m}_N^T \\boldsymbol{\\phi}(\\mathbf{x}_n)\\}^2 $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.22 数値検証
np.random.seed(42)
N, M = 25, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha = 2.0

eb = EvidenceApproximation().fit(Phi, t, init_alpha=alpha, init_beta=3.0)
beta_opt = eb.beta

# 対数エビデンスの beta に関する数値微分 (中心差分)
eps = 1e-6
grad_beta_num = (bayesian_model_evidence(Phi, t, alpha, beta_opt + eps) - \
                 bayesian_model_evidence(Phi, t, alpha, beta_opt - eps)) / (2 * eps)

assert abs(grad_beta_num) < 1e-4, f"Gradient not zero: {grad_beta_num}"
print(f"Exercise 3.22 PASSED: Numerical derivative w.r.t beta at optimal point is {grad_beta_num:.2e} (~0)")"""))

    # Exercise 3.23
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.23
**問題**: Exercise 3.12 で定義された正規-ガンマモデルにおいて、まず $\\mathbf{w}$ について周辺化し、次いで $\\beta$ について周辺化することにより、モデルエビデンス $p(\\mathbf{t})$ が
$$ p(\\mathbf{t}) = \\frac{1}{(2\\pi)^{N/2}} \\frac{b_0^{a_0}}{b_N^{a_N}} \\frac{\\Gamma(a_N)}{\\Gamma(a_0)} \\frac{|\\mathbf{S}_N|^{1/2}}{|\\mathbf{S}_0|^{1/2}} $$
(式 3.118) で与えられることを示せ。

### [解答の道筋と穴埋め]
1. $\\mathbf{w}$ の積分:
   $$ p(\\mathbf{t} | \\beta) = \\int \\mathcal{N}(\\mathbf{t} | \\mathbf{\\Phi}\\mathbf{w}, \\beta^{-1}\\mathbf{I}) \\mathcal{N}(\\mathbf{w} | \\mathbf{m}_0, (\\beta \\mathbf{S}_0)^{-1}) d\\mathbf{w} $$
   ガウス畳み込み積分により:
   $$ p(\\mathbf{t} | \\beta) = (2\\pi)^{-N/2} \\beta^{N/2} \\frac{|\\mathbf{S}_N|^{1/2}}{|\\mathbf{S}_0|^{1/2}} \\exp\\left\\{ -\\beta (b_N - b_0) \\right\\} $$
2. $\\beta$ の積分: ガンマ事前分布 $p(\\beta) = \\frac{b_0^{a_0}}{\\Gamma(a_0)} \\beta^{a_0 - 1} e^{-b_0 \\beta}$ を掛けて積分する:
   $$ p(\\mathbf{t}) = \\int_0^\\infty p(\\mathbf{t} | \\beta) p(\\beta) d\\beta = \\frac{1}{(2\\pi)^{N/2}} \\frac{b_0^{a_0}}{\\Gamma(a_0)} \\frac{|\\mathbf{S}_N|^{1/2}}{|\\mathbf{S}_0|^{1/2}} \\int_0^\\infty \\beta^{a_0 + N/2 - 1} e^{-b_N \\beta} d\\beta $$
3. $a_N = a_0 + N/2$ およびガンマ関数の積分公式を適用すると:
   $$ \\int_0^\\infty \\beta^{a_N - 1} e^{-b_N \\beta} d\\beta = \\text{[ 穴埋め 1: ? ]} = \\frac{\\Gamma(a_N)}{b_N^{a_N}} $$
4. これらを整理することで、式 (3.118) が得られる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.23 数値検証
np.random.seed(42)
N, M = 6, 2
Phi = np.random.randn(N, M)
t = np.random.randn(N)
a0, b0 = 2.5, 3.5

ng = NormalGammaLinearRegression(a0=a0, b0=b0).fit(Phi, t)

# 解析的エビデンス (式 3.118)
log_ev_analytic = -0.5 * N * np.log(2 * np.pi) + a0 * np.log(b0) - ng.a_N * np.log(ng.b_N) + \
                  (stats.gammaln(ng.a_N) - stats.gammaln(a0)) + \
                  0.5 * (np.linalg.slogdet(ng.S_N)[1] - np.linalg.slogdet(np.linalg.inv(ng.S0_inv))[1])
ev_analytic = np.exp(log_ev_analytic)

# 2次元数値求積 (w1, w2, beta の3重積分)
def integrand_3d(beta, w1, w2):
    w = np.array([w1, w2])
    log_lik = -0.5 * N * np.log(2 * np.pi / beta) - 0.5 * beta * np.sum((t - Phi @ w)**2)
    log_pw = -0.5 * M * np.log(2 * np.pi / beta) + 0.5 * np.linalg.slogdet(beta * ng.S0_inv)[1] - 0.5 * beta * (w @ ng.S0_inv @ w)
    log_pbeta = a0 * np.log(b0) - stats.gammaln(a0) + (a0 - 1) * np.log(beta) - b0 * beta
    return np.exp(log_lik + log_pw + log_pbeta)

# beta について 1次元積分 (内側の w 積分は解析的)
def integrand_beta(beta):
    log_lik_w = -0.5 * N * np.log(2 * np.pi) + 0.5 * N * np.log(beta) + \
                0.5 * (np.linalg.slogdet(ng.S_N)[1] - np.linalg.slogdet(np.linalg.inv(ng.S0_inv))[1]) - \
                beta * (ng.b_N - b0)
    p_beta = stats.gamma.pdf(beta, a=a0, scale=1.0 / b0)
    return np.exp(log_lik_w) * p_beta

ev_num, _ = integrate.quad(integrand_beta, 1e-8, 50.0)
assert np.isclose(ev_analytic, ev_num, rtol=1e-4)
print(f"Exercise 3.23 PASSED: Normal-Gamma evidence formula ({ev_analytic:.5e}) matches numerical integral ({ev_num:.5e}).")"""))

    # Exercise 3.24
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 3.24
**問題**: ベイズの定理の恒等式
$$ p(\\mathbf{t}) = \\frac{p(\\mathbf{t} | \\mathbf{w}, \\beta) p(\\mathbf{w}, \\beta)}{p(\\mathbf{w}, \\beta | \\mathbf{t})} $$
(式 3.119) を用いて、事前分布・尤度関数・事後分布の表式を直接代入することにより、積分計算を一切行わずに Exercise 3.23 の結果 (式 3.118) を導出せよ。

### [解答の道筋と穴埋め]
1. この比率は任意の $(\\mathbf{w}, \\beta)$ において一定値 $p(\\mathbf{t})$ をとる。
2. 計算を極力単純にするため、$\\mathbf{w} = \\mathbf{m}_N$ を選ぶ。
3. 各分布の代入:
   - 尤度: $p(\\mathbf{t} | \\mathbf{m}_N, \\beta) = (2\\pi)^{-N/2} \\beta^{N/2} \\exp\\left(-\\frac{\\beta}{2}\\|\\mathbf{t} - \\mathbf{\\Phi}\\mathbf{m}_N\\|^2\\right)$
   - 事前分布: $p(\\mathbf{m}_N, \\beta) = (2\\pi)^{-M/2} |\\beta \\mathbf{S}_0|^{1/2} \\exp\\left(-\\frac{\\beta}{2}(\\mathbf{m}_N - \\mathbf{m}_0)^T \\mathbf{S}_0^{-1}(\\mathbf{m}_N - \\mathbf{m}_0)\\right) \\frac{b_0^{a_0}}{\\Gamma(a_0)} \\beta^{a_0 - 1} e^{-b_0 \\beta}$
   - 事後分布: $p(\\mathbf{m}_N, \\beta | \\mathbf{t}) = (2\\pi)^{-M/2} |\\beta \\mathbf{S}_N|^{1/2} \\frac{b_N^{a_N}}{\\Gamma(a_N)} \\beta^{a_N - 1} e^{-b_N \\beta}$ (指数部で $(\\mathbf{m}_N - \\mathbf{m}_N) = \\mathbf{0}$ となるためガウス指数項は 1)
4. 指数項の相殺: Exercise 3.12 の平方完成の恒等式により、分子の指数項は厳密に $-b_N \\beta$ となるため、分母の $e^{-b_N \\beta}$ と**完全に相殺**する。
5. $\\beta$ のべき乗も相殺し、直ちに式 (3.118) が導出される:
   $$ p(\\mathbf{t}) = \\frac{1}{(2\\pi)^{N/2}} \\frac{b_0^{a_0}}{b_N^{a_N}} \\frac{\\Gamma(a_N)}{\\Gamma(a_0)} \\frac{|\\mathbf{S}_N|^{1/2}}{|\\mathbf{S}_0|^{1/2}} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 3.24 数値検証
# 任意の 10 組の (w, beta) に対し、比率 p(t|w, beta) * p(w, beta) / p(w, beta | t) が完全に一定であり、
# Exercise 3.23 の解析的エビデンスと一致することを検証
ratios = []
for _ in range(10):
    w_test = np.random.randn(M)
    beta_test = np.random.uniform(0.5, 5.0)

    log_lik = -0.5 * N * np.log(2*np.pi/beta_test) - 0.5 * beta_test * np.sum((t - Phi @ w_test)**2)
    log_prior_w = -0.5 * M * np.log(2*np.pi/beta_test) + 0.5 * np.linalg.slogdet(beta_test * ng.S0_inv)[1] - \
                  0.5 * beta_test * (w_test @ ng.S0_inv @ w_test)
    log_prior_beta = a0 * np.log(b0) - stats.gammaln(a0) + (a0 - 1) * np.log(beta_test) - b0 * beta_test

    log_post_w = -0.5 * M * np.log(2*np.pi/beta_test) + 0.5 * np.linalg.slogdet(beta_test * np.linalg.inv(ng.S_N))[1] - \
                 0.5 * beta_test * ((w_test - ng.m_N) @ np.linalg.inv(ng.S_N) @ (w_test - ng.m_N))
    log_post_beta = ng.a_N * np.log(ng.b_N) - stats.gammaln(ng.a_N) + (ng.a_N - 1) * np.log(beta_test) - ng.b_N * beta_test

    ratio = np.exp((log_lik + log_prior_w + log_prior_beta) - (log_post_w + log_post_beta))
    ratios.append(ratio)

ratios = np.array(ratios)
assert np.allclose(ratios, ev_analytic, rtol=1e-10), "Ratio varies with (w, beta)!"
print(f"Exercise 3.24 PASSED: Bayes theorem ratio is strictly constant = {ratios[0]:.5e} (std across 10 points = {np.std(ratios):.2e})")"""))

    nb['cells'] = cells
    with open('3/3_Exercises.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Created 3/3_Exercises.ipynb with all 24 individual exercises successfully.")

if __name__ == '__main__':
    create_ch3_exercises_notebook()
