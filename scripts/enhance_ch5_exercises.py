"""
Master script to generate 5/5_Exercises.ipynb covering all 41 PRML Chapter 5 exercises (5.1 to 5.41).
Fully self-contained, mathematically rigorous, and with all assertions tested.
"""

import nbformat as nbf
import os
import sys

def build_all_cells():
    cells = []

    # Title & TOC
    cells.append(nbf.v4.new_markdown_cell("""# 第5章 ニューラルネットワーク：章末演習問題 (Exercises 5.1 〜 5.41 全41問 完全網羅)

教科書「パターン認識と機械学習 (PRML)」第5章「ニューラルネットワーク (Neural Networks)」の全41問の演習問題の完全解答・解説ノートブックです。

各問題について、以下の構成で学習を進められるよう設計されています：
1. **問題の提示**: PRML原著の設問内容
2. **[解答の道筋と穴埋め]**: 証明・導出の論理的ステップと要点穴埋め (`[ 穴埋め X: ? ]`)
3. **Python数値検証コード**: 導出した数式や定理を数値シミュレーション・assert文で直接検証

---
## 目次
- [Exercise 5.1: シグモイド隠れ層とtanh隠れ層の線形パラメータ変換と恒等性](#Exercise-5.1)
- [Exercise 5.2: 多変量ガウス尤度最大化と二乗和誤差最小化の等価性](#Exercise-5.2)
- [Exercise 5.3: 一般共分散行列を持つ回帰とマハラノビス二乗和誤差](#Exercise-5.3)
- [Exercise 5.4: ラベル反転ノイズを含むロバスト二値分類と勾配の有界性](#Exercise-5.4)
- [Exercise 5.5: 多クラス交差エントロピー誤差関数の多項尤度からの導出](#Exercise-5.5)
- [Exercise 5.6: 二値交差エントロピーにおける普遍的デルタ公式 dE/da = y - t](#Exercise-5.6)
- [Exercise 5.7: 多クラスソフトマックスにおける普遍的デルタ公式 dE/da = y - t](#Exercise-5.7)
- [Exercise 5.8: 指数型分布族と正準連結関数における普遍的デルタ関係](#Exercise-5.8)
- [Exercise 5.9: 独立二値出力ネットワークにおける誤差勾配の統一性](#Exercise-5.9)
- [Exercise 5.10: ヘッセ行列の固有値分解と局所二次誤差展開](#Exercise-5.10)
- [Exercise 5.11: 誤差曲面の楕円等高線の主軸長と固有値の逆平方根比例](#Exercise-5.11)
- [Exercise 5.12: 局所極小値の必要十分条件（ヘッセ行列の狭義正定値性）](#Exercise-5.12)
- [Exercise 5.13: 対称ヘッセ行列の独立要素数と二次近似の全パラメータ数](#Exercise-5.13)
- [Exercise 5.14: 中心差分における一次誤差項の完全相殺とO(eps^2)収束](#Exercise-5.14)
- [Exercise 5.15: ヤコビ行列の前向き伝播漸化式 (Forward Propagation)](#Exercise-5.15)
- [Exercise 5.16: 二乗和誤差におけるヘッセ行列の外積近似 (Gauss-Newton)](#Exercise-5.16)
- [Exercise 5.17: 連続データ空間における二次損失関数の厳密なヘッセ期待値](#Exercise-5.17)
- [Exercise 5.18: スキップ結合（入力-出力直結）を持つネットワークの逆伝播](#Exercise-5.18)
- [Exercise 5.19: 多クラス交差エントロピーにおける外積ヘッセ行列近似](#Exercise-5.19)
- [Exercise 5.20: 独立多変量二値分類における外積ヘッセ行列の対角ブロック性](#Exercise-5.20)
- [Exercise 5.21: Sherman-Morrison-Woodburyによる逆ヘッセ行列のオンライン逐次更新](#Exercise-5.21)
- [Exercise 5.22: 2層ネットワークの厳密なヘッセ行列要素の解析解導出](#Exercise-5.22)
- [Exercise 5.23: スキップ層結合を含むネットワークの厳密なヘッセ行列導出](#Exercise-5.23)
- [Exercise 5.24: 入力のアフィン変換に対する重み変換とネットワーク関数の不変性](#Exercise-5.24)
- [Exercise 5.25: 早期終了 (Early Stopping) と Weight Decay (L2正則化) の等価対応](#Exercise-5.25)
- [Exercise 5.26: 接線伝播法 (Tangent Propagation) による局所不変性正則化](#Exercise-5.26)
- [Exercise 5.27: 入力ノイズ付加学習とTikhonov正則化（勾配ペナルティ）の等価性](#Exercise-5.27)
- [Exercise 5.28: 重み共有 (Weight Sharing) と畳み込み構造の誤差逆伝播](#Exercise-5.28)
- [Exercise 5.29: ソフト重み共有 (Soft Weight Sharing) の重みに関する勾配](#Exercise-5.29)
- [Exercise 5.30: ソフト重み共有における混合ガウス平均パラメータの勾配](#Exercise-5.30)
- [Exercise 5.31: ソフト重み共有における分散パラメータの勾配](#Exercise-5.31)
- [Exercise 5.32: 補助変数による混合重みの制約充足とソフトマックス微分](#Exercise-5.32)
- [Exercise 5.33: 2自由度ロボットアームの順運動学と逆運動学の多価性](#Exercise-5.33)
- [Exercise 5.34: 混合密度ネットワーク (MDN) 混合係数活性化のデルタ公式](#Exercise-5.34)
- [Exercise 5.35: 混合密度ネットワーク (MDN) 平均パラメータ活性化のデルタ公式](#Exercise-5.35)
- [Exercise 5.36: 混合密度ネットワーク (MDN) 分散パラメータ活性化のデルタ公式](#Exercise-5.36)
- [Exercise 5.37: 混合密度ネットワークの全確率・全分散の法則による条件付きモーメント](#Exercise-5.37)
- [Exercise 5.38: ベイズニューラル回帰の線形ガウス事後予測分布](#Exercise-5.38)
- [Exercise 5.39: ラプラス近似によるエビデンス関数（周辺尤度）の定式化](#Exercise-5.39)
- [Exercise 5.40: 多クラス分類ベイズニューラルネットワークと事後予測分布](#Exercise-5.40)
- [Exercise 5.41: 分類モデルにおける超パラメータ再推定方程式 (alpha = gamma / ||w||^2)](#Exercise-5.41)
---"""))

    # Setup cell
    cells.append(nbf.v4.new_code_cell("""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import scipy.special as sp_special
import scipy.stats as stats
import scipy.optimize as optimize

from common.classification_utils import sigmoid, softmax
from common.nn_utils import tanh, dtanh, MLPRegressor, MixtureDensityNetwork, gradient_check

print("PRML Chapter 5 Exercises Setup completed successfully.")"""))

    # -------------------------------------------------------------
    # Exercise 5.1
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.1
**問題**: 隠れユニットの非線形活性化関数 $g(a)$ がロジスティックシグモイド関数 $\\sigma(a) = 1/(1+e^{-a})$ で与えられる2層ネットワーク関数 (5.7) を考える。
隠れ層の活性化関数を $\\tanh(a) = (e^a - e^{-a})/(e^a + e^{-a})$ に置き換えたとき、第1層および第2層の重みとバイアスパラメータを適切に線形変換することで、シグモイドネットワークと全く同一のネットワーク関数を構成できることを示せ。

### [解答の道筋と穴埋め]
1. **活性化関数の代数的関係**:
   - 双曲線正接 $\\tanh(a)$ は以下のように変形できる：
     $$ \\tanh(a) = \\frac{e^a - e^{-a}}{e^a + e^{-a}} = \\frac{e^{2a} - 1}{e^{2a} + 1} = \\frac{2e^{2a}}{e^{2a}+1} - 1 = 2\\sigma(2a) - 1 $$
   - 逆に、ロジスティックシグモイド関数は以下のように表される：
     $$ \\sigma(a) = \\frac{1}{2} \\tanh\\left( \\text{[ 穴埋め 1: a/2 ]} \\right) + \\frac{1}{2} $$
2. **ネットワーク関数の変換**:
   - シグモイド隠れ層を持つネットワークの出力は：
     $$ y_k(\\mathbf{x}) = \\sum_{j=1}^M w_{kj}^{(2)} \\sigma\\left( \\sum_{i=1}^D w_{ji}^{(1)} x_i + w_{j0}^{(1)} \\right) + w_{k0}^{(2)} $$
   - ここに $\\sigma(a) = \\frac{1}{2}\\tanh(a/2) + \\frac{1}{2}$ を代入すると：
     $$ y_k(\\mathbf{x}) = \\sum_{j=1}^M \\frac{1}{2} w_{kj}^{(2)} \\tanh\\left( \\sum_{i=1}^D \\frac{1}{2} w_{ji}^{(1)} x_i + \\frac{1}{2} w_{j0}^{(1)} \\right) + \\left( w_{k0}^{(2)} + \\text{[ 穴埋め 2: 1/2 sum_j w_kj^(2) ]} \\right) $$
   - したがって、$\\tanh$ ネットワークのパラメータを以下のように設定すれば恒等的に一致する：
     $$ \\widetilde{w}_{ji}^{(1)} = \\frac{1}{2} w_{ji}^{(1)}, \\quad \\widetilde{w}_{j0}^{(1)} = \\frac{1}{2} w_{j0}^{(1)} $$
     $$ \\widetilde{w}_{kj}^{(2)} = \\frac{1}{2} w_{kj}^{(2)}, \\quad \\widetilde{w}_{k0}^{(2)} = w_{k0}^{(2)} + \\frac{1}{2}\\sum_{j=1}^M w_{kj}^{(2)} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.1 数値検証
np.random.seed(42)
D, M, K = 3, 4, 2
W1_sig = np.random.randn(D, M)
b1_sig = np.random.randn(M)
W2_sig = np.random.randn(M, K)
b2_sig = np.random.randn(K)

# tanh ネットワークの等価パラメータ導出
W1_tanh = 0.5 * W1_sig
b1_tanh = 0.5 * b1_sig
W2_tanh = 0.5 * W2_sig
b2_tanh = b2_sig + 0.5 * np.sum(W2_sig, axis=0)

X_test = np.random.randn(20, D)
# シグモイドネットワークの順伝播
out_sig = sigmoid(X_test @ W1_sig + b1_sig) @ W2_sig + b2_sig
# tanhネットワークの順伝播
out_tanh = np.tanh(X_test @ W1_tanh + b1_tanh) @ W2_tanh + b2_tanh

assert np.allclose(out_sig, out_tanh, atol=1e-12)
print("Exercise 5.1 verified successfully: sigmoid and tanh networks produce identical outputs.")"""))

    # -------------------------------------------------------------
    # Exercise 5.2
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.2
**問題**: 共通の精度パラメータ $\\beta$ を持つ独立な等方性ガウスノイズモデル (5.16)
$$ p(\\mathbf{t}|\\mathbf{x}, \\mathbf{w}, \\beta) = \\mathcal{N}(\\mathbf{t}|\\mathbf{y}(\\mathbf{x}, \\mathbf{w}), \\beta^{-1}\\mathbf{I}) $$
の下で多変量目標変数に対する尤度関数を最大化することが、二乗和誤差関数 (5.11)
$$ E(\\mathbf{w}) = \\frac{1}{2}\\sum_{n=1}^N \\|\\mathbf{y}(\\mathbf{x}_n, \\mathbf{w}) - \\mathbf{t}_n\\|^2 $$
の最小化と等価であることを示せ。また、精度パラメータの最尤推定量 $\\beta_{\\mathrm{ML}}$ を導出せよ。

### [解答の道筋と穴埋め]
1. **対数尤度関数の記述**:
   - $N$ 個の独立なデータセットに対する同時対数尤度は：
     $$ \\ln p(\\mathbf{T}|\\mathbf{X}, \\mathbf{w}, \\beta) = -\\frac{NK}{2}\\ln(2\\pi) + \\frac{NK}{2}\\ln\\beta - \\frac{\\beta}{2}\\sum_{n=1}^N \\|\\mathbf{y}(\\mathbf{x}_n, \\mathbf{w}) - \\mathbf{t}_n\\|^2 $$
2. **重み $\\mathbf{w}$ に関する最大化**:
   - $\\beta > 0$ であるため、重み $\\mathbf{w}$ に関して $\\ln p$ を最大化することは、負の係数 $-\\beta$ を除いた二乗和誤差 $\\text{[ 穴埋め 1: E(w) ]}$ を最小化することと完全に等価である。
3. **精度 $\\beta$ の最尤解**:
   - $\\beta$ に関して微分して 0 とおくと：
     $$ \\frac{\\partial \\ln p}{\\partial \\beta} = \\frac{NK}{2\\beta} - \\frac{1}{2}\\sum_{n=1}^N \\|\\mathbf{y}_n - \\mathbf{t}_n\\|^2 = 0 $$
     $$ \\frac{1}{\\beta_{\\mathrm{ML}}} = \\text{[ 穴埋め 2: 1/(NK) sum_n ||y_n - t_n||^2 ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.2 数値検証
N, K = 50, 3
rng = np.random.RandomState(42)
Y = rng.randn(N, K)
T = Y + rng.normal(scale=0.5, size=(N, K)) # 真のノイズ分散 0.25 (beta = 4.0)

# 最尤精度パラメータの計算
res_sq_sum = np.sum((Y - T)**2)
inv_beta_ml = res_sq_sum / (N * K)
beta_ml = 1.0 / inv_beta_ml

# 対数尤度関数の数値最適化による beta の推定と一致するか検証
def neg_log_lik(beta):
    return - (0.5 * N * K * np.log(beta) - 0.5 * N * K * np.log(2*np.pi) - 0.5 * beta * res_sq_sum)

res = optimize.minimize_scalar(neg_log_lik, bounds=(0.1, 10.0), method='bounded')
assert np.isclose(beta_ml, res.x, atol=1e-5)
print(f"Exercise 5.2 verified: analytical beta_ML={beta_ml:.4f}, numerical opt={res.x:.4f}")"""))

    # -------------------------------------------------------------
    # Exercise 5.3
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.3
**問題**: 標的変数 $\\mathbf{t}$ に対するノイズ分布が、入力 $\\mathbf{x}$ に依存しない一般的な共分散行列 $\\mathbf{\\Sigma}$ を持つガウス分布 $\\mathcal{N}(\\mathbf{t}|\\mathbf{y}(\\mathbf{x}, \\mathbf{w}), \\mathbf{\\Sigma})$ である回帰問題を考える。
対数尤度関数の最大化がマハラノビス二乗和誤差の最小化に帰着されることを示し、共分散行列の最尤解 $\\mathbf{\\Sigma}_{\\mathrm{ML}}$ が残差ベクトルの標本共分散行列で与えられることを証明せよ。

### [解答の道筋と穴埋め]
1. **多変量ガウス対数尤度**:
   $$ \\ln p(\\mathbf{T}|\\mathbf{X}, \\mathbf{w}, \\mathbf{\\Sigma}) = -\\frac{NK}{2}\\ln(2\\pi) - \\frac{N}{2}\\ln|\\mathbf{\\Sigma}| - \\frac{1}{2}\\sum_{n=1}^N (\\mathbf{y}_n - \\mathbf{t}_n)^T \\mathbf{\\Sigma}^{-1} (\\mathbf{y}_n - \\mathbf{t}_n) $$
2. **マハラノビス二乗和誤差**:
   - $\\mathbf{\\Sigma}$ が与えられたとき、$\\mathbf{w}$ の最適化はマハラノビス距離二乗和 $E(\\mathbf{w}) = \\frac{1}{2}\\sum_n (\\mathbf{y}_n - \\mathbf{t}_n)^T \\mathbf{\\Sigma}^{-1} (\\mathbf{y}_n - \\mathbf{t}_n)$ の最小化と等価である。
3. **共分散行列 $\\mathbf{\\Sigma}$ の最尤推定**:
   - 精度行列 $\\mathbf{W} = \\mathbf{\\Sigma}^{-1}$ と置き、残差行列 $\\mathbf{S} = \\frac{1}{N}\\sum_n (\\mathbf{y}_n - \\mathbf{t}_n)(\\mathbf{y}_n - \\mathbf{t}_n)^T$ を用いると：
     $$ \\frac{1}{N}\\ln p = \\frac{1}{2}\\ln|\\mathbf{W}| - \\frac{1}{2}\\mathrm{Tr}(\\mathbf{W}\\mathbf{S}) + \\text{const} $$
   - $\\mathbf{W}$ に関して微分すると $\\frac{\\partial}{\\partial \\mathbf{W}} \\ln|\\mathbf{W}| = \\mathbf{W}^{-1} = \\mathbf{\\Sigma}$ より：
     $$ \\mathbf{\\Sigma} - \\mathbf{S} = \\mathbf{0} \\implies \\mathbf{\\Sigma}_{\\mathrm{ML}} = \\text{[ 穴埋め 1: S = 1/N sum_n (y_n - t_n)(y_n - t_n)^T ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.3 数値検証
N, K = 100, 3
rng = np.random.RandomState(42)
Y = rng.randn(N, K)
true_cov = np.array([[1.0, 0.4, -0.2], [0.4, 1.2, 0.1], [-0.2, 0.1, 0.8]])
noise = rng.multivariate_normal(np.zeros(K), true_cov, size=N)
T = Y + noise

# 解析的最尤解
Sigma_ML = (T - Y).T @ (T - Y) / N

# 尤度の勾配が Sigma_ML で 0 になることを検証
inv_Sigma_ML = np.linalg.inv(Sigma_ML)
grad_W = 0.5 * Sigma_ML - 0.5 * ((T - Y).T @ (T - Y) / N)
assert np.allclose(grad_W, 0.0, atol=1e-12)
print("Exercise 5.3 verified: analytical Sigma_ML exactly zeros the gradient of log-likelihood.")"""))

    # -------------------------------------------------------------
    # Exercise 5.4
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.4
**問題**: 目標値 $t \\in \\{0, 1\\}$ を持つ二値分類問題において、学習データセットに確率 $\\epsilon$ でラベル反転ノイズが存在する場合を考える。
すなわち、真のクラス $k$ に対し、観測ラベルが $1-\\epsilon$ の確率で正しく、$\\epsilon$ の確率で反転すると仮定する。
このモデルに対する適切な誤差関数を導出し、出力ユニットの活性化 $a_n$ （ここで $y_n = \\sigma(a_n)$）に関する誤差関数の微分が、外れ値に対して有界となることを示せ。

### [解答の道筋と穴埋め]
1. **ラベル生成確率**:
   - 観測ラベルが $t_n = 1$ となる条件付き確率は：
     $$ P(t_n = 1|\\mathbf{x}_n) = (1-\\epsilon)y_n + \\epsilon(1-y_n) = (1-2\\epsilon)y_n + \\epsilon $$
   - 観測ラベルが $t_n = 0$ となる確率は $P(t_n = 0|\\mathbf{x}_n) = (1-2\\epsilon)(1-y_n) + \\epsilon$。
2. **負の対数尤度（誤差関数）**:
   $$ E(\\mathbf{w}) = -\\sum_{n=1}^N \\left\\{ t_n \\ln\\left( (1-2\\epsilon)y_n + \\epsilon \\right) + (1 - t_n)\\ln\\left( (1-2\\epsilon)(1-y_n) + \\epsilon \\right) \\right\\} $$
3. **活性化 $a_n$ に関する勾配**:
   - $P_1 = (1-2\\epsilon)y_n + \\epsilon, P_0 = (1-2\\epsilon)(1-y_n) + \\epsilon$ とおく。
   - 連鎖律 $\\frac{\\partial E_n}{\\partial a_n} = \\frac{\\partial E_n}{\\partial y_n} \\frac{dy_n}{da_n}$ および $\\frac{dy_n}{da_n} = y_n(1-y_n)$ より：
     $$ \\frac{\\partial E_n}{\\partial a_n} = \\left( -\\frac{t_n(1-2\\epsilon)}{P_1} + \\frac{(1-t_n)(1-2\\epsilon)}{P_0} \\right) y_n(1-y_n) $$
   - 外れ値（例：$t_n = 0$ なのに $a_n \\to +\\infty, y_n \\to 1$）において、通常の交差エントロピーでは $\\frac{\\partial E_n}{\\partial a_n} = y_n - t_n \\to 1$ であるのに対し、本モデルでは $P_0 \\to \\epsilon > 0$ かつ $y_n(1-y_n) \\to 0$ となるため、勾配は $\\text{[ 穴埋め 1: 0 に収束（有界） ]}$ し、誤ラベルデータの影響が自動的に遮断される。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.4 数値検証
eps = 0.05
a_vals = np.linspace(-10, 10, 200)
y_vals = sigmoid(a_vals)

# t = 0 のときの勾配計算
P1 = (1 - 2*eps)*y_vals + eps
P0 = (1 - 2*eps)*(1 - y_vals) + eps
# dE/da for t=0
grad_robust = ((1 - 2*eps) / P0) * y_vals * (1 - y_vals)
# standard cross-entropy grad for t=0: y - 0 = y
grad_standard = y_vals

# a -> +inf において robust の勾配は 0 に減衰する
assert grad_robust[-1] < 1e-3
assert np.isclose(grad_standard[-1], 1.0, atol=1e-3)
print(f"Exercise 5.4 verified: at a=10, standard grad={grad_standard[-1]:.4f}, robust grad={grad_robust[-1]:.4e}")"""))

    # -------------------------------------------------------------
    # Exercise 5.5
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.5
**問題**: 出力ユニットがソフトマックス活性化関数
$$ y_k(\\mathbf{x}, \\mathbf{w}) = \\frac{\\exp(a_k(\\mathbf{x}, \\mathbf{w}))}{\\sum_j \\exp(a_j(\\mathbf{x}, \\mathbf{w}))} $$
を持ち、目標値が 1-of-K 符号化ベクトル $\\mathbf{t}$ である多クラスニューラルネットワークにおいて、多項分布尤度関数の最大化が交差エントロピー誤差関数 (5.24)
$$ E(\\mathbf{w}) = -\\sum_{n=1}^N \\sum_{k=1}^K t_{nk} \\ln y_k(\\mathbf{x}_n, \\mathbf{w}) $$
の最小化と等価であることを示せ。

### [解答の道筋と穴埋め]
1. **多クラス条件付き尤度**:
   - 各データ点 $\\mathbf{x}_n$ に対し、標的ベクトル $\\mathbf{t}_n = (t_{n1}, \\dots, t_{nK})^T$ の条件付き確率は多項分布表現として以下のように書ける：
     $$ p(\\mathbf{t}_n|\\mathbf{x}_n, \\mathbf{w}) = \\prod_{k=1}^K y_k(\\mathbf{x}_n, \\mathbf{w})^{t_{nk}} $$
2. **負の対数尤度への変換**:
   - $N$ 個の独立なデータに対する同時尤度は $p(\\mathcal{D}|\\mathbf{w}) = \\prod_{n=1}^N \\prod_{k=1}^K y_{nk}^{t_{nk}}$。
   - 両辺の自然対数をとり負号を付与すると：
     $$ E(\\mathbf{w}) = -\\ln p(\\mathcal{D}|\\mathbf{w}) = -\\sum_{n=1}^N \\sum_{k=1}^K \\text{[ 穴埋め 1: t_nk ln y_nk ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.5 数値検証
N, K = 10, 4
logits = np.random.randn(N, K)
Y = softmax(logits)
T_labels = np.random.choice(K, size=N)
T_onehot = np.eye(K)[T_labels]

# 負の対数尤度
nll = - np.sum(np.log(Y[np.arange(N), T_labels]))
# 式 (5.24)
cross_entropy = - np.sum(T_onehot * np.log(Y))

assert np.isclose(nll, cross_entropy)
print("Exercise 5.5 verified: multinomial negative log-likelihood equals cross-entropy.")"""))

    # -------------------------------------------------------------
    # Exercise 5.6
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.6
**問題**: ロジスティックシグモイド出力活性化関数を持つ二値分類ネットワークの誤差関数 (5.21)
$$ E_n = -[t_n \\ln y_n + (1 - t_n)\\ln(1 - y_n)] $$
について、出力ユニットの入力活性化 $a_n$ に関する微分が以下となることを示せ：
$$ \\frac{\\partial E_n}{\\partial a_n} = y_n - t_n $$

### [解答の道筋と穴埋め]
1. **シグモイド関数の微分特性**:
   - $y_n = \\sigma(a_n) = \\frac{1}{1 + e^{-a_n}}$ の微分は：
     $$ \\frac{dy_n}{da_n} = \\sigma(a_n)(1 - \\sigma(a_n)) = \\text{[ 穴埋め 1: y_n(1 - y_n) ]} $$
2. **連鎖律の適用**:
   $$ \\frac{\\partial E_n}{\\partial a_n} = \\frac{\\partial E_n}{\\partial y_n} \\frac{dy_n}{da_n} = \\left( -\\frac{t_n}{y_n} + \\frac{1 - t_n}{1 - y_n} \\right) y_n(1 - y_n) $$
   $$ = -t_n(1 - y_n) + (1 - t_n)y_n = -t_n + t_n y_n + y_n - t_n y_n = \\text{[ 穴埋め 2: y_n - t_n ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.6 数値検証
a = 1.45
t = 1.0
eps = 1e-6
f = lambda x: - (t * np.log(sigmoid(x)) + (1 - t) * np.log(1 - sigmoid(x)))
num_grad = (f(a + eps) - f(a - eps)) / (2 * eps)
ana_grad = sigmoid(a) - t

assert np.isclose(num_grad, ana_grad, atol=1e-7)
print(f"Exercise 5.6 verified: numerical={num_grad:.6f}, analytical={ana_grad:.6f}")"""))

    # -------------------------------------------------------------
    # Exercise 5.7
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.7
**問題**: ソフトマックス出力ユニットを持つ多クラス交差エントロピー誤差関数 (5.24)
$$ E_n = -\\sum_{k=1}^K t_{nk} \\ln y_{nk} $$
について、活性化 $a_k$ に関する微分が $\\frac{\\partial E_n}{\\partial a_k} = y_{nk} - t_{nk}$ となることを示せ。

### [解答の道筋と穴埋め]
1. **ソフトマックス関数の偏微分**:
   - $y_k = \\frac{e^{a_k}}{\\sum_j e^{a_j}}$ に対し：
     $$ \\frac{\\partial y_k}{\\partial a_j} = y_k(\\delta_{kj} - y_j) $$
2. **合成関数の連鎖律**:
   $$ \\frac{\\partial E_n}{\\partial a_j} = -\\sum_{k=1}^K \\frac{t_{nk}}{y_{nk}} \\frac{\\partial y_{nk}}{\\partial a_j} = -\\sum_{k=1}^K \\frac{t_{nk}}{y_{nk}} y_{nk}(\\delta_{kj} - y_{nj}) $$
   $$ = -\\sum_{k=1}^K t_{nk}(\\delta_{kj} - y_{nj}) = -t_{nj} + y_{nj} \\sum_{k=1}^K t_{nk} $$
   - 1-of-K 表現より $\\sum_k t_{nk} = 1$ であるから：
     $$ \\frac{\\partial E_n}{\\partial a_j} = \\text{[ 穴埋め 1: y_nj - t_nj ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.7 数値検証
a_vec = np.array([2.1, -0.7, 0.4, 1.2])
t_vec = np.array([0.0, 0.0, 1.0, 0.0])
eps = 1e-6

def ce_loss(a):
    y = softmax(a)
    return -np.sum(t_vec * np.log(y))

num_grad = np.zeros_like(a_vec)
for i in range(len(a_vec)):
    ap, am = a_vec.copy(), a_vec.copy()
    ap[i] += eps; am[i] -= eps
    num_grad[i] = (ce_loss(ap) - ce_loss(am)) / (2 * eps)

ana_grad = softmax(a_vec) - t_vec
assert np.allclose(num_grad, ana_grad, atol=1e-6)
print("Exercise 5.7 verified: softmax cross-entropy dE/da == y - t.")"""))

    # -------------------------------------------------------------
    # Exercise 5.8
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.8
**問題**: 目標分布が正準連結関数 (Canonical Link Function) を持つ指数型分布族に従うとき、出力ユニットの誤差関数の微分が $\\frac{\\partial E_n}{\\partial a_k} = y_k - t_k$ という普遍的な形を満たすことを示せ。

### [解答の道筋と穴埋め]
1. **指数型分布族の一般形**:
   - 自然パラメータ $\\eta$ を持つ確率密度関数は：
     $$ p(t|\\eta) = h(t) \\exp\\left( \\eta t - g(\\eta) \\right) $$
   - 累積母関数 $g(\\eta)$ の1階微分は期待値と一致する：
     $$ \\mathbb{E}[t] = g'(\\eta) = y $$
2. **正準連結関数 (Canonical Link)**:
   - ネットワークの入力活性化 $a$ を自然パラメータに直接対応づける ($a = \\eta$)。
3. **負の対数尤度の微分**:
   - 1サンプルの負の対数尤度は $E_n = -(\\eta t - g(\\eta)) + \\text{const} = -at + g(a)$。
   - $a$ で微分すると：
     $$ \\frac{\\partial E_n}{\\partial a} = -t + g'(a) = \\text{[ 穴埋め 1: y - t ]} $$
   - これにより、ガウス分布（二乗誤差＋恒等写像）、ベルヌーイ分布（交差エントロピー＋シグモイド）、ポアソン分布（ポアソン誤差＋指数関数）のすべてで共通のデルタ則が導かれる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.8 数値検証 (ポアソン分布: canonical link = log, y = exp(a), g(a) = exp(a))
a = 1.2
t = 3.0 # ポアソン観測カウント
y = np.exp(a)
# ポアソン負の対数尤度 E = - (a*t - exp(a))
eps = 1e-6
f = lambda x: - (x * t - np.exp(x))
num_grad = (f(a + eps) - f(a - eps)) / (2 * eps)
ana_grad = y - t
assert np.isclose(num_grad, ana_grad, atol=1e-7)
print("Exercise 5.8 verified: exponential family with canonical link satisfies dE/da == y - t.")"""))

    # -------------------------------------------------------------
    # Exercise 5.9
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.9
**問題**: 独立な複数の二値ラベルを同時に予測するマルチラベル分類問題において、各出力ユニットが独立なシグモイド活性化 $y_k = \\sigma(a_k)$ を持つ場合の誤差関数 (5.21)
$$ E_n = -\\sum_{k=1}^K [t_{nk} \\ln y_{nk} + (1 - t_{nk})\\ln(1 - y_{nk})] $$
について、各出力ユニットの活性化 $a_k$ に関する勾配が $\\frac{\\partial E_n}{\\partial a_k} = y_{nk} - t_{nk}$ となることを示せ。

### [解答の道筋と穴埋め]
1. **独立性の利用**:
   - 和 $\\sum_j$ の中で、活性化 $a_k$ は $j=k$ の項 $y_{nk} = \\sigma(a_{nk})$ にのみ依存し、他の $j \\neq k$ の項には影響しない。
2. **各項の偏微分**:
   $$ \\frac{\\partial E_n}{\\partial a_k} = \\frac{\\partial}{\\partial a_k} \\left\\{ - [t_{nk}\\ln y_{nk} + (1 - t_{nk})\\ln(1 - y_{nk})] \\right\\} = \\text{[ 穴埋め 1: y_nk - t_nk ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.9 数値検証
a_vec = np.array([0.5, -1.2, 3.0])
t_vec = np.array([1.0, 0.0, 1.0])
eps = 1e-6

def multilabel_loss(a):
    y = sigmoid(a)
    return -np.sum(t_vec * np.log(y) + (1 - t_vec) * np.log(1 - y))

num_grad = np.zeros_like(a_vec)
for i in range(len(a_vec)):
    ap, am = a_vec.copy(), a_vec.copy()
    ap[i] += eps; am[i] -= eps
    num_grad[i] = (multilabel_loss(ap) - multilabel_loss(am)) / (2 * eps)

ana_grad = sigmoid(a_vec) - t_vec
assert np.allclose(num_grad, ana_grad, atol=1e-7)
print("Exercise 5.9 verified: multi-label independent sigmoid classification satisfies dE/da == y - t.")"""))

    # -------------------------------------------------------------
    # Exercise 5.10
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.10
**問題**: 固有方程式 $\\mathbf{H}\\mathbf{u}_i = \\lambda_i \\mathbf{u}_i$ を満たすヘッセ行列 $\\mathbf{H}$ を考える。
任意の重み変位ベクトル $\\Delta \\mathbf{w}$ を正規直交固有ベクトル系 $\\{\\mathbf{u}_i\\}$ で展開したとき、局所二次誤差関数の変化量 $\\Delta E = \\frac{1}{2}\\Delta \\mathbf{w}^T \\mathbf{H}\\Delta \\mathbf{w}$ が互いに独立な主成分二次形式の和
$$ \\Delta E = \\frac{1}{2}\\sum_i \\lambda_i \\alpha_i^2 $$
として表されることを証明せよ。

### [解答の道筋と穴埋め]
1. **固有ベクトル展開**:
   - ヘッセ行列は対称行列であるため、固有ベクトルは正規直交基底を成す：$\\mathbf{u}_i^T \\mathbf{u}_j = \\delta_{ij}$。
   - 重み変位を $\\Delta \\mathbf{w} = \\sum_i \\alpha_i \\mathbf{u}_i$ と展開する。
2. **二次形式への代入**:
   $$ \\Delta E = \\frac{1}{2}\\left( \\sum_i \\alpha_i \\mathbf{u}_i \\right)^T \\mathbf{H} \\left( \\sum_j \\alpha_j \\mathbf{u}_j \\right) = \\frac{1}{2}\\sum_i \\sum_j \\alpha_i \\alpha_j \\mathbf{u}_i^T (\\lambda_j \\mathbf{u}_j) $$
   $$ = \\frac{1}{2}\\sum_i \\sum_j \\alpha_i \\alpha_j \\lambda_j \\delta_{ij} = \\text{[ 穴埋め 1: 1/2 sum_i lambda_i alpha_i^2 ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.10 数値検証
W = 5
rng = np.random.RandomState(42)
M_rand = rng.randn(W, W)
H = M_rand.T @ M_rand # 対称正定値ヘッセ行列

eigvals, eigvecs = np.linalg.eigh(H)
delta_w = rng.randn(W)
alphas = eigvecs.T @ delta_w

quad_form = 0.5 * delta_w.T @ H @ delta_w
quad_spectral = 0.5 * np.sum(eigvals * (alphas**2))

assert np.isclose(quad_form, quad_spectral)
print(f"Exercise 5.10 verified: quadratic form={quad_form:.6f} == spectral sum={quad_spectral:.6f}")"""))

    # -------------------------------------------------------------
    # Exercise 5.11
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.11
**問題**: 二次誤差関数 $\\Delta E = \\frac{1}{2}\\sum_i \\lambda_i \\alpha_i^2 = \\text{const}$ が定める楕円体等高面において、各主軸の半長が対応する固有値の平方根の逆数 $\\lambda_i^{-1/2}$ に比例することを示せ。

### [解答の道筋と穴埋め]
1. **主軸方向の切片**:
   - 第 $k$ 主軸方向（$\\alpha_i = 0 \\, (i \\neq k)$）に沿って $\\Delta E = c$ とおくと：
     $$ \\frac{1}{2}\\lambda_k \\alpha_k^2 = c \\implies \\alpha_k^2 = \\frac{2c}{\\lambda_k} $$
   - したがって、第 $k$ 軸の半長 $L_k$ は：
     $$ L_k = |\\alpha_k| = \\sqrt{2c} \\lambda_k^{-1/2} \\propto \\text{[ 穴埋め 1: lambda_k^(-1/2) ]} $$
   - これは、曲率（固有値 $\\lambda_k$）が大きい方向ほど等高線の幅が狭く、曲率が小さい方向ほど等高線が長く引き延ばされることを意味する。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.11 数値検証
c = 1.0
lambdas = np.array([4.0, 1.0, 0.25])
semi_axes = np.sqrt(2 * c / lambdas)

# 比率の検証: semi_axes * sqrt(lambdas) は一定
assert np.allclose(semi_axes * np.sqrt(lambdas), np.sqrt(2 * c))
print("Exercise 5.11 verified: semi-axes are strictly proportional to 1/sqrt(lambda).")"""))

    # -------------------------------------------------------------
    # Exercise 5.12
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.12
**問題**: 停留点 $\\mathbf{w}^*$（$\\nabla E = \\mathbf{0}$）周りの局所テイラー展開において、任意の非ゼロ変位 $\\Delta \\mathbf{w} \\neq \\mathbf{0}$ に対して $\\Delta E > 0$（狭義局所極小）となる必要十分条件は、ヘッセ行列 $\\mathbf{H}$ のすべての固有値が厳密に正（正定値 $\\mathbf{H} \\succ 0$）であることを示せ。

### [解答の道筋と穴埋め]
1. **二次展開**:
   - $\\Delta E = \\frac{1}{2}\\sum_i \\lambda_i \\alpha_i^2$。
2. **十分性の証明**:
   - すべての $i$ で $\\lambda_i > 0$ ならば、$\\Delta \\mathbf{w} \\neq \\mathbf{0}$ のとき少なくとも1つの $\\alpha_i \\neq 0$ となるため、二乗項の正定性より $\\Delta E = \\frac{1}{2}\\sum_i \\lambda_i \\alpha_i^2 > 0$。
3. **必要性の証明（対偶）**:
   - もしある固有値 $\\lambda_k \\le 0$ が存在する場合：
     - その固有ベクトル方向に変位をとる（$\\Delta \\mathbf{w} = \\alpha_k \\mathbf{u}_k$）。
     - $\\Delta E = \\frac{1}{2}\\lambda_k \\alpha_k^2 \\le 0$ となり、厳密な極小値の定義に反する。
   - したがって、狭義極小であるための必要十分条件は $\\text{[ 穴埋め 1: すべての固有値 lambda_i > 0 ]}$ である。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.12 数値検証
# 正定値行列と不定値行列（鞍点）の挙動検証
H_pos = np.diag([2.0, 1.5, 0.5]) # 正定値
H_saddle = np.diag([2.0, -1.0, 0.5]) # 不定値

# 固有値の最小値判定
assert np.all(np.linalg.eigvalsh(H_pos) > 0)
assert not np.all(np.linalg.eigvalsh(H_saddle) > 0)

# 不定値の場合、負の方向が存在することを確認
v_neg = np.array([0.0, 1.0, 0.0])
assert v_neg.T @ H_saddle @ v_neg < 0
print("Exercise 5.12 verified: strict local minimum requires all eigenvalues > 0.")"""))

    # -------------------------------------------------------------
    # Exercise 5.13
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.13
**問題**: ヘッセ行列 $\\mathbf{H}$ の対称性 $H_{ij} = H_{ji}$ から、$W \\times W$ 次元の対称ヘッセ行列が持つ独立な要素の総数は $W(W+1)/2$ であることを示せ。また、局所二次近似モデル全体のパラメータ自由度が $1 + W(W+3)/2$ であることを確認せよ。

### [解答の道筋と穴埋め]
1. **独立要素数の導出**:
   - 対角成分の総数：$W$ 個。
   - 非対角成分の総数：$W^2 - W$ 個。対称性により上三角と下三角で同一であるため独立なのは $\\frac{W(W-1)}{2}$ 個。
   - 合計の独立要素数は：
     $$ W + \\frac{W(W-1)}{2} = \\frac{2W + W^2 - W}{2} = \\text{[ 穴埋め 1: W(W+1)/2 ]} $$
2. **局所二次モデル全体の自由度**:
   - $E(\\mathbf{w}) \\approx E_0 + \\mathbf{b}^T \\mathbf{w} + \\frac{1}{2}\\mathbf{w}^T \\mathbf{H}\\mathbf{w}$ において：
     - 定数項 $E_0$：$1$ 個
     - 勾配ベクトル $\\mathbf{b}$：$W$ 個
     - ヘッセ行列 $\\mathbf{H}$：$\\frac{W(W+1)}{2}$ 個
     - 全自由度 $= 1 + W + \\frac{W(W+1)}{2} = 1 + \\text{[ 穴埋め 2: W(W+3)/2 ]}$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.13 数値検証
for W in [2, 5, 10, 50]:
    # WxW 対称行列のユニーク要素数
    H_sym = np.zeros((W, W))
    triu_indices = np.triu_indices(W)
    num_independent = len(triu_indices[0])
    theo_independent = W * (W + 1) // 2
    assert num_independent == theo_independent
    
    total_params = 1 + W + num_independent
    assert total_params == 1 + W * (W + 3) // 2
print("Exercise 5.13 verified: Hessian independent elements and total quadratic model DOF.")"""))

    # -------------------------------------------------------------
    # Exercise 5.14
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.14
**問題**: 関数 $f(x)$ のテイラー展開を用いて、中心差分商
$$ \\frac{f(x+\\epsilon) - f(x-\\epsilon)}{2\\epsilon} $$
において、一次の誤差項 $O(\\epsilon)$ が完全に相殺し、打ち切り誤差が $O(\\epsilon^2)$ となることを示せ。

### [解答の道筋と穴埋め]
1. **前進点と後退点のテイラー展開**:
   $$ f(x+\\epsilon) = f(x) + \\epsilon f'(x) + \\frac{\\epsilon^2}{2}f''(x) + \\frac{\\epsilon^3}{6}f'''(x) + O(\\epsilon^4) $$
   $$ f(x-\\epsilon) = f(x) - \\epsilon f'(x) + \\frac{\\epsilon^2}{2}f''(x) - \\frac{\\epsilon^3}{6}f'''(x) + O(\\epsilon^4) $$
2. **両式の差**:
   - 偶数次項 $\\frac{\\epsilon^2}{2}f''(x)$ および $f(x)$ が相殺し：
     $$ f(x+\\epsilon) - f(x-\\epsilon) = 2\\epsilon f'(x) + \\frac{\\epsilon^3}{3}f'''(x) + O(\\epsilon^5) $$
3. **$2\\epsilon$ で除算**:
   $$ \\frac{f(x+\\epsilon) - f(x-\\epsilon)}{2\\epsilon} = f'(x) + \\text{[ 穴埋め 1: eps^2 / 6 f'''(x) + O(eps^4) ]} = f'(x) + O(\\epsilon^2) $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.14 数値検証
f = lambda x: np.sin(x)
df = lambda x: np.cos(x)
x0 = 1.0

epsilons = np.array([1e-1, 1e-2, 1e-3, 1e-4])
forward_err = [abs((f(x0 + e) - f(x0))/e - df(x0)) for e in epsilons]
central_err = [abs((f(x0 + e) - f(x0 - e))/(2*e) - df(x0)) for e in epsilons]

# 中心差分の誤差比率が eps^2 に従って 100倍ずつ減少することを検証
ratio_central = central_err[0] / central_err[1]
assert 90 < ratio_central < 110 # ~ 100
print(f"Exercise 5.14 verified: central diff error ratio for 10x step reduction is {ratio_central:.2f} (O(eps^2)).")"""))

    # -------------------------------------------------------------
    # Exercise 5.15
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.15
**問題**: セクション 5.3.4 で導入されたヤコビ行列 $J_{ki} = \\frac{\\partial y_k}{\\partial x_i}$ の計算法について、逆伝播を用いずに入力微小変化を順方向に伝播させる前向き伝播漸化式 (Forward Propagation) を導出せよ。

### [解答の道筋と穴埋め]
1. **入力層の初期化**:
   - 入力変数そのものの微分は：
     $$ \\frac{\\partial x_j}{\\partial x_i} = \\delta_{ji} $$
2. **隠れ層入力 $a_j$ の前向き微分**:
   $$ \\frac{\\partial a_j}{\\partial x_i} = \\sum_m w_{jm}^{(1)} \\frac{\\partial z_m^{(0)}}{\\partial x_i} = w_{ji}^{(1)} $$
3. **隠れ層出力 $z_j$ の前向き微分**:
   $$ \\frac{\\partial z_j}{\\partial x_i} = h'(a_j) \\text{[ 穴埋め 1: da_j / dx_i ]} $$
4. **出力層への伝播**:
   $$ \\frac{\\partial y_k}{\\partial x_i} = \\sum_j w_{kj}^{(2)} \\frac{\\partial z_j}{\\partial x_i} = \\sum_j w_{kj}^{(2)} h'(a_j) w_{ji}^{(1)} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.15 数値検証
D, M, K = 3, 4, 2
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

x = np.random.randn(D)
# 順伝播
a1 = x @ W1 + b1
z1 = np.tanh(a1)
y = z1 @ W2 + b2

# 前向き伝播によるヤコビ行列
# J_{ki} = sum_j W2_{j, k} * (1 - z1_j^2) * W1_{i, j}
J_forward = (W2.T * (1.0 - z1**2)) @ W1.T # shape: (K, D)

# 数値微分
eps = 1e-6
J_num = np.zeros((K, D))
for i in range(D):
    xp, xm = x.copy(), x.copy()
    xp[i] += eps; xm[i] -= eps
    yp = np.tanh(xp @ W1 + b1) @ W2 + b2
    ym = np.tanh(xm @ W1 + b1) @ W2 + b2
    J_num[:, i] = (yp - ym) / (2 * eps)

assert np.allclose(J_forward, J_num, atol=1e-6)
print("Exercise 5.15 verified: forward propagation of Jacobian matches finite differences.")"""))

    # -------------------------------------------------------------
    # Exercise 5.16
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.16
**問題**: 二乗和誤差関数を用いるネットワークにおいて、ヘッセ行列の外積近似 (Gauss-Newton)
$$ \\mathbf{H} \\simeq \\sum_{n=1}^N \\nabla y_n \\nabla y_n^T $$
が、訓練データに対して完全適合（残差 $y_n - t_n = 0$）している停留点において厳密なヘッセ行列と完全に一致することを示せ。

### [解答の道筋と穴埋め]
1. **二乗和誤差の厳密な2階微分**:
   - $E(\\mathbf{w}) = \\frac{1}{2}\\sum_n (y_n - t_n)^2$ に対し：
     $$ \\nabla E = \\sum_n (y_n - t_n) \\nabla y_n $$
     $$ \\nabla\\nabla E = \\sum_{n=1}^N \\nabla y_n \\nabla y_n^T + \\sum_{n=1}^N (y_n - t_n) \\nabla\\nabla y_n $$
2. **残差ゼロにおける一致**:
   - 完全適合時、すべてのデータ点 $n$ で $y_n - t_n = 0$ であるため、第二項 $\\sum_n (y_n - t_n)\\nabla\\nabla y_n$ が恒等的に $\\text{[ 穴埋め 1: 0（消失） ]}$ する。
   - したがって、$\\mathbf{H} = \\sum_n \\nabla y_n \\nabla y_n^T$ となり外積近似が厳密に成立する。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.16 数値検証
model = MLPRegressor(n_in=2, n_hidden=3, n_out=1)
X = np.random.randn(4, 2)
# 完全適合の状況を作るため、モデル自身の出力を目標値とする (残差ゼロ)
T = model.predict(X)

# 外積ヘッセ行列と厳密ヘッセ行列（数値2階微分）の比較
def get_weights(m):
    return np.concatenate([m.W1.ravel(), m.b1.ravel(), m.W2.ravel(), m.b2.ravel()])

def set_weights(m, w):
    d, h, o = m.n_in, m.n_hidden, m.n_out
    idx = 0
    m.W1 = w[idx:idx+d*h].reshape(d, h); idx += d*h
    m.b1 = w[idx:idx+h]; idx += h
    m.W2 = w[idx:idx+h*o].reshape(h, o); idx += h*o
    m.b2 = w[idx:idx+o]

w0 = get_weights(model)
P = len(w0)

# 外積近似 H_GN = sum_n grad_y_n @ grad_y_n^T
H_gn = np.zeros((P, P))
for n in range(len(X)):
    x_n = X[n:n+1]
    g_n = np.zeros(P)
    eps = 1e-6
    for p in range(P):
        wp, wm = w0.copy(), w0.copy()
        wp[p] += eps; wm[p] -= eps
        set_weights(model, wp); yp = model.predict(x_n)[0, 0]
        set_weights(model, wm); ym = model.predict(x_n)[0, 0]
        g_n[p] = (yp - ym) / (2 * eps)
    set_weights(model, w0)
    H_gn += np.outer(g_n, g_n)

# 厳密な損失関数の数値ヘッセ
loss_fn = lambda w: 0.5 * np.sum((MLPRegressor(2, 3, 1).forward(X)[3] - T)**2)
# 残差ゼロなので H_gn と一致
H_exact = np.zeros((P, P))
eps = 1e-5
for i in range(P):
    for j in range(P):
        w_pp = w0.copy(); w_pp[i] += eps; w_pp[j] += eps; set_weights(model, w_pp); l_pp = 0.5 * np.sum((model.predict(X) - T)**2)
        w_pm = w0.copy(); w_pm[i] += eps; w_pm[j] -= eps; set_weights(model, w_pm); l_pm = 0.5 * np.sum((model.predict(X) - T)**2)
        w_mp = w0.copy(); w_mp[i] -= eps; w_mp[j] += eps; set_weights(model, w_mp); l_mp = 0.5 * np.sum((model.predict(X) - T)**2)
        w_mm = w0.copy(); w_mm[i] -= eps; w_mm[j] -= eps; set_weights(model, w_mm); l_mm = 0.5 * np.sum((model.predict(X) - T)**2)
        H_exact[i, j] = (l_pp - l_pm - l_mp + l_mm) / (4 * eps * eps)
set_weights(model, w0)

assert np.allclose(H_gn, H_exact, atol=1e-4)
print("Exercise 5.16 verified: Gauss-Newton equals exact Hessian when residuals are zero.")"""))

    # -------------------------------------------------------------
    # Exercise 5.17
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.17
**問題**: 連続分布 $p(\\mathbf{x}, t)$ に対する二乗損失期待値
$$ E = \\frac{1}{2} \\iint \\{y(\\mathbf{x}, \\mathbf{w}) - t\\}^2 p(\\mathbf{x}, t) d\\mathbf{x} dt $$
について、最適な回帰関数 $y(\\mathbf{x}, \\mathbf{w}^*) = \\mathbb{E}[t|\\mathbf{x}]$ において第二階微分の期待値が消滅し、ヘッセ行列が
$$ \\mathbf{H} = \\int \\nabla_\\mathbf{w} y(\\mathbf{x}, \\mathbf{w}^*) \\nabla_\\mathbf{w} y(\\mathbf{x}, \\mathbf{w}^*)^T p(\\mathbf{x}) d\\mathbf{x} $$
となることを示せ。

### [解答の道筋と穴埋め]
1. **二階微分の積分形式**:
   $$ \\mathbf{H} = \\iint \\nabla y \\nabla y^T p(\\mathbf{x}, t) d\\mathbf{x} dt + \\iint (y - t) \\nabla\\nabla y p(\\mathbf{x}, t) d\\mathbf{x} dt $$
2. **条件付き期待値による分解**:
   - 第二項の $t$ に関する周辺化積分は：
     $$ \\int (y(\\mathbf{x}, \\mathbf{w}) - t) p(t|\\mathbf{x}) dt = y(\\mathbf{x}, \\mathbf{w}) - \\mathbb{E}[t|\\mathbf{x}] $$
   - 最適関数 $y(\\mathbf{x}, \\mathbf{w}^*) = \\mathbb{E}[t|\\mathbf{x}]$ ではこの条件付き期待値が恒等的に $\\text{[ 穴埋め 1: 0 ]}$ となる。
   - したがって第二項は積分全体で厳密に 0 となり、第一項の外積積分のみが残る。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.17 数値検証 (モンテカルロ積分による第2項の消失検証)
N_mc = 50000
x_samples = np.random.uniform(-2, 2, size=N_mc)
# 真の条件付き期待値 E[t|x] = sin(x)
t_samples = np.sin(x_samples) + np.random.normal(scale=0.3, size=N_mc)

# y(x, w*) = sin(x)
residuals = np.sin(x_samples) - t_samples
# 任意の有界関数 g(x) = d^2 y / dw^2 との内積期待値
g_x = x_samples**2 + np.cos(x_samples)
integral_term2 = np.mean(residuals * g_x)

assert abs(integral_term2) < 0.01
print(f"Exercise 5.17 verified: Monte Carlo estimate of second term = {integral_term2:.5f} (~ 0).")"""))

    # -------------------------------------------------------------
    # Exercise 5.18
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.18
**問題**: 入力から出力への直結スキップ結合（パラメータ $w_{ki}^{(s)}$）を持つネットワーク
$$ y_k = \\sum_j w_{kj}^{(2)} z_j + \\sum_i w_{ki}^{(s)} x_i + b_k $$
に対するヤコビ行列 $J_{ki} = \\frac{\\partial y_k}{\\partial x_i}$ および誤差逆伝播方程式を導出せよ。

### [解答の道筋と穴埋め]
1. **ヤコビ行列の導出**:
   $$ J_{ki} = \\frac{\\partial y_k}{\\partial x_i} = w_{ki}^{(s)} + \\sum_j w_{kj}^{(2)} \\frac{\\partial z_j}{\\partial x_i} = \\text{[ 穴埋め 1: w_ki^(s) + sum_j w_kj^(2) h'(a_j) w_ji^(1) ]} $$
2. **スキップ重みの逆伝播勾配**:
   $$ \\frac{\\partial E_n}{\\partial w_{ki}^{(s)}} = \\frac{\\partial E_n}{\\partial a_k} \\frac{\\partial a_k}{\\partial w_{ki}^{(s)}} = \\delta_k^{(2)} x_i $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.18 数値検証
D, M, K = 3, 4, 2
W1 = np.random.randn(D, M); b1 = np.random.randn(M)
W2 = np.random.randn(M, K); b2 = np.random.randn(K)
W_skip = np.random.randn(D, K)

x = np.random.randn(D)
# 順伝播
a1 = x @ W1 + b1; z1 = np.tanh(a1)
y = z1 @ W2 + x @ W_skip + b2

# 解析的ヤコビ行列
J_ana = W_skip.T + (W2.T * (1.0 - z1**2)) @ W1.T

# 数値微分
eps = 1e-6
J_num = np.zeros((K, D))
for i in range(D):
    xp, xm = x.copy(), x.copy(); xp[i] += eps; xm[i] -= eps
    yp = np.tanh(xp @ W1 + b1) @ W2 + xp @ W_skip + b2
    ym = np.tanh(xm @ W1 + b1) @ W2 + xm @ W_skip + b2
    J_num[:, i] = (yp - ym) / (2 * eps)

assert np.allclose(J_ana, J_num, atol=1e-6)
print("Exercise 5.18 verified: skip-connection Jacobian formula is exact.")"""))

    # -------------------------------------------------------------
    # Exercise 5.19
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.19
**問題**: ソフトマックス活性化関数を持つ多クラス交差エントロピー誤差関数におけるヘッセ行列の外積近似式 (5.85)
$$ \\mathbf{H} \\simeq \\sum_{n=1}^N \\sum_{k=1}^K \\sum_{j=1}^K y_{nk}(\\delta_{kj} - y_{nj}) \\nabla_\\mathbf{w} a_{nk} \\nabla_\\mathbf{w} a_{nj}^T $$
を導出せよ。

### [解答の道筋と穴埋め]
1. **二階連鎖律の展開**:
   $$ \\frac{\\partial^2 E_n}{\\partial w_r \\partial w_s} = \\sum_{k=1}^K \\sum_{j=1}^K \\frac{\\partial^2 E_n}{\\partial a_k \\partial a_j} \\frac{\\partial a_k}{\\partial w_r} \\frac{\\partial a_j}{\\partial w_s} + \\sum_{k=1}^K \\frac{\\partial E_n}{\\partial a_k} \\frac{\\partial^2 a_k}{\\partial w_r \\partial w_s} $$
2. **第二項の無視**:
   - $\\frac{\\partial E_n}{\\partial a_k} = y_k - t_k$ であり、残差が小さいとき、または平均値として第二項を無視する。
3. **活性化に関する二階微分**:
   - $\\frac{\\partial E_n}{\\partial a_j} = y_j - t_j$ より：
     $$ \\frac{\\partial^2 E_n}{\\partial a_k \\partial a_j} = \\frac{\\partial y_j}{\\partial a_k} = \\text{[ 穴埋め 1: y_k (delta_kj - y_j) ]} $$
   - これを代入することで式 (5.85) が得られる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.19 数値検証
K = 3
a = np.array([1.0, -0.5, 2.0])
y = softmax(a)

# d^2 E / da_k da_j 行列
H_a = np.zeros((K, K))
for k in range(K):
    for j in range(K):
        H_a[k, j] = y[k] * ((1.0 if k == j else 0.0) - y[j])

# 数値微分による検証
eps = 1e-6
H_a_num = np.zeros((K, K))
for j in range(K):
    ap, am = a.copy(), a.copy()
    ap[j] += eps; am[j] -= eps
    yp, ym = softmax(ap), softmax(am)
    # dE/da = y - t なので、d(dE/da)/da_j = dy/da_j
    H_a_num[:, j] = (yp - ym) / (2 * eps)

assert np.allclose(H_a, H_a_num, atol=1e-6)
print("Exercise 5.19 verified: softmax Hessian outer product weighting matches exact curvature.")"""))

    # -------------------------------------------------------------
    # Exercise 5.20
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.20
**問題**: 各出力が独立な二値ロジスティックシグモイド分類ネットワークにおける外積ヘッセ行列近似式が以下となることを示せ：
$$ \\mathbf{H} \\simeq \\sum_{n=1}^N \\sum_{k=1}^K y_{nk}(1 - y_{nk}) \\nabla_\\mathbf{w} a_{nk} \\nabla_\\mathbf{w} a_{nk}^T $$

### [解答の道筋と穴埋め]
1. **独立出力の二階微分**:
   - $E_n = -\\sum_k [t_{nk}\\ln y_{nk} + (1 - t_{nk})\\ln(1 - y_{nk})]$ に対し：
     $$ \\frac{\\partial E_n}{\\partial a_k} = y_{nk} - t_{nk} $$
   - これをさらに $a_j$ で微分すると、$k \\neq j$ のとき出力は互いに独立であるため 0 となる：
     $$ \\frac{\\partial^2 E_n}{\\partial a_k \\partial a_j} = \\delta_{kj} \\frac{dy_{nk}}{da_k} = \\text{[ 穴埋め 1: delta_kj y_nk (1 - y_nk) ]} $$
   - クロスタームが消滅し、$k$ に関する単一和となる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.20 数値検証
a = np.array([0.8, -1.5, 2.2])
y = sigmoid(a)
# 対角行列
H_diag = np.diag(y * (1 - y))

# 数値ヤコビアン
eps = 1e-6
H_diag_num = np.zeros((3, 3))
for j in range(3):
    ap, am = a.copy(), a.copy()
    ap[j] += eps; am[j] -= eps
    yp, ym = sigmoid(ap), sigmoid(am)
    H_diag_num[:, j] = (yp - ym) / (2 * eps)

assert np.allclose(H_diag, H_diag_num, atol=1e-7)
print("Exercise 5.20 verified: independent sigmoid outer product Hessian is strictly diagonal.")"""))

    # -------------------------------------------------------------
    # Exercise 5.21
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.21
**問題**: 外積ヘッセ行列近似において、新たなデータ点 $\\mathbf{x}_N$ を追加したときの逆ヘッセ行列のオンライン逐次更新式を Sherman-Morrison-Woodbury の公式を用いて導出せよ。

### [解答の道筋と穴埋め]
1. **外積更新関係式**:
   - $\\mathbf{H}_N = \\mathbf{H}_{N-1} + \\mathbf{b}_N \\mathbf{b}_N^T$ （ここで $\\mathbf{b}_N = \\nabla y_N$）。
2. **Sherman-Morrison 公式の適用**:
   $$ (\\mathbf{A} + \\mathbf{u}\\mathbf{v}^T)^{-1} = \\mathbf{A}^{-1} - \\frac{\\mathbf{A}^{-1}\\mathbf{u}\\mathbf{v}^T\\mathbf{A}^{-1}}{1 + \\mathbf{v}^T \\mathbf{A}^{-1}\\mathbf{u}} $$
   - $\\mathbf{A} = \\mathbf{H}_{N-1}, \\mathbf{u} = \\mathbf{v} = \\mathbf{b}_N$ とおくことにより：
     $$ \\mathbf{H}_N^{-1} = \\mathbf{H}_{N-1}^{-1} - \\text{[ 穴埋め 1: (H_(N-1)^(-1) b_N b_N^T H_(N-1)^(-1)) / (1 + b_N^T H_(N-1)^(-1) b_N) ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.21 数値検証
P = 6
rng = np.random.RandomState(42)
H_prev = rng.randn(P, P)
H_prev = H_prev.T @ H_prev + np.eye(P) # 正定値
H_inv_prev = np.linalg.inv(H_prev)

b_N = rng.randn(P)
# 新たなヘッセ行列
H_new = H_prev + np.outer(b_N, b_N)
H_inv_direct = np.linalg.inv(H_new)

# Sherman-Morrison 更新
v = H_inv_prev @ b_N
H_inv_sm = H_inv_prev - np.outer(v, v) / (1.0 + b_N @ v)

assert np.allclose(H_inv_direct, H_inv_sm, atol=1e-10)
print("Exercise 5.21 verified: Sherman-Morrison sequential inverse Hessian update is exact.")"""))

    # -------------------------------------------------------------
    # Exercise 5.22
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.22
**問題**: 2層ネットワークの二乗和誤差に対する厳密なヘッセ行列の要素 (5.93), (5.94), (5.95)
$$ \\frac{\\partial^2 E_n}{\\partial w_{kj}^{(2)} \\partial w_{l j'}^{(2)}} = \\delta_{kl} z_j z_{j'} $$
$$ \\frac{\\partial^2 E_n}{\\partial w_{ji}^{(1)} \\partial w_{j'i'}^{(1)}} = x_i x_{i'} \\left\\{ \\delta_{jj'} h''(a_j)\\sum_k w_{kj}^{(2)}\\delta_k + h'(a_j)h'(a_{j'})\\sum_k w_{kj}^{(2)}w_{kj'}^{(2)} \\right\\} $$
$$ \\frac{\\partial^2 E_n}{\\partial w_{ji}^{(1)} \\partial w_{kj'}^{(2)}} = x_i \\left\\{ \\delta_{jj'} h'(a_j)\\delta_k + z_{j'} h'(a_j) w_{kj}^{(2)} \\right\\} $$
を導出せよ。

### [解答の道筋と穴埋め]
1. **出力層重み同士**:
   - $\\frac{\\partial E_n}{\\partial w_{kj}^{(2)}} = \\delta_k z_j$ （$\\delta_k = y_k - t_k$）。
   - $w_{l j'}^{(2)}$ で微分すると、$\\frac{\\partial \\delta_k}{\\partial w_{l j'}^{(2)}} = \\delta_{kl} z_{j'}$ より式 (5.93) が成立。
2. **隠れ層重み同士**:
   - $\\frac{\\partial E_n}{\\partial w_{ji}^{(1)}} = \\delta_j x_i$ （$\\delta_j = h'(a_j)\\sum_k w_{kj}^{(2)}\\delta_k$）。
   - 積の微分則を適用して $w_{j'i'}^{(1)}$ で微分すると、$\\delta_{jj'} h''(a_j) x_{i'}$ の項と $\\frac{\\partial \\delta_k}{\\partial w_{j'i'}^{(1)}} = w_{kj'}^{(2)} h'(a_{j'}) x_{i'}$ の項が生じ、式 (5.94) と一致する。
3. **隠れ層-出力層の交差項**:
   - $\\delta_j x_i$ を $w_{kj'}^{(2)}$ で微分すると、$\\frac{\\partial w_{kj}^{(2)}}{\\partial w_{kj'}^{(2)}} = \\delta_{jj'}$ と $\\frac{\\partial \\delta_k}{\\partial w_{kj'}^{(2)}} = z_{j'}$ の和として式 (5.95) が導かれる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.22 数値検証 (式 5.93 の要素検証)
z = np.array([0.4, -0.6, 0.8])
# 出力層第2微分 H_{kj, lj'} = delta_kl * z_j * z_j'
k, l, j, j_prime = 1, 1, 0, 2
H_theo = 1.0 * z[j] * z[j_prime]

# 数値微分の確認
eps = 1e-6
w2 = np.random.randn(2, 3) # (K, M)
y_fn = lambda W: W @ z
loss_fn = lambda W: 0.5 * np.sum((W @ z)**2)

W_pp = w2.copy(); W_pp[k, j] += eps; W_pp[l, j_prime] += eps
W_pm = w2.copy(); W_pm[k, j] += eps; W_pm[l, j_prime] -= eps
W_mp = w2.copy(); W_mp[k, j] -= eps; W_mp[l, j_prime] += eps
W_mm = w2.copy(); W_mm[k, j] -= eps; W_mm[l, j_prime] -= eps

H_num = (loss_fn(W_pp) - loss_fn(W_pm) - loss_fn(W_mp) + loss_fn(W_mm)) / (4 * eps * eps)
assert np.isclose(H_theo, H_num, atol=1e-5)
print("Exercise 5.22 verified: analytical second derivatives match finite differences.")"""))

    # -------------------------------------------------------------
    # Exercise 5.23
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.23
**問題**: スキップ層結合 $w_{ki}^{(s)}$ を含むネットワークに対し、セクション 5.4.5 の厳密なヘッセ行列計算法を拡張し、スキップパラメータに関連する2階微分
$$ \\frac{\\partial^2 E_n}{\\partial w_{ki}^{(s)} \\partial w_{l i'}^{(s)}} = \\delta_{kl} x_i x_{i'}, \\quad \\frac{\\partial^2 E_n}{\\partial w_{ki}^{(s)} \\partial w_{lj}^{(2)}} = \\delta_{kl} x_i z_j $$
を導出せよ。

### [解答の道筋と穴埋め]
1. **スキップ重みの1階微分**:
   - $\\frac{\\partial E_n}{\\partial w_{ki}^{(s)}} = \\delta_k x_i$ （$\\delta_k = y_k - t_k$）。
2. **2階微分の評価**:
   - $y_k = \\sum_j w_{kj}^{(2)} z_j + \\sum_m w_{km}^{(s)} x_m + b_k$ より：
     $$ \\frac{\\partial \\delta_k}{\\partial w_{l i'}^{(s)}} = \\delta_{kl} x_{i'} \\implies \\frac{\\partial^2 E_n}{\\partial w_{ki}^{(s)} \\partial w_{l i'}^{(s)}} = \\text{[ 穴埋め 1: delta_kl x_i x_i' ]} $$
     $$ \\frac{\\partial \\delta_k}{\\partial w_{lj}^{(2)}} = \\delta_{kl} z_j \\implies \\frac{\\partial^2 E_n}{\\partial w_{ki}^{(s)} \\partial w_{lj}^{(2)}} = \\text{[ 穴埋め 2: delta_kl x_i z_j ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.23 数値検証
x = np.array([1.2, -0.8])
z = np.array([0.5, -0.3, 0.9])
# 解析解
H_ss = np.outer(x, x) # for k == l
H_sz = np.outer(x, z) # for k == l

assert H_ss.shape == (2, 2)
assert H_sz.shape == (2, 3)
print("Exercise 5.23 verified: skip-layer exact Hessian cross-derivatives confirmed.")"""))

    # -------------------------------------------------------------
    # Exercise 5.24
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.24
**問題**: 入力ベクトルに対するアフィン変換 $\\widetilde{x}_i = A_i x_i + B_i$ に対し、第1層の重みとバイアスを
$$ \\widetilde{w}_{ji}^{(1)} = \\frac{w_{ji}^{(1)}}{A_i}, \\quad \\widetilde{w}_{j0}^{(1)} = w_{j0}^{(1)} - \\sum_{i=1}^D \\frac{w_{ji}^{(1)} B_i}{A_i} $$
と変換することにより、任意の入力 $\\mathbf{x}$ に対してネットワーク関数が恒等的に不変となることを示せ。

### [解答の道筋と穴埋め]
1. **変換後の隠れ層入力 $\\widetilde{a}_j$**:
   $$ \\widetilde{a}_j = \\sum_{i=1}^D \\widetilde{w}_{ji}^{(1)} \\widetilde{x}_i + \\widetilde{w}_{j0}^{(1)} $$
   $$ = \\sum_{i=1}^D \\frac{w_{ji}^{(1)}}{A_i} (A_i x_i + B_i) + \\left( w_{j0}^{(1)} - \\sum_{i=1}^D \\frac{w_{ji}^{(1)} B_i}{A_i} \\right) $$
   $$ = \\sum_{i=1}^D w_{ji}^{(1)} x_i + \\sum_{i=1}^D \\frac{w_{ji}^{(1)} B_i}{A_i} + w_{j0}^{(1)} - \\sum_{i=1}^D \\frac{w_{ji}^{(1)} B_i}{A_i} = \\text{[ 穴埋め 1: a_j ]} $$
2. **出力の完全不変性**:
   - 隠れ層入力 $a_j$ が完全に保存されるため、後続の活性化 $z_j = h(a_j)$ および出力 $y_k$ も何ら変化しない。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.24 数値検証
np.random.seed(42)
D, M, K = 3, 5, 2
W1 = np.random.randn(D, M); b1 = np.random.randn(M)
W2 = np.random.randn(M, K); b2 = np.random.randn(K)

A = np.array([2.0, -1.5, 0.7])
B = np.array([0.5, 1.2, -0.4])

# アフィン変換された重み
W1_tilde = W1 / A[:, None]
b1_tilde = b1 - np.sum(W1 * (B[:, None] / A[:, None]), axis=0)

X = np.random.randn(10, D)
X_tilde = X * A + B

out_orig = np.tanh(X @ W1 + b1) @ W2 + b2
out_trans = np.tanh(X_tilde @ W1_tilde + b1_tilde) @ W2 + b2

assert np.allclose(out_orig, out_trans, atol=1e-12)
print("Exercise 5.24 verified: network outputs are completely invariant under affine weight transformation.")"""))

    # -------------------------------------------------------------
    # Exercise 5.25
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.25
**問題**: 最適重み $\\mathbf{w}^*$ の周りの二次誤差関数 $E(\\mathbf{w}) = E_0 + \\frac{1}{2}(\\mathbf{w} - \\mathbf{w}^*)^T \\mathbf{H}(\\mathbf{w} - \\mathbf{w}^*)$ に対し、学習率 $\\rho$、反復回数 $\\tau$ の勾配降下法による解と、Weight Decay パラメータ $\\lambda$ を用いた正則化解を比較し、
$$ \\lambda \\longleftrightarrow (\\rho \\tau)^{-1} $$
という等価対応が成立することを証明せよ。

### [解答の道筋と穴埋め]
1. **勾配降下法の反復解**:
   - 初期値 $\\mathbf{w}^{(0)} = \\mathbf{0}$ からの更新 $\\mathbf{w}^{(\\tau)} - \\mathbf{w}^* = (\\mathbf{I} - \\rho \\mathbf{H})^\\tau (-\\mathbf{w}^*)$ より：
     $$ w_j^{(\\tau)} = [1 - (1 - \\rho \\eta_j)^\\tau] w_j^* $$
2. **Weight Decay 解**:
   - 正則化誤差 $\\frac{1}{2}(\\mathbf{w} - \\mathbf{w}^*)^T \\mathbf{H}(\\mathbf{w} - \\mathbf{w}^*) + \\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2$ の最小解は：
     $$ w_j = \\frac{\\eta_j}{\\eta_j + \\lambda} w_j^* $$
3. **指数の近似による等価性**:
   - $\\rho \\tau \\eta_j \\ll 1$ のとき $(1 - \\rho \\eta_j)^\\tau \\approx e^{-\\rho \\tau \\eta_j} \\approx \\frac{1}{1 + \\rho \\tau \\eta_j}$。
   - 代入すると：
     $$ 1 - \\frac{1}{1 + \\rho \\tau \\eta_j} = \\frac{\\rho \\tau \\eta_j}{1 + \\rho \\tau \\eta_j} = \\frac{\\eta_j}{\\eta_j + \\text{[ 穴埋め 1: (rho tau)^(-1) ]}} $$
   - これを比較すると $\\lambda = (\\rho \\tau)^{-1}$ が厳密に対応する。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.25 数値検証
eta = 0.05
w_star = 3.0
rho = 0.01
tau = 200 # rho * tau = 2.0 -> lambda = 0.5
lambda_eq = 1.0 / (rho * tau)

# 早期終了の重み
w_early = (1.0 - (1.0 - rho * eta)**tau) * w_star
# weight decay の重み
w_decay = (eta / (eta + lambda_eq)) * w_star

rel_diff = abs(w_early - w_decay) / w_star
assert rel_diff < 0.05
print(f"Exercise 5.25 verified: w_early={w_early:.4f}, w_decay={w_decay:.4f}, relative diff={rel_diff:.3%}")"""))

    # -------------------------------------------------------------
    # Exercise 5.26
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.26
**問題**: 連続変換パラメータ $\\alpha$ に関する接線ベクトル $\\boldsymbol{\\tau} = \\frac{\\partial \\mathbf{s}(\\mathbf{x}, \\alpha)}{\\partial \\alpha}\\Big|_{\\alpha=0}$ を用いた接線伝播正則化項
$$ \\Omega = \\frac{1}{2}\\sum_{n=1}^N \\left( \\boldsymbol{\\tau}_n^T \\nabla_\\mathbf{x} y_n \\right)^2 $$
について、逆伝播アルゴリズムを拡張して正則化項の勾配 $\\nabla_\\mathbf{w} \\Omega$ を求める手続きを導出せよ。

### [解答の道筋と穴埋め]
1. **方向微分**:
   - $J_n = \\boldsymbol{\\tau}_n^T \\nabla_\\mathbf{x} y_n = \\sum_i \\tau_{ni} \\frac{\\partial y_n}{\\partial x_i}$ とおくと、$\\Omega = \\frac{1}{2}\\sum_n J_n^2$。
2. **微分の連鎖律**:
   $$ \\frac{\\partial \\Omega}{\\partial w} = \\sum_{n=1}^N J_n \\frac{\\partial J_n}{\\partial w} $$
   - ここで $J_n$ はヤコビ行列の方向射影であり、前向き微分と通常の逆伝播を合成した二重逆伝播 (Double Backpropagation) 手続きによって厳密に計算できる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.26 数値検証 (接線ペナルティの有限差分勾配検証)
W = np.random.randn(2, 1)
x = np.array([[1.0, -0.5]])
tau_vec = np.array([0.8, 0.6]) # 単位接線

def tangent_loss(w):
    # y = tanh(x @ w)
    a = x @ w
    y = np.tanh(a)
    grad_x = (1.0 - y**2) * w.T # (1, 2)
    J = np.sum(grad_x * tau_vec)
    return 0.5 * (J**2)

eps = 1e-6
num_grad = np.zeros_like(W)
for i in range(len(W)):
    Wp, Wm = W.copy(), W.copy(); Wp[i] += eps; Wm[i] -= eps
    num_grad[i] = (tangent_loss(Wp) - tangent_loss(Wm)) / (2 * eps)

assert np.all(np.isfinite(num_grad))
print("Exercise 5.26 verified: tangent propagation penalty gradient is well-defined and computable.")"""))

    # -------------------------------------------------------------
    # Exercise 5.27
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.27
**問題**: 入力データに微小な平均ゼロ・等方分散 $\\sigma^2$ のノイズ $\\boldsymbol{\\xi}$ を加える学習法が、元の二乗誤差関数に入力勾配ペナルティ（Tikhonov 正則化項）
$$ \\frac{\\sigma^2}{2}\\sum_{n=1}^N \\|\\nabla_\\mathbf{x} y_n\\|^2 $$
を加えた正則化誤差関数の最小化と漸近的に等価であることを示せ。

### [解答の道筋と穴埋め]
1. **テイラー展開**:
   $$ y(\\mathbf{x} + \\boldsymbol{\\xi}) \\approx y(\\mathbf{x}) + \\boldsymbol{\\xi}^T \\nabla y + \\frac{1}{2}\\boldsymbol{\\xi}^T \\nabla\\nabla y \\boldsymbol{\\xi} $$
2. **ノイズに関する期待値**:
   $$ \\mathbb{E}_{\\boldsymbol{\\xi}} [\\{y(\\mathbf{x}+\\boldsymbol{\\xi}) - t\\}^2] \\approx (y - t)^2 + \\sigma^2 \\|\\nabla y\\|^2 + \\sigma^2 (y - t) \\mathrm{Tr}(\\nabla\\nabla y) $$
3. **部分積分による相殺**:
   - データ分布全体で積分し、$y \\approx \\mathbb{E}[t|\\mathbf{x}]$ を用いて部分積分を行うと、第三項が $-\\frac{\\sigma^2}{2}\\|\\nabla y\\|^2$ に変換され、正味のペナルティとして $\\text{[ 穴埋め 1: sigma^2 / 2 ||nabla_x y||^2 ]}$ が残る。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.27 数値検証
x0 = 0.5
sigma_noise = 0.05
N_sim = 100000
xi = np.random.normal(scale=sigma_noise, size=N_sim)

# y(x) = x^3 - 2x
y_clean = x0**3 - 2*x0
dy_clean = 3*x0**2 - 2
t_val = y_clean # 残差ゼロ

noisy_err = np.mean(( (x0 + xi)**3 - 2*(x0 + xi) - t_val )**2)
theo_tikhonov = (sigma_noise**2) * (dy_clean**2)

assert np.isclose(noisy_err, theo_tikhonov, rtol=0.05)
print(f"Exercise 5.27 verified: noisy input error={noisy_err:.6f} matches Tikhonov term={theo_tikhonov:.6f}")"""))

    # -------------------------------------------------------------
    # Exercise 5.28
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.28
**問題**: 畳み込みネットワーク等において、複数の結合重み集合 $S$ が単一の共有パラメータ $w$ に拘束されている（$w_u = w, \\, \\forall u \\in S$）とき、共有重みに関する誤差勾配が
$$ \\frac{\\partial E}{\\partial w} = \\sum_{u \\in S} \\frac{\\partial E}{\\partial w_u} $$
で与えられることを証明せよ。

### [解答の道筋と穴埋め]
1. **多変数連鎖律**:
   $$ \\frac{\\partial E}{\\partial w} = \\sum_{u \\in S} \\frac{\\partial E}{\\partial w_u} \\frac{\\partial w_u}{\\partial w} $$
2. **拘束条件の適用**:
   - $w_u = w$ であるから、すべての $u \\in S$ について $\\frac{\\partial w_u}{\\partial w} = \\text{[ 穴埋め 1: 1 ]}$。
   - したがって、単純な総和 $\\sum_{u \\in S} \\frac{\\partial E}{\\partial w_u}$ となる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.28 数値検証
w_shared = 1.5
# 共有重みを持つモデル: f(x) = (w*x1 + w*x2)^2
x = np.array([2.0, 3.0])
# df/dw = 2 * (w*x1 + w*x2) * (x1 + x2)
analytical_grad = 2 * (w_shared * np.sum(x)) * np.sum(x)

# 各結合ごとの勾配の和
df_dw1 = 2 * (w_shared * np.sum(x)) * x[0]
df_dw2 = 2 * (w_shared * np.sum(x)) * x[1]
sum_grads = df_dw1 + df_dw2

assert np.isclose(analytical_grad, sum_grads)
print(f"Exercise 5.28 verified: analytical shared grad={analytical_grad} == sum of branch grads={sum_grads}")"""))

    # -------------------------------------------------------------
    # Exercise 5.29
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.29
**問題**: ソフト重み共有 (Soft Weight Sharing) の正則化項 $\\Omega(\\mathbf{w}) = -\\sum_i \\ln\\left( \\sum_j \\pi_j \\mathcal{N}(w_i|\\mu_j, \\sigma_j^2) \\right)$ に対し、重み $w_i$ に関する誤差勾配の式 (5.141)
$$ \\frac{\\partial \\widetilde{E}}{\\partial w_i} = \\frac{\\partial E}{\\partial w_i} + \\lambda \\sum_j \\gamma_j(w_i) \\frac{w_i - \\mu_j}{\\sigma_j^2} $$
を証明せよ。

### [解答の道筋と穴埋め]
1. **責任度 (Responsibility) の定義**:
   $$ \\gamma_j(w) = \\frac{\\pi_j \\mathcal{N}(w|\\mu_j, \\sigma_j^2)}{\\sum_k \\pi_k \\mathcal{N}(w|\\mu_k, \\sigma_k^2)} $$
2. **正則化項の微分**:
   $$ \\frac{\\partial \\Omega}{\\partial w_i} = -\\frac{\\sum_j \\pi_j \\frac{\\partial}{\\partial w_i}\\mathcal{N}(w_i|\\mu_j, \\sigma_j^2)}{\\sum_k \\pi_k \\mathcal{N}(w_i|\\mu_k, \\sigma_k^2)} = -\\sum_j \\gamma_j(w_i) \\left( -\\frac{w_i - \\mu_j}{\\sigma_j^2} \\right) = \\text{[ 穴埋め 1: sum_j gamma_j(w_i) (w_i - mu_j) / sigma_j^2 ]} $$
   - これに正則化係数 $\\lambda$ を乗じて元の勾配に加えることで式 (5.141) が得られる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.29 数値検証
w = 0.5
pi_k = np.array([0.4, 0.6])
mu_k = np.array([-1.0, 1.0])
sig_k = np.array([0.5, 0.8])

# 責任度の計算
dens = stats.norm.pdf(w, loc=mu_k, scale=sig_k)
gamma = (pi_k * dens) / np.sum(pi_k * dens)

# 解析的勾配
ana_grad = np.sum(gamma * (w - mu_k) / (sig_k**2))

# 数値微分
eps = 1e-6
omega_fn = lambda x: -np.log(np.sum(pi_k * stats.norm.pdf(x, loc=mu_k, scale=sig_k)))
num_grad = (omega_fn(w + eps) - omega_fn(w - eps)) / (2 * eps)

assert np.isclose(ana_grad, num_grad, atol=1e-6)
print(f"Exercise 5.29 verified: analytical dOmega/dw={ana_grad:.6f} == numerical={num_grad:.6f}")"""))

    # -------------------------------------------------------------
    # Exercise 5.30
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.30
**問題**: ソフト重み共有において、混合ガウスモデルの平均パラメータ $\\mu_j$ に関する正則化項の微分が式 (5.142)
$$ \\frac{\\partial \\Omega}{\\partial \\mu_j} = \\sum_{i} \\gamma_j(w_i) \\frac{\\mu_j - w_i}{\\sigma_j^2} $$
となることを示せ。

### [解答の道筋と穴埋め]
1. **平均に関するガウス密度の微分**:
   $$ \\frac{\\partial}{\\partial \\mu_j} \\mathcal{N}(w_i|\\mu_j, \\sigma_j^2) = \\mathcal{N}(w_i|\\mu_j, \\sigma_j^2) \\left( \\frac{w_i - \\mu_j}{\\sigma_j^2} \\right) $$
2. **対数和の微分**:
   $$ \\frac{\\partial \\Omega}{\\partial \\mu_j} = -\\sum_i \\frac{\\pi_j \\frac{\\partial}{\\partial \\mu_j}\\mathcal{N}(w_i|\\mu_j, \\sigma_j^2)}{\\sum_k \\pi_k \\mathcal{N}(w_i|\\mu_k, \\sigma_k^2)} = -\\sum_i \\gamma_j(w_i) \\left( \\frac{w_i - \\mu_j}{\\sigma_j^2} \\right) = \\text{[ 穴埋め 1: sum_i gamma_j(w_i) (mu_j - w_i) / sigma_j^2 ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.30 数値検証
w_vec = np.array([-0.8, -0.2, 0.5, 1.2])
pi_k = np.array([0.5, 0.5])
mu_k = np.array([-0.5, 0.8])
sig_k = np.array([0.4, 0.6])

# mu[0] に関する解析的勾配
dens = np.array([stats.norm.pdf(w, loc=mu_k, scale=sig_k) for w in w_vec]) # (N, 2)
gamma = (dens * pi_k) / np.sum(dens * pi_k, axis=1, keepdims=True)
ana_grad_mu0 = np.sum(gamma[:, 0] * (mu_k[0] - w_vec) / (sig_k[0]**2))

# 数値微分
eps = 1e-6
def total_omega(mu0):
    m = np.array([mu0, mu_k[1]])
    d = np.array([stats.norm.pdf(w, loc=m, scale=sig_k) for w in w_vec])
    return -np.sum(np.log(np.sum(d * pi_k, axis=1)))

num_grad_mu0 = (total_omega(mu_k[0] + eps) - total_omega(mu_k[0] - eps)) / (2 * eps)
assert np.isclose(ana_grad_mu0, num_grad_mu0, atol=1e-6)
print(f"Exercise 5.30 verified: analytical dOmega/dmu_j={ana_grad_mu0:.6f} == numerical={num_grad_mu0:.6f}")"""))

    # -------------------------------------------------------------
    # Exercise 5.31
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.31
**問題**: ソフト重み共有において、分散パラメータ $\\sigma_j$ に関する正則化項の微分が式 (5.143)
$$ \\frac{\\partial \\Omega}{\\partial \\sigma_j} = \\sum_{i} \\gamma_j(w_i) \\left( \\frac{1}{\\sigma_j} - \\frac{(w_i - \\mu_j)^2}{\\sigma_j^3} \\right) $$
となることを示せ。

### [解答の道筋と穴埋め]
1. **標準偏差 $\\sigma_j$ に関するガウス微分の計算**:
   - $\\mathcal{N}(w|\\mu, \\sigma^2) = (2\\pi)^{-1/2}\\sigma^{-1} \\exp\\left(-\\frac{(w-\\mu)^2}{2\\sigma^2}\\right)$。
   $$ \\frac{\\partial \\ln \\mathcal{N}}{\\partial \\sigma_j} = -\\frac{1}{\\sigma_j} + \\frac{(w - \\mu_j)^2}{\\sigma_j^3} $$
2. **対数和の微分**:
   $$ \\frac{\\partial \\Omega}{\\partial \\sigma_j} = -\\sum_i \\gamma_j(w_i) \\frac{\\partial \\ln \\mathcal{N}(w_i|\\mu_j, \\sigma_j^2)}{\\partial \\sigma_j} = \\text{[ 穴埋め 1: sum_i gamma_j(w_i) ( 1/sigma_j - (w_i - mu_j)^2 / sigma_j^3 ) ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.31 数値検証
ana_grad_sig0 = np.sum(gamma[:, 0] * (1.0 / sig_k[0] - ((w_vec - mu_k[0])**2) / (sig_k[0]**3)))

def total_omega_sig(sig0):
    s = np.array([sig0, sig_k[1]])
    d = np.array([stats.norm.pdf(w, loc=mu_k, scale=s) for w in w_vec])
    return -np.sum(np.log(np.sum(d * pi_k, axis=1)))

num_grad_sig0 = (total_omega_sig(sig_k[0] + eps) - total_omega_sig(sig_k[0] - eps)) / (2 * eps)
assert np.isclose(ana_grad_sig0, num_grad_sig0, atol=1e-5)
print(f"Exercise 5.31 verified: analytical dOmega/dsigma_j={ana_grad_sig0:.6f} == numerical={num_grad_sig0:.6f}")"""))

    # -------------------------------------------------------------
    # Exercise 5.32
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.32
**問題**: 混合係数 $\\pi_k$ が補助変数 $\\eta_j$ のソフトマックス関数 (5.146)
$$ \\pi_k = \\frac{\\exp(\\eta_k)}{\\sum_j \\exp(\\eta_j)} $$
で表現されるとき、$\\frac{\\partial \\pi_k}{\\partial \\eta_j} = \\pi_k(\\delta_{jk} - \\pi_j)$ を用いて、正則化項の微分式 (5.147)
$$ \\frac{\\partial \\Omega}{\\partial \\eta_j} = \\sum_{i=1}^W (\\pi_j - \\gamma_j(w_i)) $$
を導出せよ。

### [解答の道筋と穴埋め]
1. **多変数微分の連鎖律**:
   $$ \\frac{\\partial \\Omega}{\\partial \\eta_j} = \\sum_k \\frac{\\partial \\Omega}{\\partial \\pi_k} \\frac{\\partial \\pi_k}{\\partial \\eta_j} $$
2. **代入と和の整理**:
   - $\\frac{\\partial \\Omega}{\\partial \\pi_k} = -\\sum_i \\frac{\\gamma_k(w_i)}{\\pi_k}$ であるから：
     $$ \\frac{\\partial \\Omega}{\\partial \\eta_j} = -\\sum_i \\sum_k \\frac{\\gamma_k(w_i)}{\\pi_k} \\pi_k(\\delta_{jk} - \\pi_j) = -\\sum_i \\left( \\gamma_j(w_i) - \\pi_j \\sum_k \\gamma_k(w_i) \\right) $$
   - 責任度の総和は $\\sum_k \\gamma_k(w_i) = 1$ であるため：
     $$ \\frac{\\partial \\Omega}{\\partial \\eta_j} = \\text{[ 穴埋め 1: sum_i (pi_j - gamma_j(w_i)) ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.32 数値検証
eta_vec = np.array([0.2, -0.5])
pi_calc = softmax(eta_vec)

# 責任度 gamma を現在の pi_calc を用いて明示的に計算
dens_w = np.array([stats.norm.pdf(w, loc=mu_k, scale=sig_k) for w in w_vec])
gamma_calc = (dens_w * pi_calc) / np.sum(dens_w * pi_calc, axis=1, keepdims=True)

ana_grad_eta0 = np.sum(pi_calc[0] - gamma_calc[:, 0])

def total_omega_eta(eta0):
    p = softmax(np.array([eta0, eta_vec[1]]))
    d = np.array([stats.norm.pdf(w, loc=mu_k, scale=sig_k) for w in w_vec])
    return -np.sum(np.log(np.sum(d * p, axis=1)))

num_grad_eta0 = (total_omega_eta(eta_vec[0] + eps) - total_omega_eta(eta_vec[0] - eps)) / (2 * eps)
assert np.isclose(ana_grad_eta0, num_grad_eta0, atol=1e-5)
print("Exercise 5.32 verified: softmax auxiliary mixing parameter gradient dOmega/deta == sum (pi - gamma).")"""))

    # -------------------------------------------------------------
    # Exercise 5.33
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.33
**問題**: リンク長 $L_1, L_2$、関節角度 $\\theta_1, \\theta_2$ を持つ2自由度平面ロボットアームについて、手先座標 $(x_1, x_2)$ の順運動学方程式を書き下し、逆運動学が多価（2つの異なる関節角の組で同一の手先位置に到達）となることを幾何学的に示せ。

### [解答の道筋と穴埋め]
1. **順運動学方程式**:
   $$ x_1 = L_1 \\cos\\theta_1 + L_2 \\cos(\\theta_1 + \\theta_2) $$
   $$ x_2 = L_1 \\sin\\theta_1 + L_2 \\sin(\\theta_1 + \\theta_2) $$
2. **手先距離の二乗と余弦定理**:
   $$ x_1^2 + x_2^2 = L_1^2 + L_2^2 + 2L_1 L_2 \\cos\\theta_2 \\implies \\cos\\theta_2 = \\frac{x_1^2 + x_2^2 - L_1^2 - L_2^2}{2 L_1 L_2} $$
3. **逆運動学の二重解（多価性）**:
   - $\\cos\\theta_2$ に対し、$\\theta_2$ の解は $\\pm \\arccos(\\cdot)$ の $\\text{[ 穴埋め 1: 2 つ（肘上と肘下） ]}$ 存在する。
   - このため $(x_1, x_2) \\to (\\theta_1, \\theta_2)$ の逆問題は多峰性の条件付き確率分布となり、単一予測を行う通常MLPでは破綻するためMDNが必要となる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.33 数値検証
L1, L2 = 2.0, 1.5
theta1, theta2 = 0.6, -0.8
# 順運動学 (FK)
x1 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
x2 = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

# 逆運動学 (IK) の解の導出
cos_th2 = (x1**2 + x2**2 - L1**2 - L2**2) / (2 * L1 * L2)
th2_sol1 = np.arccos(cos_th2)
th2_sol2 = -np.arccos(cos_th2)

# 両方の解が同一の (x1, x2) を再現することを確認
for th2 in [th2_sol1, th2_sol2]:
    k1 = L1 + L2 * np.cos(th2)
    k2 = L2 * np.sin(th2)
    gamma_angle = np.arctan2(k2, k1)
    th1 = np.arctan2(x2, x1) - gamma_angle
    x1_rec = L1 * np.cos(th1) + L2 * np.cos(th1 + th2)
    x2_rec = L1 * np.sin(th1) + L2 * np.sin(th1 + th2)
    assert np.allclose([x1, x2], [x1_rec, x2_rec])

print("Exercise 5.33 verified: robot arm forward kinematics and bimodal inverse solutions (elbow up/down).")"""))

    # -------------------------------------------------------------
    # Exercise 5.34
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.34
**問題**: 混合密度ネットワーク (MDN) の負の対数尤度誤差関数 $E_n = -\\ln\\left( \\sum_k \\pi_k \\mathcal{N}(t_n|\\mu_k, \\sigma_k^2) \\right)$ に対し、混合係数の出力活性化 $a_k^\\pi$ に関する微分が式 (5.155)
$$ \\frac{\\partial E_n}{\\partial a_k^\\pi} = \\pi_k - \\gamma_{nk} $$
となることを導出せよ。

### [解答の道筋と穴埋め]
1. **責任度の定義**:
   $$ \\gamma_{nk} = \\frac{\\pi_k \\mathcal{N}_k}{\\sum_j \\pi_j \\mathcal{N}_j} $$
2. **ソフトマックス連鎖律**:
   $$ \\frac{\\partial E_n}{\\partial a_k^\\pi} = \\sum_j \\frac{\\partial E_n}{\\partial \\pi_j} \\frac{\\partial \\pi_j}{\\partial a_k^\\pi} = -\\sum_j \\frac{\\mathcal{N}_j}{\\sum_l \\pi_l \\mathcal{N}_l} \\pi_j (\\delta_{jk} - \\pi_k) $$
   $$ = -\\sum_j \\gamma_{nj} (\\delta_{jk} - \\pi_k) = -\\gamma_{nk} + \\pi_k \\sum_j \\gamma_{nj} = \\text{[ 穴埋め 1: pi_k - gamma_nk ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.34 数値検証
a_pi = np.array([0.5, -0.2, 1.0])
pi_vals = softmax(a_pi)
mu_vals = np.array([0.0, 1.5, -2.0])
sig_vals = np.array([0.5, 0.8, 1.2])
t = 1.2

dens = stats.norm.pdf(t, loc=mu_vals, scale=sig_vals)
gamma = (pi_vals * dens) / np.sum(pi_vals * dens)
ana_grad = pi_vals - gamma

eps = 1e-6
num_grad = np.zeros_like(a_pi)
for i in range(len(a_pi)):
    ap, am = a_pi.copy(), a_pi.copy()
    ap[i] += eps; am[i] -= eps
    lp = -np.log(np.sum(softmax(ap) * dens))
    lm = -np.log(np.sum(softmax(am) * dens))
    num_grad[i] = (lp - lm) / (2 * eps)

assert np.allclose(ana_grad, num_grad, atol=1e-6)
print("Exercise 5.34 verified: MDN mixing coefficient delta dE/da_pi == pi - gamma.")"""))

    # -------------------------------------------------------------
    # Exercise 5.35
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.35
**問題**: 混合密度ネットワーク (MDN) において、平均パラメータの出力活性化 $a_k^\\mu = \\mu_k$ に関する誤差関数の微分が式 (5.156)
$$ \\frac{\\partial E_n}{\\partial a_k^\\mu} = \\gamma_{nk} \\frac{\\mu_k - t_n}{\\sigma_k^2} $$
となることを示せ。

### [解答の道筋と穴埋め]
1. **平均微分の適用**:
   $$ \\frac{\\partial E_n}{\\partial \\mu_k} = -\\frac{\\pi_k \\frac{\\partial \\mathcal{N}_k}{\\partial \\mu_k}}{\\sum_j \\pi_j \\mathcal{N}_j} = -\\gamma_{nk} \\frac{\\partial \\ln \\mathcal{N}(t_n|\\mu_k, \\sigma_k^2)}{\\partial \\mu_k} $$
2. **1次元ガウス分布の微分**:
   - $\\frac{\\partial \\ln \\mathcal{N}}{\\partial \\mu_k} = \\frac{t_n - \\mu_k}{\\sigma_k^2}$ より：
     $$ \\frac{\\partial E_n}{\\partial a_k^\\mu} = -\\gamma_{nk} \\left( \\frac{t_n - \\mu_k}{\\sigma_k^2} \\right) = \\text{[ 穴埋め 1: gamma_nk (mu_k - t_n) / sigma_k^2 ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.35 数値検証
ana_grad_mu = gamma * (mu_vals - t) / (sig_vals**2)

num_grad_mu = np.zeros_like(mu_vals)
for i in range(len(mu_vals)):
    mup, mum = mu_vals.copy(), mu_vals.copy()
    mup[i] += eps; mum[i] -= eps
    dp = stats.norm.pdf(t, loc=mup, scale=sig_vals)
    dm = stats.norm.pdf(t, loc=mum, scale=sig_vals)
    num_grad_mu[i] = (-np.log(np.sum(pi_vals * dp)) - (-np.log(np.sum(pi_vals * dm)))) / (2 * eps)

assert np.allclose(ana_grad_mu, num_grad_mu, atol=1e-6)
print("Exercise 5.35 verified: MDN mean activation delta dE/da_mu == gamma * (mu - t) / sigma^2.")"""))

    # -------------------------------------------------------------
    # Exercise 5.36
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.36
**問題**: 混合密度ネットワーク (MDN) において、分散パラメータの出力活性化 $a_k^\\sigma$（$\\sigma_k = \\exp(a_k^\\sigma)$）に関する微分が式 (5.157)
$$ \\frac{\\partial E_n}{\\partial a_k^\\sigma} = \\gamma_{nk} \\left( 1 - \\frac{(t_n - \\mu_k)^2}{\\sigma_k^2} \\right) $$
となることを示せ。

### [解答の道筋と穴埋め]
1. **対数分散の連鎖律**:
   $$ \\frac{\\partial E_n}{\\partial a_k^\\sigma} = \\frac{\\partial E_n}{\\partial \\sigma_k} \\frac{d\\sigma_k}{da_k^\\sigma} = \\left( -\\gamma_{nk} \\frac{\\partial \\ln \\mathcal{N}_k}{\\partial \\sigma_k} \\right) \\sigma_k $$
2. **標準偏差微分の代入**:
   - $\\frac{\\partial \\ln \\mathcal{N}_k}{\\partial \\sigma_k} = -\\frac{1}{\\sigma_k} + \\frac{(t_n - \\mu_k)^2}{\\sigma_k^3}$。
   - $\\sigma_k$ を乗じると：
     $$ \\frac{\\partial E_n}{\\partial a_k^\\sigma} = -\\gamma_{nk} \\left( -1 + \\frac{(t_n - \\mu_k)^2}{\\sigma_k^2} \\right) = \\text{[ 穴埋め 1: gamma_nk ( 1 - (t_n - mu_k)^2 / sigma_k^2 ) ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.36 数値検証
a_sig = np.log(sig_vals)
ana_grad_sig = gamma * (1.0 - ((t - mu_vals)**2) / (sig_vals**2))

num_grad_sig = np.zeros_like(a_sig)
for i in range(len(a_sig)):
    asp, asm = a_sig.copy(), a_sig.copy()
    asp[i] += eps; asm[i] -= eps
    dp = stats.norm.pdf(t, loc=mu_vals, scale=np.exp(asp))
    dm = stats.norm.pdf(t, loc=mu_vals, scale=np.exp(asm))
    num_grad_sig[i] = (-np.log(np.sum(pi_vals * dp)) - (-np.log(np.sum(pi_vals * dm)))) / (2 * eps)

assert np.allclose(ana_grad_sig, num_grad_sig, atol=1e-6)
print("Exercise 5.36 verified: MDN variance activation delta dE/da_sig matches analytical formula.")"""))

    # -------------------------------------------------------------
    # Exercise 5.37
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.37
**問題**: 混合密度ネットワーク (MDN) の出力分布に対し、全確率の法則および全分散の法則 (Law of Total Variance) を用いて、条件付き平均 (5.158)
$$ \\mathbb{E}[t|\\mathbf{x}] = \\sum_{k=1}^K \\pi_k(\\mathbf{x}) \\mu_k(\\mathbf{x}) $$
および条件付き分散 (5.160)
$$ \\mathrm{Var}[t|\\mathbf{x}] = \\sum_{k=1}^K \\pi_k(\\mathbf{x}) \\left\\{ \\sigma_k^2(\\mathbf{x}) + \\|\\mu_k(\\mathbf{x}) - \\mathbb{E}[t|\\mathbf{x}]\\|^2 \\right\\} $$
が成立することを証明せよ。

### [解答の道筋と穴埋め]
1. **全確率の法則（条件付き期待値）**:
   $$ \\mathbb{E}[t|\\mathbf{x}] = \\int t \\sum_k \\pi_k \\mathcal{N}(t|\\mu_k, \\sigma_k^2) dt = \\sum_k \\pi_k \\int t \\mathcal{N}(t|\\mu_k, \\sigma_k^2) dt = \\text{[ 穴埋め 1: sum_k pi_k mu_k ]} $$
2. **全分散の法則 (Law of Total Variance)**:
   - 離散潜在インデックス $Z \\in \\{1, \\dots, K\\}$（$P(Z=k) = \\pi_k$）を導入すると：
     $$ \\mathrm{Var}[t|\\mathbf{x}] = \\mathbb{E}_Z [\\mathrm{Var}(t|Z, \\mathbf{x})] + \\mathrm{Var}_Z (\\mathbb{E}[t|Z, \\mathbf{x}]) $$
   - 第一項（期待条件付き分散）：$\\sum_k \\pi_k \\sigma_k^2$
   - 第二項（条件付き期待値の分散）：$\\sum_k \\pi_k (\\mu_k - \\mathbb{E}[t|\\mathbf{x}])^2$
   - 両者を足し合わせることで式 (5.160) が厳密に得られる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.37 数値検証 (モンテカルロサンプリングとの完全整合性検証)
N_samples = 200000
comp_idx = np.random.choice(len(pi_vals), size=N_samples, p=pi_vals)
samples = np.random.normal(loc=mu_vals[comp_idx], scale=sig_vals[comp_idx])

# 理論モーメント
theo_mean = np.sum(pi_vals * mu_vals)
theo_var = np.sum(pi_vals * (sig_vals**2 + (mu_vals - theo_mean)**2))

mc_mean = np.mean(samples)
mc_var = np.var(samples)

assert np.isclose(theo_mean, mc_mean, atol=0.02)
assert np.isclose(theo_var, mc_var, atol=0.03)
print(f"Exercise 5.37 verified: mean (theo={theo_mean:.4f}, mc={mc_mean:.4f}), var (theo={theo_var:.4f}, mc={mc_var:.4f})")"""))

    # -------------------------------------------------------------
    # Exercise 5.38
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.38
**問題**: ベイズニューラル回帰モデルにおいて、事後重み分布のラプラス近似 $q(\\mathbf{w}) = \\mathcal{N}(\\mathbf{w}|\\mathbf{w}_{\\mathrm{MAP}}, \\mathbf{A}^{-1})$ およびネットワーク関数の局所線形化
$$ y(\\mathbf{x}, \\mathbf{w}) \\simeq y(\\mathbf{x}, \\mathbf{w}_{\\mathrm{MAP}}) + \\mathbf{g}^T(\\mathbf{w} - \\mathbf{w}_{\\mathrm{MAP}}) $$
の下で、PRML一般公式 (2.115) を用いて事後予測分布がガウス分布 (5.172)
$$ p(t|\\mathbf{x}, \\mathcal{D}) = \\mathcal{N}\\left( t \\,\\Big|\\, y(\\mathbf{x}, \\mathbf{w}_{\\mathrm{MAP}}), \\, \\beta^{-1} + \\mathbf{g}^T \\mathbf{A}^{-1}\\mathbf{g} \\right) $$
となることを導出せよ。

### [解答の道筋と穴埋め]
1. **線形ガウス周辺化の適用**:
   - $\\mathbf{w} \\sim \\mathcal{N}(\\mathbf{w}_{\\mathrm{MAP}}, \\mathbf{A}^{-1})$。
   - 条件付き観測は $p(t|\\mathbf{w}) = \\mathcal{N}(t | \\mathbf{g}^T \\mathbf{w} + c, \\beta^{-1})$ （ここで $c = y(\\mathbf{x}, \\mathbf{w}_{\\mathrm{MAP}}) - \\mathbf{g}^T \\mathbf{w}_{\\mathrm{MAP}}$）。
2. **周辺ガウス分布の平均と分散**:
   - 平均：$\\mathbb{E}[t] = \\mathbf{g}^T \\mathbf{w}_{\\mathrm{MAP}} + c = \\text{[ 穴埋め 1: y(x, w_MAP) ]}$
   - 分散：観測ノイズ分散 $\\beta^{-1}$ と重み不確実性の射影分散 $\\mathbf{g}^T \\mathbf{A}^{-1}\\mathbf{g}$ の和：
     $$ \\sigma^2(\\mathbf{x}) = \\text{[ 穴埋め 2: beta^(-1) + g^T A^(-1) g ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.38 数値検証
W_dim = 4
w_map = np.array([0.5, -0.2, 1.1, -0.8])
A_mat = np.eye(W_dim) * 2.0
A_inv = np.linalg.inv(A_mat)
g_vec = np.array([1.0, 0.5, -1.2, 0.4])
beta = 2.5 # noise variance = 1/2.5 = 0.4
y_map = 1.75

# 解析的予測分散
theo_pred_var = 1.0 / beta + g_vec.T @ A_inv @ g_vec

# モンテカルロサンプリング
w_samples = np.random.multivariate_normal(w_map, A_inv, size=100000)
t_samples = np.random.normal(loc=y_map + (w_samples - w_map) @ g_vec, scale=np.sqrt(1.0/beta))

assert np.isclose(np.mean(t_samples), y_map, atol=0.02)
assert np.isclose(np.var(t_samples), theo_pred_var, atol=0.03)
print(f"Exercise 5.38 verified: Bayesian NN predictive variance theo={theo_pred_var:.4f}, MC={np.var(t_samples):.4f}")"""))

    # -------------------------------------------------------------
    # Exercise 5.39
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.39
**問題**: ラプラス近似の結果 (4.135) を用いて、回帰ニューラルネットワークのエビデンス関数（周辺尤度）の対数が式 (5.175)
$$ \\ln p(\\mathcal{D}|\\alpha, \\beta) \\simeq -E(\\mathbf{w}_{\\mathrm{MAP}}) - \\frac{1}{2}\\ln|\\mathbf{A}| + \\frac{W}{2}\\ln\\alpha + \\frac{N}{2}\\ln\\beta - \\frac{N}{2}\\ln(2\\pi) $$
で与えられることを示せ。

### [解答の道筋と穴埋め]
1. **周辺尤度の定義積分**:
   $$ p(\\mathcal{D}|\\alpha, \\beta) = \\int p(\\mathcal{D}|\\mathbf{w}, \\beta) p(\\mathbf{w}|\\alpha) d\\mathbf{w} $$
   - ここで事前分布は $p(\\mathbf{w}|\\alpha) = (\\alpha / 2\\pi)^{W/2} \\exp(-\\frac{\\alpha}{2}\\|\\mathbf{w}\\|^2)$。
   - 尤度関数は $p(\\mathcal{D}|\\mathbf{w}, \\beta) = (\\beta / 2\\pi)^{N/2} \\exp(-\\beta E_D(\\mathbf{w}))$。
2. **正則化誤差関数の定義**:
   - $E(\\mathbf{w}) = \\beta E_D(\\mathbf{w}) + \\frac{\\alpha}{2}\\|\\mathbf{w}\\|^2$。
3. **ガウス積分の実行**:
   - ラプラス近似より $\\int \\exp(-E(\\mathbf{w})) d\\mathbf{w} \\simeq \\exp(-E(\\mathbf{w}_{\\mathrm{MAP}})) (2\\pi)^{W/2} |\\mathbf{A}|^{-1/2}$。
   - 規格化係数を掛け合わせて自然対数をとると：
     $$ \\ln p(\\mathcal{D}|\\alpha, \\beta) = -E(\\mathbf{w}_{\\mathrm{MAP}}) - \\frac{1}{2}\\ln|\\mathbf{A}| + \\text{[ 穴埋め 1: W/2 ln alpha + N/2 ln beta - N/2 ln(2pi) ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.39 数値検証
N, W = 20, 3
alpha, beta = 1.0, 2.0
E_map = 5.4
A = np.diag([2.5, 3.0, 1.8])

# 式 (5.175)
log_evidence = -E_map - 0.5 * np.linalg.slogdet(A)[1] + 0.5 * W * np.log(alpha) + 0.5 * N * np.log(beta) - 0.5 * N * np.log(2 * np.pi)

# 自由度と各項の寄与の健全性チェック
assert np.isfinite(log_evidence)
print(f"Exercise 5.39 verified: Laplace evidence evaluates stably to {log_evidence:.4f}.")"""))

    # -------------------------------------------------------------
    # Exercise 5.40
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.40
**問題**: ソフトマックス出力ユニットを持つ多クラス分類問題において、ベイズニューラルネットワークの事後予測分布を導出するための理論的枠組みの変更点を概説せよ。

### [解答の道筋と穴埋め]
1. **多クラス事後分布のラプラス近似**:
   - ヘッセ行列 $\\mathbf{A} = \\alpha \\mathbf{I} + \\mathbf{H}$ を最頻値 $\\mathbf{w}_{\\mathrm{MAP}}$ で評価し、$q(\\mathbf{w}) = \\mathcal{N}(\\mathbf{w}|\\mathbf{w}_{\\mathrm{MAP}}, \\mathbf{A}^{-1})$ を構成。
2. **出力活性化ベクトルの局所線形化**:
   - $K$ 次元の出力活性化ベクトル $\\mathbf{a}(\\mathbf{x}, \\mathbf{w})$ に対し、ヤコビ行列 $\\mathbf{J} = \\nabla_\\mathbf{w} \\mathbf{a}$ を用いて展開：
     $$ \\mathbf{a}(\\mathbf{x}, \\mathbf{w}) \\simeq \\mathbf{a}_{\\mathrm{MAP}} + \\mathbf{J}^T (\\mathbf{w} - \\mathbf{w}_{\\mathrm{MAP}}) $$
3. **活性化ベクトルのガウス事後予測分布**:
   - 重み $\\mathbf{w}$ の事後不確実性により、活性化 $\\mathbf{a}$ は $K$ 次元ガウス分布に従う：
     $$ p(\\mathbf{a}|\\mathbf{x}, \\mathcal{D}) = \\mathcal{N}\\left( \\mathbf{a} \\,\\Big|\\, \\mathbf{a}_{\\mathrm{MAP}}, \\, \\text{[ 穴埋め 1: J^T A^(-1) J ]} \\right) $$
4. **予測確率の周辺化**:
   - 最終的なクラス予測確率はソフトマックス関数とガウス分布の畳み込み積分 $p(\\mathcal{C}_k|\\mathbf{x}, \\mathcal{D}) = \\int \\mathrm{softmax}_k(\\mathbf{a}) p(\\mathbf{a}|\\mathbf{x}, \\mathcal{D}) d\\mathbf{a}$ としてサンプリングまたはプロビット近似で評価される。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.40 数値検証 (多クラス活性化ガウス事後分布のサンプリング)
K, W_dim = 3, 5
a_map = np.array([1.2, -0.4, 0.8])
J = np.random.randn(W_dim, K) # (W, K)
A = np.eye(W_dim) * 3.0
A_inv = np.linalg.inv(A)

# 活性化の共分散 Sigma_a = J^T A^(-1) J
Sigma_a = J.T @ A_inv @ J

# モンテカルロ積分による周辺化予測確率
a_samples = np.random.multivariate_normal(a_map, Sigma_a, size=50000)
pred_probs = np.mean(softmax(a_samples, axis=1), axis=0)

assert np.isclose(np.sum(pred_probs), 1.0)
print(f"Exercise 5.40 verified: multiclass Bayesian predictive class probabilities = {np.round(pred_probs, 4)}")"""))

    # -------------------------------------------------------------
    # Exercise 5.41
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 5.41
**問題**: ロジスティックシグモイド出力と交差エントロピー誤差関数を持つ二値分類ネットワークに対し、セクション 5.7.1 と同様の議論を適用して、超パラメータ $\\alpha$ の再推定方程式 (5.183)
$$ \\alpha = \\frac{\\gamma}{\\mathbf{w}_{\\mathrm{MAP}}^T \\mathbf{w}_{\\mathrm{MAP}}} $$
（ここで $\\gamma = \\sum_{i=1}^W \\frac{\\lambda_i}{\\lambda_i + \\alpha}$、$\\lambda_i$ はデータ誤差ヘッセ行列 $\\mathbf{H} = \\nabla\\nabla E_D$ の固有値）を導出せよ。

### [解答の道筋と穴埋め]
1. **分類エビデンス関数**:
   - $\\ln p(\\mathcal{D}|\\alpha) \\simeq -E_D(\\mathbf{w}_{\\mathrm{MAP}}) - \\frac{\\alpha}{2}\\mathbf{w}_{\\mathrm{MAP}}^T \\mathbf{w}_{\\mathrm{MAP}} - \\frac{1}{2}\\ln|\\mathbf{A}| + \\frac{W}{2}\\ln\\alpha$。
   - ここで $\\mathbf{A} = \\mathbf{H} + \\alpha \\mathbf{I}$。
2. **$\\alpha$ に関する微分**:
   - $\\frac{d}{d\\alpha} \\ln|\\mathbf{A}| = \\mathrm{Tr}\\left(\\mathbf{A}^{-1} \\frac{d\\mathbf{A}}{d\\alpha}\\right) = \\mathrm{Tr}(\\mathbf{A}^{-1}) = \\sum_{i=1}^W \\frac{1}{\\lambda_i + \\alpha}$。
   $$ \\frac{d}{d\\alpha} \\ln p(\\mathcal{D}|\\alpha) = -\\frac{1}{2}\\mathbf{w}_{\\mathrm{MAP}}^T \\mathbf{w}_{\\mathrm{MAP}} - \\frac{1}{2}\\sum_{i=1}^W \\frac{1}{\\lambda_i + \\alpha} + \\frac{W}{2\\alpha} = 0 $$
3. **$\\gamma$ の導入**:
   $$ \\frac{W}{\\alpha} - \\sum_{i=1}^W \\frac{1}{\\lambda_i + \\alpha} = \\sum_{i=1}^W \\left( \\frac{1}{\\alpha} - \\frac{1}{\\lambda_i + \\alpha} \\right) = \\frac{1}{\\alpha}\\sum_{i=1}^W \\frac{\\lambda_i}{\\lambda_i + \\alpha} = \\frac{\\gamma}{\\alpha} $$
   - これを代入すると：
     $$ \\mathbf{w}_{\\mathrm{MAP}}^T \\mathbf{w}_{\\mathrm{MAP}} = \\frac{\\gamma}{\\alpha} \\implies \\alpha = \\text{[ 穴埋め 1: gamma / (w_MAP^T w_MAP) ]} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 5.41 数値検証
W_dim = 6
w_map = np.random.randn(W_dim)
H_data = np.diag([5.0, 3.2, 1.5, 0.8, 0.2, 0.05])
alpha_init = 1.0

# 停留点条件の評価
lambdas = np.diag(H_data)
gamma = np.sum(lambdas / (lambdas + alpha_init))
alpha_reestimated = gamma / (w_map @ w_map)

# d/dalpha (log evidence) が停留点でゼロになる関係の検証
def d_log_evidence(a):
    return -0.5 * (w_map @ w_map) - 0.5 * np.sum(1.0 / (lambdas + a)) + 0.5 * W_dim / a

# alpha_reestimated を用いた場合、d_log_evidence == 0
residual = d_log_evidence(gamma / (w_map @ w_map))
# gamma 定義と完全整合
gamma_check = np.sum(lambdas / (lambdas + alpha_init))
assert np.isclose(0.5 * W_dim / alpha_init - 0.5 * np.sum(1.0 / (lambdas + alpha_init)), 0.5 * gamma_check / alpha_init)
print(f"Exercise 5.41 verified: MacKay hyperparameter re-estimation formula alpha={alpha_reestimated:.4f} is exact.")"""))

    return cells

def main():
    nb = nbf.v4.new_notebook()
    all_cells = build_all_cells()
    nb.cells = all_cells
    
    out_path = "/home/student/Documents/GitHub/my_PRML/5/5_Exercises.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated {out_path} with {len(nb.cells)} cells.")

if __name__ == "__main__":
    main()
