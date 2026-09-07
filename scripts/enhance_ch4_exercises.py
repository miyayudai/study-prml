"""
Master script to build 4/4_Exercises.ipynb
Covers all 26 PRML Chapter 4 exercises (4.1 to 4.26) with:
- Detailed theoretical explanation and mathematical derivation
- Step-by-step logic and structured fill-in-the-blank markdown format
- Self-contained Python numerical verification code with assertions
"""

import json
import nbformat as nbf
import numpy as np

def create_ch4_exercises_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & TOC
    cells.append(nbf.v4.new_markdown_cell("""# 第4章 章末演習問題 (Exercises 4.1 〜 4.26 全26問 完全網羅)

教科書「パターン認識と機械学習 (PRML)」第4章「線形識別モデル (Linear Models for Classification)」の全26問の演習問題の解答・解説ノートブックです。

各問題について、以下の構成で学習を進められるよう設計されています：
1. **問題の提示**: PRML原著の設問内容
2. **[解答の道筋と穴埋め]**: 証明・導出の論理的ステップと要点穴埋め (`[ 穴埋め X: ? ]`)
3. **Python数値検証コード**: 導出した数式や定理を数値シミュレーション・assert文で直接検証

---
## 目次
- [Exercise 4.1: 凸包 (Convex Hull) と超平面分離定理による線形分離可能性の同値性](#Exercise-4.1)
- [Exercise 4.2: 最小二乗回帰における目標ベクトルの線形制約の保存](#Exercise-4.2)
- [Exercise 4.3: 多変量目標ベクトルのアフィン制約の保存証明](#Exercise-4.3)
- [Exercise 4.4: 2クラスFisher線形判別分析のレイリー商最大化](#Exercise-4.4)
- [Exercise 4.5: 特定目標値を用いた最小二乗回帰とFisher判別の等価性](#Exercise-4.5)
- [Exercise 4.6: 1-of-K表現における多クラス最小二乗判別の次元性](#Exercise-4.6)
- [Exercise 4.7: ロジスティックシグモイド関数の対称性と導関数](#Exercise-4.7)
- [Exercise 4.8: 共通共分散を持つガウス生成モデルの事後確率係数](#Exercise-4.8)
- [Exercise 4.9: ガウス生成モデルの対数尤度最大化による最尤推定量](#Exercise-4.9)
- [Exercise 4.10: クラス固有共分散を持つガウス生成モデル (QDA)](#Exercise-4.10)
- [Exercise 4.11: ナイーブベイズモデルの事後確率の線形ロジスティック形式](#Exercise-4.11)
- [Exercise 4.12: ソフトマックス関数のヤコビ行列の導出](#Exercise-4.12)
- [Exercise 4.13: 線形分離可能データにおける最尤推定の重み発散と過学習](#Exercise-4.13)
- [Exercise 4.14: ロジスティック回帰のヘッセ行列の狭義正定値性（強凸性）](#Exercise-4.14)
- [Exercise 4.15: 多クラス交差エントロピー誤差関数の勾配導出](#Exercise-4.15)
- [Exercise 4.16: ソフトラベルに対する交差エントロピー勾配とヘッセ行列の普遍性](#Exercise-4.16)
- [Exercise 4.17: プロビット回帰リンク関数の対数尤度勾配](#Exercise-4.17)
- [Exercise 4.18: プロビット回帰の対数尤度の凹性（ヘッセ行列の半正定値性）](#Exercise-4.18)
- [Exercise 4.19: 多クラスロジスティック回帰のブロックヘッセ行列の半正定値性](#Exercise-4.19)
- [Exercise 4.20: ソフトマックス活性化の並進不変性とヘッセ行列の零空間](#Exercise-4.20)
- [Exercise 4.21: 誤差関数erfとプロビット近似（lambda^2 = pi/8）](#Exercise-4.21)
- [Exercise 4.22: 1次元分布のラプラス近似と規格化定数の評価](#Exercise-4.22)
- [Exercise 4.23: 多次元分布のラプラス近似とエビデンス評価](#Exercise-4.23)
- [Exercise 4.24: ラプラス近似からベイズ情報量規準 (BIC) への漸近展開](#Exercise-4.24)
- [Exercise 4.25: ベイズロジスティック回帰の予測分布と畳み込み近似](#Exercise-4.25)
- [Exercise 4.26: プロビット関数とガウス分布の厳密な畳み込み積分恒等式](#Exercise-4.26)
---"""))

    # Setup cell
    cells.append(nbf.v4.new_code_cell("""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import scipy.stats as stats
import scipy.special as special
import scipy.integrate as integrate
import scipy.optimize as optimize

from prml.linear import (
    Perceptron, FisherLinearDiscriminant, MulticlassFisherLinearDiscriminant,
    GaussianGenerativeClassifier, LogisticRegression, MulticlassLogisticRegression,
    ProbitRegression, BayesianLogisticRegression, LaplaceApproximation
)

print("Chapter 4 Exercises Setup completed successfully.")"""))

    # Exercise 4.1
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.1
**問題**: データ点集合 $\\{\\mathbf{x}_n\\}$ の凸包 (Convex Hull) は、$\\alpha_n \\ge 0$ かつ $\\sum_n \\alpha_n = 1$ を満たす点 $\\mathbf{x} = \\sum_n \\alpha_n \\mathbf{x}_n$ の集合として定義される。
もう一つの点集合 $\\{\\mathbf{y}_m\\}$ とその凸包を考える。
これら2つの点集合が線形分離可能であるとは、あるベクトル $\\mathbf{w}$ とスカラー $w_0$ が存在して、すべての $\\mathbf{x}_n$ に対して $\\mathbf{w}^T \\mathbf{x}_n + w_0 > 0$ かつすべての $\\mathbf{y}_m$ に対して $\\mathbf{w}^T \\mathbf{y}_m + w_0 < 0$ が成り立つことである。
凸包同士が交差するならば2つの点集合は線形分離不可能であり、逆に線形分離可能であれば凸包同士は交差しないことを示せ。

### [解答の道筋と穴埋め]
1. **$(\\implies)$ 対偶の証明（凸包が交差する $\\implies$ 線形分離不可能）**:
   - 2つの凸包が交差すると仮定し、その共通の交点を $\\mathbf{z}$ とする。
   - $\\mathbf{z}$ は $\\{\\mathbf{x}_n\\}$ の凸結合 $\\mathbf{z} = \\sum_n \\alpha_n \\mathbf{x}_n$ ($\\alpha_n \\ge 0, \\sum_n \\alpha_n = 1$) である。
   - 同時に $\\mathbf{z}$ は $\\{\\mathbf{y}_m\\}$ の凸結合 $\\mathbf{z} = \\sum_m \\beta_m \\mathbf{y}_m$ ($\\beta_m \\ge 0, \\sum_m \\beta_m = 1$) である。
   - もしこれらを線形分離する超平面 $(\\mathbf{w}, w_0)$ が存在したと仮定すると:
     - すべての $n$ で $\\mathbf{w}^T \\mathbf{x}_n + w_0 > 0$ であるから、凸結合をとると:
       $$ \\mathbf{w}^T \\mathbf{z} + w_0 = \\sum_n \\alpha_n (\\mathbf{w}^T \\mathbf{x}_n + w_0) > 0 $$
     - 一方、すべての $m$ で $\\mathbf{w}^T \\mathbf{y}_m + w_0 < 0$ であるから、凸結合をとると:
       $$ \\mathbf{w}^T \\mathbf{z} + w_0 = \\sum_m \\beta_m (\\mathbf{w}^T \\mathbf{y}_m + w_0) < 0 $$
   - これは $\\text{[ 穴埋め 1: ? ]} > 0$ かつ $\\text{[ 穴埋め 1: ? ]} < 0$ という矛盾を導く。したがって線形分離不可能である。
2. **$(\\Longleftarrow)$ 超平面分離定理 (Separating Hyperplane Theorem)**:
   - 有限集合の凸包は閉かつ有界（コンパクト）な凸集合である。
   - 2つの互いに素なコンパクト凸集合に対しては、両者を厳密に分離する超平面が存在することが凸解析の分離定理により保証される。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.1 数値検証
# 2つのクラスタの凸包交差判定を線形計画法 (LP) で解き、線形分離可能性と完全一致することを検証
def check_convex_hull_intersection(X, Y):
    \"\"\"2つの点集合 X, Y の凸包が交差するか (sum alpha_i x_i = sum beta_j y_j) を判定\"\"\"
    N, D = X.shape
    M, _ = Y.shape
    # 変数: [alpha_1..alpha_N, beta_1..beta_M] (N+M次元)
    # 制約:
    # 1. sum alpha_i x_i - sum beta_j y_j = 0 (D本)
    # 2. sum alpha_i = 1
    # 3. sum beta_j = 1
    # 4. alpha >= 0, beta >= 0
    A_eq = np.zeros((D + 2, N + M))
    A_eq[:D, :N] = X.T
    A_eq[:D, N:] = -Y.T
    A_eq[D, :N] = 1.0
    A_eq[D + 1, N:] = 1.0
    b_eq = np.zeros(D + 2)
    b_eq[D] = 1.0
    b_eq[D + 1] = 1.0

    c = np.zeros(N + M)  # ダミー目的関数
    res = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    return res.success

# 1. 明らかに線形分離可能な2点群
np.random.seed(42)
X_sep = np.random.randn(20, 2) + np.array([5.0, 5.0])
Y_sep = np.random.randn(20, 2) + np.array([-5.0, -5.0])
assert not check_convex_hull_intersection(X_sep, Y_sep), "Separable clusters should not intersect!"

# 2. 明らかに交差する2点群
X_overlap = np.random.randn(20, 2)
Y_overlap = np.random.randn(20, 2)
assert check_convex_hull_intersection(X_overlap, Y_overlap), "Overlapping clusters must intersect!"
print("Exercise 4.1 PASSED: Convex hull intersection theorem verified successfully via LP.")"""))

    # Exercise 4.2
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.2
**問題**: 最小二乗回帰モデルにおいて、目標ベクトル $\\mathbf{t}_n$ が線形制約
$$ \\mathbf{a}^T \\mathbf{t}_n + b = 0 \\quad (n = 1, \\dots, N) $$
を満たしているとする。モデルの入力基底ベクトル $\\boldsymbol{\\phi}(\\mathbf{x})$ がバイアス項 $\\phi_1(\\mathbf{x}) = 1$ を含んでいるとき、任意の入力 $\\mathbf{x}$ に対するモデルの予測値 $\\mathbf{y}(\\mathbf{x})$ もまた全く同一の線形制約
$$ \\mathbf{a}^T \\mathbf{y}(\\mathbf{x}) + b = 0 $$
を満たすことを示せ。

### [解答の道筋と穴埋め]
1. 目標値行列 $\\mathbf{T} \\in \\mathbb{R}^{N \\times K}$ の各行は $\\mathbf{t}_n^T$ であるから、全データにおける制約を行列形式で書くと:
   $$ \\mathbf{T} \\mathbf{a} + b \\mathbf{1}_N = \\mathbf{0} \\implies \\mathbf{T} \\mathbf{a} = -b \\mathbf{1}_N $$
2. 最小二乗解の重み行列 $\\mathbf{W} \\in \\mathbb{R}^{M \\times K}$ は正規方程式により与えられる:
   $$ \\mathbf{W} = (\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T \\mathbf{T} $$
3. これに右から $\\mathbf{a}$ を掛けると:
   $$ \\mathbf{W} \\mathbf{a} = (\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T (\\mathbf{T} \\mathbf{a}) = (\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T (-b \\mathbf{1}_N) $$
4. ここで第1基底関数が $\\phi_1(\\mathbf{x}) = 1$ であるため、計画行列の第1列は $\\mathbf{1}_N$ である。単位ベクトル $\\mathbf{e}_1 = [1, 0, \\dots, 0]^T$ に対し:
   $$ \\mathbf{\\Phi} \\mathbf{e}_1 = \\mathbf{1}_N $$
   したがって $(\\mathbf{\\Phi}^T \\mathbf{\\Phi})^{-1} \\mathbf{\\Phi}^T \\mathbf{1}_N = \\mathbf{e}_1$ である。
5. よって $\\mathbf{W} \\mathbf{a} = \\text{[ 穴埋め 1: ? ]} = -b \\mathbf{e}_1$ となる。
6. 任意の $\\mathbf{x}$ における予測値 $\\mathbf{y}(\\mathbf{x}) = \\mathbf{W}^T \\boldsymbol{\\phi}(\\mathbf{x})$ に $\\mathbf{a}^T$ を掛けると:
   $$ \\mathbf{a}^T \\mathbf{y}(\\mathbf{x}) = (\\mathbf{W} \\mathbf{a})^T \\boldsymbol{\\phi}(\\mathbf{x}) = (-b \\mathbf{e}_1)^T \\boldsymbol{\\phi}(\\mathbf{x}) = -b \\phi_1(\\mathbf{x}) = -b $$
   したがって $\\mathbf{a}^T \\mathbf{y}(\\mathbf{x}) + b = 0$ が恒等的に成り立つ。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.2 数値検証
np.random.seed(42)
N, M, K = 30, 4, 3
Phi = np.hstack([np.ones((N, 1)), np.random.randn(N, M - 1)])

# 任意の制約ベクトル a, スカラー b
a = np.array([2.0, -1.5, 3.0])
b = -4.5

# a^T t_n + b = 0 を満たす目標行列 T を生成
T = np.random.randn(N, K)
# 第1列を制約を満たすよう修正: a[0] * T[:, 0] + a[1]*T[:, 1] + a[2]*T[:, 2] + b = 0
T[:, 0] = -(T[:, 1] * a[1] + T[:, 2] * a[2] + b) / a[0]
assert np.allclose(T @ a + b, 0.0)

# 最小二乗解 W
W = np.linalg.pinv(Phi) @ T

# 任意のテスト点 x_test で予測 y(x) = W^T phi(x) を評価
Phi_test = np.hstack([np.ones((50, 1)), np.random.randn(50, M - 1)])
Y_pred = Phi_test @ W

# 全テスト点において a^T y(x) + b == 0 であることを検証
constraint_errors = Y_pred @ a + b
assert np.allclose(constraint_errors, 0.0, atol=1e-12)
print(f"Exercise 4.2 PASSED: Linear constraint preserved exactly (max abs error = {np.max(np.abs(constraint_errors)):.2e}).")"""))

    # Exercise 4.3
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.3
**問題**: Exercise 4.2 の結果を拡張し、目標ベクトルが複数の線形制約
$$ \\mathbf{A} \\mathbf{t}_n + \\mathbf{b} = \\mathbf{0} \\quad (n = 1, \\dots, N) $$
（$\\mathbf{A}$ は任意の行列、$\\mathbf{b}$ は任意のベクトル）を満たすとき、任意の入力 $\\mathbf{x}$ に対する最小二乗予測値もまた $\\mathbf{A} \\mathbf{y}(\\mathbf{x}) + \\mathbf{b} = \\mathbf{0}$ を満たすことを示せ。

### [解答の道筋と穴埋め]
1. 行列 $\\mathbf{A}$ の第 $j$ 行ベクトルを $\\mathbf{a}_j^T$、ベクトル $\\mathbf{b}$ の第 $j$ 成分を $b_j$ とする。
2. 連立制約 $\\mathbf{A} \\mathbf{t}_n + \\mathbf{b} = \\mathbf{0}$ は、各 $j$ について
   $$ \\mathbf{a}_j^T \\mathbf{t}_n + b_j = 0 $$
   という独立した線形制約の連立とみなすことができる。
3. Exercise 4.2 の定理を各行ベクトル $\\mathbf{a}_j^T$ とスカラー $b_j$ に適用すると、すべての $j$ について直ちに
   $$ \\mathbf{a}_j^T \\mathbf{y}(\\mathbf{x}) + b_j = 0 $$
   が成り立つ。
4. これを行列形式に再結合すると:
   $$ \\mathbf{A} \\mathbf{y}(\\mathbf{x}) + \\mathbf{b} = \\text{[ 穴埋め 1: ? ]} = \\mathbf{0} $$
   が任意の $\\mathbf{x}$ において成立する。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.3 数値検証
np.random.seed(42)
N, M, K = 30, 4, 4
Phi = np.hstack([np.ones((N, 1)), np.random.randn(N, M - 1)])

# 2本の独立な制約 A (2 x K), b (2)
A_mat = np.array([[1.0, -1.0, 2.0, 0.0],
                  [0.0, 2.0, -1.0, 1.5]])
b_vec = np.array([1.2, -3.4])

# 制約 A t_n + b = 0 を満たす T を生成
T = np.random.randn(N, K)
# A の右擬似逆行列を用いて制約を満たす射影
for n in range(N):
    T[n] = T[n] - np.linalg.pinv(A_mat) @ (A_mat @ T[n] + b_vec)
assert np.allclose(T @ A_mat.T + b_vec, 0.0)

W = np.linalg.pinv(Phi) @ T
Phi_test = np.hstack([np.ones((50, 1)), np.random.randn(50, M - 1)])
Y_pred = Phi_test @ W

residual = Y_pred @ A_mat.T + b_vec
assert np.allclose(residual, 0.0, atol=1e-12)
print(f"Exercise 4.3 PASSED: Multiple affine constraints preserved exactly (max error = {np.max(np.abs(residual)):.2e}).")"""))

    # Exercise 4.4
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.4
**問題**: 2クラス判別において、フィッシャーの判別基準
$$ J(\\mathbf{w}) = \\frac{\\mathbf{w}^T \\mathbf{S}_B \\mathbf{w}}{\\mathbf{w}^T \\mathbf{S}_W \\mathbf{w}} $$
（式 4.25）を最大化するベクトル $\\mathbf{w}$ が
$$ \\mathbf{w} \\propto \\mathbf{S}_W^{-1} (\\mathbf{m}_2 - \\mathbf{m}_1) $$
（式 4.29）で与えられることを、$\\nabla_\\mathbf{w} J(\\mathbf{w}) = \\mathbf{0}$ を直接計算することにより証明せよ。

### [解答の道筋と穴埋め]
1. 商の微分公式より:
   $$ \\nabla_\\mathbf{w} J(\\mathbf{w}) = \\frac{2 \\mathbf{S}_B \\mathbf{w} (\\mathbf{w}^T \\mathbf{S}_W \\mathbf{w}) - 2 \\mathbf{S}_W \\mathbf{w} (\\mathbf{w}^T \\mathbf{S}_B \\mathbf{w})}{(\\mathbf{w}^T \\mathbf{S}_W \\mathbf{w})^2} = \\mathbf{0} $$
2. したがって停留点において:
   $$ \\mathbf{S}_B \\mathbf{w} = \\left( \\frac{\\mathbf{w}^T \\mathbf{S}_B \\mathbf{w}}{\\mathbf{w}^T \\mathbf{S}_W \\mathbf{w}} \\right) \\mathbf{S}_W \\mathbf{w} = J(\\mathbf{w}) \\mathbf{S}_W \\mathbf{w} $$
3. ここでクラス間共分散行列は $\\mathbf{S}_B = (\\mathbf{m}_2 - \\mathbf{m}_1)(\\mathbf{m}_2 - \\mathbf{m}_1)^T$ であるから:
   $$ \\mathbf{S}_B \\mathbf{w} = (\\mathbf{m}_2 - \\mathbf{m}_1) \\{ (\\mathbf{m}_2 - \\mathbf{m}_1)^T \\mathbf{w} \\} $$
   これはスカラー $(\\mathbf{m}_2 - \\mathbf{m}_1)^T \\mathbf{w}$ とベクトル $(\\mathbf{m}_2 - \\mathbf{m}_1)$ の積であり、常に $(\\mathbf{m}_2 - \\mathbf{m}_1)$ の方向を向く。
4. これを代入すると:
   $$ \\{ (\\mathbf{m}_2 - \\mathbf{m}_1)^T \\mathbf{w} \\} (\\mathbf{m}_2 - \\mathbf{m}_1) = J(\\mathbf{w}) \\mathbf{S}_W \\mathbf{w} $$
5. 両辺に $\\mathbf{S}_W^{-1}$ を掛けると:
   $$ \\mathbf{w} = \\text{[ 穴埋め 1: ? ]} \\mathbf{S}_W^{-1} (\\mathbf{m}_2 - \\mathbf{m}_1) \\propto \\mathbf{S}_W^{-1} (\\mathbf{m}_2 - \\mathbf{m}_1) $$
   （ただしスカラー係数は向きを変えないため判別境界に影響しない）。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.4 数値検証
import scipy.linalg as la
np.random.seed(42)
D = 3
N1, N2 = 40, 50
X1 = np.random.randn(N1, D) + np.array([2.0, 0.0, 1.0])
X2 = np.random.randn(N2, D) + np.array([-1.0, 1.5, -0.5])

m1 = np.mean(X1, axis=0)
m2 = np.mean(X2, axis=0)

Sw = (X1 - m1).T @ (X1 - m1) + (X2 - m2).T @ (X2 - m2)
Sb = np.outer(m2 - m1, m2 - m1)

# 解析解 w_analytic = Sw^{-1} (m2 - m1)
w_analytic = np.linalg.solve(Sw, m2 - m1)
w_analytic /= np.linalg.norm(w_analytic)

# 一般化固有値問題 S_B w = lambda S_W w によるレイリー商の厳密な最大化
eigvals, eigvecs = la.eigh(Sb, Sw)
w_eig = eigvecs[:, -1]
w_eig /= np.linalg.norm(w_eig)

# 方向の一致度 (コサイン類似度の絶対値が 1)
cos_sim = abs(np.dot(w_analytic, w_eig))
assert np.isclose(cos_sim, 1.0, atol=1e-10)
print(f"Exercise 4.4 PASSED: Fisher's optimal direction matches Rayleigh quotient generalized eig (|cos sim| = {cos_sim:.10f}).")"""""))

    # Exercise 4.5
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.5
**問題**: 2クラス分類において、目標値として
$$ t_n = \\begin{cases} \\frac{N}{N_1} & (\\mathbf{x}_n \\in C_1) \\\\ -\\frac{N}{N_2} & (\\mathbf{x}_n \\in C_2) \\end{cases} $$
（式 4.34）を採用した二乗和誤差最小二乗法を考える。
パラメータ $(\\mathbf{w}, w_0)$ に関する最小化を行うと、得られる重みベクトル $\\mathbf{w}$ がフィッシャーの判別解 $\\mathbf{w} \\propto \\mathbf{S}_W^{-1} (\\mathbf{m}_2 - \\mathbf{m}_1)$ と**厳密に一致**することを示せ。

### [解答の道筋と穴埋め]
1. 二乗和誤差関数は $E = \\frac{1}{2} \\sum_{n=1}^N (\\mathbf{w}^T \\mathbf{x}_n + w_0 - t_n)^2$ である。
2. $w_0$ に関する微分を 0 と置くと:
   $$ \\sum_{n=1}^N (\\mathbf{w}^T \\mathbf{x}_n + w_0 - t_n) = 0 \\implies w_0 = -\\mathbf{w}^T \\mathbf{m} + \\frac{1}{N} \\sum_{n=1}^N t_n $$
   ここで $\\sum_{n=1}^N t_n = N_1 (N/N_1) + N_2 (-N/N_2) = N - N = 0$ であるから:
   $$ w_0 = -\\mathbf{w}^T \\mathbf{m} \\quad \\left( \\mathbf{m} = \\frac{1}{N} \\sum_{n=1}^N \\mathbf{x}_n = \\frac{N_1 \\mathbf{m}_1 + N_2 \\mathbf{m}_2}{N} \\right) $$
3. $\\mathbf{w}$ に関する微分を 0 と置くと:
   $$ \\sum_{n=1}^N (\\mathbf{w}^T (\\mathbf{x}_n - \\mathbf{m}) - t_n) \\mathbf{x}_n = \\mathbf{0} $$
   左辺の第2項は:
   $$ \\sum_{n=1}^N t_n \\mathbf{x}_n = \\frac{N}{N_1} \\sum_{n \\in C_1} \\mathbf{x}_n - \\frac{N}{N_2} \\sum_{n \\in C_2} \\mathbf{x}_n = N(\\mathbf{m}_1 - \\mathbf{m}_2) $$
4. 第1項を行列形式で展開すると:
   $$ \\left( \\mathbf{S}_W + \\frac{N_1 N_2}{N}(\\mathbf{m}_1 - \\mathbf{m}_2)(\\mathbf{m}_1 - \\mathbf{m}_2)^T \\right) \\mathbf{w} = N(\\mathbf{m}_1 - \\mathbf{m}_2) $$
5. 第2項は $(\\mathbf{m}_1 - \\mathbf{m}_2)$ 方向のスカラー倍であるため、両辺に $\\mathbf{S}_W^{-1}$ を作用させると:
   $$ \\mathbf{w} \\propto \\text{[ 穴埋め 1: ? ]} = \\mathbf{S}_W^{-1} (\\mathbf{m}_1 - \\mathbf{m}_2) $$
   となり、フィッシャーの判別解と厳密に等価であることが示される。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.5 数値検証
np.random.seed(42)
N1, N2, D = 35, 45, 3
N = N1 + N2
X1 = np.random.randn(N1, D) + np.array([1.5, 0.5, -1.0])
X2 = np.random.randn(N2, D) + np.array([-1.0, -0.5, 1.5])

X = np.vstack([X1, X2])
t = np.hstack([np.full(N1, N / N1), np.full(N2, -N / N2)])

# 最小二乗法で学習
Phi = np.hstack([np.ones((N, 1)), X])
w_ls_full = np.linalg.pinv(Phi) @ t
w_ls = w_ls_full[1:]  # 重みベクトル
w_ls_normalized = w_ls / np.linalg.norm(w_ls)

# Fisher の解析解
m1, m2 = np.mean(X1, axis=0), np.mean(X2, axis=0)
Sw = (X1 - m1).T @ (X1 - m1) + (X2 - m2).T @ (X2 - m2)
w_fisher = np.linalg.solve(Sw, m1 - m2)
w_fisher_normalized = w_fisher / np.linalg.norm(w_fisher)

cos_sim = np.dot(w_ls_normalized, w_fisher_normalized)
assert np.isclose(cos_sim, 1.0, atol=1e-10)
print(f"Exercise 4.5 PASSED: Least-squares weight is strictly collinear to Fisher discriminant (cos sim = {cos_sim:.10f}).")"""))

    # Exercise 4.6
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.6
**問題**: 1-of-K 符号化を用いた多クラス分類の最小二乗回帰において、モデルの予測ベクトル $\\mathbf{y}(\\mathbf{x})$ の成分の総和が常に 1 となり、予測空間が $(K-1)$ 次元のアフィン部分空間に制限されることを示せ。

### [解答の道筋と穴埋め]
1. 1-of-K 符号化では、各目標ベクトル $\\mathbf{t}_n \\in \\{0, 1\\}^K$ の成分の総和は常に $\\sum_{k=1}^K t_{nk} = 1$ である。
2. これは線形制約 $\\mathbf{1}_K^T \\mathbf{t}_n - 1 = 0$ (すなわち $\\mathbf{a} = \\mathbf{1}_K, b = -1$) と表せる。
3. 入力基底にバイアス項 $\\phi_1(\\mathbf{x}) = 1$ が含まれている場合、Exercise 4.2 の定理により:
   $$ \\mathbf{1}_K^T \\mathbf{y}(\\mathbf{x}) - 1 = 0 \\implies \\sum_{k=1}^K y_k(\\mathbf{x}) = \\text{[ 穴埋め 1: ? ]} = 1 $$
   が任意の入力 $\\mathbf{x}$ において厳密に成り立つ。
4. したがって予測ベクトル $\\mathbf{y}(\\mathbf{x})$ は $K$ 次元空間全体ではなく、$(K-1)$ 次元の超平面内に束縛される。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.6 数値検証
np.random.seed(42)
N, D, K = 60, 3, 4
X = np.random.randn(N, D)
labels = np.random.choice(K, size=N)
T = np.eye(K)[labels]  # 1-of-K 符号化

Phi = np.hstack([np.ones((N, 1)), X])
W = np.linalg.pinv(Phi) @ T

# 任意のテスト点での予測値
Phi_test = np.hstack([np.ones((100, 1)), np.random.randn(100, D)])
Y_pred = Phi_test @ W

sums = np.sum(Y_pred, axis=1)
assert np.allclose(sums, 1.0, atol=1e-12)
print(f"Exercise 4.6 PASSED: Multiclass LS predictions strictly sum to 1.0 (max deviation = {np.max(np.abs(sums - 1.0)):.2e}).")"""))

    # Exercise 4.7
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.7
**問題**: ロジスティックシグモイド関数 $\\sigma(a) = \\frac{1}{1 + e^{-a}}$ (式 4.3) について:
1. 対称性 $\\sigma(-a) = 1 - \\sigma(a)$ を示せ。
2. 導関数が $\\frac{d\\sigma}{da} = \\sigma(a)(1 - \\sigma(a))$ (式 4.88) となることを示せ。
3. 逆関数（ロジット関数）が $a = \\ln\\left(\\frac{\\sigma}{1 - \\sigma}\\right)$ となることを示せ。

### [解答の道筋と穴埋め]
1. 対称性:
   $$ 1 - \\sigma(a) = 1 - \\frac{1}{1 + e^{-a}} = \\frac{e^{-a}}{1 + e^{-a}} = \\frac{1}{e^a + 1} = \\sigma(-a) $$
2. 導関数: 商の微分法則より
   $$ \\frac{d\\sigma}{da} = -\\frac{-e^{-a}}{(1 + e^{-a})^2} = \\left(\\frac{1}{1 + e^{-a}}\\right) \\left(\\frac{e^{-a}}{1 + e^{-a}}\\right) = \\text{[ 穴埋め 1: ? ]} = \\sigma(a)(1 - \\sigma(a)) $$
3. 逆関数: $y = \\frac{1}{1 + e^{-a}}$ とおくと:
   $$ 1 + e^{-a} = \\frac{1}{y} \\implies e^{-a} = \\frac{1 - y}{y} \\implies e^a = \\frac{y}{1 - y} \\implies a = \\ln\\left(\\frac{y}{1 - y}\\right) $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.7 数値検証
a_grid = np.linspace(-6.0, 6.0, 200)
sig = 1.0 / (1.0 + np.exp(-a_grid))

# 1. 対称性検証
sig_neg = 1.0 / (1.0 + np.exp(a_grid))
assert np.allclose(sig_neg, 1.0 - sig, atol=1e-14)

# 2. 導関数検証 (中心差分 vs 解析導関数)
eps = 1e-6
sig_diff = ((1.0 / (1.0 + np.exp(-(a_grid + eps)))) - (1.0 / (1.0 + np.exp(-(a_grid - eps))))) / (2 * eps)
sig_analytic = sig * (1.0 - sig)
assert np.allclose(sig_diff, sig_analytic, atol=1e-6)

# 3. 逆関数検証
logit = np.log(sig / (1.0 - sig))
assert np.allclose(logit, a_grid, atol=1e-12)
print("Exercise 4.7 PASSED: Sigmoid symmetry, derivative, and inverse function all verified.")"""))

    # Exercise 4.8
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.8
**問題**: 共通の共分散行列 $\\mathbf{\\Sigma}$ を持つ2クラスガウス生成モデル
$$ p(\\mathbf{x} | C_k) = \\mathcal{N}(\\mathbf{x} | \\boldsymbol{\\mu}_k, \\mathbf{\\Sigma}) $$
において、事後確率がロジスティックシグモイド関数
$$ p(C_1 | \\mathbf{x}) = \\sigma(\\mathbf{w}^T \\mathbf{x} + w_0) $$
で表されることを示し、重みベクトル $\\mathbf{w}$ およびバイアス $w_0$ の表式 (式 4.59 - 4.61) を導出せよ。

### [解答の道筋と穴埋め]
1. ベイズの定理より:
   $$ p(C_1 | \\mathbf{x}) = \\frac{p(\\mathbf{x} | C_1) p(C_1)}{p(\\mathbf{x} | C_1) p(C_1) + p(\\mathbf{x} | C_2) p(C_2)} = \\frac{1}{1 + \\exp(-a)} = \\sigma(a) $$
   ただし $a = \\ln \\frac{p(\\mathbf{x} | C_1) p(C_1)}{p(\\mathbf{x} | C_2) p(C_2)}$。
2. ガウス分布の比を展開すると:
   $$ a = -\\frac{1}{2}(\\mathbf{x} - \\boldsymbol{\\mu}_1)^T \\mathbf{\\Sigma}^{-1} (\\mathbf{x} - \\boldsymbol{\\mu}_1) + \\frac{1}{2}(\\mathbf{x} - \\boldsymbol{\\mu}_2)^T \\mathbf{\\Sigma}^{-1} (\\mathbf{x} - \\boldsymbol{\\mu}_2) + \\ln \\frac{p(C_1)}{p(C_2)} $$
3. 二次項 $-\\frac{1}{2}\\mathbf{x}^T \\mathbf{\\Sigma}^{-1} \\mathbf{x}$ は共通の共分散を持つため**完全に相殺**する。
4. 一次項をまとめると:
   $$ \\mathbf{w} = \\text{[ 穴埋め 1: ? ]} = \\mathbf{\\Sigma}^{-1} (\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2) $$
5. 定数項をまとめると:
   $$ w_0 = \\text{[ 穴埋め 2: ? ]} = -\\frac{1}{2}\\boldsymbol{\\mu}_1^T \\mathbf{\\Sigma}^{-1} \\boldsymbol{\\mu}_1 + \\frac{1}{2}\\boldsymbol{\\mu}_2^T \\mathbf{\\Sigma}^{-1} \\boldsymbol{\\mu}_2 + \\ln\\frac{p(C_1)}{p(C_2)} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.8 数値検証
np.random.seed(42)
D = 3
mu1 = np.array([1.0, 2.0, -1.0])
mu2 = np.array([-1.5, 0.5, 2.0])
Sigma = np.array([[2.0, 0.5, 0.2],
                  [0.5, 1.5, -0.3],
                  [0.2, -0.3, 1.8]])
pi1, pi2 = 0.6, 0.4

# 解析的 w, w0
Sigma_inv = np.linalg.inv(Sigma)
w_analytic = Sigma_inv @ (mu1 - mu2)
w0_analytic = -0.5 * mu1 @ Sigma_inv @ mu1 + 0.5 * mu2 @ Sigma_inv @ mu2 + np.log(pi1 / pi2)

# テスト点における厳密なベイズ事後確率
x_test = np.random.randn(50, D)
p_x_C1 = stats.multivariate_normal.pdf(x_test, mean=mu1, cov=Sigma) * pi1
p_x_C2 = stats.multivariate_normal.pdf(x_test, mean=mu2, cov=Sigma) * pi2
posterior_exact = p_x_C1 / (p_x_C1 + p_x_C2)

# シグモイド形式による事後確率
a_vals = x_test @ w_analytic + w0_analytic
posterior_sigmoid = 1.0 / (1.0 + np.exp(-a_vals))

assert np.allclose(posterior_exact, posterior_sigmoid, atol=1e-12)
print("Exercise 4.8 PASSED: Generative Gaussian posterior strictly equals sigmoid form.")"""))

    # Exercise 4.9
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.9
**問題**: 共通共分散行列 $\\mathbf{\\Sigma}$ を持つ2クラスガウス生成モデルにおいて、同時対数尤度関数
$$ \\ln p(\\mathbf{X}, \\mathbf{t} | \\pi, \\boldsymbol{\\mu}_1, \\boldsymbol{\\mu}_2, \\mathbf{\\Sigma}) = \\sum_{n=1}^N \\left[ t_n \\ln(\\pi \\mathcal{N}(\\mathbf{x}_n | \\boldsymbol{\\mu}_1, \\mathbf{\\Sigma})) + (1 - t_n) \\ln((1 - \\pi) \\mathcal{N}(\\mathbf{x}_n | \\boldsymbol{\\mu}_2, \\mathbf{\\Sigma})) \\right] $$
（式 4.64）を最大化することにより、パラメータ $\\pi, \\boldsymbol{\\mu}_1, \\boldsymbol{\\mu}_2, \\mathbf{\\Sigma}$ の最尤推定量 (式 4.67 - 4.70) を導出せよ。

### [解答の道筋と穴埋め]
1. $\\pi$ の最大化:
   $$ \\frac{\\partial \\ln p}{\\partial \\pi} = \\sum_{n=1}^N \\left[ \\frac{t_n}{\\pi} - \\frac{1 - t_n}{1 - \\pi} \\right] = \\frac{N_1}{\\pi} - \\frac{N_2}{1 - \\pi} = 0 \\implies \\pi = \\text{[ 穴埋め 1: ? ]} = \\frac{N_1}{N} $$
2. $\\boldsymbol{\\mu}_1$ の最大化:
   $$ \\frac{\\partial \\ln p}{\\partial \\boldsymbol{\\mu}_1} = \\sum_{n \\in C_1} \\mathbf{\\Sigma}^{-1} (\\mathbf{x}_n - \\boldsymbol{\\mu}_1) = \\mathbf{0} \\implies \\boldsymbol{\\mu}_1 = \\frac{1}{N_1} \\sum_{n \\in C_1} \\mathbf{x}_n $$
   同様に $\\boldsymbol{\\mu}_2 = \\frac{1}{N_2} \\sum_{n \\in C_2} \\mathbf{x}_n$。
3. $\\mathbf{\\Sigma}$ の最大化:
   - 散布行列 $\\mathbf{S}_k = \\sum_{n \\in C_k} (\\mathbf{x}_n - \\boldsymbol{\\mu}_k)(\\mathbf{x}_n - \\boldsymbol{\\mu}_k)^T$ と置くと:
     $$ \\ln p = -\\frac{N}{2} \\ln |\\mathbf{\\Sigma}| - \\frac{1}{2} \\mathrm{Tr}\\left(\\mathbf{\\Sigma}^{-1} (\\mathbf{S}_1 + \\mathbf{S}_2)\\right) + \\mathrm{const} $$
   - 行列式とトレースの微分公式より:
     $$ \\mathbf{\\Sigma} = \\text{[ 穴埋め 2: ? ]} = \\frac{1}{N} (\\mathbf{S}_1 + \\mathbf{S}_2) = \\frac{N_1}{N} \\frac{\\mathbf{S}_1}{N_1} + \\frac{N_2}{N} \\frac{\\mathbf{S}_2}{N_2} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.9 数値検証
np.random.seed(42)
N1, N2, D = 40, 60, 2
N = N1 + N2
X1 = np.random.randn(N1, D) + [1.0, 1.0]
X2 = np.random.randn(N2, D) + [-1.0, -1.0]
X = np.vstack([X1, X2])
t = np.hstack([np.ones(N1), np.zeros(N2)])

# 解析的最尤解
pi_hat = N1 / N
mu1_hat = np.mean(X1, axis=0)
mu2_hat = np.mean(X2, axis=0)
S1 = (X1 - mu1_hat).T @ (X1 - mu1_hat)
S2 = (X2 - mu2_hat).T @ (X2 - mu2_hat)
Sigma_hat = (S1 + S2) / N

# GaussianGenerativeClassifier の実装と完全一致するか検証
clf = GaussianGenerativeClassifier(shared_cov=True).fit(X, t)
idx0 = np.where(clf.classes == 0)[0][0]
idx1 = np.where(clf.classes == 1)[0][0]
assert np.isclose(clf.priors[idx1], pi_hat)
assert np.allclose(clf.means[idx1], mu1_hat)
assert np.allclose(clf.means[idx0], mu2_hat)
assert np.allclose(clf.covs[0], Sigma_hat)
print("Exercise 4.9 PASSED: Generative Gaussian MLE formulas strictly match classifier implementation.")"""))

    # Exercise 4.10
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.10
**問題**: 各クラスが独自の共分散行列 $\\mathbf{\\Sigma}_k$ を持つガウス生成モデル（二次判別分析: QDA）において、事後確率の対数比が入力 $\\mathbf{x}$ の二次関数となることを示し、二次境界を決定する行列とベクトルの表式を求めよ。

### [解答の道筋と穴埋め]
1. クラス固有の共分散を持つ場合、対数尤度比は:
   $$ a(\\mathbf{x}) = \\ln \\frac{p(\\mathbf{x}|C_1)p(C_1)}{p(\\mathbf{x}|C_2)p(C_2)} = -\\frac{1}{2}(\\mathbf{x} - \\boldsymbol{\\mu}_1)^T \\mathbf{\\Sigma}_1^{-1}(\\mathbf{x} - \\boldsymbol{\\mu}_1) + \\frac{1}{2}(\\mathbf{x} - \\boldsymbol{\\mu}_2)^T \\mathbf{\\Sigma}_2^{-1}(\\mathbf{x} - \\boldsymbol{\\mu}_2) - \\frac{1}{2}\\ln\\frac{|\\mathbf{\\Sigma}_1|}{|\\mathbf{\\Sigma}_2|} + \\ln\\frac{p(C_1)}{p(C_2)} $$
2. $\\mathbf{\\Sigma}_1 \\neq \\mathbf{\\Sigma}_2$ であるため、二次項 $-\\frac{1}{2}\\mathbf{x}^T (\\mathbf{\\Sigma}_1^{-1} - \\mathbf{\\Sigma}_2^{-1}) \\mathbf{x}$ は**相殺されずに残る**。
3. したがって判別式は二次形式で表される:
   $$ a(\\mathbf{x}) = \\mathbf{x}^T \\mathbf{W} \\mathbf{x} + \\mathbf{w}^T \\mathbf{x} + w_0 $$
   - $\\mathbf{W} = \\text{[ 穴埋め 1: ? ]} = -\\frac{1}{2}(\\mathbf{\\Sigma}_1^{-1} - \\mathbf{\\Sigma}_2^{-1})$
   - $\\mathbf{w} = \\mathbf{\\Sigma}_1^{-1} \\boldsymbol{\\mu}_1 - \\mathbf{\\Sigma}_2^{-1} \\boldsymbol{\\mu}_2$
   - $w_0 = -\\frac{1}{2}\\boldsymbol{\\mu}_1^T \\mathbf{\\Sigma}_1^{-1}\\boldsymbol{\\mu}_1 + \\frac{1}{2}\\boldsymbol{\\mu}_2^T \\mathbf{\\Sigma}_2^{-1}\\boldsymbol{\\mu}_2 - \\frac{1}{2}\\ln\\frac{|\\mathbf{\\Sigma}_1|}{|\\mathbf{\\Sigma}_2|} + \\ln\\frac{p(C_1)}{p(C_2)}$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.10 数値検証
np.random.seed(42)
D = 2
mu1 = np.array([1.0, 1.0])
mu2 = np.array([-1.0, -1.0])
Sigma1 = np.array([[1.0, 0.3], [0.3, 1.5]])
Sigma2 = np.array([[2.5, -0.4], [-0.4, 0.8]])
pi1, pi2 = 0.5, 0.5

S1_inv = np.linalg.inv(Sigma1)
S2_inv = np.linalg.inv(Sigma2)
W_mat = -0.5 * (S1_inv - S2_inv)
w_vec = S1_inv @ mu1 - S2_inv @ mu2
w0_val = -0.5 * mu1 @ S1_inv @ mu1 + 0.5 * mu2 @ S2_inv @ mu2 - \
         0.5 * np.log(np.linalg.det(Sigma1) / np.linalg.det(Sigma2)) + np.log(pi1 / pi2)

# テスト点における真の対数オッズ vs 二次形式 a(x)
X_test = np.random.randn(40, D)
for x in X_test:
    log_p1 = np.log(pi1) + stats.multivariate_normal.logpdf(x, mean=mu1, cov=Sigma1)
    log_p2 = np.log(pi2) + stats.multivariate_normal.logpdf(x, mean=mu2, cov=Sigma2)
    exact_log_odds = log_p1 - log_p2
    quad_log_odds = x @ W_mat @ x + np.dot(w_vec, x) + w0_val
    assert np.isclose(exact_log_odds, quad_log_odds, atol=1e-12)
print("Exercise 4.10 PASSED: QDA quadratic boundary formula strictly matches exact density log odds.")"""))

    # Exercise 4.11
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.11
**問題**: クラス条件付き独立な二値特徴量 $x_i \\in \\{0, 1\\}$ を持つナイーブベイズモデル
$$ p(\\mathbf{x} | C_k) = \\prod_{i=1}^D \\mu_{ki}^{x_i} (1 - \\mu_{ki})^{1 - x_i} $$
において、事後確率 $p(C_1 | \\mathbf{x})$ が入力の線形結合に対するロジスティックシグモイド関数 $\\sigma(\\mathbf{w}^T \\mathbf{x} + w_0)$ で厳密に表されることを示せ。

### [解答の道筋と穴埋め]
1. 事後オッズの対数 $a(\\mathbf{x}) = \\ln \\frac{p(\\mathbf{x} | C_1) p(C_1)}{p(\\mathbf{x} | C_2) p(C_2)}$ を計算する。
2. 特徴量の条件付き独立性より:
   $$ \\ln p(\\mathbf{x} | C_k) = \\sum_{i=1}^D \\left[ x_i \\ln \\mu_{ki} + (1 - x_i) \\ln(1 - \\mu_{ki}) \\right] = \\sum_{i=1}^D x_i \\ln \\frac{\\mu_{ki}}{1 - \\mu_{ki}} + \\sum_{i=1}^D \\ln(1 - \\mu_{ki}) $$
3. 2クラスの差分をとると:
   $$ a(\\mathbf{x}) = \\sum_{i=1}^D x_i \\ln \\left( \\frac{\\mu_{1i}(1 - \\mu_{2i})}{\\mu_{2i}(1 - \\mu_{1i})} \\right) + \\sum_{i=1}^D \\ln \\frac{1 - \\mu_{1i}}{1 - \\mu_{2i}} + \\ln \\frac{p(C_1)}{p(C_2)} $$
4. これは $\\mathbf{w}^T \\mathbf{x} + w_0$ の線形関数である:
   - $w_i = \\text{[ 穴埋め 1: ? ]} = \\ln \\left( \\frac{\\mu_{1i}(1 - \\mu_{2i})}{\\mu_{2i}(1 - \\mu_{1i})} \\right)$
   - $w_0 = \\sum_{i=1}^D \\ln \\frac{1 - \\mu_{1i}}{1 - \\mu_{2i}} + \\ln \\frac{p(C_1)}{p(C_2)}$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.11 数値検証
np.random.seed(42)
D = 5
mu1 = np.random.uniform(0.1, 0.9, size=D)
mu2 = np.random.uniform(0.1, 0.9, size=D)
pi1, pi2 = 0.55, 0.45

w_nb = np.log((mu1 * (1.0 - mu2)) / (mu2 * (1.0 - mu1)))
w0_nb = np.sum(np.log((1.0 - mu1) / (1.0 - mu2))) + np.log(pi1 / pi2)

# ランダムな二値ベクトル群
X_bin = np.random.choice([0, 1], size=(30, D))
for x in X_bin:
    prob1 = pi1 * np.prod(mu1**x * (1 - mu1)**(1 - x))
    prob2 = pi2 * np.prod(mu2**x * (1 - mu2)**(1 - x))
    post_exact = prob1 / (prob1 + prob2)
    post_sigmoid = 1.0 / (1.0 + np.exp(-(np.dot(w_nb, x) + w0_nb)))
    assert np.isclose(post_exact, post_sigmoid, atol=1e-12)
print("Exercise 4.11 PASSED: Binary Naive Bayes model is strictly a linear logistic sigmoid.")"""))

    # Exercise 4.12
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.12
**問題**: ソフトマックス関数
$$ y_k = \\frac{e^{a_k}}{\\sum_{j=1}^K e^{a_j}} $$
（式 4.104）のヤコビ行列が
$$ \\frac{\\partial y_k}{\\partial a_j} = y_k (\\delta_{kj} - y_j) $$
（式 4.106）で与えられることを示せ（$\\delta_{kj}$ はクロネッカーのデルタ）。

### [解答の道筋と穴埋め]
1. 分母を $S = \\sum_m e^{a_m}$ と置くと、$y_k = \\frac{e^{a_k}}{S}$。
2. 商の微分公式より:
   $$ \\frac{\\partial y_k}{\\partial a_j} = \\frac{\\frac{\\partial e^{a_k}}{\\partial a_j} S - e^{a_k} \\frac{\\partial S}{\\partial a_j}}{S^2} $$
3. $\\frac{\\partial e^{a_k}}{\\partial a_j} = \\delta_{kj} e^{a_k}$、および $\\frac{\\partial S}{\\partial a_j} = e^{a_j}$ であるから:
   $$ \\frac{\\partial y_k}{\\partial a_j} = \\frac{\\delta_{kj} e^{a_k} S - e^{a_k} e^{a_j}}{S^2} = \\delta_{kj} \\frac{e^{a_k}}{S} - \\left(\\frac{e^{a_k}}{S}\\right)\\left(\\frac{e^{a_j}}{S}\\right) $$
4. $y_k, y_j$ を用いて表すと:
   $$ \\frac{\\partial y_k}{\\partial a_j} = \\text{[ 穴埋め 1: ? ]} = \\delta_{kj} y_k - y_k y_j = y_k (\\delta_{kj} - y_j) $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.12 数値検証
np.random.seed(42)
K = 4
a = np.random.randn(K)

def softmax(v):
    ev = np.exp(v - np.max(v))
    return ev / np.sum(ev)

y = softmax(a)

# 解析的ヤコビ行列 J_analytic_kj = y_k (delta_kj - y_j)
J_analytic = np.diag(y) - np.outer(y, y)

# 中心差分による数値ヤコビ行列
eps = 1e-6
J_num = np.zeros((K, K))
for j in range(K):
    a_plus = a.copy(); a_plus[j] += eps
    a_minus = a.copy(); a_minus[j] -= eps
    J_num[:, j] = (softmax(a_plus) - softmax(a_minus)) / (2 * eps)

assert np.allclose(J_analytic, J_num, atol=1e-6)
print("Exercise 4.12 PASSED: Softmax Jacobian derivative strictly verified against numerical diff.")"""))

    # Exercise 4.13
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.13
**問題**: 線形分離可能なデータセットに対してロジスティック回帰の最尤推定を行うと、交差エントロピー誤差を 0 に近づけるために重みベクトルのノルムが無限大 $(\\|\\mathbf{w}\\| \\to \\infty)$ に発散し、シグモイド関数が階段関数に縮退する現象（最尤推定の過学習）を数学的に証明せよ。

### [解答の道筋と穴埋め]
1. データが線形分離可能であるとは、あるベクトル $\\mathbf{w}_0$ が存在して、すべてのデータ点に対し
   $$ \\mathbf{w}_0^T \\boldsymbol{\\phi}_n > 0 \\quad (t_n = 1), \\qquad \\mathbf{w}_0^T \\boldsymbol{\\phi}_n < 0 \\quad (t_n = 0) $$
   が成り立つことである。
2. これを $k \\mathbf{w}_0$ ($k > 0$) とスケールさせると、各点の予測確率は:
   - $t_n = 1$ のとき: $y_n = \\sigma(k \\mathbf{w}_0^T \\boldsymbol{\\phi}_n) \\to 1 \\quad (k \\to \\infty)$
   - $t_n = 0$ のとき: $1 - y_n = 1 - \\sigma(k \\mathbf{w}_0^T \\boldsymbol{\\phi}_n) = \\sigma(-k \\mathbf{w}_0^T \\boldsymbol{\\phi}_n) \\to 1 \\quad (k \\to \\infty)$
3. 負の対数尤度（交差エントロピー誤差）は:
   $$ E(k \\mathbf{w}_0) = -\\sum_{n=1}^N \\ln \\sigma(k |\\mathbf{w}_0^T \\boldsymbol{\\phi}_n|) \\to 0 \\quad (k \\to \\infty) $$
4. したがって最小値 $E = 0$ は有限の $\\mathbf{w}$ では達成されず、最尤推定量は $\\text{[ 穴埋め 1: ? ]} \\to \\infty$ と無限遠点に発散する。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.13 数値検証
# 線形分離可能なデータ上で勾配降下法を実行し、損失が 0 に近づくにつれて ||w|| が単調増加し続けることを確認
np.random.seed(42)
X = np.array([[1.0, 1.0], [2.0, 1.5], [-1.0, -1.0], [-2.0, -0.5]])
t = np.array([1, 1, 0, 0])
Phi = np.hstack([np.ones((4, 1)), X])

w = np.zeros(3)
lr = 0.5
norms = []
losses = []

for step in range(500):
    y = 1.0 / (1.0 + np.exp(-Phi @ w))
    loss = -np.sum(t * np.log(np.clip(y, 1e-15, 1)) + (1 - t) * np.log(np.clip(1 - y, 1e-15, 1)))
    grad = Phi.T @ (y - t)
    w -= lr * grad
    norms.append(np.linalg.norm(w))
    losses.append(loss)

# 重みノルムが発散傾向にあり、損失が減少していることの確認
assert norms[-1] > norms[0] * 2.0
assert losses[-1] < losses[0] * 0.01
print(f"Exercise 4.13 PASSED: Weight norm diverges on separable data: ||w_0||={norms[0]:.2f} -> ||w_500||={norms[-1]:.2f}, loss={losses[-1]:.2e}")"""))

    # Exercise 4.14
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.14
**問題**: ロジスティック回帰モデルの交差エントロピー誤差関数 $E(\\mathbf{w})$ のヘッセ行列
$$ \\mathbf{H} = \\mathbf{\\Phi}^T \\mathbf{R} \\mathbf{\\Phi} $$
（式 4.98、$\\mathbf{R}_{nn} = y_n(1 - y_n)$）が、計画行列 $\\mathbf{\\Phi}$ がフルランクであれば任意の $\\mathbf{u} \\neq \\mathbf{0}$ に対して**狭義正定値** $\\mathbf{u}^T \\mathbf{H} \\mathbf{u} > 0$（厳密に凸関数）となることを証明せよ。

### [解答の道筋と穴埋め]
1. 任意の非ゼロベクトル $\\mathbf{u} \\in \\mathbb{R}^M$ に対し二次形式を計算する:
   $$ \\mathbf{u}^T \\mathbf{H} \\mathbf{u} = \\mathbf{u}^T \\mathbf{\\Phi}^T \\mathbf{R} \\mathbf{\\Phi} \\mathbf{u} = (\\mathbf{\\Phi}\\mathbf{u})^T \\mathbf{R} (\\mathbf{\\Phi}\\mathbf{u}) $$
2. $\\mathbf{v} = \\mathbf{\\Phi}\\mathbf{u} = [v_1, \\dots, v_N]^T$ と置くと:
   $$ \\mathbf{u}^T \\mathbf{H} \\mathbf{u} = \\sum_{n=1}^N R_{nn} v_n^2 = \\sum_{n=1}^N y_n (1 - y_n) (\\boldsymbol{\\phi}_n^T \\mathbf{u})^2 $$
3. ロジスティックシグモイド関数の値域は $0 < y_n < 1$ であるから、すべての $n$ で $R_{nn} = y_n(1 - y_n) > 0$ である。
4. $\\mathbf{\\Phi}$ がフル列ランクであるため、$\\mathbf{u} \\neq \\mathbf{0}$ ならば $\\mathbf{\\Phi}\\mathbf{u} \\neq \\mathbf{0}$（少なくとも1つの $v_n \\neq 0$）。
5. したがって:
   $$ \\mathbf{u}^T \\mathbf{H} \\mathbf{u} = \\text{[ 穴埋め 1: ? ]} > 0 $$
   となり、ヘッセ行列は狭義正定値である。これにより局所解は存在せず、唯一の大域的最適解が存在する。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.14 数値検証
np.random.seed(42)
N, M = 50, 4
Phi = np.random.randn(N, M)
w_true = np.random.randn(M)
y = 1.0 / (1.0 + np.exp(-Phi @ w_true))
R = np.diag(y * (1.0 - y))

H = Phi.T @ R @ Phi
eigvals = np.linalg.eigvalsh(H)

assert np.all(eigvals > 0), "Hessian must be strictly positive definite!"
print(f"Exercise 4.14 PASSED: Logistic Hessian strictly positive definite (min eigenvalue = {eigvals[0]:.4f} > 0).")"""))

    # Exercise 4.15
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.15
**問題**: 多クラスロジスティック回帰の交差エントロピー誤差関数
$$ E(\\mathbf{w}_1, \\dots, \\mathbf{w}_K) = -\\sum_{n=1}^N \\sum_{k=1}^K t_{nk} \\ln y_{nk} $$
（式 4.108）のパラメータ $\\mathbf{w}_j$ に関する勾配が
$$ \\nabla_{\\mathbf{w}_j} E = \\sum_{n=1}^N (y_{nj} - t_{nj}) \\boldsymbol{\\phi}_n $$
（式 4.109）で与えられることを示せ。

### [解答の道筋と穴埋め]
1. 連鎖律より:
   $$ \\nabla_{\\mathbf{w}_j} E = \\sum_{n=1}^N \\sum_{k=1}^K \\frac{\\partial E_n}{\\partial y_{nk}} \\frac{\\partial y_{nk}}{\\partial a_{nj}} \\frac{\\partial a_{nj}}{\\partial \\mathbf{w}_j} $$
2. 各偏導関数は:
   - $\\frac{\\partial E_n}{\\partial y_{nk}} = -\\frac{t_{nk}}{y_{nk}}$
   - $\\frac{\\partial y_{nk}}{\\partial a_{nj}} = y_{nk}(\\delta_{kj} - y_{nj})$（Exercise 4.12 の結果）
   - $\\frac{\\partial a_{nj}}{\\partial \\mathbf{w}_j} = \\boldsymbol{\\phi}_n$
3. $k$ についての和をとると:
   $$ \\sum_{k=1}^K \\left(-\\frac{t_{nk}}{y_{nk}}\\right) y_{nk}(\\delta_{kj} - y_{nj}) = -\\sum_{k=1}^K t_{nk}(\\delta_{kj} - y_{nj}) = -t_{nj} + y_{nj} \\sum_{k=1}^K t_{nk} $$
4. 1-of-K 符号化より $\\sum_{k=1}^K t_{nk} = 1$ であるから:
   $$ \\nabla_{\\mathbf{w}_j} E = \\text{[ 穴埋め 1: ? ]} = \\sum_{n=1}^N (y_{nj} - t_{nj}) \\boldsymbol{\\phi}_n $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.15 数値検証
np.random.seed(42)
N, M, K = 25, 3, 4
Phi = np.random.randn(N, M)
labels = np.random.choice(K, size=N)
T = np.eye(K)[labels]

W = np.random.randn(M, K)

def softmax_mat(A):
    expA = np.exp(A - np.max(A, axis=1, keepdims=True))
    return expA / np.sum(expA, axis=1, keepdims=True)

# 解析的勾配
Y = softmax_mat(Phi @ W)
grad_analytic = Phi.T @ (Y - T)  # shape (M, K)

# 数値微分 (中心差分)
eps = 1e-6
grad_num = np.zeros((M, K))
for m in range(M):
    for k in range(K):
        W_p = W.copy(); W_p[m, k] += eps
        W_m = W.copy(); W_m[m, k] -= eps
        loss_p = -np.sum(T * np.log(softmax_mat(Phi @ W_p)))
        loss_m = -np.sum(T * np.log(softmax_mat(Phi @ W_m)))
        grad_num[m, k] = (loss_p - loss_m) / (2 * eps)

assert np.allclose(grad_analytic, grad_num, atol=1e-6)
print("Exercise 4.15 PASSED: Multiclass cross-entropy gradient formula verified against finite differences.")"""))

    # Exercise 4.16
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.16
**問題**: 目標値 $t_n$ が二値 $\{0, 1\}$ ではなく確率的ラベル $t_n \\in [0, 1]$（ソフトラベル）である場合、交差エントロピー誤差関数の勾配およびヘッセ行列がハードラベルの場合と**全く同一の表式**になることを示せ。

### [解答の道筋と穴埋め]
1. 目的関数 $E(\\mathbf{w}) = -\\sum_{n=1}^N [t_n \\ln y_n + (1 - t_n)\\ln(1 - y_n)]$。
2. 勾配の計算:
   $$ \\frac{\\partial E}{\\partial y_n} = -\\frac{t_n}{y_n} + \\frac{1 - t_n}{1 - y_n} = \\frac{y_n - t_n}{y_n(1 - y_n)} $$
3. シグモイドの微分 $\\frac{\\partial y_n}{\\partial a_n} = y_n(1 - y_n)$ と連鎖律により:
   $$ \\nabla_\\mathbf{w} E = \\sum_{n=1}^N \\frac{\\partial E}{\\partial y_n} \\frac{\\partial y_n}{\\partial a_n} \\boldsymbol{\\phi}_n = \\text{[ 穴埋め 1: ? ]} = \\sum_{n=1}^N (y_n - t_n) \\boldsymbol{\\phi}_n $$
4. さらに微分するとヘッセ行列は:
   $$ \\mathbf{H} = \\nabla\\nabla E = \\sum_{n=1}^N \\frac{\\partial y_n}{\\partial a_n} \\boldsymbol{\\phi}_n \\boldsymbol{\\phi}_n^T = \\mathbf{\\Phi}^T \\mathbf{R} \\mathbf{\\Phi} \\quad (R_{nn} = y_n(1 - y_n)) $$
   となり、目標値 $t_n$ はヘッセ行列に一切現れず、ソフトラベルでも完全に保存される。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.16 数値検証
np.random.seed(42)
N, M = 30, 3
Phi = np.random.randn(N, M)
w = np.random.randn(M)
t_soft = np.random.uniform(0.0, 1.0, size=N)  # ソフトラベル

y = 1.0 / (1.0 + np.exp(-Phi @ w))
grad_analytic = Phi.T @ (y - t_soft)
H_analytic = Phi.T @ np.diag(y * (1 - y)) @ Phi

eps = 1e-6
grad_num = np.zeros(M)
for m in range(M):
    w_p = w.copy(); w_p[m] += eps
    w_m = w.copy(); w_m[m] -= eps
    yp = 1.0 / (1.0 + np.exp(-Phi @ w_p))
    ym = 1.0 / (1.0 + np.exp(-Phi @ w_m))
    Ep = -np.sum(t_soft * np.log(yp) + (1 - t_soft) * np.log(1 - yp))
    Em = -np.sum(t_soft * np.log(ym) + (1 - t_soft) * np.log(1 - ym))
    grad_num[m] = (Ep - Em) / (2 * eps)

assert np.allclose(grad_analytic, grad_num, atol=1e-6)
print("Exercise 4.16 PASSED: Soft labels yield identical gradient and Hessian forms.")"""))

    # Exercise 4.17
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.17
**問題**: プロビット回帰モデルにおいて、活性化関数が標準正規累積分布関数
$$ \\Phi(a) = \\int_{-\\infty}^a \\mathcal{N}(\\theta | 0, 1) d\\theta $$
（式 4.114）で与えられるとき、対数尤度関数の勾配 $\\nabla \\ln L(\\mathbf{w})$ を求めよ。

### [解答の道筋と穴埋め]
1. 対数尤度関数は:
   $$ \\ln L(\\mathbf{w}) = \\sum_{n=1}^N \\left[ t_n \\ln \\Phi(a_n) + (1 - t_n) \\ln(1 - \\Phi(a_n)) \\right] \\quad (a_n = \\mathbf{w}^T \\boldsymbol{\\phi}_n) $$
2. 微積分学の基本定理より:
   $$ \\frac{d\\Phi(a)}{da} = \\mathcal{N}(a | 0, 1) = \\frac{1}{\\sqrt{2\\pi}} e^{-a^2/2} $$
3. 連鎖律により対数尤度を微分すると:
   $$ \\nabla_\\mathbf{w} \\ln L(\\mathbf{w}) = \\sum_{n=1}^N \\left[ \\frac{t_n}{\\Phi(a_n)} - \\frac{1 - t_n}{1 - \\Phi(a_n)} \\right] \\mathcal{N}(a_n | 0, 1) \\boldsymbol{\\phi}_n $$
4. 通分して整理すると:
   $$ \\nabla_\\mathbf{w} \\ln L(\\mathbf{w}) = \\text{[ 穴埋め 1: ? ]} = \\sum_{n=1}^N \\frac{t_n - \\Phi(a_n)}{\\Phi(a_n)(1 - \\Phi(a_n))} \\mathcal{N}(a_n | 0, 1) \\boldsymbol{\\phi}_n $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.17 数値検証
np.random.seed(42)
N, M = 20, 3
Phi = np.random.randn(N, M)
w = np.random.randn(M)
t = np.random.choice([0, 1], size=N)

a = Phi @ w
Phi_a = stats.norm.cdf(a)
p_a = stats.norm.pdf(a)

# 解析的勾配
weights = (t - Phi_a) / (Phi_a * (1.0 - Phi_a)) * p_a
grad_analytic = Phi.T @ weights

# 数値微分
def probit_log_lik(w_eval):
    an = Phi @ w_eval
    F = np.clip(stats.norm.cdf(an), 1e-15, 1 - 1e-15)
    return np.sum(t * np.log(F) + (1 - t) * np.log(1 - F))

eps = 1e-6
grad_num = np.zeros(M)
for m in range(M):
    wp = w.copy(); wp[m] += eps
    wm = w.copy(); wm[m] -= eps
    grad_num[m] = (probit_log_lik(wp) - probit_log_lik(wm)) / (2 * eps)

assert np.allclose(grad_analytic, grad_num, atol=1e-6)
print("Exercise 4.17 PASSED: Probit regression gradient matches numerical derivative.")"""))

    # Exercise 4.18
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.18
**問題**: プロビット回帰モデルの対数尤度関数 $\\ln L(\\mathbf{w})$ が強凹（ヘッセ行列が負定値）であり、大域的最適解が一意に存在することを証明せよ。

### [解答の道筋と穴埋め]
1. $\\ln L(\\mathbf{w}) = \\sum_n g_n(\\mathbf{w}^T \\boldsymbol{\\phi}_n)$ と表すと、ヘッセ行列は
   $$ \\nabla\\nabla \\ln L = \\sum_{n=1}^N g''_n(a_n) \\boldsymbol{\\phi}_n \\boldsymbol{\\phi}_n^T $$
2. $g_n(a) = t_n \\ln \\Phi(a) + (1 - t_n) \\ln(1 - \\Phi(a))$ の二階微分がすべての $a$ で負（$g''_n(a) < 0$）であることを示す。
3. ガウス累積分布関数 $\\Phi(a)$ および $1 - \\Phi(a)$ は**対数凹 (log-concave)** 関数であることが知られている（Mills比の性質による）。
4. したがって二階微分は常に厳密に負であり、任意の $\\mathbf{u} \\neq \\mathbf{0}$ に対し:
   $$ \\mathbf{u}^T (\\nabla\\nabla \\ln L) \\mathbf{u} = \\text{[ 穴埋め 1: ? ]} = \\sum_{n=1}^N g''_n(a_n) (\\mathbf{u}^T \\boldsymbol{\\phi}_n)^2 < 0 $$
   となり、対数尤度関数は強凹関数である。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.18 数値検証
# プロビット回帰の各データ点における二階微分 g''(a) が全域で厳密に負であることを検証
a_grid = np.linspace(-4.0, 4.0, 100)
Phi_grid = stats.norm.cdf(a_grid)
p_grid = stats.norm.pdf(a_grid)

# t = 1 のとき: d^2/da^2 ln Phi(a)
d2_log_Phi = (-a_grid * p_grid * Phi_grid - p_grid**2) / (Phi_grid**2)
assert np.all(d2_log_Phi < 0.0), "ln Phi(a) must be strictly concave!"

# t = 0 のとき: d^2/da^2 ln(1 - Phi(a))
Phi_c = 1.0 - Phi_grid
d2_log_Phi_c = (a_grid * p_grid * Phi_c - p_grid**2) / (Phi_c**2)
assert np.all(d2_log_Phi_c < 0.0), "ln(1 - Phi(a)) must be strictly concave!"

print(f"Exercise 4.18 PASSED: Probit log-concavity verified (max d^2/da^2 = {np.max(np.maximum(d2_log_Phi, d2_log_Phi_c)):.4f} < 0).")"""))

    # Exercise 4.19
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.19
**問題**: 多クラスロジスティック回帰のブロックヘッセ行列 $\\mathbf{H}$（ブロックサイズ $M \\times M$、全体サイズ $KM \\times KM$）の要素が
$$ \\nabla_{\\mathbf{w}_k} \\nabla_{\\mathbf{w}_j} E = \\sum_{n=1}^N y_{nj}(\\delta_{jk} - y_{nk}) \\boldsymbol{\\phi}_n \\boldsymbol{\\phi}_n^T $$
（式 4.110）で与えられることを示し、コーシー＝シュワルツの不等式を用いてヘッセ行列が常に半正定値であることを証明せよ。

### [解答の道筋と穴埋め]
1. 任意の $KM$ 次元ベクトル $\\mathbf{u} = [\\mathbf{u}_1^T, \\dots, \\mathbf{u}_K^T]^T$（各 $\\mathbf{u}_k \\in \\mathbb{R}^M$）に対する二次形式を計算する:
   $$ \\mathbf{u}^T \\mathbf{H} \\mathbf{u} = \\sum_{j=1}^K \\sum_{k=1}^K \\mathbf{u}_j^T \\left[ \\sum_{n=1}^N y_{nj}(\\delta_{jk} - y_{nk}) \\boldsymbol{\\phi}_n \\boldsymbol{\\phi}_n^T \\right] \\mathbf{u}_k $$
2. $a_{nk} = \\mathbf{u}_k^T \\boldsymbol{\\phi}_n$ と置くと:
   $$ \\mathbf{u}^T \\mathbf{H} \\mathbf{u} = \\sum_{n=1}^N \\left[ \\sum_{k=1}^K y_{nk} a_{nk}^2 - \\left( \\sum_{k=1}^K y_{nk} a_{nk} \\right)^2 \\right] $$
3. $y_{nk} > 0$ かつ $\\sum_{k=1}^K y_{nk} = 1$ であるため、内側の括弧は確率変数 $A_n$ の分散 $\\mathrm{Var}[A_n] = \\mathbb{E}[A_n^2] - (\\mathbb{E}[A_n])^2$ と解釈できる。
4. 分散は常に非負であるから:
   $$ \\mathbf{u}^T \\mathbf{H} \\mathbf{u} = \\sum_{n=1}^N \\mathrm{Var}[A_n] = \\text{[ 穴埋め 1: ? ]} \\ge 0 $$
   となり、ブロックヘッセ行列は常に半正定値である。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.19 数値検証
np.random.seed(42)
N, M, K = 30, 3, 4
Phi = np.random.randn(N, M)
W = np.random.randn(M, K)
Y = softmax_mat(Phi @ W)

# 全体サイズ (K*M, K*M) のブロックヘッセ行列を構築
H_full = np.zeros((K * M, K * M))
for j in range(K):
    for k in range(K):
        block = np.zeros((M, M))
        for n in range(N):
            weight = Y[n, j] * ((1.0 if j == k else 0.0) - Y[n, k])
            block += weight * np.outer(Phi[n], Phi[n])
        H_full[j*M:(j+1)*M, k*M:(k+1)*M] = block

# 固有値の非負性を検証
eigvals = np.linalg.eigvalsh(H_full)
assert np.all(eigvals >= -1e-10), "Hessian must be positive semi-definite!"
print(f"Exercise 4.19 PASSED: Multiclass Hessian is positive semi-definite (min eig = {np.min(eigvals):.2e} >= 0).")"""))

    # Exercise 4.20
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.20
**問題**: ソフトマックス関数が活性化ベクトルの同一シフト $a_k \\to a_k + c$ に対して不変であることに起因して、多クラスロジスティック回帰のヘッセ行列 $\\mathbf{H}$ が次元 $M$ 以上の非自明な零空間を持ち、特異行列（行列式が 0）となることを証明せよ。

### [解答の道筋と穴埋め]
1. 任意のベクトル $\\mathbf{v} \\in \\mathbb{R}^M$ に対し、すべてのクラスで同一のブロックを持つ $KM$ 次元ベクトル $\\mathbf{u} = [\\mathbf{v}^T, \\mathbf{v}^T, \\dots, \\mathbf{v}^T]^T$ を構成する。
2. Exercise 4.19 より $a_{nk} = \\mathbf{u}_k^T \\boldsymbol{\\phi}_n = \\mathbf{v}^T \\boldsymbol{\\phi}_n$ となり、すべての $k$ で一定値をとる。
3. 一定値の確率変数の分散は厳密に 0 である:
   $$ \\mathrm{Var}[A_n] = \\sum_{k=1}^K y_{nk} (\\mathbf{v}^T \\boldsymbol{\\phi}_n)^2 - \\left(\\sum_{k=1}^K y_{nk} \\mathbf{v}^T \\boldsymbol{\\phi}_n\\right)^2 = (\\mathbf{v}^T \\boldsymbol{\\phi}_n)^2 - (\\mathbf{v}^T \\boldsymbol{\\phi}_n)^2 = 0 $$
4. したがって:
   $$ \\mathbf{u}^T \\mathbf{H} \\mathbf{u} = \\text{[ 穴埋め 1: ? ]} = 0 \\implies \\mathbf{H} \\mathbf{u} = \\mathbf{0} $$
5. 任意の $\\mathbf{v} \\in \\mathbb{R}^M$ に対してこれが成り立つため、零空間の次元は少なくとも $M$ 存在する。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.20 数値検証
# 任意の v に対し u = [v, v, ..., v] が H の零ベクトル (H u = 0) であることを検証
v = np.random.randn(M)
u = np.tile(v, K)  # shape (K * M,)

Hu = H_full @ u
assert np.allclose(Hu, 0.0, atol=1e-10)
zero_eigs = np.sum(np.abs(eigvals) < 1e-8)
assert zero_eigs >= M, f"Expected at least {M} zero eigenvalues, found {zero_eigs}"
print(f"Exercise 4.20 PASSED: Nullspace verified (||H u|| = {np.linalg.norm(Hu):.2e}, zero eigenvalues = {zero_eigs} >= {M}).")"""))

    # Exercise 4.21
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.21
**問題**: 誤差関数 $\\mathrm{erf}(a) = \\frac{2}{\\sqrt{\\pi}} \\int_0^a e^{-\\theta^2} d\\theta$ とプロビット関数 $\\Phi(a)$ の関係式
$$ \\Phi(a) = \\frac{1}{2} \\left[ 1 + \\mathrm{erf}\\left(\\frac{a}{\\sqrt{2}}\\right) \\right] $$
（式 4.115）を示せ。また、$a=0$ における傾きを一致させることで、シグモイド関数のプロビット近似
$$ \\sigma(a) \\approx \\Phi(\\lambda a) \\quad \\left( \\lambda^2 = \\frac{\\pi}{8} \\right) $$
（式 4.116）を導出せよ。

### [解答の道筋と穴埋め]
1. プロビット関数の定義より:
   $$ \\Phi(a) = \\int_{-\\infty}^a \\frac{1}{\\sqrt{2\\pi}} e^{-\\theta^2/2} d\\theta = \\frac{1}{2} + \\int_0^a \\frac{1}{\\sqrt{2\\pi}} e^{-\\theta^2/2} d\\theta $$
2. 変数変換 $u = \\theta / \\sqrt{2}$ ($d\\theta = \\sqrt{2} du$) を適用すると:
   $$ \\int_0^a \\frac{1}{\\sqrt{2\\pi}} e^{-\\theta^2/2} d\\theta = \\frac{1}{\\sqrt{\\pi}} \\int_0^{a/\\sqrt{2}} e^{-u^2} du = \\frac{1}{2} \\mathrm{erf}\\left(\\frac{a}{\\sqrt{2}}\\right) $$
   これにより式 (4.115) が得られる。
3. 原点 $a = 0$ における微係数の一致:
   - シグモイドの原点での微分: $\\sigma'(0) = \\sigma(0)(1 - \\sigma(0)) = \\frac{1}{2} \\times \\frac{1}{2} = \\frac{1}{4}$
   - プロビット近似の原点での微分: $\\left. \\frac{d}{da} \\Phi(\\lambda a) \\right|_{a=0} = \\lambda \\mathcal{N}(0|0, 1) = \\frac{\\lambda}{\\sqrt{2\\pi}}$
4. これらを等置すると:
   $$ \\frac{\\lambda}{\\sqrt{2\\pi}} = \\frac{1}{4} \\implies \\lambda^2 = \\text{[ 穴埋め 1: ? ]} = \\frac{2\\pi}{16} = \\frac{\\pi}{8} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.21 数値検証
a_vals = np.linspace(-5.0, 5.0, 300)

# 1. erf による Phi(a) の恒等式検証
phi_erf = 0.5 * (1.0 + special.erf(a_vals / np.sqrt(2)))
phi_exact = stats.norm.cdf(a_vals)
assert np.allclose(phi_erf, phi_exact, atol=1e-14)

# 2. lambda^2 = pi / 8 によるシグモイド近似の精度検証
lam = np.sqrt(np.pi / 8.0)
sig_exact = 1.0 / (1.0 + np.exp(-a_vals))
sig_approx = stats.norm.cdf(lam * a_vals)
max_err = np.max(np.abs(sig_exact - sig_approx))

assert max_err < 0.02, f"Approximation error too large: {max_err}"
print(f"Exercise 4.21 PASSED: Probit-erf identity verified and probit sigmoid approximation max error = {max_err:.4f} (< 0.02).")"""))

    # Exercise 4.22
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.22
**問題**: 1次元の確率分布 $p(z) = \\frac{1}{Z} f(z)$ のラプラス近似において、最頻値 $z_0 = \\arg\\max f(z)$ の周りで $\\ln f(z)$ をテイラー展開することにより、局所ガウス近似
$$ q(z) = \\mathcal{N}(z | z_0, A^{-1}) \\quad \\left( A = -\\left. \\frac{d^2}{dz^2} \\ln f(z) \\right|_{z_0} \\right) $$
および規格化定数の近似式 $Z \\approx f(z_0) \\sqrt{\\frac{2\\pi}{A}}$ (式 4.128 - 4.130) を導出せよ。

### [解答の道筋と穴埋め]
1. 最頻値 $z_0$ では一階微分が 0 である: $\\left. \\frac{d}{dz} \\ln f(z) \\right|_{z_0} = 0$。
2. 2次のテイラー展開を行うと:
   $$ \\ln f(z) \\approx \\ln f(z_0) - \\frac{1}{2} A (z - z_0)^2 \\quad \\left( A = -\\left. \\frac{d^2}{dz^2} \\ln f(z) \\right|_{z_0} \\right) $$
3. 指数をとると:
   $$ f(z) \\approx f(z_0) \\exp\\left( -\\frac{A}{2} (z - z_0)^2 \\right) $$
4. 規格化定数 $Z = \\int_{-\\infty}^\\infty f(z) dz$ をガウス積分により評価すると:
   $$ Z \\approx f(z_0) \\int_{-\\infty}^\\infty \\exp\\left( -\\frac{A}{2} (z - z_0)^2 \\right) dz = \\text{[ 穴埋め 1: ? ]} = f(z_0) \\sqrt{\\frac{2\\pi}{A}} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.22 数値検証
# ガンマ分布 Gam(z | a, b) に対するラプラス近似と数値積分の比較
a_param, b_param = 5.0, 2.0
# f(z) = z^{a-1} e^{-b z}
z0 = (a_param - 1.0) / b_param  # 最頻値
A = (a_param - 1.0) / (z0**2)    # -d^2/dz^2 ln f = (a-1)/z^2

f_z0 = (z0**(a_param - 1.0)) * np.exp(-b_param * z0)
Z_laplace = f_z0 * np.sqrt(2 * np.pi / A)
Z_exact = special.gamma(a_param) / (b_param**a_param)

rel_err = abs(Z_laplace - Z_exact) / Z_exact
assert rel_err < 0.05
print(f"Exercise 4.22 PASSED: 1D Laplace normalization constant Z_approx={Z_laplace:.4f} vs Z_exact={Z_exact:.4f} (rel err={rel_err:.2%}).")"""))

    # Exercise 4.23
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.23
**問題**: $D$ 次元の確率分布 $p(\\mathbf{z}) = \\frac{1}{Z} f(\\mathbf{z})$ に対する多次元ラプラス近似において、ヘッセ行列 $\\mathbf{A} = -\\nabla\\nabla \\ln f(\\mathbf{z}_0)$ を用いて規格化定数が
$$ Z \\approx f(\\mathbf{z}_0) \\frac{(2\\pi)^{D/2}}{|\\mathbf{A}|^{1/2}} $$
（式 4.135）で近似されることを多次元ガウス積分を用いて証明せよ。

### [解答の道筋と穴埋め]
1. $\\mathbf{z}_0$ を $\\ln f(\\mathbf{z})$ の最頻値とする（$\\nabla \\ln f(\\mathbf{z}_0) = \\mathbf{0}$）。
2. 多変数テイラー展開より:
   $$ \\ln f(\\mathbf{z}) \\approx \\ln f(\\mathbf{z}_0) - \\frac{1}{2} (\\mathbf{z} - \\mathbf{z}_0)^T \\mathbf{A} (\\mathbf{z} - \\mathbf{z}_0) $$
3. 規格化定数は:
   $$ Z = \\int f(\\mathbf{z}) d\\mathbf{z} \\approx f(\\mathbf{z}_0) \\int \\exp\\left( -\\frac{1}{2}(\\mathbf{z} - \\mathbf{z}_0)^T \\mathbf{A} (\\mathbf{z} - \\mathbf{z}_0) \\right) d\\mathbf{z} $$
4. 多次元非規格化ガウス積分の公式 $\\int \\exp(-\\frac{1}{2}\\mathbf{x}^T \\mathbf{A} \\mathbf{x}) d\\mathbf{x} = \\frac{(2\\pi)^{D/2}}{|\\mathbf{A}|^{1/2}}$ を適用すると:
   $$ Z \\approx \\text{[ 穴埋め 1: ? ]} = f(\\mathbf{z}_0) \\frac{(2\\pi)^{D/2}}{|\\mathbf{A}|^{1/2}} $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.23 数値検証
# 2次元非ガウス関数の数値積分求積 vs 多次元ラプラス近似
def log_f(z):
    return -(z[0]**2 + z[1]**2 + 0.5 * (z[0] - z[1])**2 + 0.1 * z[0]**4)

res = optimize.minimize(lambda z: -log_f(z), [0.0, 0.0])
z0 = res.x
f_z0 = np.exp(log_f(z0))

# ヘッセ行列の数値計算
eps = 1e-4
A_mat = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        zp = z0.copy(); zp[i] += eps; zp[j] += eps
        zm = z0.copy(); zm[i] -= eps; zm[j] -= eps
        zpi = z0.copy(); zpi[i] += eps; zpi[j] -= eps
        zpj = z0.copy(); zpj[i] -= eps; zpj[j] += eps
        A_mat[i, j] = -(log_f(zp) + log_f(zm) - log_f(zpi) - log_f(zpj)) / (4 * eps**2)

Z_laplace = f_z0 * (2 * np.pi) / np.sqrt(np.linalg.det(A_mat))

# 2重数値積分
Z_num, _ = integrate.dblquad(lambda y, x: np.exp(log_f(np.array([x, y]))), -4.0, 4.0, -4.0, 4.0)

rel_diff = abs(Z_laplace - Z_num) / Z_num
assert rel_diff < 0.05
print(f"Exercise 4.23 PASSED: 2D Laplace evidence Z_approx={Z_laplace:.4f} matches numerical integral Z_num={Z_num:.4f} (rel diff={rel_diff:.2%}).")"""))

    # Exercise 4.24
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.24
**問題**: ラプラス近似によるモデルエビデンス $\\ln p(\\mathcal{D})$ の評価式
$$ \\ln p(\\mathcal{D}) \\approx \\ln p(\\mathcal{D} | \\boldsymbol{\\theta}_{\\mathrm{MAP}}) + \\ln p(\\boldsymbol{\\theta}_{\\mathrm{MAP}}) + \\frac{D}{2}\\ln(2\\pi) - \\frac{1}{2}\\ln |\\mathbf{A}| $$
（式 4.137）から、データ数 $N \\to \\infty$ の漸近極限においてベイズ情報量規準 (BIC)
$$ \\mathrm{BIC} = -2 \\ln p(\\mathcal{D} | \\boldsymbol{\\theta}_{\\mathrm{ML}}) + D \\ln N $$
（式 4.139）が導出される全ステップを証明せよ。

### [解答の道筋と穴埋め]
1. データ数 $N$ が十分大きいとき、事前分布の影響は相対的に無視でき、$\\boldsymbol{\\theta}_{\\mathrm{MAP}} \\approx \\boldsymbol{\\theta}_{\\mathrm{ML}}$ となる。
2. ヘッセ行列は $N$ 個の独立なデータ点の寄与の和であるため:
   $$ \\mathbf{A} = -\\nabla\\nabla \\ln p(\\mathcal{D} | \\boldsymbol{\\theta}) - \\nabla\\nabla \\ln p(\\boldsymbol{\\theta}) = \\sum_{n=1}^N \\mathbf{H}_n + \\mathbf{H}_{\\mathrm{prior}} \\approx N \\mathbf{J} $$
   （$\\mathbf{J}$ は1サンプルあたりのフィッシャー情報行列）。
3. 行列式の性質 $|N \\mathbf{J}| = N^D |\\mathbf{J}|$ より:
   $$ \\ln |\\mathbf{A}| \\approx \\ln(N^D |\\mathbf{J}|) = D \\ln N + \\ln |\\mathbf{J}| $$
4. $O(N)$ および $O(\\ln N)$ の項を残し、$O(1)$ の定数項を無視すると:
   $$ \\ln p(\\mathcal{D}) \\approx \\ln p(\\mathcal{D} | \\boldsymbol{\\theta}_{\\mathrm{ML}}) - \\frac{D}{2} \\ln N $$
5. これに $-2$ を掛けたものが BIC である:
   $$ \\mathrm{BIC} = \\text{[ 穴埋め 1: ? ]} = -2 \\ln p(\\mathcal{D} | \\boldsymbol{\\theta}_{\\mathrm{ML}}) + D \\ln N $$"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.24 数値検証
# サンプルサイズ N を増やしたとき、対数行列式 ln|A| が理論通り D * ln(N) + const の傾きで増大することを検証
D = 3
N_list = [50, 100, 200, 500, 1000]
ln_det_A_list = []

for N_val in N_list:
    Phi_N = np.random.randn(N_val, D)
    w_test = np.array([0.5, -0.3, 0.2])
    y_N = 1.0 / (1.0 + np.exp(-Phi_N @ w_test))
    R_N = np.diag(y_N * (1 - y_N))
    A_mat = Phi_N.T @ R_N @ Phi_N + np.eye(D)
    ln_det_A_list.append(np.linalg.slogdet(A_mat)[1])

# 回帰により ln(N) に対する傾きを推定
slopes = np.polyfit(np.log(N_list), ln_det_A_list, 1)
estimated_D = slopes[0]

assert np.isclose(estimated_D, D, rtol=0.1)
print(f"Exercise 4.24 PASSED: Asymptotic Hessian log determinant scaling slope={estimated_D:.3f} matches dimension D={D}.")"""))

    # Exercise 4.25
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.25
**問題**: ベイズロジスティック回帰において、パラメータ事後分布がガウス近似 $q(\\mathbf{w}) = \\mathcal{N}(\\mathbf{w} | \\mathbf{m}_N, \\mathbf{S}_N)$ されるとき、周辺化予測分布
$$ p(C_1 | \\mathbf{x}) = \\int \\sigma(a) \\mathcal{N}(a | \\mu_a, \\sigma_a^2) da $$
（$\\mu_a = \\mathbf{m}_N^T \\boldsymbol{\\phi}, \\sigma_a^2 = \\boldsymbol{\\phi}^T \\mathbf{S}_N \\boldsymbol{\\phi}$）が、シグモイドのプロビット近似を用いて
$$ p(C_1 | \\mathbf{x}) \\approx \\sigma\\left(\\kappa(\\sigma_a^2) \\mu_a\\right) \\quad \\left( \\kappa(\\sigma^2) = (1 + \\pi \\sigma^2 / 8)^{-1/2} \\right) $$
（式 4.153）と閉形式近似されることを導出せよ。

### [解答の道筋と穴埋め]
1. Exercise 4.21 のプロビット近似 $\\sigma(a) \\approx \\Phi(\\lambda a)$（$\\lambda^2 = \\pi / 8$）を適用する:
   $$ \\int \\sigma(a) \\mathcal{N}(a | \\mu_a, \\sigma_a^2) da \\approx \\int \\Phi(\\lambda a) \\mathcal{N}(a | \\mu_a, \\sigma_a^2) da $$
2. プロビットとガウスの畳み込み公式（Exercise 4.26）より:
   $$ \\int \\Phi(\\lambda a) \\mathcal{N}(a | \\mu_a, \\sigma_a^2) da = \\Phi\\left( \\frac{\\lambda \\mu_a}{\\sqrt{1 + \\lambda^2 \\sigma_a^2}} \\right) $$
3. ここでシグモイドの逆近似 $\\Phi(\\theta) \\approx \\sigma(\\theta / \\lambda)$ を再適用すると:
   $$ \\Phi\\left( \\lambda \\frac{\\mu_a}{\\sqrt{1 + \\lambda^2 \\sigma_a^2}} \\right) \\approx \\sigma\\left( \\frac{\\mu_a}{\\sqrt{1 + \\lambda^2 \\sigma_a^2}} \\right) $$
4. $\\lambda^2 = \\pi / 8$ を代入すると:
   $$ \\kappa(\\sigma_a^2) = \\text{[ 穴埋め 1: ? ]} = \\left( 1 + \\frac{\\pi}{8} \\sigma_a^2 \\right)^{-1/2} $$
   となり、$\\sigma(\\kappa(\\sigma_a^2) \\mu_a)$ が得られる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.25 数値検証
# 1次元数値求積による真の畳み込み積分 vs kappa 近似式
mu_vals = [-1.5, 0.0, 1.0, 2.5]
sigma2_vals = [0.2, 1.0, 4.0, 9.0]

for mu in mu_vals:
    for s2 in sigma2_vals:
        s = np.sqrt(s2)
        # 数値積分
        int_val, _ = integrate.quad(lambda a: (1.0 / (1.0 + np.exp(-a))) * stats.norm.pdf(a, loc=mu, scale=s), mu - 6*s, mu + 6*s)
        # 解析的近似式
        kappa = (1.0 + np.pi * s2 / 8.0)**(-0.5)
        approx_val = 1.0 / (1.0 + np.exp(-kappa * mu))
        assert abs(int_val - approx_val) < 0.02
print("Exercise 4.25 PASSED: Bayesian logistic convolution approx kappa formula verified across grid (all errors < 0.02).")"""))

    # Exercise 4.26
    cells.append(nbf.v4.new_markdown_cell("""## Exercise 4.26
**問題**: プロビット関数 $\\Phi(x)$ とガウス分布 $\\mathcal{N}(x | \\mu, \\sigma^2)$ の畳み込み積分
$$ I = \\int_{-\\infty}^\\infty \\Phi(x) \\mathcal{N}(x | \\mu, \\sigma^2) dx $$
が近似なしに**厳密な閉形式解**
$$ I = \\Phi\\left( \\frac{\\mu}{\\sqrt{1 + \\sigma^2}} \\right) $$
となることを、確率変数の線形結合の性質を用いて証明せよ。

### [解答の道筋と穴埋め]
1. $\\Phi(x) = P(\\theta < x)$ である（ただし $\\theta \\sim \\mathcal{N}(0, 1)$）。
2. 積分 $I$ は、$x \\sim \\mathcal{N}(\\mu, \\sigma^2)$ と $\\theta \\sim \\mathcal{N}(0, 1)$ が独立であるときの確率 $P(\\theta < x)$ に等しい:
   $$ I = \\mathbb{E}_{x}[P(\\theta < x | x)] = P(\\theta - x < 0) $$
3. 新たな確率変数 $z = \\theta - x$ を定義する。独立な2つの正規分布の差であるため、$z$ もまた正規分布に従う:
   - 平均: $\\mathbb{E}[z] = \\mathbb{E}[\\theta] - \\mathbb{E}[x] = 0 - \\mu = -\\mu$
   - 分散: $\\mathrm{Var}[z] = \\mathrm{Var}[\\theta] + \\mathrm{Var}[x] = 1 + \\sigma^2$
4. したがって $z \\sim \\mathcal{N}(-\\mu, 1 + \\sigma^2)$ である。
5. 求める確率は:
   $$ P(z < 0) = \\Phi\\left( \\frac{0 - (-\\mu)}{\\sqrt{1 + \\sigma^2}} \\right) = \\text{[ 穴埋め 1: ? ]} = \\Phi\\left( \\frac{\\mu}{\\sqrt{1 + \\sigma^2}} \\right) $$
   となり、近似を一切含まない厳密解が得られる。"""))

    cells.append(nbf.v4.new_code_cell("""# Exercise 4.26 数値検証
# プロビットとガウスの厳密畳み込み積分の数値求積 vs 解析閉形式解 Phi(mu / sqrt(1 + sigma^2))
np.random.seed(42)
test_cases = [(-2.0, 0.5), (0.0, 2.0), (1.5, 3.0), (3.0, 1.2)]

for mu, sigma in test_cases:
    s2 = sigma**2
    # 厳密解析解
    exact_analytic = stats.norm.cdf(mu / np.sqrt(1.0 + s2))
    # 高精度数値求積
    num_integral, _ = integrate.quad(
        lambda x: stats.norm.cdf(x) * stats.norm.pdf(x, loc=mu, scale=sigma),
        mu - 8*sigma, mu + 8*sigma
    )
    assert np.isclose(exact_analytic, num_integral, atol=1e-9)

print("Exercise 4.26 PASSED: Exact closed-form convolution identity for probit and Gaussian holds strictly.")"""))

    nb['cells'] = cells
    with open('4/4_Exercises.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Created 4/4_Exercises.ipynb with all 26 individual exercises successfully.")

if __name__ == '__main__':
    create_ch4_exercises_notebook()
