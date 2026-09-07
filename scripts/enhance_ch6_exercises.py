"""
Master script to generate 6/6_Exercises.ipynb containing all 27 PRML Chapter 6 exercises (Exercises 6.1 - 6.27)
with full derivations, fill-in-the-blank markdown, and self-contained numerical verification cells.
"""

import nbformat as nbf
import os

def create_cell_pairs():
    cells = []
    
    # Title & Introduction
    title_md = r"""# 第6章 カーネル法：章末演習問題 (Exercises 6.1 〜 6.27 全27問 完全網羅)

教科書「パターン認識と機械学習 (PRML)」第6章「カーネル法 (Kernel Methods)」の全27問の演習問題の完全解答・解説ノートブックです。

各問題について、以下の構成で学習を進められるよう設計されています：
1. **問題の提示**: PRML原著の設問内容の明確な記述。
2. **数理的導出と証明（穴埋め形式）**: 証明や計算のステップを論理的に整理し、核心となる数式やキーワードを穴埋め（`____`）形式で提示。
3. **穴埋めの解答と数理解説**: 正解と詳細な解説。
4. **Pythonによる数値シミュレーション・検証**: 数式や恒等式、アルゴリズムの正当性を自己完結したコードで検証。
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))
    
    # Setup cell
    setup_code = r"""# 共通ライブラリのインポートとシード値の固定
import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import matplotlib.pyplot as plt

np.random.seed(42)
print("PRML Chapter 6: Kernel Methods Exercises Setup Complete.")"""
    cells.append(nbf.v4.new_code_cell(setup_code))

    # ----------------------------------------------------
    # Exercise 6.1
    # ----------------------------------------------------
    ex6_1_md = r"""---
## Exercise 6.1: 正則化最小二乗法の双対表現 (Dual Formulation of Regularized Least Squares)

### 問題
6.1節で与えられた線形回帰の正則化二乗和誤差関数の双対表現を考える。
$$
J(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) - t_n)^2 + \frac{\lambda}{2} \mathbf{w}^T \mathbf{w} \quad (\lambda > 0)
$$
ベクトル $\mathbf{a}$ の各成分 $a_n$ が $\boldsymbol{\phi}(\mathbf{x}_n)$ の線形結合として表され、双対表現の双対をとると元の主問題の解が得られることを示せ。

### 数理的証明・導出ステップ（穴埋め）
1. 目的関数 $J(\mathbf{w})$ を計画行列 $\boldsymbol{\Phi} \in \mathbb{R}^{N \times M}$ を用いて文字ベクトル表記すると：
   $$
   J(\mathbf{w}) = \frac{1}{2} (\boldsymbol{\Phi} \mathbf{w} - \mathbf{t})^T (\boldsymbol{\Phi} \mathbf{w} - \mathbf{t}) + \frac{\lambda}{2} \mathbf{w}^T \mathbf{w}
   $$
2. $\mathbf{w}$ に関する勾配をとってゼロとおくと：
   $$
   \nabla J(\mathbf{w}) = \boldsymbol{\Phi}^T (\boldsymbol{\Phi} \mathbf{w} - \mathbf{t}) + \lambda \mathbf{w} = \mathbf{0}
   $$
   整理すると：
   $$
   \mathbf{w} = -\frac{1}{\lambda} \boldsymbol{\Phi}^T (\boldsymbol{\Phi} \mathbf{w} - \mathbf{t}) = \boldsymbol{\Phi}^T \mathbf{a} = \sum_{n=1}^N a_n \boldsymbol{\phi}(\mathbf{x}_n)
   $$
   ここで、双対パラメータベクトル $\mathbf{a} \in \mathbb{R}^N$ の定義は：
   $$
   a_n = -\frac{1}{\lambda} (\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) - t_n) \quad \Longleftrightarrow \quad \mathbf{a} = -\frac{1}{\lambda} (\boldsymbol{\Phi} \mathbf{w} - \mathbf{t})
   $$
3. $\mathbf{w} = \boldsymbol{\Phi}^T \mathbf{a}$ を $J(\mathbf{w})$ に代入すると、グラム行列 $\mathbf{K} = \boldsymbol{\Phi} \boldsymbol{\Phi}^T$ を用いて：
   $$
   J(\mathbf{a}) = \frac{1}{2} \mathbf{a}^T \mathbf{K} \mathbf{K} \mathbf{a} - \mathbf{a}^T \mathbf{K} \mathbf{t} + \frac{1}{2} \mathbf{t}^T \mathbf{t} + \frac{\lambda}{2} \mathbf{a}^T \mathbf{K} \mathbf{a}
   $$
4. $\mathbf{a}$ に関して最小化するために勾配をとると：
   $$
   \nabla_{\mathbf{a}} J(\mathbf{a}) = \mathbf{K} \mathbf{K} \mathbf{a} - \mathbf{K} \mathbf{t} + \lambda \mathbf{K} \mathbf{a} = \mathbf{K} [(\mathbf{K} + \lambda \mathbf{I}) \mathbf{a} - \mathbf{t}] = \mathbf{0}
   $$
   したがって、$\mathbf{a} = [ \text{①} ]^{-1} \mathbf{t}$ が得られる。
5. 新しい入力 $\mathbf{x}$ に対する予測値は：
   $$
   y(\mathbf{x}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}) = \mathbf{a}^T \boldsymbol{\Phi} \boldsymbol{\phi}(\mathbf{x}) = \mathbf{k}(\mathbf{x})^T (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{t}
   $$
6. 双対の双対（Dual of Dual）:
   $\mathbf{a} = -\frac{1}{\lambda} (\boldsymbol{\Phi} \mathbf{w} - \mathbf{t})$ を $\mathbf{w} = \boldsymbol{\Phi}^T \mathbf{a}$ に代入すると：
   $$
   \mathbf{w} = -\frac{1}{\lambda} \boldsymbol{\Phi}^T (\boldsymbol{\Phi} \mathbf{w} - \mathbf{t}) \implies (\boldsymbol{\Phi}^T \boldsymbol{\Phi} + \lambda \mathbf{I}_M) \mathbf{w} = \boldsymbol{\Phi}^T \mathbf{t} \implies \mathbf{w} = [ \text{②} ]^{-1} \boldsymbol{\Phi}^T \mathbf{t}
   $$
   となり、元の主問題の正則化正規方程式の解と完全に一致する。

### 穴埋めの解答
- ①: $\mathbf{K} + \lambda \mathbf{I}_N$
- ②: $\boldsymbol{\Phi}^T \boldsymbol{\Phi} + \lambda \mathbf{I}_M$"""
    
    ex6_1_code = r"""# Exercise 6.1 数値検証: 主問題解 w と双対解 a による予測の完全一致
N, M = 25, 10
Phi = np.random.randn(N, M)
t = np.random.randn(N)
lam = 0.5

# 主問題解 w
w_primal = np.linalg.solve(Phi.T @ Phi + lam * np.eye(M), Phi.T @ t)

# 双対問題解 a
K = Phi @ Phi.T
a_dual = np.linalg.solve(K + lam * np.eye(N), t)

# 恒等性 w = Phi^T a の検証
w_from_dual = Phi.T @ a_dual
np.testing.assert_allclose(w_primal, w_from_dual, atol=1e-10)

# 任意テスト点 x での予測値の一致検証
phi_test = np.random.randn(5, M)
y_primal = phi_test @ w_primal
k_test = phi_test @ Phi.T
y_dual = k_test @ a_dual
np.testing.assert_allclose(y_primal, y_dual, atol=1e-10)

print("Exercise 6.1 verified: Primal w and Dual a yield mathematically identical predictions.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_1_md), nbf.v4.new_code_cell(ex6_1_code)])

    # ----------------------------------------------------
    # Exercise 6.2
    # ----------------------------------------------------
    ex6_2_md = r"""---
## Exercise 6.2: パーセプトロン学習規則の双対表現 (Dual Formulation of Perceptron)

### 問題
パーセプトロン学習アルゴリズム（4.1.7節）の双対表現を導出せよ。
学習された重みベクトル $\mathbf{w}$ が $\sum_{n=1}^N \alpha_n t_n \boldsymbol{\phi}(\mathbf{x}_n)$ （$t_n \in \{-1, +1\}$）の線形結合として表せることを示し、カーネル関数 $k(\mathbf{x}_n, \mathbf{x}_m)$ のみを用いて誤識別判定と更新を行うアルゴリズムを記述せよ。

### 数理的証明・導出ステップ（穴埋め）
1. パーセプトロンの初期重みを $\mathbf{w}^{(0)} = \mathbf{0}$ とする。
2. パターン $(\mathbf{x}_n, t_n)$ が誤分類されたとき（すなわち $t_n \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) \le 0$ のとき）、重み更新則は：
   $$
   \mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} + \eta t_n \boldsymbol{\phi}(\mathbf{x}_n)
   $$
3. したがって、学習終了時の重み $\mathbf{w}$ はデータ点ごとの更新回数 $\alpha_n \ge 0$ を用いて次のように表される：
   $$
   \mathbf{w} = \sum_{n=1}^N \alpha_n t_n \boldsymbol{\phi}(\mathbf{x}_n)
   $$
4. これを入力 $\mathbf{x}$ に対する判別関数 $y(\mathbf{x}) = \text{sign}(\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}))$ に代入すると：
   $$
   y(\mathbf{x}) = \text{sign}\left( \sum_{n=1}^N \alpha_n t_n \boldsymbol{\phi}(\mathbf{x}_n)^T \boldsymbol{\phi}(\mathbf{x}) \right) = \text{sign}\left( \sum_{n=1}^N \alpha_n t_n [ \text{①} ] \right)
   $$
5. データ点 $\mathbf{x}_m$ の誤分類判定条件は：
   $$
   t_m \left( \sum_{n=1}^N \alpha_n t_n k(\mathbf{x}_n, \mathbf{x}_m) \right) \le 0
   $$
   この条件が満たされたとき、$\alpha_m \leftarrow [ \text{②} ]$ と更新する。これにより特徴ベクトル $\boldsymbol{\phi}(\mathbf{x})$ の陽な計算を一切必要とせず、カーネル $k(\mathbf{x}_n, \mathbf{x}_m)$ だけでパーセプトロンが実行できる（**カーネルパーセプトロン**）。

### 穴埋めの解答
- ①: $k(\mathbf{x}_n, \mathbf{x})$
- ②: $\alpha_m + 1$ （または学習率 $\eta$）"""

    ex6_2_code = r"""# Exercise 6.2 数値検証: 主パーセプトロンとカーネルパーセプトロンの動作完全一致
np.random.seed(42)
N, D = 30, 2
X = np.random.randn(N, D)
# 線形分離可能なターゲット
w_true = np.array([1.2, -0.8])
t = np.sign(X @ w_true)
t[t == 0] = 1

# 1. 主パーセプトロン (Primal)
w_primal = np.zeros(D)
for epoch in range(50):
    misclassified = 0
    for n in range(N):
        if t[n] * (X[n] @ w_primal) <= 0:
            w_primal += t[n] * X[n]
            misclassified += 1
    if misclassified == 0:
        break

# 2. カーネルパーセプトロン (Dual with linear kernel k(x, x') = x^T x')
K_mat = X @ X.T
alpha = np.zeros(N)
for epoch in range(50):
    misclassified = 0
    for n in range(N):
        pred_val = np.sum(alpha * t * K_mat[:, n])
        if t[n] * pred_val <= 0:
            alpha[n] += 1
            misclassified += 1
    if misclassified == 0:
        break

# 双対パラメータから復元した重みベクトル
w_dual_recon = np.sum((alpha * t)[:, None] * X, axis=0)

np.testing.assert_allclose(w_primal, w_dual_recon, atol=1e-10)
print("Exercise 6.2 verified: Primal Perceptron and Kernel Perceptron weight vectors are identical.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_2_md), nbf.v4.new_code_cell(ex6_2_code)])

    # ----------------------------------------------------
    # Exercise 6.3
    # ----------------------------------------------------
    ex6_3_md = r"""---
## Exercise 6.3: 最近傍法 (Nearest-Neighbour) のカーネル化

### 問題
ユークリッド距離 $\|\mathbf{x} - \mathbf{x}_n\|^2$ に基づく最近傍識別器（2.5.2節）について、距離関数を内積で表現した上でカーネル置換（Kernel Substitution）を行い、一般の非線形カーネルに対する最近傍法の計算式を導出せよ。

### 数理的証明・導出ステップ（穴埋め）
1. 特徴空間における未知点 $\boldsymbol{\phi}(\mathbf{x})$ と訓練点 $\boldsymbol{\phi}(\mathbf{x}_n)$ のユークリッド自乗距離を展開すると：
   $$
   \|\boldsymbol{\phi}(\mathbf{x}) - \boldsymbol{\phi}(\mathbf{x}_n)\|^2 = \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}) - 2 \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}_n) + \boldsymbol{\phi}(\mathbf{x}_n)^T \boldsymbol{\phi}(\mathbf{x}_n)
   $$
2. 内積 $\boldsymbol{\phi}(\mathbf{u})^T \boldsymbol{\phi}(\mathbf{v})$ をカーネル関数 $k(\mathbf{u}, \mathbf{v})$ に置換すると：
   $$
   \text{dist}_k^2(\mathbf{x}, \mathbf{x}_n) = [ \text{①} ] - 2 k(\mathbf{x}, \mathbf{x}_n) + [ \text{②} ]
   $$
3. 最近傍決定則は、このカーネル化自乗距離を最小化するインデックス $n^* = \arg\min_n \text{dist}_k^2(\mathbf{x}, \mathbf{x}_n)$ を見つけ、そのクラスラベル $t_{n^*}$ を割り当てることである。
4. なお、定常カーネル（$k(\mathbf{x}, \mathbf{x}) = \text{const}$）の場合、自乗距離の最小化は $k(\mathbf{x}, \mathbf{x}_n)$ の**最大化**と等価になる。

### 穴埋めの解答
- ①: $k(\mathbf{x}, \mathbf{x})$
- ②: $k(\mathbf{x}_n, \mathbf{x}_n)$"""

    ex6_3_code = r"""# Exercise 6.3 数値検証: 特徴空間ユークリッド距離とカーネル表現の一致
# 多項式カーネル k(x, x') = (x^T x' + 1)^2 に対する明示的特徴写像
# 2次元入力 x = (x1, x2) に対し phi(x) = (x1^2, sqrt(2)*x1*x2, x2^2, sqrt(2)*x1, sqrt(2)*x2, 1)
x = np.array([1.2, -0.7])
xn = np.array([-0.5, 1.4])

def phi_poly2(v):
    return np.array([v[0]**2, np.sqrt(2)*v[0]*v[1], v[1]**2, np.sqrt(2)*v[0], np.sqrt(2)*v[1], 1.0])

def k_poly2(u, v):
    return (np.dot(u, v) + 1.0)**2

# 明示的特徴空間での距離自乗
dist_feature_sq = np.sum((phi_poly2(x) - phi_poly2(xn))**2)

# カーネルによる距離自乗
dist_kernel_sq = k_poly2(x, x) - 2 * k_poly2(x, xn) + k_poly2(xn, xn)

assert np.isclose(dist_feature_sq, dist_kernel_sq, atol=1e-10)
print(f"Exercise 6.3 verified: Feature dist^2 ({dist_feature_sq:.6f}) == Kernel dist^2 ({dist_kernel_sq:.6f})")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_3_md), nbf.v4.new_code_cell(ex6_3_code)])

    # ----------------------------------------------------
    # Exercise 6.4
    # ----------------------------------------------------
    ex6_4_md = r"""---
## Exercise 6.4: 正要素行列と正定値性の反例 (Counterexamples for Matrix Positivity)

### 問題
付録Cでは、全ての要素が正であるにもかかわらず負の固有値を持ち、したがって正定値ではない対称行列の例 $\begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$ が示されている。
その逆の性質、すなわち「全ての固有値が正（正定値）であるにもかかわらず、少なくとも1つの負の要素を持つ $2 \times 2$ 行列」の具体例を挙げよ。

### 数理的証明・導出ステップ（穴埋め）
1. 対称 $2 \times 2$ 行列 $\mathbf{A} = \begin{pmatrix} a & b \\ b & c \end{pmatrix}$ が正定値（$\mathbf{A} \succ 0$）であるための必要十分条件は：
   - 対角要素が正: $a > 0, c > 0$
   - 行列式が正: $\det(\mathbf{A}) = [ \text{①} ] > 0$
2. 非対角要素 $b$ を負（$b < 0$）に選んでも、$a c > b^2$ を満たせば正定値行列となる。
3. 例えば $a = 2, c = 2, b = -1$ と選ぶと：
   $$
   \mathbf{A} = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}
   $$
   この行列の非対角要素は $-1 < 0$ である。
4. 固有値を計算すると：
   $$
   \det(\lambda \mathbf{I} - \mathbf{A}) = (\lambda - 2)^2 - 1 = 0 \implies \lambda_1 = 3, \quad \lambda_2 = 1
   $$
   全ての固有値が厳密に正であり、正定値である。

### 穴埋めの解答
- ①: $a c - b^2$"""

    ex6_4_code = r"""# Exercise 6.4 数値検証: 正定値かつ負の要素を持つ行列
A = np.array([[2.0, -1.0],
              [-1.0, 2.0]])

# 1. 負の要素が存在することの確認
assert np.any(A < 0)

# 2. 固有値がすべて厳密に正（正定値）であることの確認
eigvals = np.linalg.eigvalsh(A)
assert np.all(eigvals > 0)
print(f"Exercise 6.4 verified: Matrix A has negative elements (A[0,1]={A[0,1]}) and strictly positive eigenvalues: {eigvals}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_4_md), nbf.v4.new_code_cell(ex6_4_code)])

    # ----------------------------------------------------
    # Exercise 6.5
    # ----------------------------------------------------
    ex6_5_md = r"""---
## Exercise 6.5: 有効なカーネルの構築規則 (6.13) & (6.14) の検証

### 問題
有効なカーネル $k_1(\mathbf{x}, \mathbf{x}')$ に対し、以下の変換が有効なカーネルを生成すること（式 6.13 および 6.14）を証明せよ：
1. $k(\mathbf{x}, \mathbf{x}') = c \, k_1(\mathbf{x}, \mathbf{x}') \quad (c > 0)$
2. $k(\mathbf{x}, \mathbf{x}') = f(\mathbf{x}) \, k_1(\mathbf{x}, \mathbf{x}') \, f(\mathbf{x}')$ （$f(\mathbf{x})$ は任意の実数値関数）

### 数理的証明・導出ステップ（穴埋め）
1. 任意の入力点集合 $\{\mathbf{x}_n\}_{n=1}^N$ に対する $k_1$ のグラム行列を $\mathbf{K}_1$ とする。$k_1$ は有効なカーネルなので、任意のベクトル $\mathbf{v} \in \mathbb{R}^N$ に対し $\mathbf{v}^T \mathbf{K}_1 \mathbf{v} \ge 0$ である。
2. **規則 1 ($k = c k_1$)**:
   グラム行列は $\mathbf{K} = c \mathbf{K}_1$ となる。
   任意の $\mathbf{v}$ に対し：
   $$
   \mathbf{v}^T \mathbf{K} \mathbf{v} = \mathbf{v}^T (c \mathbf{K}_1) \mathbf{v} = c (\mathbf{v}^T \mathbf{K}_1 \mathbf{v}) \ge 0 \quad (\because c > 0)
   $$
   よって $\mathbf{K}$ は半正定値である。
3. **規則 2 ($k = f(\mathbf{x}) k_1(\mathbf{x}, \mathbf{x}') f(\mathbf{x}')$)**:
   対角行列 $\mathbf{D} = \text{diag}(f(\mathbf{x}_1), \dots, f(\mathbf{x}_N))$ を導入すると、グラム行列の要素は $K_{nm} = f(\mathbf{x}_n) (\mathbf{K}_1)_{nm} f(\mathbf{x}_m)$ だから、
   $$
   \mathbf{K} = \mathbf{D} \mathbf{K}_1 \mathbf{D}
   $$
   任意の $\mathbf{v} \in \mathbb{R}^N$ に対し、$\mathbf{u} = \mathbf{D} \mathbf{v}$ とおくと：
   $$
   \mathbf{v}^T \mathbf{K} \mathbf{v} = \mathbf{v}^T \mathbf{D} \mathbf{K}_1 \mathbf{D} \mathbf{v} = [ \text{①} ]^T \mathbf{K}_1 [ \text{①} ] = \mathbf{u}^T \mathbf{K}_1 \mathbf{u} \ge 0
   $$
   よって $\mathbf{K}$ は常に半正定値であり、有効なカーネルである。

### 穴埋めの解答
- ①: $\mathbf{u}$ （または $\mathbf{D} \mathbf{v}$）"""

    ex6_5_code = r"""# Exercise 6.5 数値検証: スケーリングと関数変調による半正定値性
np.random.seed(42)
N = 10
X = np.random.randn(N, 2)
# ベースカーネル (RBF)
dists = np.sum((X[:, None, :] - X[None, :, :])**2, axis=-1)
K1 = np.exp(-0.5 * dists)

# 1. c * K1
c = 3.5
K_c = c * K1
assert np.all(np.linalg.eigvalsh(K_c) >= -1e-10)

# 2. f(x) * K1 * f(x')
f_vals = np.sin(X[:, 0]) + 2.0 * X[:, 1]**2
D = np.diag(f_vals)
K_f = D @ K1 @ D
assert np.all(np.linalg.eigvalsh(K_f) >= -1e-10)

print("Exercise 6.5 verified: Scaled kernel and modulated kernel are strictly positive semidefinite.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_5_md), nbf.v4.new_code_cell(ex6_5_code)])

    # ----------------------------------------------------
    # Exercise 6.6
    # ----------------------------------------------------
    ex6_6_md = r"""---
## Exercise 6.6: 多項式核と指数カーネルの正当性 (6.15) & (6.16)

### 問題
有効なカーネル $k_1(\mathbf{x}, \mathbf{x}')$ に対し、以下が有効なカーネルであることを証明せよ：
1. $k(\mathbf{x}, \mathbf{x}') = q(k_1(\mathbf{x}, \mathbf{x}'))$ （$q(\cdot)$ は非負係数を持つ多項式）
2. $k(\mathbf{x}, \mathbf{x}') = \exp(k_1(\mathbf{x}, \mathbf{x}'))$

### 数理的証明・導出ステップ（穴埋め）
1. カーネルの積 $k_1(\mathbf{x}, \mathbf{x}') k_2(\mathbf{x}, \mathbf{x}')$ は**シューアの積定理 (Schur Product Theorem)** により半正定値行列の要素ごとのアダマール積となり、有効なカーネルである。
2. したがって、$k_1^m(\mathbf{x}, \mathbf{x}')$ も任意の自然数 $m$ に対し有効なカーネルである。
3. 多項式 $q(t) = \sum_{m=0}^M c_m t^m$ において各係数 $c_m \ge 0$ であれば、有効なカーネルの正係数線形結合となるため、$q(k_1)$ は半正定値である。
4. 指数関数はマクローリン展開により：
   $$
   \exp(k_1(\mathbf{x}, \mathbf{x}')) = \sum_{m=0}^\infty \frac{1}{m!} [ \text{①} ]
   $$
   全ての係数 $1/m! > 0$ は正であり、半正定値行列の収束する極限は半正定値であるため、$\exp(k_1)$ は常に有効なカーネルである。

### 穴埋めの解答
- ①: $k_1^m(\mathbf{x}, \mathbf{x}')$"""

    ex6_6_code = r"""# Exercise 6.6 数値検証: 多項式核と指数核の半正定値性
np.random.seed(42)
N = 12
X = np.random.randn(N, 3)
K1 = X @ X.T  # 線形カーネル (PSD)

# 1. 多項式 q(t) = 2 + 3*t + 1.5*t^2 + 0.5*t^3
K_poly = 2.0 + 3.0 * K1 + 1.5 * (K1**2) + 0.5 * (K1**3)
eig_poly = np.linalg.eigvalsh(K_poly)
assert np.all(eig_poly >= -1e-9)

# 2. 指数カーネル exp(K1)
K_exp = np.exp(K1)
eig_exp = np.linalg.eigvalsh(K_exp)
assert np.all(eig_exp >= -1e-9)

print(f"Exercise 6.6 verified: Poly min eig={np.min(eig_poly):.6e}, Exp min eig={np.min(eig_exp):.6e} >= 0.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_6_md), nbf.v4.new_code_cell(ex6_6_code)])

    # ----------------------------------------------------
    # Exercise 6.7
    # ----------------------------------------------------
    ex6_7_md = r"""---
## Exercise 6.7: カーネルの和と積の正当性 (6.17) & (6.18)

### 問題
有効なカーネル $k_1(\mathbf{x}, \mathbf{x}')$ および $k_2(\mathbf{x}, \mathbf{x}')$ に対し、以下が有効なカーネルであることを証明せよ：
1. $k(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') + k_2(\mathbf{x}, \mathbf{x}')$
2. $k(\mathbf{x}, \mathbf{x}') = k_1(\mathbf{x}, \mathbf{x}') k_2(\mathbf{x}, \mathbf{x}')$

### 数理的証明・導出ステップ（穴埋め）
1. **和の規則**:
   グラム行列は $\mathbf{K} = \mathbf{K}_1 + \mathbf{K}_2$ である。
   任意の $\mathbf{v} \in \mathbb{R}^N$ に対し、
   $$
   \mathbf{v}^T \mathbf{K} \mathbf{v} = \mathbf{v}^T \mathbf{K}_1 \mathbf{v} + \mathbf{v}^T \mathbf{K}_2 \mathbf{v} \ge 0 + 0 = 0
   $$
2. **積の規則（シューアの積定理）**:
   $\mathbf{K}_1$ は半正定値行列なので、固有値分解により $\mathbf{K}_1 = \sum_{i=1}^N \lambda_i \mathbf{u}_i \mathbf{u}_i^T$ （$\lambda_i \ge 0$）と表せる。
   積行列の要素は $K_{nm} = (\mathbf{K}_1)_{nm} (\mathbf{K}_2)_{nm} = \sum_{i=1}^N \lambda_i u_{ni} u_{mi} (\mathbf{K}_2)_{nm}$ である。
   任意のベクトル $\mathbf{v}$ に対し、$\mathbf{z}_i = \mathbf{v} \circ \mathbf{u}_i$ （要素ごとの積 $z_{in} = v_n u_{ni}$）とおくと：
   $$
   \mathbf{v}^T \mathbf{K} \mathbf{v} = \sum_{n=1}^N \sum_{m=1}^N v_n v_m \sum_{i=1}^N \lambda_i u_{ni} u_{mi} (\mathbf{K}_2)_{nm} = \sum_{i=1}^N \lambda_i [ \text{①} ]^T \mathbf{K}_2 [ \text{①} ]
   $$
   $\lambda_i \ge 0$ かつ $\mathbf{K}_2 \succeq 0$ より、各項は非負であるため全体の総和も非負となる。

### 穴埋めの解答
- ①: $\mathbf{z}_i$ （または $\mathbf{v} \circ \mathbf{u}_i$）"""

    ex6_7_code = r"""# Exercise 6.7 数値検証: カーネルの和とアダマール積の半正定値性
np.random.seed(42)
N = 10
X = np.random.randn(N, 2)
K1 = X @ X.T
dists = np.sum((X[:, None, :] - X[None, :, :])**2, axis=-1)
K2 = np.exp(-0.5 * dists)

# 和
K_sum = K1 + K2
assert np.all(np.linalg.eigvalsh(K_sum) >= -1e-10)

# アダマール積
K_prod = K1 * K2
assert np.all(np.linalg.eigvalsh(K_prod) >= -1e-10)

print("Exercise 6.7 verified: Sum and Hadamard product of Gram matrices are positive semidefinite.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_7_md), nbf.v4.new_code_cell(ex6_7_code)])

    # ----------------------------------------------------
    # Exercise 6.8
    # ----------------------------------------------------
    ex6_8_md = r"""---
## Exercise 6.8: 特徴写像合成と半正定値対称双線形形式 (6.19) & (6.20)

### 問題
有効なカーネル構築規則 (6.19) および (6.20) を検証せよ：
1. $k(\mathbf{x}, \mathbf{x}') = k_3(\boldsymbol{\phi}(\mathbf{x}), \boldsymbol{\phi}(\mathbf{x}'))$ （$k_3$ は $\mathbb{R}^M$ 上の有効なカーネル）
2. $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{A} \mathbf{x}'$ （$\mathbf{A} \in \mathbb{R}^{D \times D}$ は半正定値対称行列）

### 数理的証明・導出ステップ（穴埋め）
1. **規則 1**:
   変換された点列 $\mathbf{z}_n = \boldsymbol{\phi}(\mathbf{x}_n) \in \mathbb{R}^M$ を考えると、グラム行列は $K_{nm} = k_3(\mathbf{z}_n, \mathbf{z}_m)$ となる。$k_3$ は有効なカーネルなので、任意の点列に対するグラム行列は半正定値である。
2. **規則 2**:
   $\mathbf{A}$ は半正定値対称行列であるから、コレスキー分解または固有値分解により $\mathbf{A} = \mathbf{L} \mathbf{L}^T$ と因数分解できる。
   したがって：
   $$
   k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{A} \mathbf{x}' = \mathbf{x}^T \mathbf{L} \mathbf{L}^T \mathbf{x}' = (\mathbf{L}^T \mathbf{x})^T (\mathbf{L}^T \mathbf{x}') = [ \text{①} ]^T [ \text{②} ]
   $$
   ここで特徴写像を $\boldsymbol{\psi}(\mathbf{x}) = \mathbf{L}^T \mathbf{x}$ と定義すれば、ユークリッド空間上の標準内積表現となるため、有効なカーネルである。

### 穴埋めの解答
- ①: $\mathbf{L}^T \mathbf{x}$
- ②: $\mathbf{L}^T \mathbf{x}'$"""

    ex6_8_code = r"""# Exercise 6.8 数値検証: A = L L^T によるカーネル合成
D = 4
L = np.random.randn(D, 3)
A = L @ L.T  # 半正定値行列

N = 10
X = np.random.randn(N, D)

# カーネル行列 K_nm = x_n^T A x_m
K_A = X @ A @ X.T
eigvals = np.linalg.eigvalsh(K_A)
assert np.all(eigvals >= -1e-10)

# 特徴空間写像 psi(x) = L^T x との内積との一致
Psi = X @ L
K_psi = Psi @ Psi.T
np.testing.assert_allclose(K_A, K_psi, atol=1e-12)

print("Exercise 6.8 verified: Bilinear kernel x^T A x' is strictly PSD and equals (L^T x)^T (L^T x').")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_8_md), nbf.v4.new_code_cell(ex6_8_code)])

    # ----------------------------------------------------
    # Exercise 6.9
    # ----------------------------------------------------
    ex6_9_md = r"""---
## Exercise 6.9: 直積空間上のカーネルの和と積 (6.21) & (6.22)

### 問題
入力ベクトルが直和分解 $\mathbf{x} = (\mathbf{x}_a, \mathbf{x}_b)$ されているとき、以下が有効なカーネルであることを証明せよ：
1. $k(\mathbf{x}, \mathbf{x}') = k_a(\mathbf{x}_a, \mathbf{x}'_a) + k_b(\mathbf{x}_b, \mathbf{x}'_b)$
2. $k(\mathbf{x}, \mathbf{x}') = k_a(\mathbf{x}_a, \mathbf{x}'_a) \, k_b(\mathbf{x}_b, \mathbf{x}'_b)$

### 数理的証明・導出ステップ（穴埋め）
1. 入力空間 $\mathcal{X} = \mathcal{X}_a \times \mathcal{X}_b$ において、$\tilde{k}_a(\mathbf{x}, \mathbf{x}') = k_a(\mathbf{x}_a, \mathbf{x}'_a)$ は $\mathbf{x} \mapsto \mathbf{x}_a$ という射影特徴写像 $\boldsymbol{\phi}(\mathbf{x}) = \mathbf{x}_a$ と $k_a$ の合成とみなせる（式 6.19）。
2. 同様に $\tilde{k}_b(\mathbf{x}, \mathbf{x}') = k_b(\mathbf{x}_b, \mathbf{x}'_b)$ も $\mathcal{X}$ 上の有効なカーネルである。
3. したがって、$k(\mathbf{x}, \mathbf{x}') = \tilde{k}_a(\mathbf{x}, \mathbf{x}') + \tilde{k}_b(\mathbf{x}, \mathbf{x}')$ は有効なカーネルの和（式 6.17）より有効である。
4. また、$k(\mathbf{x}, \mathbf{x}') = \tilde{k}_a(\mathbf{x}, \mathbf{x}') \tilde{k}_b(\mathbf{x}, \mathbf{x}')$ は有効なカーネルの積（式 6.18）より有効である。
   特徴写像で表すと：
   $$
   \boldsymbol{\phi}(\mathbf{x}) = \boldsymbol{\phi}_a(\mathbf{x}_a) \otimes \boldsymbol{\phi}_b(\mathbf{x}_b)
   $$
   となり、クロネッカー積（テンソル積）特徴空間における内積と一致する。

### 穴埋めの解答
- 核心概念: 射影特徴空間合成 (6.19) と和 (6.17) / 積 (6.18) の組み合わせ"""

    ex6_9_code = r"""# Exercise 6.9 数値検証: 直積空間上のカーネルの半正定値性
N = 10
Xa = np.random.randn(N, 2)
Xb = np.random.randn(N, 3)

Ka = Xa @ Xa.T
Kb = np.exp(-0.5 * np.sum((Xb[:, None, :] - Xb[None, :, :])**2, axis=-1))

# 和
K_sum = Ka + Kb
assert np.all(np.linalg.eigvalsh(K_sum) >= -1e-10)

# 積
K_prod = Ka * Kb
assert np.all(np.linalg.eigvalsh(K_prod) >= -1e-10)

print("Exercise 6.9 verified: Kernels on partitioned direct product spaces are valid PSD.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_9_md), nbf.v4.new_code_cell(ex6_9_code)])

    # ----------------------------------------------------
    # Exercise 6.10
    # ----------------------------------------------------
    ex6_10_md = r"""---
## Exercise 6.10: ランク1カーネルによる関数の学習

### 問題
目的関数 $f(\mathbf{x})$ を学習するためのカーネルとして $k(\mathbf{x}, \mathbf{x}') = f(\mathbf{x}) f(\mathbf{x}')$ を用いると、このカーネルに基づく線形学習モデルの予測関数は常に $f(\mathbf{x})$ に比例する解を見出すことを証明せよ。

### 数理的証明・導出ステップ（穴埋め）
1. カーネル関数 $k(\mathbf{x}, \mathbf{x}') = f(\mathbf{x}) f(\mathbf{x}')$ に対する訓練データのグラム行列は：
   $$
   \mathbf{K} = \mathbf{f} \mathbf{f}^T \quad \left( \mathbf{f} = (f(\mathbf{x}_1), \dots, f(\mathbf{x}_N))^T \right)
   $$
   これは階数1（ランク1）の半正定値対称行列である。
2. 双対正則化最小二乗法の解ベクトル $\mathbf{a} = (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{t}$ を考える。
   シャーマン・モリソンの公式より：
   $$
   (\lambda \mathbf{I} + \mathbf{f} \mathbf{f}^T)^{-1} = \frac{1}{\lambda} \mathbf{I} - \frac{\frac{1}{\lambda^2} \mathbf{f} \mathbf{f}^T}{1 + \frac{1}{\lambda} \mathbf{f}^T \mathbf{f}} = \frac{1}{\lambda} \mathbf{I} - \frac{\mathbf{f} \mathbf{f}^T}{\lambda (\lambda + \|\mathbf{f}\|^2)}
   $$
3. これに $\mathbf{f}^T$ を左から掛けると：
   $$
   \mathbf{f}^T (\mathbf{K} + \lambda \mathbf{I})^{-1} = \frac{\mathbf{f}^T}{\lambda} - \frac{\|\mathbf{f}\|^2 \mathbf{f}^T}{\lambda (\lambda + \|\mathbf{f}\|^2)} = \frac{(\lambda + \|\mathbf{f}\|^2 - \|\mathbf{f}\|^2) \mathbf{f}^T}{\lambda (\lambda + \|\mathbf{f}\|^2)} = \frac{1}{[ \text{①} ]} \mathbf{f}^T
   $$
4. 新しい入力 $\mathbf{x}$ に対するカーネルベクトルは $\mathbf{k}(\mathbf{x}) = f(\mathbf{x}) \mathbf{f}$ であるから、予測値 $y(\mathbf{x})$ は：
   $$
   y(\mathbf{x}) = \mathbf{k}(\mathbf{x})^T (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{t} = f(\mathbf{x}) \mathbf{f}^T (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{t} = \left( \frac{\mathbf{f}^T \mathbf{t}}{\lambda + \|\mathbf{f}\|^2} \right) f(\mathbf{x})
   $$
   係数 $c = \frac{\mathbf{f}^T \mathbf{t}}{\lambda + \|\mathbf{f}\|^2}$ はスカラー定数であるから、予測関数は厳密に $y(\mathbf{x}) \propto f(\mathbf{x})$ となる。

### 穴埋めの解答
- ①: $\lambda + \|\mathbf{f}\|^2$"""

    ex6_10_code = r"""# Exercise 6.10 数値検証: k(x, x') = f(x) f(x') による予測が f(x) に比例することの検証
np.random.seed(42)
N = 15
X_train = np.random.uniform(-2, 2, N)
def target_f(x):
    return np.sin(2.0 * x) + 0.5 * x

f_train = target_f(X_train)
t = f_train + np.random.normal(0, 0.2, N)
lam = 0.1

K = np.outer(f_train, f_train)
a = np.linalg.solve(K + lam * np.eye(N), t)

# 任意テスト点での予測
X_test = np.linspace(-2, 2, 20)
f_test = target_f(X_test)
K_test = np.outer(f_test, f_train)
y_pred = K_test @ a

# 理論係数 c
c_analytic = np.dot(f_train, t) / (lam + np.sum(f_train**2))
np.testing.assert_allclose(y_pred, c_analytic * f_test, atol=1e-12)

print(f"Exercise 6.10 verified: y_pred = {c_analytic:.6f} * f(x) for all test points.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_10_md), nbf.v4.new_code_cell(ex6_10_code)])

    # ----------------------------------------------------
    # Exercise 6.11
    # ----------------------------------------------------
    ex6_11_md = r"""---
## Exercise 6.11: ガウスカーネルの無限次元特徴空間展開

### 問題
1次元ガウスカーネル $k(x, x') = \exp\left( -\frac{(x - x')^2}{2\sigma^2} \right)$ について、式 (6.25) の分解を用い、中間項をベキ級数展開することで、このカーネルが無限次元特徴ベクトルの内積として表現できることを示せ。

### 数理的証明・導出ステップ（穴埋め）
1. 指数部を展開すると：
   $$
   -\frac{(x - x')^2}{2\sigma^2} = -\frac{x^2}{2\sigma^2} + \frac{x x'}{\sigma^2} - \frac{x'^2}{2\sigma^2}
   $$
   したがって：
   $$
   k(x, x') = \exp\left(-\frac{x^2}{2\sigma^2}\right) \exp\left(\frac{x x'}{\sigma^2}\right) \exp\left(-\frac{x'^2}{2\sigma^2}\right)
   $$
2. 中央の因子 $\exp\left(\frac{x x'}{\sigma^2}\right)$ をテイラー展開（ベキ級数展開）すると：
   $$
   \exp\left(\frac{x x'}{\sigma^2}\right) = \sum_{m=0}^\infty \frac{1}{m!} \left(\frac{x x'}{\sigma^2}\right)^m = \sum_{m=0}^\infty \frac{x^m x'^m}{m! \sigma^{2m}}
   $$
3. これを代入して整理すると：
   $$
   k(x, x') = \sum_{m=0}^\infty \left[ \frac{1}{\sqrt{m!} \sigma^m} \exp\left(-\frac{x^2}{2\sigma^2}\right) x^m \right] \left[ \frac{1}{\sqrt{m!} \sigma^m} \exp\left(-\frac{x'^2}{2\sigma^2}\right) x'^m \right]
   $$
4. したがって、無限次元特徴ベクトル $\boldsymbol{\phi}(x) = (\phi_0(x), \phi_1(x), \phi_2(x), \dots)^T$ の第 $m$ 成分を：
   $$
   \phi_m(x) = [ \text{①} ] \exp\left(-\frac{x^2}{2\sigma^2}\right) x^m
   $$
   と定義すれば、$k(x, x') = \boldsymbol{\phi}(x)^T \boldsymbol{\phi}(x') = \sum_{m=0}^\infty \phi_m(x) \phi_m(x')$ と厳密に無限次元内積で表現される。

### 穴埋めの解答
- ①: $\frac{1}{\sqrt{m!} \sigma^m}$"""

    ex6_11_code = r"""# Exercise 6.11 数値検証: ガウスカーネルの級数打ち切り近似と真値の一致
import math

x, x_prime = 0.7, -0.4
sigma = 1.0

k_exact = np.exp(- (x - x_prime)**2 / (2.0 * sigma**2))

# 級数を M=25 項まで計算
M_terms = 25
phi_x = np.zeros(M_terms)
phi_xp = np.zeros(M_terms)
for m in range(M_terms):
    coeff = 1.0 / (np.sqrt(float(math.factorial(m))) * (sigma**m))
    phi_x[m] = coeff * np.exp(- x**2 / (2 * sigma**2)) * (x**m)
    phi_xp[m] = coeff * np.exp(- x_prime**2 / (2 * sigma**2)) * (x_prime**m)

k_approx = np.dot(phi_x, phi_xp)
assert np.isclose(k_exact, k_approx, atol=1e-12)
print(f"Exercise 6.11 verified: Exact RBF={k_exact:.10f}, Truncated Series={k_approx:.10f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_11_md), nbf.v4.new_code_cell(ex6_11_code)])

    # ----------------------------------------------------
    # Exercise 6.12
    # ----------------------------------------------------
    ex6_12_md = r"""---
## Exercise 6.12: 集合カーネルとベキ集合特徴空間

### 問題
有限集合 $D$ の部分集合全体の集合（ベキ集合 $\mathcal{P}(D)$）を考える。
2つの部分集合 $A_1, A_2 \subseteq D$ に対し、集合カーネルが次のように定義される：
$$
k(A_1, A_2) = 2^{|A_1 \cap A_2|}
$$
このカーネルが、部分集合 $U \subseteq D$ でインデックス付けされた $2^{|D|}$ 次元の特徴写像 $\phi_U(A) = \mathbb{I}(U \subseteq A)$ の内積として表現できることを示せ。

### 数理的証明・導出ステップ（穴埋め）
1. 特徴ベクトル $\boldsymbol{\phi}(A)$ の各成分は、$D$ の任意の部分集合 $U \subseteq D$ に対し次のように定義される：
   $$
   \phi_U(A) = \begin{cases} 1 & (U \subseteq A) \\ 0 & (\text{otherwise}) \end{cases}
   $$
2. 2つの部分集合 $A_1, A_2$ に対する特徴ベクトルの内積は：
   $$
   \langle \boldsymbol{\phi}(A_1), \boldsymbol{\phi}(A_2) \rangle = \sum_{U \subseteq D} \phi_U(A_1) \phi_U(A_2)
   $$
3. 積 $\phi_U(A_1) \phi_U(A_2) = 1$ となるのは、$U \subseteq A_1$ かつ $U \subseteq A_2$、すなわち：
   $$
   U \subseteq [ \text{①} ]
   $$
   のときであり、それ以外では $0$ である。
4. したがって、内積の値は「$A_1 \cap A_2$ のすべての部分集合の個数」に等しい。
   有限集合 $S$ の部分集合の総数は $2^{|S|}$ であるから：
   $$
   \sum_{U \subseteq (A_1 \cap A_2)} 1 = 2^{|A_1 \cap A_2|} = k(A_1, A_2)
   $$
   となり、カーネル $k(A_1, A_2)$ が $2^{|D|}$ 次元特徴空間での内積と完全に一致することが証明された。

### 穴埋めの解答
- ①: $A_1 \cap A_2$"""

    ex6_12_code = r"""# Exercise 6.12 数値検証: 集合カーネル 2^|A1 \cap A2| と 2^|D| 次元特徴空間内積の一致
from itertools import chain, combinations

D_set = {1, 2, 3, 4}
all_subsets = list(chain.from_iterable(combinations(D_set, r) for r in range(len(D_set)+1)))
all_subsets = [set(s) for s in all_subsets]

A1 = {1, 2, 3}
A2 = {2, 3, 4}

# カーネル直接計算
k_val = 2 ** len(A1.intersection(A2))

# 2^|D| 次元特徴ベクトル phi(A)
phi_A1 = np.array([1.0 if U.issubset(A1) else 0.0 for U in all_subsets])
phi_A2 = np.array([1.0 if U.issubset(A2) else 0.0 for U in all_subsets])

inner_prod = np.dot(phi_A1, phi_A2)
assert k_val == inner_prod == 4  # |{2, 3}| = 2 -> 2^2 = 4
print(f"Exercise 6.12 verified: Kernel={k_val} == Feature Inner Product={inner_prod}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_12_md), nbf.v4.new_code_cell(ex6_12_code)])

    # ----------------------------------------------------
    # Exercise 6.13
    # ----------------------------------------------------
    ex6_13_md = r"""---
## Exercise 6.13: フィッシャーカーネルのパラメータ非線形変換不変性

### 問題
フィッシャーカーネル（Fisher Kernel; 式 6.33）
$$
k(\mathbf{x}, \mathbf{x}') = \mathbf{g}(\boldsymbol{\theta}, \mathbf{x})^T \mathbf{F}^{-1} \mathbf{g}(\boldsymbol{\theta}, \mathbf{x}')
$$
は、正則で微分可能なパラメータの非線形変換 $\boldsymbol{\theta} \to \boldsymbol{\psi}(\boldsymbol{\theta})$ に対して不変であることを証明せよ。

### 数理的証明・導出ステップ（穴埋め）
1. フィッシャースコアベクトルは $\mathbf{g}(\boldsymbol{\theta}, \mathbf{x}) = \nabla_{\boldsymbol{\theta}} \log p(\mathbf{x} | \boldsymbol{\theta})$ で定義される。
   ヤコビアン行列を $\mathbf{J} = \frac{\partial \boldsymbol{\theta}}{\partial \boldsymbol{\psi}}$ ($J_{ij} = \frac{\partial \theta_i}{\partial \psi_j}$) とすると、連鎖律より：
   $$
   \mathbf{g}(\boldsymbol{\psi}, \mathbf{x}) = \nabla_{\boldsymbol{\psi}} \log p(\mathbf{x} | \boldsymbol{\theta}(\boldsymbol{\psi})) = \mathbf{J}^T \nabla_{\boldsymbol{\theta}} \log p(\mathbf{x} | \boldsymbol{\theta}) = [ \text{①} ]
   $$
2. フィッシャー情報行列 $\mathbf{F}_{\boldsymbol{\psi}}$ は：
   $$
   \mathbf{F}_{\boldsymbol{\psi}} = \mathbb{E}_{\mathbf{x}} [\mathbf{g}(\boldsymbol{\psi}, \mathbf{x}) \mathbf{g}(\boldsymbol{\psi}, \mathbf{x})^T] = \mathbf{J}^T \mathbb{E}_{\mathbf{x}} [\mathbf{g}(\boldsymbol{\theta}, \mathbf{x}) \mathbf{g}(\boldsymbol{\theta}, \mathbf{x})^T] \mathbf{J} = \mathbf{J}^T \mathbf{F}_{\boldsymbol{\theta}} \mathbf{J}
   $$
3. その逆行列は：
   $$
   \mathbf{F}_{\boldsymbol{\psi}}^{-1} = (\mathbf{J}^T \mathbf{F}_{\boldsymbol{\theta}} \mathbf{J})^{-1} = \mathbf{J}^{-1} \mathbf{F}_{\boldsymbol{\theta}}^{-1} (\mathbf{J}^T)^{-1}
   $$
4. 変換後のフィッシャーカーネル $k_{\boldsymbol{\psi}}(\mathbf{x}, \mathbf{x}')$ に代入すると：
   $$
   k_{\boldsymbol{\psi}}(\mathbf{x}, \mathbf{x}') = (\mathbf{J}^T \mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x}))^T [\mathbf{J}^{-1} \mathbf{F}_{\boldsymbol{\theta}}^{-1} (\mathbf{J}^T)^{-1}] (\mathbf{J}^T \mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x}'))
   $$
   $$
   = \mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x})^T \mathbf{J} \mathbf{J}^{-1} \mathbf{F}_{\boldsymbol{\theta}}^{-1} (\mathbf{J}^T)^{-1} \mathbf{J}^T \mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x}') = \mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x})^T \mathbf{F}_{\boldsymbol{\theta}}^{-1} \mathbf{g}_{\boldsymbol{\theta}}(\mathbf{x}') = k_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{x}')
   $$
   となり、完全にパラメータ化の選び方に不変であることが示された。

### 穴埋めの解答
- ①: $\mathbf{J}^T \mathbf{g}(\boldsymbol{\theta}, \mathbf{x})$"""

    ex6_13_code = r"""# Exercise 6.13 数値検証: パラメータ再パラメータ化に対するフィッシャーカーネル不変性
# 1次元指数分布 p(x | theta) = theta * exp(-theta * x)
# 再パラメータ化 psi = theta^2 (d theta / d psi = 1 / (2 * sqrt(psi)) = 1 / (2 * theta))
theta = 2.5
psi = theta**2
x1, x2 = 0.8, 1.4

# 1. theta 空間
# log p = log(theta) - theta * x -> g_theta = 1/theta - x
g1_theta = 1.0 / theta - x1
g2_theta = 1.0 / theta - x2
# F_theta = E[(1/theta - x)^2] = Var[x] = 1 / theta^2
F_theta = 1.0 / (theta**2)
k_theta = g1_theta * (1.0 / F_theta) * g2_theta

# 2. psi 空間
# log p = 0.5 * log(psi) - sqrt(psi) * x -> g_psi = 1/(2*psi) - x / (2*sqrt(psi))
g1_psi = 1.0 / (2.0 * psi) - x1 / (2.0 * np.sqrt(psi))
g2_psi = 1.0 / (2.0 * psi) - x2 / (2.0 * np.sqrt(psi))
# J = d theta / d psi = 1 / (2 * theta)
J = 1.0 / (2.0 * theta)
F_psi = (J**2) * F_theta
k_psi = g1_psi * (1.0 / F_psi) * g2_psi

np.testing.assert_allclose(k_theta, k_psi, atol=1e-12)
print(f"Exercise 6.13 verified: k_theta={k_theta:.8f} == k_psi={k_psi:.8f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_13_md), nbf.v4.new_code_cell(ex6_13_code)])

    # ----------------------------------------------------
    # Exercise 6.14
    # ----------------------------------------------------
    ex6_14_md = r"""---
## Exercise 6.14: ガウス分布に対するフィッシャーカーネルの導出

### 問題
共分散行列 $\mathbf{S}$ が既知で固定された平均パラメータ $\boldsymbol{\mu}$ を持つ多変量ガウス分布 $p(\mathbf{x} | \boldsymbol{\mu}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \mathbf{S})$ に対するフィッシャーカーネルの具体的な形を求めよ。

### 数理的証明・導出ステップ（穴埋め）
1. 対数尤度関数は：
   $$
   \log p(\mathbf{x} | \boldsymbol{\mu}) = -\frac{D}{2} \log(2\pi) - \frac{1}{2} \log |\mathbf{S}| - \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{S}^{-1} (\mathbf{x} - \boldsymbol{\mu})
   $$
2. $\boldsymbol{\mu}$ に関するフィッシャースコアは：
   $$
   \mathbf{g}(\boldsymbol{\mu}, \mathbf{x}) = \nabla_{\boldsymbol{\mu}} \log p(\mathbf{x} | \boldsymbol{\mu}) = \mathbf{S}^{-1} (\mathbf{x} - \boldsymbol{\mu})
   $$
3. フィッシャー情報行列 $\mathbf{F}$ は：
   $$
   \mathbf{F} = \mathbb{E}_{\mathbf{x}} [\mathbf{g} \mathbf{g}^T] = \mathbf{S}^{-1} \mathbb{E}_{\mathbf{x}} [(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T] \mathbf{S}^{-1} = \mathbf{S}^{-1} \mathbf{S} \mathbf{S}^{-1} = [ \text{①} ]
   $$
   したがって $\mathbf{F}^{-1} = \mathbf{S}$ である。
4. フィッシャーカーネルは：
   $$
   k(\mathbf{x}, \mathbf{x}') = \mathbf{g}(\boldsymbol{\mu}, \mathbf{x})^T \mathbf{F}^{-1} \mathbf{g}(\boldsymbol{\mu}, \mathbf{x}') = [\mathbf{S}^{-1} (\mathbf{x} - \boldsymbol{\mu})]^T \mathbf{S} [\mathbf{S}^{-1} (\mathbf{x}' - \boldsymbol{\mu})]
   $$
   $$
   = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{S}^{-1} \mathbf{S} \mathbf{S}^{-1} (\mathbf{x}' - \boldsymbol{\mu}) = [ \text{②} ]
   $$
   これは平均 $\boldsymbol{\mu}$ を原点とし、精度行列 $\mathbf{S}^{-1}$ で重み付けしたマハラノビス内積（双線形カーネル）となる。

### 穴埋めの解答
- ①: $\mathbf{S}^{-1}$
- ②: $(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{S}^{-1} (\mathbf{x}' - \boldsymbol{\mu})$"""

    ex6_14_code = r"""# Exercise 6.14 数値検証: 多変量ガウス分布のフィッシャーカーネル
D = 3
mu = np.array([0.5, -1.0, 2.0])
A = np.random.randn(D, D)
S = A @ A.T + np.eye(D)  # 共分散行列
S_inv = np.linalg.inv(S)

x1 = np.array([1.0, 0.0, 1.5])
x2 = np.array([-0.5, -1.2, 2.5])

# スコアベクトル
g1 = S_inv @ (x1 - mu)
g2 = S_inv @ (x2 - mu)
# F^-1 = S
k_fisher = g1.T @ S @ g2

# 解析解
k_analytic = (x1 - mu).T @ S_inv @ (x2 - mu)
assert np.isclose(k_fisher, k_analytic, atol=1e-12)
print(f"Exercise 6.14 verified: Fisher Kernel={k_fisher:.8f} == Analytic Mahalanobis={k_analytic:.8f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_14_md), nbf.v4.new_code_cell(ex6_14_code)])

    # ----------------------------------------------------
    # Exercise 6.15
    # ----------------------------------------------------
    ex6_15_md = r"""---
## Exercise 6.15: 2×2 グラム行列式とコーシー・シュワルツの不等式

### 問題
有効な正定値カーネル $k(\mathbf{x}, \mathbf{x}')$ に対し、$2 \times 2$ グラム行列の行列式を考慮することにより、以下のコーシー・シュワルツの不等式（式 6.96）が成立することを証明せよ：
$$
k(\mathbf{x}_1, \mathbf{x}_2)^2 \le k(\mathbf{x}_1, \mathbf{x}_1) k(\mathbf{x}_2, \mathbf{x}_2)
$$

### 数理的証明・導出ステップ（穴埋め）
1. 任意の2点 $\mathbf{x}_1, \mathbf{x}_2$ に対するグラム行列は対称行列であり：
   $$
   \mathbf{K} = \begin{pmatrix} k(\mathbf{x}_1, \mathbf{x}_1) & k(\mathbf{x}_1, \mathbf{x}_2) \\ k(\mathbf{x}_2, \mathbf{x}_1) & k(\mathbf{x}_2, \mathbf{x}_2) \end{pmatrix}
   $$
2. $k$ は有効なカーネルであるため、グラム行列 $\mathbf{K}$ は半正定値行列（$\mathbf{K} \succeq 0$）である。
3. 半正定値行列のすべての主小行列式および全体の行列式は非負であるため：
   $$
   \det(\mathbf{K}) = [ \text{①} ] - k(\mathbf{x}_1, \mathbf{x}_2)^2 \ge 0
   $$
4. 移項することにより直ちに：
   $$
   k(\mathbf{x}_1, \mathbf{x}_2)^2 \le k(\mathbf{x}_1, \mathbf{x}_1) k(\mathbf{x}_2, \mathbf{x}_2)
   $$
   が得られる。特徴空間の内積 $\langle \boldsymbol{\phi}(\mathbf{x}_1), \boldsymbol{\phi}(\mathbf{x}_2) \rangle^2 \le \|\boldsymbol{\phi}(\mathbf{x}_1)\|^2 \|\boldsymbol{\phi}(\mathbf{x}_2)\|^2$ と完全に同値である。

### 穴埋めの解答
- ①: $k(\mathbf{x}_1, \mathbf{x}_1) k(\mathbf{x}_2, \mathbf{x}_2)$"""

    ex6_15_code = r"""# Exercise 6.15 数値検証: コーシー・シュワルツ不等式の検証
def rbf_k(x1, x2):
    return np.exp(-np.sum((x1 - x2)**2))

x1 = np.array([0.5, 1.2])
x2 = np.array([-0.3, 0.8])

k11 = rbf_k(x1, x1)
k22 = rbf_k(x2, x2)
k12 = rbf_k(x1, x2)

assert k12**2 <= k11 * k22
print(f"Exercise 6.15 verified: k(x1, x2)^2 = {k12**2:.6f} <= k11 * k22 = {k11 * k22:.6f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_15_md), nbf.v4.new_code_cell(ex6_15_code)])

    # ----------------------------------------------------
    # Exercise 6.16
    # ----------------------------------------------------
    ex6_16_md = r"""---
## Exercise 6.16: リプレゼンター定理 (Representer Theorem) の証明

### 問題
パラメータ $\mathbf{w}$ に依存する誤差関数が次の形を持つとする：
$$
J(\mathbf{w}) = f(\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_1), \dots, \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_N)) + g(\mathbf{w}^T \mathbf{w})
$$
ここで $g(\cdot)$ は狭義単調増加関数とする。このとき、$J(\mathbf{w})$ を最小化する任意の重みベクトル $\mathbf{w}$ は、データ点の特徴ベクトルの線形結合 $\mathbf{w} = \sum_{n=1}^N a_n \boldsymbol{\phi}(\mathbf{x}_n)$ として展開できることを示せ。

### 数理的証明・導出ステップ（穴埋め）
1. データ空間の張る部分空間 $\mathcal{S} = \text{span}\{\boldsymbol{\phi}(\mathbf{x}_1), \dots, \boldsymbol{\phi}(\mathbf{x}_N)\}$ と、その直交補空間 $\mathcal{S}^\perp$ を考える。
2. 任意の重みベクトル $\mathbf{w}$ は次のように一意に直交分解できる：
   $$
   \mathbf{w} = \mathbf{w}_\parallel + \mathbf{w}_\perp \quad (\mathbf{w}_\parallel \in \mathcal{S}, \quad \mathbf{w}_\perp \in \mathcal{S}^\perp)
   $$
3. 任意のデータ点 $\mathbf{x}_n$ に対し、直交性より $\mathbf{w}_\perp^T \boldsymbol{\phi}(\mathbf{x}_n) = 0$ であるから：
   $$
   \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) = (\mathbf{w}_\parallel + \mathbf{w}_\perp)^T \boldsymbol{\phi}(\mathbf{x}_n) = [ \text{①} ]
   $$
   したがって、データ適合項 $f$ の値は $\mathbf{w}_\perp$ に一切依存せず、$f(\dots \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) \dots) = f(\dots \mathbf{w}_\parallel^T \boldsymbol{\phi}(\mathbf{x}_n) \dots)$ である。
4. 一方、正則化項内のノルム自乗を展開すると：
   $$
   \mathbf{w}^T \mathbf{w} = \|\mathbf{w}_\parallel + \mathbf{w}_\perp\|^2 = \|\mathbf{w}_\parallel\|^2 + \|\mathbf{w}_\perp\|^2 \ge \|\mathbf{w}_\parallel\|^2
   $$
   等号成立は $\mathbf{w}_\perp = \mathbf{0}$ のときに限られる。
5. $g$ は狭義単調増加関数であるため：
   $$
   g(\mathbf{w}^T \mathbf{w}) \ge g(\|\mathbf{w}_\parallel\|^2)
   $$
   であり、$\mathbf{w}_\perp \ne \mathbf{0}$ のとき厳密に $J(\mathbf{w}) > J(\mathbf{w}_\parallel)$ となる。
6. したがって最小解では必ず $\mathbf{w}_\perp = \mathbf{0}$ でなければならず、$\mathbf{w} = \mathbf{w}_\parallel \in \mathcal{S}$、すなわち $\mathbf{w} = \sum_{n=1}^N a_n \boldsymbol{\phi}(\mathbf{x}_n)$ の形でなければならない。

### 穴埋めの解答
- ①: $\mathbf{w}_\parallel^T \boldsymbol{\phi}(\mathbf{x}_n)$"""

    ex6_16_code = r"""# Exercise 6.16 数値検証: 直交成分 w_perp が必ずノルムと正則化損失を増大させることの検証
N, D = 8, 20
Phi = np.random.randn(N, D)
# S の正規直交基底 (mode='complete' で (D, D) 行列を得る)
Q, R = np.linalg.qr(Phi.T, mode='complete')
Q_parallel = Q[:, :N]
Q_perp = Q[:, N:]

w_parallel = Q_parallel @ np.random.randn(N)
w_perp = Q_perp @ np.random.randn(D - N)
w_total = w_parallel + w_perp

# データ適合値の一致
np.testing.assert_allclose(Phi @ w_total, Phi @ w_parallel, atol=1e-12)

# 正則化項の狭義増大
assert np.sum(w_total**2) > np.sum(w_parallel**2)
print("Exercise 6.16 verified: w_perp does not change predictions but strictly increases penalty.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_16_md), nbf.v4.new_code_cell(ex6_16_code)])

    # ----------------------------------------------------
    # Exercise 6.17
    # ----------------------------------------------------
    ex6_17_md = r"""---
## Exercise 6.17: 入力ノイズ平均化誤差の変分法とナダラヤ・ワトソン解の導出

### 問題
入力ノイズ $\boldsymbol{\xi} \sim \nu(\boldsymbol{\xi})$ を伴う二乗和誤差関数（式 6.39）
$$
E = \frac{1}{2} \sum_{n=1}^N \int \{y(\mathbf{x}_n - \boldsymbol{\xi}) - t_n\}^2 \nu(\boldsymbol{\xi}) d\boldsymbol{\xi}
$$
を変分法（付録D）を用いて関数 $y(\mathbf{x})$ に関して最小化し、最適解が式 (6.40) のナダラヤ・ワトソン核回帰モデル
$$
y(\mathbf{x}) = \frac{\sum_{n=1}^N t_n \nu(\mathbf{x} - \mathbf{x}_n)}{\sum_{m=1}^N \nu(\mathbf{x} - \mathbf{x}_m)}
$$
となることを証明せよ。

### 数理的証明・導出ステップ（穴埋め）
1. 積分変数を $\mathbf{z} = \mathbf{x}_n - \boldsymbol{\xi}$ と置換すると、$\boldsymbol{\xi} = \mathbf{x}_n - \mathbf{z}$ より：
   $$
   E[y] = \frac{1}{2} \sum_{n=1}^N \int \{y(\mathbf{z}) - t_n\}^2 \nu(\mathbf{x}_n - \mathbf{z}) d\mathbf{z}
   $$
2. 被積分関数のダミー変数を $\mathbf{x}$ に書き換え、積分と総和を入れ替えると：
   $$
   E[y] = \int \left[ \frac{1}{2} \sum_{n=1}^N \{y(\mathbf{x}) - t_n\}^2 \nu(\mathbf{x}_n - \mathbf{x}) \right] d\mathbf{x}
   $$
3. $y(\mathbf{x})$ に関する変分導関数（汎関数微分）をとってゼロとおくと：
   $$
   \frac{\delta E}{\delta y(\mathbf{x})} = \sum_{n=1}^N [ \text{①} ] \nu(\mathbf{x}_n - \mathbf{x}) = 0
   $$
4. $y(\mathbf{x})$ について展開して解くと：
   $$
   y(\mathbf{x}) \sum_{n=1}^N \nu(\mathbf{x}_n - \mathbf{x}) - \sum_{n=1}^N t_n \nu(\mathbf{x}_n - \mathbf{x}) = 0
   $$
   $$
   y(\mathbf{x}) = \frac{\sum_{n=1}^N t_n \nu(\mathbf{x} - \mathbf{x}_n)}{\sum_{m=1}^N \nu(\mathbf{x} - \mathbf{x}_m)} = \sum_{n=1}^N k(\mathbf{x}, \mathbf{x}_n) t_n
   $$
   ここでノイズ分布が対称（$\nu(-\boldsymbol{\xi}) = \nu(\boldsymbol{\xi})$）であるとし、等価カーネル $k(\mathbf{x}, \mathbf{x}_n) = \frac{\nu(\mathbf{x} - \mathbf{x}_n)}{\sum_m \nu(\mathbf{x} - \mathbf{x}_m)}$ は $\sum_n k(\mathbf{x}, \mathbf{x}_n) = 1$ を満たす。

### 穴埋めの解答
- ①: $y(\mathbf{x}) - t_n$"""

    ex6_17_code = r"""# Exercise 6.17 数値検証: 入力ノイズ平均化変分最適解とナダラヤ・ワトソン予測の一致
N = 10
X = np.random.uniform(-2, 2, N)
t = np.sin(X) + np.random.normal(0, 0.1, N)
sigma_noise = 0.5

def nu(xi):
    return stats.norm.pdf(xi, loc=0, scale=sigma_noise)

# ナダラヤ・ワトソン核回帰予測
x_eval = 0.3
weights = nu(x_eval - X)
y_nw = np.sum(weights * t) / np.sum(weights)

# 変分導関数の勾配が y_nw で厳密にゼロになることの確認
grad_at_nw = np.sum((y_nw - t) * nu(X - x_eval))
assert np.isclose(grad_at_nw, 0.0, atol=1e-12)
print(f"Exercise 6.17 verified: dE/dy({x_eval}) = {grad_at_nw:.4e} == 0 at Nadaraya-Watson solution.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_17_md), nbf.v4.new_code_cell(ex6_17_code)])

    # ----------------------------------------------------
    # Exercise 6.18
    # ----------------------------------------------------
    ex6_18_md = r"""---
## Exercise 6.18: ナダラヤ・ワトソンモデルの条件付き分布・期待値・条件付き分散

### 問題
入力 $x$ と目標変数 $t$ が等方性ガウス分布コンポーネント（共分散 $\sigma^2 \mathbf{I}$）を持つナダラヤ・ワトソンモデルを考える。
条件付き確率密度 $p(t | x)$、条件付き期待値 $\mathbb{E}[t | x]$、および条件付き分散 $\text{var}[t | x]$ の解析的表現を等価カーネル $k(x, x_n)$ を用いて表せ。

### 数理的証明・導出ステップ（穴埋め）
1. パリツェン窓型密度推定による同時確率密度 $p(x, t)$ は：
   $$
   p(x, t) = \frac{1}{N} \sum_{n=1}^N \mathcal{N}(x | x_n, \sigma^2) \mathcal{N}(t | t_n, \sigma^2)
   $$
2. 周辺確率密度 $p(x) = \int p(x, t) dt = \frac{1}{N} \sum_{n=1}^N \mathcal{N}(x | x_n, \sigma^2)$ より、条件付き密度は：
   $$
   p(t | x) = \frac{p(x, t)}{p(x)} = \sum_{n=1}^N \left( \frac{\mathcal{N}(x | x_n, \sigma^2)}{\sum_{m=1}^N \mathcal{N}(x | x_m, \sigma^2)} \right) \mathcal{N}(t | t_n, \sigma^2) = \sum_{n=1}^N k(x, x_n) \mathcal{N}(t | t_n, \sigma^2)
   $$
3. 条件付き期待値は：
   $$
   \mathbb{E}[t | x] = \int t \, p(t | x) dt = \sum_{n=1}^N k(x, x_n) \int t \, \mathcal{N}(t | t_n, \sigma^2) dt = [ \text{①} ]
   $$
4. 条件付き二次モーメントは $\int t^2 \mathcal{N}(t | t_n, \sigma^2) dt = \sigma^2 + t_n^2$ より：
   $$
   \mathbb{E}[t^2 | x] = \sum_{n=1}^N k(x, x_n) (\sigma^2 + t_n^2) = \sigma^2 + \sum_{n=1}^N k(x, x_n) t_n^2
   $$
5. 条件付き分散は全分散の法則（式 5.158 と同型）より：
   $$
   \text{var}[t | x] = \mathbb{E}[t^2 | x] - (\mathbb{E}[t | x])^2 = \sigma^2 + \sum_{n=1}^N k(x, x_n) t_n^2 - \left( \sum_{n=1}^N k(x, x_n) t_n \right)^2
   $$
   $$
   = [ \text{②} ]
   $$

### 穴埋めの解答
- ①: $\sum_{n=1}^N k(x, x_n) t_n$
- ②: $\sigma^2 + \sum_{n=1}^N k(x, x_n) (t_n - \mathbb{E}[t | x])^2$"""

    ex6_18_code = r"""# Exercise 6.18 数値検証: ナダラヤ・ワトソンの条件付き期待値と全分散の数値積分一致
x_train = np.array([-1.0, 0.0, 1.0])
t_train = np.array([0.5, -0.2, 1.2])
sigma = 0.4

x_eval = 0.2
weights = stats.norm.pdf(x_eval, loc=x_train, scale=sigma)
k_vec = weights / np.sum(weights)

mean_analytic = np.sum(k_vec * t_train)
var_analytic = sigma**2 + np.sum(k_vec * (t_train - mean_analytic)**2)

# 数値積分による検証
import scipy.integrate as integrate

t_grid = np.linspace(-3, 4, 1000)
p_t_given_x = np.zeros_like(t_grid)
for n in range(len(x_train)):
    p_t_given_x += k_vec[n] * stats.norm.pdf(t_grid, loc=t_train[n], scale=sigma)

mean_num = integrate.trapezoid(t_grid * p_t_given_x, t_grid)
var_num = integrate.trapezoid((t_grid - mean_num)**2 * p_t_given_x, t_grid)

assert np.isclose(mean_analytic, mean_num, atol=1e-4)
assert np.isclose(var_analytic, var_num, atol=1e-4)
print(f"Exercise 6.18 verified: Analytic Mean={mean_analytic:.5f} (Num={mean_num:.5f}), Var={var_analytic:.5f} (Num={var_num:.5f})")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_18_md), nbf.v4.new_code_cell(ex6_18_code)])

    # ----------------------------------------------------
    # Exercise 6.19
    # ----------------------------------------------------
    ex6_19_md = r"""---
## Exercise 6.19: 入力および出力ノイズ問題からのナダラヤ・ワトソンモデルの導出

### 問題
真の入力点 $z_n$ が未観測であり、ノイズを含んだ観測値 $x_n = z_n + \xi_n$ （$\xi_n \sim g(\xi)$）と目標値 $t_n = y(z_n) + \epsilon_n$ が与えられた回帰問題を考える。
ノイズ分布に関する期待二乗和誤差を変分法で最小化することにより、ナダラヤ・ワトソンカーネル回帰解が得られることを示せ。

### 数理的証明・導出ステップ（穴埋め）
1. 誤差関数をノイズ分布 $g(\boldsymbol{\xi})$ で平均化した期待損失は：
   $$
   E = \frac{1}{2} \sum_{n=1}^N \int \{y(x_n - \boldsymbol{\xi}_n) - t_n\}^2 g(\boldsymbol{\xi}_n) d\boldsymbol{\xi}_n
   $$
2. これは演習 6.17 の誤差関数式 (6.99) と数学的に完全に同一の形式である。
3. したがって変分法による定常条件 $\frac{\delta E}{\delta y(\mathbf{x})} = 0$ は直ちに：
   $$
   y(\mathbf{x}) = \frac{\sum_{n=1}^N t_n g(\mathbf{x} - \mathbf{x}_n)}{\sum_{m=1}^N g(\mathbf{x} - \mathbf{x}_m)} = \sum_{n=1}^N k(\mathbf{x}, \mathbf{x}_n) t_n
   $$
   を導き、観測された入力ノイズ分布 $g$ がそのまま等価カーネルのプロファイルとなる。

### 穴埋めの解答
- 核心概念: 演習6.17の変分導出の一致と、ノイズ密度 $g$ による等価カーネルの形成"""

    ex6_19_code = r"""# Exercise 6.19 数値検証: 未知潜在点 z のノイズ下での変分最適性と等価カーネル
N = 8
Z_true = np.random.uniform(-1, 1, N)
t = 2.0 * Z_true + np.random.normal(0, 0.1, N)
# 観測入力 X = Z + xi
xi = np.random.normal(0, 0.3, N)
X_obs = Z_true + xi

def g(x):
    return stats.norm.pdf(x, loc=0, scale=0.3)

x_test = 0.5
k_weights = g(x_test - X_obs)
y_sol = np.sum(k_weights * t) / np.sum(k_weights)

# 停留点条件 sum_n (y - t_n) g(x_test - x_n) = 0
residual = np.sum((y_sol - t) * g(x_test - X_obs))
assert np.isclose(residual, 0.0, atol=1e-12)
print(f"Exercise 6.19 verified: Variational optimality residual = {residual:.4e} == 0.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_19_md), nbf.v4.new_code_cell(ex6_19_code)])

    # ----------------------------------------------------
    # Exercise 6.20
    # ----------------------------------------------------
    ex6_20_md = r"""---
## Exercise 6.20: ガウス過程回帰の予測平均と予測分散の導出 (6.66) & (6.67)

### 問題
同時ガウス分布の条件付き分布の公式（2.81および2.82）を用いて、ガウス過程回帰における新しい入力 $\mathbf{x}_{N+1}$ に対する条件付き予測分布 $p(t_{N+1} | \mathbf{t}_N)$ の平均 $m(\mathbf{x}_{N+1})$（式 6.66）および分散 $\sigma^2(\mathbf{x}_{N+1})$（式 6.67）を導出せよ。

### 数理的証明・導出ステップ（穴埋め）
1. 訓練目標値 $\mathbf{t}_N = (t_1, \dots, t_N)^T$ とテスト目標値 $t_{N+1}$ の結合事前分布は平均ゼロの多変量正規分布である：
   $$
   p(\mathbf{t}_{N+1}) = \mathcal{N}\left( \mathbf{0}, \mathbf{C}_{N+1} \right)
   $$
2. 共分散行列 $\mathbf{C}_{N+1}$ をブロック分割すると：
   $$
   \mathbf{C}_{N+1} = \begin{pmatrix} \mathbf{C}_N & \mathbf{k} \\ \mathbf{k}^T & c \end{pmatrix}
   $$
   ここで $\mathbf{C}_N = \mathbf{K} + \sigma^2 \mathbf{I}_N$、$k_n = k(\mathbf{x}_n, \mathbf{x}_{N+1})$、$c = k(\mathbf{x}_{N+1}, \mathbf{x}_{N+1}) + \sigma^2$ である。
3. ガウス分布の分割条件付き分布公式（2.81, 2.82）
   $\boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} (\mathbf{x}_b - \boldsymbol{\mu}_b)$、
   $\boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} \boldsymbol{\Sigma}_{ba}$
   を適用すると：
   - 予測平均（式 6.66）:
     $$
     m(\mathbf{x}_{N+1}) = [ \text{①} ]^T \mathbf{C}_N^{-1} \mathbf{t}_N
     $$
   - 予測分散（式 6.67）:
     $$
     \sigma^2(\mathbf{x}_{N+1}) = c - [ \text{①} ]^T \mathbf{C}_N^{-1} [ \text{①} ] = k(\mathbf{x}_{N+1}, \mathbf{x}_{N+1}) + \sigma^2 - \mathbf{k}^T \mathbf{C}_N^{-1} \mathbf{k}
     $$

### 穴埋めの解答
- ①: $\mathbf{k}$"""

    ex6_20_code = r"""# Exercise 6.20 数値検証: 多変量ガウス条件付き分布公式とGP回帰の完全一致
N = 5
X_train = np.random.randn(N, 2)
t_train = np.random.randn(N)
x_new = np.random.randn(1, 2)
noise_var = 0.2

def rbf(x1, x2):
    return np.exp(-0.5 * np.sum((x1[:, None, :] - x2[None, :, :])**2, axis=-1))

K_NN = rbf(X_train, X_train)
C_N = K_NN + noise_var * np.eye(N)
k_vec = rbf(X_train, x_new).ravel()
c_scalar = rbf(x_new, x_new)[0, 0] + noise_var

# 1. 式 (6.66), (6.67)
m_gp = k_vec.T @ np.linalg.solve(C_N, t_train)
var_gp = c_scalar - k_vec.T @ np.linalg.solve(C_N, k_vec)

# 2. 全体共分散行列からの条件付き分布計算
C_full = np.empty((N + 1, N + 1))
C_full[:N, :N] = C_N
C_full[:N, N] = k_vec
C_full[N, :N] = k_vec
C_full[N, N] = c_scalar
inv_C_full = np.linalg.inv(C_full)
# シューア補元公式による検証
var_schur = 1.0 / inv_C_full[-1, -1]
np.testing.assert_allclose(var_gp, var_schur, atol=1e-12)

print(f"Exercise 6.20 verified: Predictive mean={m_gp:.6f}, variance={var_gp:.6f} matches Schur complement.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_20_md), nbf.v4.new_code_cell(ex6_20_code)])

    # ----------------------------------------------------
    # Exercise 6.21
    # ----------------------------------------------------
    ex6_21_md = r"""---
## Exercise 6.21: ガウス過程回帰とベイズ線形回帰の予測分布の等価性証明

### 問題
カーネル関数が固定された非線形基底関数 $\boldsymbol{\phi}(\mathbf{x})$ によって $k(\mathbf{x}, \mathbf{x}') = \alpha^{-1} \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}')$ と定義されるガウス過程回帰モデルを考える。
行列の恒等式 (C.6) および (C.7)（ウッドベリーの公式）を用いて、このGP回帰の予測分布が第3章のベイズ線形回帰の予測分布（式 3.58 および 3.59）と厳密に同一であることを示せ。

### 数理的証明・導出ステップ（穴埋め）
1. 計画行列 $\boldsymbol{\Phi} \in \mathbb{R}^{N \times M}$、目標ノイズ精度 $\beta = \sigma^{-2}$ とする。
   GPの共分散行列とベクトルは：
   $$
   \mathbf{C}_N = \alpha^{-1} \boldsymbol{\Phi} \boldsymbol{\Phi}^T + \beta^{-1} \mathbf{I}_N, \quad \mathbf{k}(\mathbf{x}) = \alpha^{-1} \boldsymbol{\Phi} \boldsymbol{\phi}(\mathbf{x})
   $$
2. **予測平均の等価性**:
   行列の恒等式 (C.6): $(\mathbf{P}^{-1} + \mathbf{B}^T \mathbf{R}^{-1} \mathbf{B})^{-1} \mathbf{B}^T \mathbf{R}^{-1} = \mathbf{P} \mathbf{B}^T (\mathbf{B} \mathbf{P} \mathbf{B}^T + \mathbf{R})^{-1}$ において、
   $\mathbf{P} = \alpha^{-1} \mathbf{I}_M$, $\mathbf{R} = \beta^{-1} \mathbf{I}_N$, $\mathbf{B} = \boldsymbol{\Phi}$ とおくと：
   $$
   \mathbf{C}_N^{-1} \boldsymbol{\Phi} = (\alpha^{-1} \boldsymbol{\Phi} \boldsymbol{\Phi}^T + \beta^{-1} \mathbf{I}_N)^{-1} \boldsymbol{\Phi} = \beta \boldsymbol{\Phi} (\alpha \mathbf{I}_M + \beta \boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} = \beta \boldsymbol{\Phi} \mathbf{S}_N
   $$
   したがって GP の予測平均は：
   $$
   m(\mathbf{x}) = \mathbf{k}(\mathbf{x})^T \mathbf{C}_N^{-1} \mathbf{t} = \alpha^{-1} \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\Phi}^T \mathbf{C}_N^{-1} \mathbf{t} = \boldsymbol{\phi}(\mathbf{x})^T [ \beta \mathbf{S}_N \boldsymbol{\Phi}^T \mathbf{t} ] = \boldsymbol{\phi}(\mathbf{x})^T \mathbf{m}_N
   $$
   となり、ベイズ線形回帰の平均 $\mathbf{m}_N^T \boldsymbol{\phi}(\mathbf{x})$ と完全に一致する。
3. **予測分散の等価性**:
   行列の恒等式 (C.7): $(\mathbf{A} + \mathbf{B} \mathbf{D}^{-1} \mathbf{C})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1} \mathbf{B} (\mathbf{D} + \mathbf{C} \mathbf{A}^{-1} \mathbf{B})^{-1} \mathbf{C} \mathbf{A}^{-1}$ を用いると：
   $$
   \mathbf{k}(\mathbf{x})^T \mathbf{C}_N^{-1} \mathbf{k}(\mathbf{x}) = \alpha^{-1} \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}) - \boldsymbol{\phi}(\mathbf{x})^T \mathbf{S}_N \boldsymbol{\phi}(\mathbf{x})
   $$
   したがって GP の予測分散は：
   $$
   \sigma^2(\mathbf{x}) = k(\mathbf{x}, \mathbf{x}) + \beta^{-1} - \mathbf{k}^T \mathbf{C}_N^{-1} \mathbf{k} = \alpha^{-1} \boldsymbol{\phi}^T \boldsymbol{\phi} + \beta^{-1} - (\alpha^{-1} \boldsymbol{\phi}^T \boldsymbol{\phi} - \boldsymbol{\phi}^T \mathbf{S}_N \boldsymbol{\phi}) = \frac{1}{\beta} + \boldsymbol{\phi}(\mathbf{x})^T \mathbf{S}_N \boldsymbol{\phi}(\mathbf{x})
   $$
   となり、式 (3.59) の $\sigma_N^2(\mathbf{x})$ と完全に一致する。

### 穴埋めの解答
- 核心等価性: ウッドベリーの公式による $O(N^3)$ GP予測と $O(M^3)$ ベイズ線形回帰予測の双対代数的一致"""

    ex6_21_code = r"""# Exercise 6.21 数値検証: GP回帰とベイズ線形回帰の予測分布の完全数値一致
from common import BayesianLinearRegression

N, M = 20, 5
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha_val, beta_val = 1.5, 3.0

# 1. ベイズ線形回帰
blr = BayesianLinearRegression(alpha=alpha_val, beta=beta_val)
blr.fit(Phi, t)
phi_test = np.random.randn(4, M)
m_blr, std_blr = blr.predict(phi_test)
var_blr = std_blr ** 2

# 2. ガウス過程回帰
# k(x, x') = 1/alpha * phi(x)^T phi(x')
K = (1.0 / alpha_val) * (Phi @ Phi.T)
C_N = K + (1.0 / beta_val) * np.eye(N)
k_test = (1.0 / alpha_val) * (phi_test @ Phi.T)

m_gp = k_test @ np.linalg.solve(C_N, t)
var_gp = np.zeros(4)
for i in range(4):
    c_self = (1.0 / alpha_val) * np.dot(phi_test[i], phi_test[i]) + (1.0 / beta_val)
    var_gp[i] = c_self - k_test[i] @ np.linalg.solve(C_N, k_test[i])

np.testing.assert_allclose(m_blr, m_gp, atol=1e-12)
np.testing.assert_allclose(var_blr, var_gp, atol=1e-12)

print("Exercise 6.21 verified: GP and Bayesian Linear Regression predictions match to machine precision.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_21_md), nbf.v4.new_code_cell(ex6_21_code)])

    # ----------------------------------------------------
    # Exercise 6.22
    # ----------------------------------------------------
    ex6_22_md = r"""---
## Exercise 6.22: 複数テスト点に対する結合予測分布と周辺化整合性

### 問題
$N$ 個の訓練点 $\mathbf{x}_1, \dots, \mathbf{x}_N$ と $L$ 個のテスト点 $\mathbf{x}_{N+1}, \dots, \mathbf{x}_{N+L}$ に対するテスト目標値ベクトル $\mathbf{t}_* = (t_{N+1}, \dots, t_{N+L})^T$ の結合予測分布 $p(\mathbf{t}_* | \mathbf{t}_N)$ を導出せよ。
さらに、その任意の1成分 $t_j$ ($N+1 \le j \le N+L$) に関する周辺分布が、単一テスト点のGP回帰結果（式 6.66 および 6.67）と厳密に一致することを示せ。

### 数理的証明・導出ステップ（穴埋め）
1. 訓練データ $\mathbf{t}_N \in \mathbb{R}^N$ とテストデータ $\mathbf{t}_* \in \mathbb{R}^L$ の結合事前分布は：
   $$
   p(\mathbf{t}_N, \mathbf{t}_*) = \mathcal{N}\left( \mathbf{0}, \begin{pmatrix} \mathbf{C}_N & \mathbf{K}_* \\ \mathbf{K}_*^T & \mathbf{C}_* \end{pmatrix} \right)
   $$
   ここで $(\mathbf{K}_*)_{nl} = k(\mathbf{x}_n, \mathbf{x}_{N+l})$、$(\mathbf{C}_*)_{ll'} = k(\mathbf{x}_{N+l}, \mathbf{x}_{N+l'}) + \sigma^2 \delta_{ll'}$ である。
2. ガウス分布の条件付き分布の公式（2.81, 2.82）より、結合予測分布は：
   $$
   p(\mathbf{t}_* | \mathbf{t}_N) = \mathcal{N}\left( \mathbf{t}_* | \boldsymbol{\mu}_*, \boldsymbol{\Sigma}_* \right)
   $$
   - 結合予測平均ベクトル:
     $$
     \boldsymbol{\mu}_* = [ \text{①} ]^T \mathbf{C}_N^{-1} \mathbf{t}_N
     $$
   - 結合予測共分散行列:
     $$
     \boldsymbol{\Sigma}_* = \mathbf{C}_* - [ \text{①} ]^T \mathbf{C}_N^{-1} [ \text{①} ]
     $$
3. 周辺化整合性:
   多変量正規分布 $\mathcal{N}(\boldsymbol{\mu}_*, \boldsymbol{\Sigma}_*)$ から任意の単一変数 $t_j$ を周辺化して取り出す操作は、平均ベクトルの第 $j$ 成分 $\mu_{*j}$ と共分散行列の対角要素 $(\boldsymbol{\Sigma}_*)_{jj}$ をそのまま抽出することに等しい。
   $$
   \mu_{*j} = \mathbf{k}_j^T \mathbf{C}_N^{-1} \mathbf{t}_N = m(\mathbf{x}_j), \quad (\boldsymbol{\Sigma}_*)_{jj} = c_j - \mathbf{k}_j^T \mathbf{C}_N^{-1} \mathbf{k}_j = \sigma^2(\mathbf{x}_j)
   $$
   これは単一テスト点に対する式 (6.66) および (6.67) と完全に一致する。

### 穴埋めの解答
- ①: $\mathbf{K}_*$"""

    ex6_22_code = r"""# Exercise 6.22 数値検証: 結合テスト予測分布の対角成分と単独テスト予測の一致
N, L = 10, 4
X_tr = np.random.randn(N, 2)
t_tr = np.random.randn(N)
X_te = np.random.randn(L, 2)
noise = 0.15

def rbf_mat(A, B):
    return np.exp(-0.5 * np.sum((A[:, None, :] - B[None, :, :])**2, axis=-1))

C_N = rbf_mat(X_tr, X_tr) + noise * np.eye(N)
K_star = rbf_mat(X_tr, X_te)
C_star = rbf_mat(X_te, X_te) + noise * np.eye(L)

# 結合予測分布
mu_joint = K_star.T @ np.linalg.solve(C_N, t_tr)
Sigma_joint = C_star - K_star.T @ np.linalg.solve(C_N, K_star)

# 1点ごとの単独予測
for l in range(L):
    k_l = K_star[:, l]
    c_l = C_star[l, l]
    m_single = k_l @ np.linalg.solve(C_N, t_tr)
    var_single = c_l - k_l @ np.linalg.solve(C_N, k_l)
    assert np.isclose(mu_joint[l], m_single, atol=1e-12)
    assert np.isclose(Sigma_joint[l, l], var_single, atol=1e-12)

print("Exercise 6.22 verified: Joint GP predictive marginals strictly match single test point GP formulas.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_22_md), nbf.v4.new_code_cell(ex6_22_code)])

    # ----------------------------------------------------
    # Exercise 6.23
    # ----------------------------------------------------
    ex6_23_md = r"""---
## Exercise 6.23: 多次元目標変数に対するガウス過程回帰

### 問題
目標変数 $\mathbf{t}$ が $D$ 次元のベクトルである場合のガウス過程回帰モデルを考える。
テスト入力 $\mathbf{x}_{N+1}$ に対する条件付き分布 $p(\mathbf{t}_{N+1} | \mathbf{T})$ （$\mathbf{T} \in \mathbb{R}^{N \times D}$）を記述せよ。

### 数理的証明・導出ステップ（穴埋め）
1. 各目標変数の次元 $d \in \{1, \dots, D\}$ が入力 $\mathbf{x}$ が与えられた下で独立であると仮定する場合、事前分布は $D$ 個の独立なガウス過程の積となる：
   $$
   p(\mathbf{T}) = \prod_{d=1}^D \mathcal{N}(\mathbf{t}_{:, d} | \mathbf{0}, \mathbf{C}_N)
   $$
2. テスト入力 $\mathbf{x}_{N+1}$ に対する各次元 $t_{N+1, d}$ の条件付き分布は、スカラーGP回帰の結果を適用して：
   $$
   p(t_{N+1, d} | \mathbf{t}_{:, d}) = \mathcal{N}(t_{N+1, d} | \mathbf{k}^T \mathbf{C}_N^{-1} \mathbf{t}_{:, d}, \, \sigma^2(\mathbf{x}_{N+1}))
   $$
3. これらをベクトル形式にまとめると：
   $$
   p(\mathbf{t}_{N+1} | \mathbf{T}) = \mathcal{N}\left(\mathbf{t}_{N+1} \,|\, \mathbf{M}_{N+1}, \, \sigma^2(\mathbf{x}_{N+1}) [ \text{①} ] \right)
   $$
   ここで予測平均ベクトルは：
   $$
   \mathbf{M}_{N+1} = \mathbf{T}^T \mathbf{C}_N^{-1} \mathbf{k} \in \mathbb{R}^D
   $$
   であり、スカラー予測分散 $\sigma^2(\mathbf{x}_{N+1}) = k(\mathbf{x}_{N+1}, \mathbf{x}_{N+1}) + \sigma^2 - \mathbf{k}^T \mathbf{C}_N^{-1} \mathbf{k}$ はすべての次元で共通となる。

### 穴埋めの解答
- ①: $\mathbf{I}_D$"""

    ex6_23_code = r"""# Exercise 6.23 数値検証: 多次元出力 GP 回帰の平均ベクトルと共分散
N, D = 12, 3
X_tr = np.random.randn(N, 2)
T_tr = np.random.randn(N, D)
x_new = np.random.randn(1, 2)

C_N = rbf_mat(X_tr, X_tr) + 0.1 * np.eye(N)
k_vec = rbf_mat(X_tr, x_new).ravel()
c_val = rbf_mat(x_new, x_new)[0, 0] + 0.1

# 多次元予測平均 M = T^T C_N^-1 k
mean_multi = T_tr.T @ np.linalg.solve(C_N, k_vec)
var_scalar = c_val - k_vec @ np.linalg.solve(C_N, k_vec)

# 各次元の独立 GP 回帰との比較
for d in range(D):
    m_d = k_vec @ np.linalg.solve(C_N, T_tr[:, d])
    assert np.isclose(mean_multi[d], m_d, atol=1e-12)

print(f"Exercise 6.23 verified: Multidimensional GP mean shape={mean_multi.shape}, var={var_scalar:.6f}.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_23_md), nbf.v4.new_code_cell(ex6_23_code)])

    # ----------------------------------------------------
    # Exercise 6.24
    # ----------------------------------------------------
    ex6_24_md = r"""---
## Exercise 6.24: 対角重み行列と正定値行列の和の正定値性

### 問題
1. 各対角要素が $0 < W_{ii} < 1$ を満たす対角行列 $\mathbf{W}$ が正定値であることを示せ。
2. 2つの正定値行列 $\mathbf{A} \succ 0$ と $\mathbf{B} \succ 0$ の和 $\mathbf{A} + \mathbf{B}$ が正定値であることを示せ。

### 数理的証明・導出ステップ（穴埋め）
1. **対角行列 $\mathbf{W}$ の正定値性**:
   任意の非ゼロベクトル $\mathbf{v} \in \mathbb{R}^N \setminus \{\mathbf{0}\}$ に対し：
   $$
   \mathbf{v}^T \mathbf{W} \mathbf{v} = \sum_{i=1}^N W_{ii} v_i^2
   $$
   仮定より各 $W_{ii} > 0$ であり、かつ $\mathbf{v} \ne \mathbf{0}$ より少なくとも1つの $v_i^2 > 0$ が存在する。したがって $\sum_{i=1}^N W_{ii} v_i^2 > 0$ となり、$\mathbf{W}$ は厳密に正定値である。
2. **正定値行列の和の正定値性**:
   $\mathbf{A} \succ 0$ かつ $\mathbf{B} \succ 0$ であるから、任意の $\mathbf{v} \ne \mathbf{0}$ に対し：
   $$
   \mathbf{v}^T \mathbf{A} \mathbf{v} > 0 \quad \text{かつ} \quad \mathbf{v}^T \mathbf{B} \mathbf{v} > 0
   $$
   したがって：
   $$
   \mathbf{v}^T (\mathbf{A} + \mathbf{B}) \mathbf{v} = \mathbf{v}^T \mathbf{A} \mathbf{v} + \mathbf{v}^T \mathbf{B} \mathbf{v} > 0 + 0 = 0
   $$
   となり、$\mathbf{A} + \mathbf{B}$ も厳密に正定値である。
3. この結果により、ガウス過程分類のヘッセ行列 $-\nabla\nabla \Psi = \mathbf{W} + \mathbf{C}_N^{-1}$ は常に狭義正定値となり、事後分布の対数確率密度が狭義上に凸で唯一の大域的極大点（モード）を持つことが保証される。

### 穴埋めの解答
- 核心概念: 2次形式の非ゼロ正値性と、事後分布の厳密凹性 (Strict Concavity)"""

    ex6_24_code = r"""# Exercise 6.24 数値検証: 対角行列 W と正定値行列の和の正定値性
N = 10
# 0 < W_ii < 1
W_diag = np.random.uniform(0.01, 0.99, N)
W = np.diag(W_diag)
assert np.all(np.linalg.eigvalsh(W) > 0)

# 正定値行列 A, B
M1 = np.random.randn(N, N)
A = M1 @ M1.T + 0.1 * np.eye(N)
M2 = np.random.randn(N, N)
B = M2 @ M2.T + 0.2 * np.eye(N)

assert np.all(np.linalg.eigvalsh(A) > 0)
assert np.all(np.linalg.eigvalsh(B) > 0)

A_plus_B = A + B
eig_sum = np.linalg.eigvalsh(A_plus_B)
assert np.all(eig_sum > 0)
print(f"Exercise 6.24 verified: Min eigenvalue of (A + B) = {np.min(eig_sum):.6f} > 0.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_24_md), nbf.v4.new_code_cell(ex6_24_code)])

    # ----------------------------------------------------
    # Exercise 6.25
    # ----------------------------------------------------
    ex6_25_md = r"""---
## Exercise 6.25: ガウス過程分類の Newton-Raphson 事後モード更新式 (6.83) の導出

### 問題
ニュートン・ラフソン反復公式（4.92）を用いて、ガウス過程分類（GPC）モデルにおける潜在変数ベクトル $\mathbf{a}_N$ の事後分布のモード $\mathbf{a}_N^*$ を探索する反復更新式 (6.83)
$$
\mathbf{a}_N^{(\text{new})} = \mathbf{C}_N (\mathbf{I} + \mathbf{W} \mathbf{C}_N)^{-1} (\mathbf{t}_N - \boldsymbol{\sigma}_N + \mathbf{W} \mathbf{a}_N)
$$
を導出せよ。

### 数理的証明・導出ステップ（穴埋め）
1. 事後分布の対数確率密度は：
   $$
   \Psi(\mathbf{a}_N) = \log p(\mathbf{a}_N | \mathbf{t}_N) = -\frac{1}{2} \mathbf{a}_N^T \mathbf{C}_N^{-1} \mathbf{a}_N + \sum_{n=1}^N [t_n \log \sigma(a_n) + (1 - t_n) \log(1 - \sigma(a_n))] + \text{const}
   $$
2. $\mathbf{a}_N$ に関する勾配ベクトル $\mathbf{g}$ は：
   $$
   \mathbf{g} = \nabla \Psi(\mathbf{a}_N) = \mathbf{t}_N - \boldsymbol{\sigma}_N - \mathbf{C}_N^{-1} \mathbf{a}_N \quad (\sigma_n = \sigma(a_n))
   $$
3. ヘッセ行列 $\mathbf{H}$ は：
   $$
   \mathbf{H} = \nabla \nabla \Psi(\mathbf{a}_N) = - [ \text{①} ] - \mathbf{C}_N^{-1}
   $$
   ここで $\mathbf{W} = \text{diag}(\sigma_n (1 - \sigma_n))$ である。
4. ニュートン・ラフソン更新 $\mathbf{a}^{(\text{new})} = \mathbf{a} - \mathbf{H}^{-1} \mathbf{g}$ に代入すると：
   $$
   \mathbf{a}^{(\text{new})} = \mathbf{a} + (\mathbf{W} + \mathbf{C}_N^{-1})^{-1} (\mathbf{t}_N - \boldsymbol{\sigma}_N - \mathbf{C}_N^{-1} \mathbf{a})
   $$
5. 行列の因数分解恒等式 $(\mathbf{W} + \mathbf{C}_N^{-1})^{-1} = \mathbf{C}_N (\mathbf{I} + \mathbf{W} \mathbf{C}_N)^{-1}$ を用いると：
   $$
   \mathbf{a}^{(\text{new})} = \mathbf{a} + \mathbf{C}_N (\mathbf{I} + \mathbf{W} \mathbf{C}_N)^{-1} (\mathbf{t}_N - \boldsymbol{\sigma}_N) - \mathbf{C}_N (\mathbf{I} + \mathbf{W} \mathbf{C}_N)^{-1} \mathbf{C}_N^{-1} \mathbf{a}
   $$
   $$
   = \mathbf{a} - (\mathbf{I} + \mathbf{C}_N \mathbf{W})^{-1} \mathbf{a} + \mathbf{C}_N (\mathbf{I} + \mathbf{W} \mathbf{C}_N)^{-1} (\mathbf{t}_N - \boldsymbol{\sigma}_N)
   $$
   共通因子 $\mathbf{C}_N (\mathbf{I} + \mathbf{W} \mathbf{C}_N)^{-1}$ でくくると：
   $$
   \mathbf{a}^{(\text{new})} = \mathbf{C}_N (\mathbf{I} + \mathbf{W} \mathbf{C}_N)^{-1} [ \mathbf{W} \mathbf{a}_N + \mathbf{t}_N - \boldsymbol{\sigma}_N ]
   $$
   となり、式 (6.83) が得られる。

### 穴埋めの解答
- ①: $\mathbf{W}$"""

    ex6_25_code = r"""# Exercise 6.25 数値検証: GPC Newton-Raphson の収束と最適点勾配ゼロ検証
from common.classification_utils import sigmoid

N = 10
X = np.random.randn(N, 2)
t = np.random.choice([0.0, 1.0], size=N)
C_N = rbf_mat(X, X) + 1e-4 * np.eye(N)

# ニュートン・ラフソン反復 (式 6.83)
a = np.zeros(N)
for iteration in range(25):
    sig = sigmoid(a)
    W_diag = sig * (1.0 - sig)
    W = np.diag(W_diag)
    
    # 式 (6.83)
    inv_term = np.linalg.inv(np.eye(N) + W @ C_N)
    a_new = C_N @ inv_term @ (W @ a + t - sig)
    
    diff = np.max(np.abs(a_new - a))
    a = a_new
    if diff < 1e-8:
        break

# 最適点 a* での勾配 grad = t - sig - C_N^-1 a がゼロであることを検証
sig_opt = sigmoid(a)
grad_opt = t - sig_opt - np.linalg.solve(C_N, a)
assert np.all(np.abs(grad_opt) < 1e-6)
print(f"Exercise 6.25 verified: Mode converged in {iteration+1} steps, max |grad| = {np.max(np.abs(grad_opt)):.4e} < 1e-6.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_25_md), nbf.v4.new_code_cell(ex6_25_code)])

    # ----------------------------------------------------
    # Exercise 6.26
    # ----------------------------------------------------
    ex6_26_md = r"""---
## Exercise 6.26: ガウス過程分類の潜在事後予測分布の平均・分散 (6.87) & (6.88)

### 問題
線形ガウスモデルの事後予測分布の公式（2.115）を用いて、ガウス過程分類におけるテスト点潜在変数 $a_{N+1}$ の事後分布 $p(a_{N+1} | \mathbf{t}_N)$ の平均（式 6.87）
$$
\mathbb{E}[a_{N+1} | \mathbf{t}_N] = \mathbf{k}^T (\mathbf{t}_N - \boldsymbol{\sigma}_N^*)
$$
および分散（式 6.88）
$$
\text{var}[a_{N+1} | \mathbf{t}_N] = c - \mathbf{k}^T (\mathbf{W}^{-1} + \mathbf{C}_N)^{-1} \mathbf{k}
$$
を導出せよ。

### 数理的証明・導出ステップ（穴埋め）
1. ラプラス近似の下で、訓練潜在変数の事後分布は $p(\mathbf{a}_N | \mathbf{t}_N) \simeq \mathcal{N}(\mathbf{a}_N | \mathbf{a}_N^*, \mathbf{A}^{-1})$ と近似される。ここでヘッセ行列の逆行列は $\mathbf{A}^{-1} = (\mathbf{C}_N^{-1} + \mathbf{W})^{-1}$ である。
2. 条件付き事前分布 $p(a_{N+1} | \mathbf{a}_N)$ は正規分布 $\mathcal{N}(a_{N+1} | \mathbf{k}^T \mathbf{C}_N^{-1} \mathbf{a}_N, \, c - \mathbf{k}^T \mathbf{C}_N^{-1} \mathbf{k})$ である。
3. 線形ガウスモデルの畳み込み積分（公式 2.115）を用いると、平均は：
   $$
   \mathbb{E}[a_{N+1} | \mathbf{t}_N] = \mathbf{k}^T \mathbf{C}_N^{-1} \mathbf{a}_N^*
   $$
   モード点 $\mathbf{a}_N^*$ において勾配はゼロであるから、$\mathbf{g} = \mathbf{t}_N - \boldsymbol{\sigma}_N^* - \mathbf{C}_N^{-1} \mathbf{a}_N^* = \mathbf{0}$、すなわち $\mathbf{C}_N^{-1} \mathbf{a}_N^* = [ \text{①} ]$ が成立する。代入すると式 (6.87) が得られる。
4. 分散は：
   $$
   \text{var}[a_{N+1} | \mathbf{t}_N] = c - \mathbf{k}^T \mathbf{C}_N^{-1} \mathbf{k} + \mathbf{k}^T \mathbf{C}_N^{-1} (\mathbf{C}_N^{-1} + \mathbf{W})^{-1} \mathbf{C}_N^{-1} \mathbf{k}
   $$
   $$
   = c - \mathbf{k}^T [\mathbf{C}_N^{-1} - \mathbf{C}_N^{-1} (\mathbf{C}_N^{-1} + \mathbf{W})^{-1} \mathbf{C}_N^{-1}] \mathbf{k}
   $$
5. ウッドベリーの公式（恒等式 C.7）$(\mathbf{C}_N + \mathbf{W}^{-1})^{-1} = \mathbf{C}_N^{-1} - \mathbf{C}_N^{-1} (\mathbf{C}_N^{-1} + \mathbf{W})^{-1} \mathbf{C}_N^{-1}$ を適用すると：
   $$
   \text{var}[a_{N+1} | \mathbf{t}_N] = c - \mathbf{k}^T [ \text{②} ]^{-1} \mathbf{k}
   $$
   となり、式 (6.88) が得られる。

### 穴埋めの解答
- ①: $\mathbf{t}_N - \boldsymbol{\sigma}_N^*$
- ②: $\mathbf{W}^{-1} + \mathbf{C}_N$"""

    ex6_26_code = r"""# Exercise 6.26 数値検証: GPC 潜在事後分布平均・分散の代数的一致
x_new = np.random.randn(1, 2)
k_vec = rbf_mat(X, x_new).ravel()
c_val = rbf_mat(x_new, x_new)[0, 0]

# 式 (6.87)
mean_analytic = k_vec @ (t - sig_opt)
# 定義 k^T C_N^-1 a*
mean_def = k_vec @ np.linalg.solve(C_N, a)
assert np.isclose(mean_analytic, mean_def, atol=1e-6)

# 式 (6.88)
W_inv = np.diag(1.0 / W_diag)
var_analytic = c_val - k_vec @ np.linalg.solve(W_inv + C_N, k_vec)

# 畳み込み分散公式: c - k^T C_N^-1 k + k^T C_N^-1 (C_N^-1 + W)^-1 C_N^-1 k
A_inv = np.linalg.inv(np.linalg.inv(C_N) + W)
var_conv = c_val - k_vec @ np.linalg.solve(C_N, k_vec) + k_vec @ np.linalg.solve(C_N, A_inv @ np.linalg.solve(C_N, k_vec))
assert np.isclose(var_analytic, var_conv, atol=1e-6)

print(f"Exercise 6.26 verified: Predictive mean={mean_analytic:.6f}, variance={var_analytic:.6f} matches convolution.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_26_md), nbf.v4.new_code_cell(ex6_26_code)])

    # ----------------------------------------------------
    # Exercise 6.27
    # ----------------------------------------------------
    ex6_27_md = r"""---
## Exercise 6.27: ラプラス近似 GPC における対数周辺尤度とその超パラメータ勾配

### 問題
ガウス過程分類におけるラプラス近似の対数周辺尤度（式 6.90）
$$
\log p(\mathbf{t}_N | \boldsymbol{\theta}) \simeq \Psi(\mathbf{a}_N^*) - \frac{1}{2} \log |\mathbf{I}_N + \mathbf{C}_N \mathbf{W}|
$$
を導出せよ。
さらに、カーネルの超パラメータ $\theta_j$ に関する対数尤度の勾配各項（式 6.91, 6.92, 6.94）
$$
\frac{\partial}{\partial \theta_j} \log p(\mathbf{t}_N | \boldsymbol{\theta}) = \frac{1}{2} \mathbf{a}_N^{*T} \mathbf{C}_N^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta_j} \mathbf{C}_N^{-1} \mathbf{a}_N^* - \frac{1}{2} \text{Tr}\left( (\mathbf{I} + \mathbf{C}_N \mathbf{W})^{-1} \mathbf{W} \frac{\partial \mathbf{C}_N}{\partial \theta_j} \right)
$$
を導出せよ。

### 数理的証明・導出ステップ（穴埋め）
1. 目的関数 $p(\mathbf{t}_N | \boldsymbol{\theta}) = \int p(\mathbf{t}_N | \mathbf{a}_N) p(\mathbf{a}_N | \boldsymbol{\theta}) d\mathbf{a}_N = \int \exp(\Psi(\mathbf{a}_N)) d\mathbf{a}_N$ に対し、モード $\mathbf{a}_N^*$ 周りで2次展開を行うと：
   $$
   \Psi(\mathbf{a}_N) \simeq \Psi(\mathbf{a}_N^*) - \frac{1}{2} (\mathbf{a}_N - \mathbf{a}_N^*)^T \mathbf{A} (\mathbf{a}_N - \mathbf{a}_N^*) \quad (\mathbf{A} = \mathbf{C}_N^{-1} + \mathbf{W})
   $$
2. ガウス積分を実行すると：
   $$
   \int \exp\left(-\frac{1}{2} (\mathbf{a} - \mathbf{a}^*)^T \mathbf{A} (\mathbf{a} - \mathbf{a}^*)\right) d\mathbf{a} = (2\pi)^{N/2} |\mathbf{A}|^{-1/2}
   $$
   したがって対数周辺尤度は：
   $$
   \log p(\mathbf{t}_N | \boldsymbol{\theta}) \simeq \Psi(\mathbf{a}_N^*) - \frac{1}{2} \log |\mathbf{A}| + \frac{N}{2} \log(2\pi)
   $$
   ここで事前分布の正規化項を含めて整理すると：
   $$
   |\mathbf{A}| |\mathbf{C}_N| = |(\mathbf{C}_N^{-1} + \mathbf{W}) \mathbf{C}_N| = |[ \text{①} ]|
   $$
   これにより式 (6.90) の $\log p(\mathbf{t}_N | \boldsymbol{\theta}) \simeq \Psi(\mathbf{a}_N^*) - \frac{1}{2} \log |\mathbf{I}_N + \mathbf{C}_N \mathbf{W}|$ が得られる。
3. **超パラメータ勾配**:
   全微分 $\frac{d}{d\theta_j} \log p$ をとるとき、モード点では $\nabla_{\mathbf{a}_N} \Psi(\mathbf{a}_N^*) = \mathbf{0}$ であるため、包絡線定理（Envelope Theorem）により陰関数微分項 $\frac{\partial \mathbf{a}_N^*}{\partial \theta_j}$ の寄与は一次のオーダーで完全にゼロとなる。
   - $\Psi(\mathbf{a}_N^*)$ の陽な依存性からの微分：
     $$
     \frac{\partial \Psi}{\partial \theta_j} = -\frac{1}{2} \mathbf{a}_N^{*T} \frac{\partial \mathbf{C}_N^{-1}}{\partial \theta_j} \mathbf{a}_N^* - \frac{1}{2} \frac{\partial \log |\mathbf{C}_N|}{\partial \theta_j} = \frac{1}{2} \mathbf{a}_N^{*T} \mathbf{C}_N^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta_j} \mathbf{C}_N^{-1} \mathbf{a}_N^* - \frac{1}{2} \text{Tr}\left(\mathbf{C}_N^{-1} \frac{\partial \mathbf{C}_N}{\partial \theta_j}\right)
     $$
   - 行列式項の微分：
     $$
     -\frac{1}{2} \frac{\partial}{\partial \theta_j} \log |\mathbf{I} + \mathbf{C}_N \mathbf{W}| = -\frac{1}{2} \text{Tr}\left( (\mathbf{I} + \mathbf{C}_N \mathbf{W})^{-1} \mathbf{W} \frac{\partial \mathbf{C}_N}{\partial \theta_j} \right)
     $$
   これらを足し合わせることで式 (6.91)-(6.94) の解析的勾配が得られる。

### 穴埋めの解答
- ①: $\mathbf{I}_N + \mathbf{C}_N \mathbf{W}$"""

    ex6_27_code = r"""# Exercise 6.27 数値検証: ラプラス近似対数周辺尤度と完全解析的勾配 (陽項 + 陰項) の一致
theta_val = 1.2
dC = rbf_mat(X, X)

def compute_log_marginal_lik(th):
    C = th * dC + 1e-3 * np.eye(N)
    a = np.zeros(N)
    for _ in range(50):
        s = sigmoid(a)
        W = np.diag(s * (1 - s))
        inv_term = np.linalg.inv(np.eye(N) + W @ C)
        a_new = C @ inv_term @ (W @ a + t - s)
        if np.max(np.abs(a_new - a)) < 1e-12:
            a = a_new
            break
        a = a_new
    s = sigmoid(a)
    W = np.diag(s * (1 - s))
    quad = -0.5 * a @ np.linalg.solve(C, a)
    det = -0.5 * np.linalg.slogdet(np.eye(N) + C @ W)[1]
    ll = np.sum(t * np.log(np.clip(s, 1e-12, 1-1e-12)) + (1-t) * np.log(np.clip(1-s, 1e-12, 1-1e-12)))
    return quad + det + ll, a, C, W, s

log_lik, a_opt, C_opt, W_opt, s_opt = compute_log_marginal_lik(theta_val)

# 1. 陽項 (式 6.91)
C_inv_a = np.linalg.solve(C_opt, a_opt)
term_explicit_1 = 0.5 * C_inv_a @ dC @ C_inv_a
term_explicit_2 = -0.5 * np.trace(np.linalg.solve(np.eye(N) + C_opt @ W_opt, W_opt @ dC))
grad_explicit = term_explicit_1 + term_explicit_2

# 2. 最頻値変動による陰項 (式 6.92 - 6.94)
da_dtheta = np.linalg.solve(np.eye(N) + W_opt @ C_opt, dC @ (t - s_opt))
inv_I_CW = np.linalg.inv(np.eye(N) + C_opt @ W_opt)
dW_da = s_opt * (1.0 - s_opt) * (1.0 - 2.0 * s_opt)
M_mat = inv_I_CW @ C_opt
d_det_da = np.diag(M_mat) * dW_da
grad_implicit = -0.5 * np.dot(d_det_da, da_dtheta)

total_analytic = grad_explicit + grad_implicit

# 3. 有限差分数値勾配
eps = 1e-5
log_lik_p, _, _, _, _ = compute_log_marginal_lik(theta_val + eps)
log_lik_m, _, _, _, _ = compute_log_marginal_lik(theta_val - eps)
numeric_grad = (log_lik_p - log_lik_m) / (2 * eps)

np.testing.assert_allclose(total_analytic, numeric_grad, rtol=1e-2, atol=1e-3)
print(f"Exercise 6.27 verified: Analytic Grad={total_analytic:.6f} (Explicit={grad_explicit:.6f}, Implicit={grad_implicit:.6f}) == Numeric Grad={numeric_grad:.6f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex6_27_md), nbf.v4.new_code_cell(ex6_27_code)])

    return cells

def main():
    nb = nbf.v4.new_notebook()
    nb.cells = create_cell_pairs()
    
    out_path = "/home/student/Documents/GitHub/my_PRML/6/6_Exercises.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated {out_path} with {len(nb.cells)} cells.")

if __name__ == "__main__":
    main()
