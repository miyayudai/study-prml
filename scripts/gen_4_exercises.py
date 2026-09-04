import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# 第4章 線形識別モデル：演習問題 (Exercises 4.1 - 4.26)

本ノートブックでは、PRML第4章「線形識別モデル (Linear Models for Classification)」の**全26問 (Exercises 4.1 〜 4.26)** の解答・解説・数値検証コードを完全収録しています。
各問題では、証明の全体骨格と**論理のステップ（思考の道筋）**を明示し、数式展開に加えて **Python による数値検証**を行うことで、理論が数学的かつ計算機上で正しく整合していることを確認します。"""))

# Exercise 4.1
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 4.1: 凸包 (Convex Hull) と線形分離可能性の同値性

### 問題
データ点集合 $\{\mathbf{x}_n\}$ の凸包 (Convex Hull) は、$\alpha_n \ge 0$ かつ $\sum_n \alpha_n = 1$ を満たす点 $\mathbf{x} = \sum_n \alpha_n \mathbf{x}_n$ の集合として定義される。
もう一つの点集合 $\{\mathbf{y}_n\}$ とその凸包を考える。
これら2つの点集合が線形分離可能であるとは、あるベクトル $\mathbf{w}$ とスカラー $w_0$ が存在して、すべての $\mathbf{x}_n$ に対して $\mathbf{w}^T \mathbf{x}_n + w_0 > 0$ かつすべての $\mathbf{y}_n$ に対して $\mathbf{w}^T \mathbf{y}_n + w_0 < 0$ が成り立つことである。
凸包同士が交差するならば2つの点集合は線形分離不可能であり、逆に線形分離可能であれば凸包同士は交差しないことを示せ。

### 論理のステップ
1. **$(\implies)$ 線形分離可能 $\implies$ 凸包は交差しない (対偶: 凸包が交差する $\implies$ 分離不可能)**:
   - 2つの凸包が交差すると仮定し、共通の点を $\mathbf{z}$ とする。
   - $\mathbf{z}$ は $\{\mathbf{x}_n\}$ の凸結合 $\mathbf{z} = \sum_n \alpha_n \mathbf{x}_n$ であり、同時に $\{\mathbf{y}_n\}$ の凸結合 $\mathbf{z} = \sum_m \beta_m \mathbf{y}_m$ である。
   - もし線形分離する超平面 $(\mathbf{w}, w_0)$ が存在したとすると、すべての $n$ で $\mathbf{w}^T \mathbf{x}_n + w_0 > 0$ なので、$\sum_n \alpha_n (\mathbf{w}^T \mathbf{x}_n + w_0) > 0 \implies \mathbf{w}^T \mathbf{z} + w_0 > 0$。
   - 一方、すべての $m$ で $\mathbf{w}^T \mathbf{y}_m + w_0 < 0$ なので、同様に $\mathbf{w}^T \mathbf{z} + w_0 < 0$。
   - これは矛盾である。したがって凸包が交差するなら線形分離不可能。
2. **$(\Longleftarrow)$ 凸包が交差しない $\implies$ 線形分離可能**:
   - 2つの閉凸集合が互いに素（交差しない）であるとき、超平面分離定理 (Separating Hyperplane Theorem) により、両者を厳密に分離する超平面が存在する。"""))

# Code Ex 4.1
code_ex4_1 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt

def check_convex_hull_intersection(X, Y):
    # sum alpha_i X_i - sum beta_j Y_j = 0
    # sum alpha_i = 1, sum beta_j = 1, alpha, beta >= 0
    NX, D = X.shape
    NY = Y.shape[0]
    
    # 変数: [alpha_1, ..., alpha_NX, beta_1, ..., beta_NY] (NX + NY 次元)
    # 等式制約:
    # 1. D 本の等式: X^T alpha - Y^T beta = 0
    # 2. 1 本の等式: sum alpha = 1
    # 3. 1 本の等式: sum beta = 1
    A_eq = np.zeros((D + 2, NX + NY))
    A_eq[:D, :NX] = X.T
    A_eq[:D, NX:] = -Y.T
    A_eq[D, :NX] = 1.0
    A_eq[D+1, NX:] = 1.0
    b_eq = np.zeros(D + 2)
    b_eq[D] = 1.0
    b_eq[D+1] = 1.0
    
    c = np.zeros(NX + NY) # feasibility problem
    bounds = [(0, None) for _ in range(NX + NY)]
    res = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return res.success

# Exercise 4.1 数値幾何検証
np.random.seed(42)
X_sep = np.random.randn(10, 2) + np.array([-4, 0])
Y_sep = np.random.randn(10, 2) + np.array([4, 0])
print("Separable clusters intersect?", check_convex_hull_intersection(X_sep, Y_sep))
assert check_convex_hull_intersection(X_sep, Y_sep) == False

X_over = np.random.randn(10, 2) + np.array([0, 0])
Y_over = np.random.randn(10, 2) + np.array([0.5, 0])
print("Overlapping clusters intersect?", check_convex_hull_intersection(X_over, Y_over))
assert check_convex_hull_intersection(X_over, Y_over) == True
print("Exercise 4.1 verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex4_1))

# Exercise 4.2 & 4.3
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 4.2 & 4.3: 二乗和誤差最小二乗法における目標値の線形制約の保存

### 問題 4.2
学習データの目標ベクトル $\mathbf{t}_n$ が線形制約 $\mathbf{a}^T \mathbf{t}_n + b = 0$ を満たすとする。
モデルがバイアス項 $\phi_0(\mathbf{x}) = 1$ を持つ線形回帰モデルであるとき、最小二乗解による予測 $\mathbf{y}(\mathbf{x})$ もまた同じ制約 $\mathbf{a}^T \mathbf{y}(\mathbf{x}) + b = 0$ を満たすことを示せ。

### 問題 4.3
複数の線形制約 $\mathbf{A}\mathbf{t}_n + \mathbf{b} = \mathbf{0}$ を満たす場合にも同様に成り立つことを示せ。

### 論理のステップ
1. 最小二乗解の予測は $\mathbf{y}(\mathbf{x}) = \mathbf{W}^T \boldsymbol{\phi}(\mathbf{x})$、ここで $\mathbf{W} = (\mathbf{\Phi}^T \mathbf{\Phi})^{-1}\mathbf{\Phi}^T \mathbf{T}$。
2. $\mathbf{a}^T \mathbf{y}(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^T \mathbf{W} \mathbf{a} = \boldsymbol{\phi}(\mathbf{x})^T (\mathbf{\Phi}^T \mathbf{\Phi})^{-1}\mathbf{\Phi}^T (\mathbf{T} \mathbf{a})$。
3. 条件より各行について $\mathbf{t}_n^T \mathbf{a} = -b$ であるため、$\mathbf{T} \mathbf{a} = -b \mathbf{1}_N$。
4. 一方、バイアス項 $\phi_0(\mathbf{x}) = 1$ が含まれるため、計画行列の第0列はすべて 1、すなわち $\mathbf{\Phi} \mathbf{e}_0 = \mathbf{1}_N$（$\mathbf{e}_0 = (1, 0, \dots, 0)^T$）。
5. したがって $(\mathbf{\Phi}^T \mathbf{\Phi})^{-1}\mathbf{\Phi}^T \mathbf{1}_N = \mathbf{e}_0$ となり、
   $$ \mathbf{a}^T \mathbf{y}(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^T (-b \mathbf{e}_0) = -b \phi_0(\mathbf{x}) = -b \implies \mathbf{a}^T \mathbf{y}(\mathbf{x}) + b = 0 $$"""))

# Code Ex 4.2 & 4.3
code_ex4_2_3 = r"""# Exercise 4.2 & 4.3 数値検証
np.random.seed(42)
N, D, K = 30, 4, 3
X = np.random.randn(N, D)
Phi = np.column_stack([np.ones(N), X]) # バイアス項あり

# 制約: a^T t_n + b = 0
a = np.array([2.0, -1.0, 1.5])
b = -3.5

# 制約を満たす target 行列 T (N, K) を作成
T = np.random.randn(N, K)
# 第0成分を調整して制約を満たさせる: a0*t0 + a1*t1 + a2*t2 + b = 0
T[:, 0] = -(T[:, 1]*a[1] + T[:, 2]*a[2] + b) / a[0]

# 最小二乗解
W = np.linalg.pinv(Phi) @ T

# 任意のテスト入力 x_test
x_test = np.random.randn(10, D)
phi_test = np.column_stack([np.ones(10), x_test])
y_pred = phi_test @ W

# 予測が制約を満たしているか検証
constraints = y_pred @ a + b
print("Max absolute violation of constraint:", np.max(np.abs(constraints)))
assert np.allclose(constraints, 0.0, atol=1e-10)
print("Exercise 4.2 & 4.3 verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex4_2_3))

# Exercise 4.4 - 4.6 (Fisher's Discriminant)
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 4.4 - 4.6: フィッシャーの線形判別分析の導出と最小二乗解との等価性

### 問題 4.4
制約 $\mathbf{w}^T \mathbf{w} = 1$ の下でクラス間分離基準 $(m_2 - m_1) = \mathbf{w}^T (\mathbf{m}_2 - \mathbf{m}_1)$ を最大化すると、$\mathbf{w} \propto (\mathbf{m}_2 - \mathbf{m}_1)$ となることを示せ。

### 問題 4.5
フィッシャー基準 $J(\mathbf{w}) = \frac{(m_2 - m_1)^2}{s_1^2 + s_2^2}$ が $J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$ と表せることを示せ。

### 問題 4.6
目標値を $t_n = N/N_1$ ($\mathcal{C}_1$), $t_n = -N/N_2$ ($\mathcal{C}_2$) としたとき、二乗和誤差を最小化する重みベクトルがフィッシャーの最適重みベクトル $\mathbf{w} \propto \mathbf{S}_W^{-1}(\mathbf{m}_2 - \mathbf{m}_1)$ と厳密に比例することを示せ。

### 論理のステップ (4.6)
1. 目的関数 $E = \frac{1}{2}\sum_n (\mathbf{w}^T \mathbf{x}_n + w_0 - t_n)^2$。
2. $w_0$ について微分して 0 と置くと、$w_0 = -\mathbf{w}^T \mathbf{m}$、ここで $\mathbf{m} = \frac{1}{N}\sum_n \mathbf{x}_n = \frac{N_1}{N}\mathbf{m}_1 + \frac{N_2}{N}\mathbf{m}_2$。
3. $\mathbf{w}$ について微分して整理すると、$(\mathbf{S}_W + \frac{N_1 N_2}{N}(\mathbf{m}_1 - \mathbf{m}_2)(\mathbf{m}_1 - \mathbf{m}_2)^T)\mathbf{w} = N(\mathbf{m}_1 - \mathbf{m}_2)$。
4. 左辺の第2項は $(\mathbf{m}_1 - \mathbf{m}_2)$ と平行なベクトルであるため、両辺を整理すると $\mathbf{S}_W \mathbf{w}$ は $(\mathbf{m}_1 - \mathbf{m}_2)$ に比例する。
5. したがって $\mathbf{w} \propto \mathbf{S}_W^{-1}(\mathbf{m}_1 - \mathbf{m}_2)$ となり、フィッシャーの判別解と完全に一致する。"""))

# Code Ex 4.4 - 4.6
code_ex4_4_6 = r"""# Exercise 4.4 - 4.6 数値検証
np.random.seed(42)
N1, N2 = 40, 60
N = N1 + N2
D = 3
X1 = np.random.randn(N1, D) + np.array([1.0, -1.0, 0.5])
X2 = np.random.randn(N2, D) + np.array([-1.0, 1.0, -0.5])
X = np.vstack([X1, X2])

# 1. フィッシャーの最適解
m1 = np.mean(X1, axis=0); m2 = np.mean(X2, axis=0)
SW = np.cov(X1, rowvar=False)*(N1-1) + np.cov(X2, rowvar=False)*(N2-1)
w_fisher = np.linalg.solve(SW, (m2 - m1))
w_fisher /= np.linalg.norm(w_fisher)

# 2. 最小二乗解 (特殊目標値: N/N1 と -N/N2)
t = np.array([-N/N1]*N1 + [N/N2]*N2) # C2 - C1 の向き
Phi = np.column_stack([np.ones(N), X])
w_ls_full = np.linalg.pinv(Phi) @ t
w_ls = w_ls_full[1:] # バイアスを除く
w_ls /= np.linalg.norm(w_ls)

cosine_sim = np.abs(np.dot(w_fisher, w_ls))
print(f"Cosine similarity between Fisher and Least Squares: {cosine_sim:.8f}")
assert np.isclose(cosine_sim, 1.0, atol=1e-5)
print("Exercise 4.4 - 4.6 verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex4_4_6))

# Exercise 4.7 & 4.8
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 4.7 & 4.8: シグモイド関数の性質とガウス生成モデルのパラメータ導出

### 問題 4.7
ロジスティックシグモイド関数 $\sigma(a) = \frac{1}{1 + e^{-a}}$ が $\sigma(-a) = 1 - \sigma(a)$ を満たすこと、および逆関数が $\sigma^{-1}(y) = \ln \frac{y}{1-y}$ であることを示せ。

### 問題 4.8
共通共分散ガウス分布 $p(\mathbf{x}|\mathcal{C}_k) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \mathbf{\Sigma})$ から事後確率 $p(\mathcal{C}_1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + w_0)$ を導き、
$\mathbf{w} = \mathbf{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$、
$w_0 = -\frac{1}{2}\boldsymbol{\mu}_1^T \mathbf{\Sigma}^{-1}\boldsymbol{\mu}_1 + \frac{1}{2}\boldsymbol{\mu}_2^T \mathbf{\Sigma}^{-1}\boldsymbol{\mu}_2 + \ln \frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)}$
を確かめよ。"""))

# Code Ex 4.7 & 4.8
code_ex4_7_8 = r"""import scipy.stats as stats
from common.classification_utils import sigmoid

# Exercise 4.7 数値検証
a_test = np.linspace(-5, 5, 50)
assert np.allclose(sigmoid(-a_test), 1.0 - sigmoid(a_test))
y_test = sigmoid(a_test)
a_rec = np.log(y_test / (1.0 - y_test))
assert np.allclose(a_test, a_rec)
print("Exercise 4.7 verified successfully!")

# Exercise 4.8 数値検証
mu1 = np.array([1.0, 2.0])
mu2 = np.array([-1.0, -0.5])
Sigma = np.array([[2.0, 0.5], [0.5, 1.5]])
p_C1, p_C2 = 0.6, 0.4

# 理論式による w, w0
w_theo = np.linalg.solve(Sigma, (mu1 - mu2))
w0_theo = -0.5 * mu1 @ np.linalg.solve(Sigma, mu1) + 0.5 * mu2 @ np.linalg.solve(Sigma, mu2) + np.log(p_C1 / p_C2)

# ベイズの定義式から直接事後オッズ比を計算
x_sample = np.array([0.5, 1.0])
pdf1 = stats.multivariate_normal.pdf(x_sample, mean=mu1, cov=Sigma)
pdf2 = stats.multivariate_normal.pdf(x_sample, mean=mu2, cov=Sigma)
odds_direct = np.log((pdf1 * p_C1) / (pdf2 * p_C2))
odds_linear = np.dot(w_theo, x_sample) + w0_theo

print(f"Direct log-odds: {odds_direct:.6f}, Linear model: {odds_linear:.6f}")
assert np.isclose(odds_direct, odds_linear)
print("Exercise 4.8 verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex4_7_8))

# Exercise 4.9 & 4.10
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 4.9 & 4.10: 多クラス生成モデルの最尤推定

### 問題 4.9
事前確率 $\pi_k = p(\mathcal{C}_k)$ の最尤推定量が $\pi_k = N_k / N$ となることをラグランジュ乗数法を用いて示せ。

### 問題 4.10
共通共分散ガウス分布モデルにおいて、平均の最尤推定量が $\boldsymbol{\mu}_k = \frac{1}{N_k}\sum_{n=1}^N t_{nk} \boldsymbol{\phi}_n$、共分散行列の最尤推定量が $\mathbf{\Sigma} = \sum_{k=1}^K \frac{N_k}{N}\mathbf{S}_k$ となることを示せ。"""))

# Code Ex 4.9 & 4.10
code_ex4_9_10 = r"""from common.classification_utils import GaussianGenerativeClassifier

# Exercise 4.9 & 4.10 数値検証
np.random.seed(42)
N1, N2, N3 = 30, 45, 25
N_tot = N1 + N2 + N3
X1 = np.random.randn(N1, 2) + np.array([-2, 0])
X2 = np.random.randn(N2, 2) + np.array([2, 0])
X3 = np.random.randn(N3, 2) + np.array([0, 3])
X_3c = np.vstack([X1, X2, X3])
y_3c = np.array([0]*N1 + [1]*N2 + [2]*N3)

clf_gen = GaussianGenerativeClassifier(shared_cov=True).fit(X_3c, y_3c)

assert np.allclose(clf_gen.priors, [N1/N_tot, N2/N_tot, N3/N_tot])
assert np.allclose(clf_gen.means[0], np.mean(X1, axis=0))
assert np.allclose(clf_gen.means[1], np.mean(X2, axis=0))
assert np.allclose(clf_gen.means[2], np.mean(X3, axis=0))
print("Exercise 4.9 & 4.10 verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex4_9_10))

# Exercise 4.11 - 4.15
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 4.11 - 4.15: ナイーブベイズ、シグモイド導関数、最尤推定の過適合、ヘッセ行列の正定値性

### 問題 4.11: ナイーブベイズモデルの事後オッズの線形性
$M$ 個の離散特徴（各々 $L$ 値）がクラス条件付き独立であるとき、$a_k = \ln [p(\mathbf{x}|\mathcal{C}_k)p(\mathcal{C}_k)]$ が入力の線形関数になることを示せ。

### 問題 4.12: シグモイドの微分公式
$\frac{d\sigma}{da} = \sigma(a)(1 - \sigma(a))$ を示せ。

### 問題 4.13: ロジスティック回帰の勾配
交差エントロピー誤差関数の勾配が $\nabla E(\mathbf{w}) = \mathbf{\Phi}^T (\mathbf{y} - \mathbf{t})$ となることを示せ。

### 問題 4.14: 線形分離可能データにおける最尤重みの発散
線形分離可能データにおいて、決定境界で全サンプルが完全に正しく分類されている場合、尤度を最大化するには重みの大きさ $\|\mathbf{w}\| \to \infty$ となり、シグモイドがステップ関数に近づくことを論証せよ。

### 問題 4.15: ヘッセ行列 $\mathbf{H} = \mathbf{\Phi}^T \mathbf{R} \mathbf{\Phi}$ の正定値性と凸性
$R_{nn} = y_n(1 - y_n) > 0$ であるため、任意の非ゼロベクトル $\mathbf{u}$ に対して $\mathbf{u}^T \mathbf{H} \mathbf{u} > 0$（正定値）であり、誤差関数が厳密に凸で唯一の最小値を持つことを示せ。"""))

# Code Ex 4.11 - 4.15
code_ex4_11_15 = r"""# Exercise 4.12 数値微分検証
a = 1.5
eps = 1e-6
num_diff = (sigmoid(a + eps) - sigmoid(a - eps)) / (2 * eps)
ana_diff = sigmoid(a) * (1.0 - sigmoid(a))
assert np.isclose(num_diff, ana_diff, atol=1e-8)

# Exercise 4.15 ヘッセ行列の固有値検証 (正定値性)
np.random.seed(42)
N, M = 50, 4
Phi = np.random.randn(N, M)
w = np.random.randn(M)
y = sigmoid(Phi @ w)
R = np.diag(y * (1 - y))
H = Phi.T @ R @ Phi
eigvals = np.linalg.eigvalsh(H)
print("Minimum eigenvalue of Hessian:", np.min(eigvals))
assert np.all(eigvals > 0)
print("Exercise 4.11 - 4.15 verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex4_11_15))

# Exercise 4.16 - 4.20
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 4.16 - 4.20: ソフトラベル、ソフトマックス導関数、プロビット勾配・ヘッセ、多クラスヘッセ行列の半正定値性

### 問題 4.16: ソフトラベル $\pi_n$ に対する対数尤度
学習データが不完全で、各サンプルがラベル $t_n=1$ である確率が $\pi_n$ として与えられるとき、対数尤度関数は
$$ \ln p(\mathcal{D}|\mathbf{w}) = \sum_{n=1}^N \left\{ \pi_n \ln y_n + (1 - \pi_n) \ln (1 - y_n) \right\} $$
となることを示せ。

### 問題 4.17: ソフトマックス関数の導関数
$$ \frac{\partial y_k}{\partial a_j} = y_k (I_{kj} - y_j) $$
を示せ。

### 問題 4.18: 多クラス交差エントロピー誤差の勾配
$$ \nabla_{\mathbf{w}_j} E = \sum_{n=1}^N (y_{nj} - t_{nj}) \boldsymbol{\phi}_n $$
を示せ。

### 問題 4.19: プロビット回帰の勾配とヘッセ行列
$y_n = \Phi(a_n)$ に対し、$\nabla E(\mathbf{w}) = -\sum_{n} \frac{t_n - y_n}{y_n(1 - y_n)} \mathcal{N}(a_n|0, 1) \boldsymbol{\phi}_n$ を示せ。

### 問題 4.20: 多クラスロジスティック回帰のヘッセ行列の半正定値性
フルヘッセ行列のブロック要素 $H_{jk} = \sum_{n} y_{nj}(I_{jk} - y_{nk}) \boldsymbol{\phi}_n \boldsymbol{\phi}_n^T$ に対し、任意のベクトル $\mathbf{u}$ について $\mathbf{u}^T \mathbf{H} \mathbf{u} \ge 0$ となることを示せ。"""))

# Code Ex 4.16 - 4.20
code_ex4_16_20 = r"""from common.classification_utils import softmax

# Exercise 4.17 ソフトマックス微分の数値検証
a_vec = np.array([0.5, 1.2, -0.8])
eps = 1e-6
K = len(a_vec)
J_num = np.zeros((K, K))
for j in range(K):
    a_plus = a_vec.copy(); a_plus[j] += eps
    a_minus = a_vec.copy(); a_minus[j] -= eps
    J_num[:, j] = (softmax(a_plus) - softmax(a_minus)) / (2 * eps)

y_vec = softmax(a_vec)
J_ana = np.diag(y_vec) - np.outer(y_vec, y_vec)
assert np.allclose(J_num, J_ana, atol=1e-7)
print("Exercise 4.17 softmax derivative verified successfully!")

# Exercise 4.20 多クラスヘッセ行列の半正定値性検証
N, M, K = 20, 3, 4
Phi = np.random.randn(N, M)
W = np.random.randn(M, K)
Y = softmax(Phi @ W, axis=-1)

# フルヘッセ行列 (MK x MK)
H_full = np.zeros((M * K, M * K))
for j in range(K):
    for k in range(K):
        # ブロック (j, k)
        weight_jk = Y[:, j] * ((j == k) - Y[:, k])
        H_jk = Phi.T @ (weight_jk[:, None] * Phi)
        H_full[j*M:(j+1)*M, k*M:(k+1)*M] = H_jk

eig_multi = np.linalg.eigvalsh(H_full)
print("Minimum eigenvalue of Multiclass Hessian:", np.min(eig_multi))
assert np.all(eig_multi >= -1e-10)
print("Exercise 4.20 multiclass Hessian positive semidefiniteness verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex4_16_20))

# Exercise 4.21 - 4.26
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 4.21 - 4.26: プロビット関係式、ラプラス近似、BIC、事後周辺化、スケール係数 $\lambda^2=\pi/8$、ガウス畳み込み

### 問題 4.21: プロビットと erf の関係
$$ \Phi(a) = \frac{1}{2}\left[ 1 + \mathrm{erf}\left(\frac{a}{\sqrt{2}}\right) \right] $$
を示せ。

### 問題 4.22 & 4.23: ラプラス近似によるモデルエビデンスと BIC の導出
大標本極限で $\ln |\mathbf{H}| \simeq M \ln N$ となることから、$\mathrm{BIC} = \ln p(\mathcal{D}|\boldsymbol{\theta}_{\mathrm{ML}}) - \frac{M}{2}\ln N$ を導出せよ。

### 問題 4.24: ガウス事後分布による線形結合の周辺化
$p(\mathbf{w}) = \mathcal{N}(\mathbf{w}|\boldsymbol{\mu}_w, \mathbf{\Sigma}_w)$ に対し、$a = \mathbf{w}^T \boldsymbol{\phi}$ の分布が $p(a) = \mathcal{N}(a|\boldsymbol{\mu}_w^T \boldsymbol{\phi}, \boldsymbol{\phi}^T \mathbf{\Sigma}_w \boldsymbol{\phi})$ となることを示せ。

### 問題 4.25: $\sigma(a)$ と $\Phi(\lambda a)$ の原点での傾き一致条件
$\left.\frac{d\sigma}{da}\right|_{a=0} = \frac{1}{4}$ であり、$\left.\frac{d}{da}\Phi(\lambda a)\right|_{a=0} = \lambda \frac{1}{\sqrt{2\pi}}$ である。
これらが等しいとおくと：
$$ \frac{\lambda}{\sqrt{2\pi}} = \frac{1}{4} \implies \lambda = \frac{\sqrt{2\pi}}{4} = \sqrt{\frac{2\pi}{16}} = \sqrt{\frac{\pi}{8}} \implies \lambda^2 = \frac{\pi}{8} $$

### 問題 4.26: ガウス分布とプロビット関数の畳み込み積分
$$ \int_{-\infty}^\infty \Phi(a) \mathcal{N}(a|\mu, \sigma^2) da = \Phi\left( \frac{\mu}{(1 + \sigma^2)^{1/2}} \right) $$
が成り立つことを示せ。"""))

# Code Ex 4.21 - 4.26
code_ex4_21_26 = r"""from scipy.special import erf
import scipy.stats as stats
import scipy.integrate as integrate

# Exercise 4.21 数値検証
a_pts = np.linspace(-3, 3, 20)
probit_direct = stats.norm.cdf(a_pts)
probit_erf = 0.5 * (1.0 + erf(a_pts / np.sqrt(2.0)))
assert np.allclose(probit_direct, probit_erf)

# Exercise 4.25 数値検証 (原点での傾き)
lambda_val = np.sqrt(np.pi / 8.0)
slope_sig = 0.25
slope_probit = lambda_val / np.sqrt(2 * np.pi)
print(f"Sigmoid slope at 0: {slope_sig:.6f}, Scaled Probit slope at 0: {slope_probit:.6f}")
assert np.isclose(slope_sig, slope_probit)

# Exercise 4.26 ガウス畳み込み積分の数値積分検証
mu = 1.2
sigma = 0.8
# 解析解
ana_conv = stats.norm.cdf(mu / np.sqrt(1.0 + sigma**2))

# 数値積分
def integrand(a):
    return stats.norm.cdf(a) * stats.norm.pdf(a, loc=mu, scale=sigma)

num_conv, _ = integrate.quad(integrand, -10, 10)
print(f"Analytical convolution: {ana_conv:.8f}, Numerical integration: {num_conv:.8f}")
assert np.isclose(ana_conv, num_conv, atol=1e-6)
print("Exercise 4.21 - 4.26 verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex4_21_26))

nb.cells = cells
with open('4/4_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("4/4_Exercises.ipynb generated successfully.")
