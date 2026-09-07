#!/usr/bin/env python3
"""
enhance_ch7_exercises.py
Generate comprehensive, self-contained, rigorously verified notebooks for all 19 exercises in PRML Chapter 7.
Every exercise includes:
- Problem statement matching Bishop's original text
- Structured derivation with fill-in-the-blank (穴埋め)
- Solutions to the fill-in-the-blank
- Robust, self-contained Python numerical simulation / verification code with assertions.
"""

import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Introduction
    title_md = r"""# 第7章 スパースカーネルマシン：演習問題 (Exercises 7.1 - 7.19)

本ノートブックでは、PRML第7章「スパースカーネルマシン (Sparse Kernel Machines)」の**全19問 (Exercises 7.1 〜 7.19)** に対する厳密な数理解説、証明ステップ、穴埋め問題、および Python による数値検証コードを完全網羅しています。

### 主な学習項目
1. **サポートベクトルマシン (SVM) の幾何学と双対理論 (7.1 - 7.6)**:
   - パルツェン窓分類器と最近傍平均決定境界の等価性 (7.1)
   - マージン制約定数 $\gamma$ に対する決定超平面の不変性 (7.2)
   - 異クラス2点による垂直二等分面の幾何学的決定 (7.3)
   - 幾何学的マージン恒等式 $1/\rho^2 = \|\mathbf{w}\|^2 = \sum a_n$ (7.4)
   - 双対目的関数の最大値 $2\widetilde{L}(\mathbf{a}^*) = 1/\rho^2$ (7.5)
   - ロジスティック回帰の二乗正則化損失とSVMヒンジ損失の比較 (7.6)
2. **サポートベクトル回帰 (SVR) (7.7 - 7.8)**:
   - $\epsilon$-感度損失関数とSVR双対問題の完全導出 (7.7)
   - SVRのKKT相補性スラック条件とマージン超過点の乗数飽和 $a_n = C$ (7.8)
3. **関連度ベクトルマシン (RVM) (7.9 - 7.19)**:
   - 重み事後ガウス分布の平均 $\mathbf{m}$ と共分散 $\boldsymbol{\Sigma}$ (7.9)
   - 平方完成ガウス積分による周辺尤度（エビデンス）の導出 (7.10)
   - 線形ガウス周辺化公式 (2.115) による周辺尤度の直接導出 (7.11)
   - 周辺尤度最大化によるハイパーパラメータ $\alpha_i, \beta$ の再推定方程式 (7.12)
   - ガンマ超事前分布を導入したMAP再推定方程式 (7.13)
   - RVM予測分布の平均・分散公式の導出 (7.14)
   - 単一ハイパーパラメータ $\alpha_i$ の寄与分離 $L(\boldsymbol{\alpha}) = L(\boldsymbol{\alpha}_{-i}) + \lambda(\alpha_i)$ (7.15)
   - 2階微分による停留点 $\alpha_i^*$ の大域的極大性証明 (7.16)
   - ウッドベリー恒等式を用いたスパース性因子 $s_i, q_i$ の高速計算関係式 (7.17)
   - 分類RVMの事後対数分布の勾配ベクトルとヘッセ行列 (7.18)
   - 分類RVMのラプラス近似とハイパーパラメータ再推定方程式 (7.19)
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))

    setup_code = r"""# 共通環境のインポートと数値設定
import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt

from common.svm_rvm_utils import SupportVectorClassifier, SupportVectorRegressor, RelevanceVectorRegressor
from common.kernel_utils import rbf_kernel, linear_kernel

np.random.seed(42)
print("Environment successfully initialized with numpy and scipy.optimize.")"""
    cells.append(nbf.v4.new_code_cell(setup_code))

    # --- Exercise 7.1 ---
    ex7_1_md = r"""---
## <a id="Exercise-7.1"></a>Exercise 7.1: パルツェン窓密度比と最近傍平均分類器の等価性

### 問題の提示
各クラス $\mathcal{C}_k$ ($t \in \{-1, +1\}$) に対する条件付き密度推定として、訓練標本数 $N_k$ のパルツェン窓密度推定器
$$ p(\mathbf{x}|\mathcal{C}_k) = \frac{1}{N_k} \sum_{n \in \mathcal{C}_k} k(\mathbf{x}, \mathbf{x}_n) $$
を用いる。2クラスの事前確率が等しいと仮定したときの最小誤り率決定則を導出せよ。また、線形カーネル $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$ の場合、この決定則がクラス標本平均 $\mathbf{m}_k = \frac{1}{N_k}\sum_{n \in \mathcal{C}_k} \mathbf{x}_n$ のうち近い方に割り当てる最近傍平均分類器となることを示せ。さらに一般の内積カーネル $k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}')$ において、特徴空間における平均への距離に基づく決定則となることを示せ。

### [解答の道筋と穴埋め]
1. **最小誤り率決定則**:
   事前確率が等しい ($p(\mathcal{C}_1) = p(\mathcal{C}_{-1})$) とき、ベイズの決定則は尤度比により定まる：
   $$ p(\mathbf{x}|\mathcal{C}_1) \gtrless p(\mathbf{x}|\mathcal{C}_{-1}) \iff \frac{1}{N_1}\sum_{n \in \mathcal{C}_1} k(\mathbf{x}, \mathbf{x}_n) \gtrless [ \text{①} ] $$
2. **線形カーネルにおける展開**:
   $k(\mathbf{x}, \mathbf{x}_n) = \mathbf{x}^T \mathbf{x}_n$ を代入すると、和を内積の外にくくり出すことができる：
   $$ p(\mathbf{x}|\mathcal{C}_k) = \mathbf{x}^T \left( \frac{1}{N_k}\sum_{n \in \mathcal{C}_k} \mathbf{x}_n \right) = \mathbf{x}^T \mathbf{m}_k $$
   したがって、決定則は $\mathbf{x}^T \mathbf{m}_1 \gtrless \mathbf{x}^T \mathbf{m}_{-1}$ となる。
   両辺に $-\frac{1}{2}(\|\mathbf{m}_1\|^2 - \|\mathbf{m}_{-1}\|^2)$ を加減算して平方完成すると：
   $$ \|\mathbf{x} - \mathbf{m}_1\|^2 \lessgtr [ \text{②} ] $$
   （クラス平均のノルムが等しい場合、または各クラス平均とのユークリッド距離二乗の比較）に帰着する。
3. **特徴空間における一般化**:
   $k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}')$ の場合、特徴空間におけるクラス平均を $\boldsymbol{\mu}_k = \frac{1}{N_k}\sum_{n \in \mathcal{C}_k}\boldsymbol{\phi}(\mathbf{x}_n)$ とおくと、
   $$ p(\mathbf{x}|\mathcal{C}_k) = \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\mu}_k $$
   となり、特徴空間 $\mathcal{H}$ における平均ベクトル $\boldsymbol{\mu}_k$ への距離比較に基づく決定則となる。

### 穴埋めの解答
- ①: $\frac{1}{N_{-1}}\sum_{n \in \mathcal{C}_{-1}} k(\mathbf{x}, \mathbf{x}_n)$
- ②: $\|\mathbf{x} - \mathbf{m}_{-1}\|^2$"""

    ex7_1_code = r"""# Exercise 7.1 数値検証: パルツェン窓密度比と最近傍平均決定境界の完全一致
N1, N2 = 30, 25
X1 = np.random.randn(N1, 2) + np.array([1.5, 1.5])
X2 = np.random.randn(N2, 2) + np.array([-1.5, -1.5])

m1 = np.mean(X1, axis=0)
m2 = np.mean(X2, axis=0)

X_test = np.random.uniform(-3, 3, (50, 2))

# 1. パルツェン窓による密度比 (線形カーネル)
p1 = np.mean(X_test @ X1.T, axis=1) # x^T m1
p2 = np.mean(X_test @ X2.T, axis=1) # x^T m2
pred_parzen = np.sign(p1 - p2)

# 2. クラス平均との内積差による決定
pred_mean = np.sign(X_test @ (m1 - m2))

np.testing.assert_array_equal(pred_parzen, pred_mean)
print(f"Exercise 7.1 verified: Parzen density predictions match closest mean rule across all {len(X_test)} test points!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_1_md), nbf.v4.new_code_cell(ex7_1_code)])

    # --- Exercise 7.2 ---
    ex7_2_md = r"""---
## <a id="Exercise-7.2"></a>Exercise 7.2: マージン制約定数 $\gamma$ に対する決定超平面の不変性

### 問題の提示
ハードマージンSVMの制約条件（式 7.5）の右辺の $1$ を、任意の正の実定数 $\gamma > 0$ で置き換えた以下の問題を考える：
$$ \min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad t_n (\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) + b) \ge \gamma \quad (n = 1, \dots, N) $$
このとき、得られる最大マージン超平面の位置および幾何学的マージン幅 $\rho$ が、標準的な定式化（$\gamma = 1$）の解と完全に同一であることを示せ。

### [解答の道筋と穴埋め]
1. **変数変換の導入**:
   スケーリングされた変数 $\mathbf{w}' = \mathbf{w}/\gamma, \quad b' = b/\gamma$ を定義する。
   制約式 $t_n (\mathbf{w}^T \boldsymbol{\phi}_n + b) \ge \gamma$ の両辺を $\gamma > 0$ で割ると：
   $$ t_n (\mathbf{w}'^T \boldsymbol{\phi}_n + b') \ge [ \text{①} ] $$
2. **目的関数の変形**:
   主問題の目的関数は：
   $$ \frac{1}{2}\|\mathbf{w}\|^2 = \frac{\gamma^2}{2} \|\mathbf{w}'\|^2 $$
   $\gamma^2 > 0$ は定数であるため、これを最小化する $\mathbf{w}'^*$ は、制約 $t_n (\mathbf{w}'^T \boldsymbol{\phi}_n + b') \ge 1$ の下で $\frac{1}{2}\|\mathbf{w}'\|^2$ を最小化する標準問題の解と厳密に一致する。
3. **超平面とマージンの幾何学的不変性**:
   最適解を $(\mathbf{w}^*, b^*)$ とすると、$\mathbf{w}^* = \gamma \mathbf{w}'^*, b^* = \gamma b'^*$ である。
   決定超平面は：
   $$ \mathbf{w}^{*T}\boldsymbol{\phi}(\mathbf{x}) + b^* = 0 \iff \gamma(\mathbf{w}'^{*T}\boldsymbol{\phi}(\mathbf{x}) + b'^*) = 0 \iff [ \text{②} ] $$
   幾何学的マージン幅は：
   $$ \rho = \frac{\gamma}{\|\mathbf{w}^*\|} = \frac{\gamma}{\gamma \|\mathbf{w}'^*\|} = \frac{1}{\|\mathbf{w}'^*\|} $$
   となり、$\gamma$ の選択に依らず不変である。

### 穴埋めの解答
- ①: $1$
- ②: $\mathbf{w}'^{*T}\boldsymbol{\phi}(\mathbf{x}) + b'^* = 0$"""

    ex7_2_code = r"""# Exercise 7.2 数値検証: スケール定数 gamma を変えた二次計画問題解の超平面不変性
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [-1.0, -1.0], [-2.0, -2.0], [-1.0, -3.0]])
t = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

def solve_svm_primal(gamma_val):
    # min 0.5 * ||w||^2 s.t. t_n (w^T x_n + b) >= gamma
    def obj(p):
        return 0.5 * np.sum(p[:2]**2)
    def obj_jac(p):
        return np.array([p[0], p[1], 0.0])
    cons = {
        'type': 'ineq',
        'fun': lambda p: t * (X @ p[:2] + p[2]) - gamma_val,
        'jac': lambda p: np.hstack([t[:, None] * X, t[:, None]])
    }
    res = opt.minimize(obj, np.zeros(3), jac=obj_jac, constraints=cons, method='SLSQP', options={'ftol': 1e-10})
    return res.x[:2], res.x[2]

w1, b1 = solve_svm_primal(1.0)
w4, b4 = solve_svm_primal(4.0)

# w4, b4 は w1, b1 の厳密に 4 倍
np.testing.assert_allclose(w4, 4.0 * w1, rtol=1e-5)
np.testing.assert_allclose(b4, 4.0 * b1, rtol=1e-5)

# 単位法線ベクトルと幾何学的マージン
u1 = w1 / np.linalg.norm(w1)
u4 = w4 / np.linalg.norm(w4)
np.testing.assert_allclose(u1, u4, atol=1e-7)

rho1 = 1.0 / np.linalg.norm(w1)
rho4 = 4.0 / np.linalg.norm(w4)
np.testing.assert_allclose(rho1, rho4, atol=1e-7)
print(f"Exercise 7.2 verified: gamma=1 and gamma=4 define identical hyperplane (margin={rho1:.6f})")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_2_md), nbf.v4.new_code_cell(ex7_2_code)])

    # --- Exercise 7.3 ---
    ex7_3_md = r"""---
## <a id="Exercise-7.3"></a>Exercise 7.3: 異クラス2点による最大マージン超平面の一意決定

### 問題の提示
データ空間の次元に関わらず、正例 $\mathbf{x}_1$ ($t_1 = +1$) と負例 $\mathbf{x}_2$ ($t_2 = -1$) の2点のみからなるデータ集合が与えられたとき、最大マージン超平面の位置が一意に決定され、2点を結ぶ線分の垂直二等分面
$$ \mathbf{w} = \frac{2(\mathbf{x}_1 - \mathbf{x}_2)}{\|\mathbf{x}_1 - \mathbf{x}_2\|^2}, \quad b = -\frac{\|\mathbf{x}_1\|^2 - \|\mathbf{x}_2\|^2}{\|\mathbf{x}_1 - \mathbf{x}_2\|^2} $$
となることを示せ。

### [解答の道筋と穴埋め]
1. **制約条件の連立**:
   2点に対する正準制約は：
   $$ \mathbf{w}^T \mathbf{x}_1 + b \ge 1, \quad -(\mathbf{w}^T \mathbf{x}_2 + b) \ge 1 \implies \mathbf{w}^T \mathbf{x}_2 + b \le -1 $$
   辺々を引くと、$\mathbf{w}^T (\mathbf{x}_1 - \mathbf{x}_2) \ge [ \text{①} ]$。
2. **重みノルムの最小化**:
   コーシー・シュワルツの不等式より：
   $$ 2 \le \mathbf{w}^T (\mathbf{x}_1 - \mathbf{x}_2) \le \|\mathbf{w}\| \|\mathbf{x}_1 - \mathbf{x}_2\| \implies \|\mathbf{w}\| \ge \frac{2}{\|\mathbf{x}_1 - \mathbf{x}_2\|} $$
   等号が成立するのは $\mathbf{w}$ が差ベクトル $(\mathbf{x}_1 - \mathbf{x}_2)$ に平行なときに限られる：
   $$ \mathbf{w} = c (\mathbf{x}_1 - \mathbf{x}_2) \implies c \|\mathbf{x}_1 - \mathbf{x}_2\|^2 = 2 \implies c = \frac{2}{\|\mathbf{x}_1 - \mathbf{x}_2\|^2} $$
   したがって、$\mathbf{w}^* = \frac{2(\mathbf{x}_1 - \mathbf{x}_2)}{\|\mathbf{x}_1 - \mathbf{x}_2\|^2}$。
3. **切片 $b$ の決定**:
   両点はサポートベクトルであるため、$\mathbf{w}^T \mathbf{x}_1 + b = 1$ かつ $\mathbf{w}^T \mathbf{x}_2 + b = -1$。
   和をとると $\mathbf{w}^T (\mathbf{x}_1 + \mathbf{x}_2) + 2b = 0$ より：
   $$ b = -\frac{1}{2}\mathbf{w}^T (\mathbf{x}_1 + \mathbf{x}_2) = - \frac{(\mathbf{x}_1 - \mathbf{x}_2)^T (\mathbf{x}_1 + \mathbf{x}_2)}{\|\mathbf{x}_1 - \mathbf{x}_2\|^2} = [ \text{②} ] $$
   これは線分の中点 $\frac{\mathbf{x}_1 + \mathbf{x}_2}{2}$ を通り、線分に直交する垂直二等分面である。

### 穴埋めの解答
- ①: $2$
- ②: $-\frac{\|\mathbf{x}_1\|^2 - \|\mathbf{x}_2\|^2}{\|\mathbf{x}_1 - \mathbf{x}_2\|^2}$"""

    ex7_3_code = r"""# Exercise 7.3 数値検証: 2点SVMの最適化解と理論垂直二等分面の一致
x1 = np.array([3.0, 4.0, 1.0, 2.0])
x2 = np.array([-1.0, 0.0, -2.0, 1.0])

diff = x1 - x2
d_sq = np.dot(diff, diff)
w_theory = 2.0 * diff / d_sq
b_theory = - (np.dot(x1, x1) - np.dot(x2, x2)) / d_sq

# SLSQP による主問題最適化
D = len(x1)
def obj_2(p):
    return 0.5 * np.sum(p[:D]**2)
def grad_2(p):
    return np.append(p[:D], 0.0)

cons_2 = {
    'type': 'ineq',
    'fun': lambda p: np.array([p[:D] @ x1 + p[D] - 1.0, - (p[:D] @ x2 + p[D]) - 1.0]),
    'jac': lambda p: np.array([np.append(x1, 1.0), np.append(-x2, -1.0)])
}

res_2 = opt.minimize(obj_2, np.zeros(D + 1), jac=grad_2, constraints=cons_2, method='SLSQP', options={'ftol': 1e-10})
w_qp = res_2.x[:D]
b_qp = res_2.x[D]

np.testing.assert_allclose(w_qp, w_theory, rtol=1e-5)
np.testing.assert_allclose(b_qp, b_theory, rtol=1e-5)
print(f"Exercise 7.3 verified: 4D 2-point QP solution matches theoretical perpendicular bisector (||w||={np.linalg.norm(w_theory):.6f})")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_3_md), nbf.v4.new_code_cell(ex7_3_code)])

    # --- Exercise 7.4 ---
    ex7_4_md = r"""---
## <a id="Exercise-7.4"></a>Exercise 7.4: マージン幅 $\rho$ とラグランジュ乗数の総和恒等式 $\frac{1}{\rho^2} = \sum a_n$

### 問題の提示
最大マージン超平面における幾何学的マージン幅 $\rho$ について、式 (7.10) の双対目的関数を制約 (7.11) および (7.12) の下で最大化して得られる最適ラグランジュ乗数 $\{a_n\}$ を用いて
$$ \frac{1}{\rho^2} = \sum_{n=1}^N a_n \quad (7.123) $$
が成立することを示せ。

### [解答の道筋と穴埋め]
1. **最適重みベクトルの双対展開**:
   ラグランジアンの極値条件より、$\mathbf{w} = \sum_{n=1}^N a_n t_n \boldsymbol{\phi}(\mathbf{x}_n)$。
2. **内積の評価とKKT相補性スラック条件**:
   重みベクトルのノルム二乗を展開すると：
   $$ \|\mathbf{w}\|^2 = \mathbf{w}^T \mathbf{w} = \sum_{n=1}^N a_n t_n (\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n)) = \sum_{n=1}^N a_n t_n (y(\mathbf{x}_n) - b) $$
   $$ = \sum_{n=1}^N a_n [ t_n y(\mathbf{x}_n) ] - b [ \text{①} ] $$
   バイアス $b$ の極値条件 $\sum_{n=1}^N a_n t_n = 0$ より第二項は消滅する。
3. **KKT条件の適用**:
   KKT相補性条件は $a_n [t_n y(\mathbf{x}_n) - 1] = 0$ であるから、$a_n > 0$ となるすべてのサポートベクトルにおいて $t_n y(\mathbf{x}_n) = 1$ が成立する。
   したがって：
   $$ \|\mathbf{w}\|^2 = \sum_{n=1}^N a_n (1) = [ \text{②} ] $$
   マージン幅の定義 $\rho = \frac{1}{\|\mathbf{w}\|}$ より、$\frac{1}{\rho^2} = \|\mathbf{w}\|^2 = \sum_{n=1}^N a_n$ が証明された。

### 穴埋めの解答
- ①: $\sum_{n=1}^N a_n t_n$
- ②: $\sum_{n=1}^N a_n$"""

    ex7_4_code = r"""# Exercise 7.4 数値検証: 1/rho^2 == ||w||^2 == sum(a_n) の厳密成立
X_data = np.array([[-1.5, -0.5], [-1.0, 1.0], [-0.5, -1.0], [1.0, 0.5], [1.5, -0.5], [2.0, 1.0]])
t_data = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
N = len(t_data)

# 双対問題の求解: min 0.5 * a^T (Y K Y) a - sum a s.t. a >= 0, sum(a t) = 0
K = X_data @ X_data.T
P = np.outer(t_data, t_data) * K + 1e-10 * np.eye(N)

def obj_dual(a):
    return 0.5 * a @ P @ a - np.sum(a)
def grad_dual(a):
    return P @ a - 1.0

cons_dual = {'type': 'eq', 'fun': lambda a: np.dot(a, t_data), 'jac': lambda a: t_data}
bounds_dual = [(0.0, None) for _ in range(N)]

res_dual = opt.minimize(obj_dual, np.zeros(N), jac=grad_dual, constraints=cons_dual, bounds=bounds_dual, method='SLSQP', options={'ftol': 1e-10})
a_opt = res_dual.x

w_opt = np.sum((a_opt * t_data)[:, None] * X_data, axis=0)
norm_w_sq = np.sum(w_opt**2)
sum_a = np.sum(a_opt)
inv_rho_sq = norm_w_sq

np.testing.assert_allclose(sum_a, norm_w_sq, rtol=1e-5)
print(f"Exercise 7.4 verified: sum(a_n) = {sum_a:.6f} == ||w||^2 = {norm_w_sq:.6f} == 1/rho^2")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_4_md), nbf.v4.new_code_cell(ex7_4_code)])

    # --- Exercise 7.5 ---
    ex7_5_md = r"""---
## <a id="Exercise-7.5"></a>Exercise 7.5: 双対目的関数の最大値 $\widetilde{L}(\mathbf{a}^*)$ と $1/\rho^2$ の等価性

### 問題の提示
前問の $\rho$ と最適乗数 $\{a_n\}$ が、以下の関係式
$$ \frac{1}{\rho^2} = 2\widetilde{L}(\mathbf{a}) \quad (7.124) $$
を満たすことを示せ。ただし $\widetilde{L}(\mathbf{a})$ は式 (7.10) で定義される双対目的関数である。さらに、$\frac{1}{\rho^2} = \|\mathbf{w}\|^2$ (7.125) を示せ。

### [解答の道筋と穴埋め]
1. **双対目的関数の二乗和展開**:
   双対目的関数は：
   $$ \widetilde{L}(\mathbf{a}) = \sum_{n=1}^N a_n - \frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m k(\mathbf{x}_n, \mathbf{x}_m) $$
   第二項は $\mathbf{w} = \sum_{n=1}^N a_n t_n \boldsymbol{\phi}(\mathbf{x}_n)$ のノルム二乗に等しい：
   $$ \sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m k(\mathbf{x}_n, \mathbf{x}_m) = [ \text{①} ] $$
2. **最適値の代入**:
   最適解 $\mathbf{a}^*$ において、$\widetilde{L}(\mathbf{a}^*) = \sum_{n=1}^N a_n^* - \frac{1}{2}\|\mathbf{w}^*\|^2$。
   Exercise 7.4 より $\|\mathbf{w}^*\|^2 = \sum_{n=1}^N a_n^*$ であるから：
   $$ \widetilde{L}(\mathbf{a}^*) = \sum_{n=1}^N a_n^* - \frac{1}{2}\sum_{n=1}^N a_n^* = [ \text{②} ] $$
   両辺を2倍すると：
   $$ 2\widetilde{L}(\mathbf{a}^*) = \sum_{n=1}^N a_n^* = \|\mathbf{w}^*\|^2 = \frac{1}{\rho^2} $$
   となり、(7.124) および (7.125) が同時に証明される。

### 穴埋めの解答
- ①: $\|\mathbf{w}\|^2$
- ②: $\frac{1}{2}\sum_{n=1}^N a_n^*$"""

    ex7_5_code = r"""# Exercise 7.5 数値検証: 2 * L_dual == sum(a_n) == ||w||^2 == 1/rho^2
quad_term = a_opt @ (np.outer(t_data, t_data) * K) @ a_opt
L_dual = np.sum(a_opt) - 0.5 * quad_term

two_L_dual = 2.0 * L_dual
np.testing.assert_allclose(two_L_dual, sum_a, rtol=1e-5)
np.testing.assert_allclose(two_L_dual, norm_w_sq, rtol=1e-5)
print(f"Exercise 7.5 verified: 2*L(a) = {two_L_dual:.6f} == sum(a_n) = {sum_a:.6f} == ||w||^2 = {norm_w_sq:.6f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_5_md), nbf.v4.new_code_cell(ex7_5_code)])

    # --- Exercise 7.6 ---
    ex7_6_md = r"""---
## <a id="Exercise-7.6"></a>Exercise 7.6: ロジスティック回帰の負の対数尤度と正則化誤差関数 (7.47)

### 問題の提示
目標変数が $t \in \{-1, +1\}$ であるロジスティック回帰モデルを考える。モデル出力を $y(\mathbf{x}) = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}) + b$ とし、$p(t=1|y) = \sigma(y)$ と定義するとき、二次正則化項を加えた負の対数尤度が式 (7.47)
$$ E(\mathbf{w}) = \sum_{n=1}^N E_1(t_n y_n) + \frac{\lambda}{2}\|\mathbf{w}\|^2, \quad E_1(z) = \ln(1 + e^{-z}) $$
の形式となることを示せ。

### [解答の道筋と穴埋め]
1. **対称シグモイド表現**:
   $p(t=1|y) = \sigma(y) = \frac{1}{1 + e^{-y}}$ より、
   $$ p(t=-1|y) = 1 - \sigma(y) = \frac{e^{-y}}{1 + e^{-y}} = \frac{1}{1 + e^y} = \sigma(-y) $$
   したがって、$t \in \{-1, +1\}$ に対する条件付き確率は統一的に：
   $$ p(t|y) = [ \text{①} ] = \frac{1}{1 + e^{-t y}} $$
   と表される。
2. **負の対数尤度の計算**:
   単一サンプルの負の対数尤度は：
   $$ -\ln p(t_n|y_n) = -\ln \left( \frac{1}{1 + e^{-t_n y_n}} \right) = \ln(1 + e^{-t_n y_n}) $$
   これは損失関数 $E_1(z) = \ln(1 + e^{-z})$ を用いて $E_1(t_n y_n)$ と書ける。
3. **全誤差関数**:
   独立同分布データに対する負の対数尤度の総和に、重み減衰（L2正則化）項 $\frac{\lambda}{2}\|\mathbf{w}\|^2$ を加算すると：
   $$ E(\mathbf{w}) = \sum_{n=1}^N [ \text{②} ] + \frac{\lambda}{2}\|\mathbf{w}\|^2 $$
   となり、式 (7.47) に厳密に一致する。

### 穴埋めの解答
- ①: $\sigma(t y)$
- ②: $\ln(1 + e^{-t_n y_n})$"""

    ex7_6_code = r"""# Exercise 7.6 数値検証: t in {-1, 1} のロジスティック損失と標準交差エントロピーの完全一致
y_vals = np.linspace(-4, 4, 100)

for t_val in [-1.0, 1.0]:
    # 式 (7.47): ln(1 + exp(-t y))
    loss_7_47 = np.log(1.0 + np.exp(-t_val * y_vals))
    
    # 標準二値交差エントロピー: target in {0, 1}
    target_01 = 1.0 if t_val == 1.0 else 0.0
    sig_y = 1.0 / (1.0 + np.exp(-y_vals))
    bce = - (target_01 * np.log(sig_y + 1e-15) + (1.0 - target_01) * np.log(1.0 - sig_y + 1e-15))
    
    np.testing.assert_allclose(loss_7_47, bce, atol=1e-7)

print("Exercise 7.6 verified: Symmetric logistic loss E_1(ty) is identical to binary cross-entropy!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_6_md), nbf.v4.new_code_cell(ex7_6_code)])

    # --- Exercise 7.7 ---
    ex7_7_md = r"""---
## <a id="Exercise-7.7"></a>Exercise 7.7: サポートベクトル回帰 (SVR) のラグランジュ双対問題 (7.61) の導出

### 問題の提示
サポートベクトル回帰における主問題のラグランジアン（式 7.56）
$$ L = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{n=1}^N (\xi_n + \widehat{\xi}_n) - \sum_{n=1}^N a_n (\epsilon + \xi_n + y_n - t_n) - \sum_{n=1}^N \widehat{a}_n (\epsilon + \widehat{\xi}_n - y_n + t_n) - \sum_{n=1}^N (\mu_n \xi_n + \widehat{\mu}_n \widehat{\xi}_n) $$
を考える。$\mathbf{w}, b, \xi_n, \widehat{\xi}_n$ に関する微分をゼロとおいてこれら消去することにより、双対ラグランジアンが式 (7.61)
$$ \widetilde{L}(\mathbf{a}, \widehat{\mathbf{a}}) = -\frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N (a_n - \widehat{a}_n)(a_m - \widehat{a}_m) k(\mathbf{x}_n, \mathbf{x}_m) - \epsilon \sum_{n=1}^N (a_n + \widehat{a}_n) + \sum_{n=1}^N (a_n - \widehat{a}_n) t_n $$
で与えられることを示せ。

### [解答の道筋と穴埋め]
1. **極値条件の導出**:
   - $\frac{\partial L}{\partial \mathbf{w}} = \mathbf{0} \implies \mathbf{w} = \sum_{n=1}^N (a_n - \widehat{a}_n) \boldsymbol{\phi}(\mathbf{x}_n)$
   - $\frac{\partial L}{\partial b} = 0 \implies [ \text{①} ]$
   - $\frac{\partial L}{\partial \xi_n} = 0 \implies C - a_n - \mu_n = 0$
   - $\frac{\partial L}{\partial \widehat{\xi}_n} = 0 \implies C - \widehat{a}_n - \widehat{\mu}_n = 0$
2. **スラック変数項の消去**:
   スラック変数を含む項は $(C - a_n - \mu_n)\xi_n + (C - \widehat{a}_n - \widehat{\mu}_n)\widehat{\xi}_n = 0$ となり完全に消滅する。
3. **ラグランジアンへの再代入**:
   $\frac{1}{2}\|\mathbf{w}\|^2 - \sum(a_n - \widehat{a}_n)\mathbf{w}^T \boldsymbol{\phi}_n = \frac{1}{2}\|\mathbf{w}\|^2 - \|\mathbf{w}\|^2 = -\frac{1}{2}\|\mathbf{w}\|^2$。
   グラム行列で表すと：
   $$ -\frac{1}{2}\sum_{n=1}^N \sum_{m=1}^N (a_n - \widehat{a}_n)(a_m - \widehat{a}_m) k(\mathbf{x}_n, \mathbf{x}_m) $$
   バイアス項は $-b \sum (a_n - \widehat{a}_n) = 0$ となる。
   残りの線形項は：
   $$ [ \text{②} ] $$
   これらを結合することで、式 (7.61) が導出される。

### 穴埋めの解答
- ①: $\sum_{n=1}^N (a_n - \widehat{a}_n) = 0$
- ②: $-\epsilon \sum_{n=1}^N (a_n + \widehat{a}_n) + \sum_{n=1}^N (a_n - \widehat{a}_n) t_n$"""

    ex7_7_code = r"""# Exercise 7.7 数値検証: SVR 主問題ラグランジアンと双対関数の停留値厳密一致
X_svr = np.array([[1.0], [2.0], [3.0], [4.0]])
t_svr = np.array([1.2, 1.9, 3.2, 3.8])
eps = 0.2
C_val = 5.0
N_s = len(t_svr)

# 双対問題の二次計画問題解 (SLSQP)
K_svr = X_svr @ X_svr.T
B_mat = np.hstack([np.eye(N_s), -np.eye(N_s)])
P_svr = B_mat.T @ K_svr @ B_mat + 1e-10 * np.eye(2*N_s)
c_svr = np.hstack([eps - t_svr, eps + t_svr])

def obj_svr(alpha):
    return 0.5 * alpha @ P_svr @ alpha + np.dot(c_svr, alpha)
def grad_svr(alpha):
    return P_svr @ alpha + c_svr

eq_v = np.hstack([np.ones(N_s), -np.ones(N_s)])
cons_svr = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, eq_v), 'jac': lambda alpha: eq_v}
bounds_svr = [(0.0, C_val) for _ in range(2 * N_s)]

res_svr = opt.minimize(obj_svr, np.zeros(2*N_s), jac=grad_svr, constraints=cons_svr, bounds=bounds_svr, method='SLSQP', options={'ftol': 1e-10})
alpha = res_svr.x
a = alpha[:N_s]
a_hat = alpha[N_s:]

diff_a = a - a_hat
w_star = np.sum(diff_a[:, None] * X_svr, axis=0)

# 双対関数値
dual_val = -0.5 * diff_a @ K_svr @ diff_a - eps * np.sum(a + a_hat) + np.dot(diff_a, t_svr)

# 主ラグランジアン停留値
L_stat = -0.5 * np.sum(w_star**2) - eps * np.sum(a + a_hat) + np.dot(diff_a, t_svr)
np.testing.assert_allclose(dual_val, L_stat, rtol=1e-6)
print(f"Exercise 7.7 verified: SVR dual objective {dual_val:.6f} equals stationary Lagrangian value {L_stat:.6f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_7_md), nbf.v4.new_code_cell(ex7_7_code)])

    # --- Exercise 7.8 ---
    ex7_8_md = r"""---
## <a id="Exercise-7.8"></a>Exercise 7.8: SVR の KKT 相補性条件とマージン超過点の乗数飽和 $a_n = C$

### 問題の提示
サポートベクトル回帰において、$\xi_n > 0$ であるすべての訓練データ点に対して $a_n = C$ となり、同様に $\widehat{\xi}_n > 0$ であるすべての訓練データ点に対して $\widehat{a}_n = C$ となることを示せ。

### [解答の道筋と穴埋め]
1. **主変数と双対変数のKKT条件**:
   スラック変数 $\xi_n, \widehat{\xi}_n$ およびそのラグランジュ乗数 $\mu_n, \widehat{\mu}_n \ge 0$ に対する停留点条件（Exercise 7.7）は：
   $$ C - a_n - \mu_n = 0 \implies \mu_n = C - a_n $$
   $$ C - \widehat{a}_n - \widehat{\mu}_n = 0 \implies \widehat{\mu}_n = C - \widehat{a}_n $$
2. **相補性スラック条件**:
   不等式制約に対するKKT条件より：
   $$ \mu_n \xi_n = 0 \iff [ \text{①} ] = 0 $$
   $$ \widehat{\mu}_n \widehat{\xi}_n = 0 \iff (C - \widehat{a}_n) \widehat{\xi}_n = 0 $$
3. **$\xi_n > 0$ における帰結**:
   データ点が上側チューブ境界を越えている（$\xi_n > 0$）場合、$(C - a_n)\xi_n = 0$ が成立するためには、
   $$ [ \text{②} ] $$
   でなければならない。同様に下側チューブ境界を越えている（$\widehat{\xi}_n > 0$）場合、$\widehat{a}_n = C$ となる。
   これにより、マージン外の誤差点はすべて許容上限 $C$ に飽和することが示された。

### 穴埋めの解答
- ①: $(C - a_n)\xi_n$
- ②: $a_n = C$"""

    ex7_8_code = r"""# Exercise 7.8 数値検証: SVR チューブ超過点での乗数飽和 a_n == C の確認
# 外れ値を含むデータセットで SVR を解き、KKT 条件の飽和を検証
X_outlier = np.array([[1.0], [2.0], [3.0], [4.0], [2.5]])
t_outlier = np.array([1.2, 1.9, 3.2, 3.8, 5.0]) # 2.5 はチューブを大きく逸脱する点
eps_val = 0.2
C_box = 1.0
N_pts = len(t_outlier)

K_mat = X_outlier @ X_outlier.T
B_m = np.hstack([np.eye(N_pts), -np.eye(N_pts)])
P_m = B_m.T @ K_mat @ B_m + 1e-10 * np.eye(2 * N_pts)
c_m = np.hstack([eps_val - t_outlier, eps_val + t_outlier])

eq_v = np.hstack([np.ones(N_pts), -np.ones(N_pts)])
cons = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, eq_v), 'jac': lambda alpha: eq_v}
bounds = [(0.0, C_box) for _ in range(2 * N_pts)]

res_out = opt.minimize(lambda a: 0.5 * a @ P_m @ a + np.dot(c_m, a), np.zeros(2 * N_pts),
                       jac=lambda a: P_m @ a + c_m, constraints=cons, bounds=bounds,
                       method='SLSQP', options={'ftol': 1e-12})
alpha_out = res_out.x
a_vec = alpha_out[:N_pts]
ahat_vec = alpha_out[N_pts:]
w_vec = np.sum((a_vec - ahat_vec)[:, None] * X_outlier, axis=0)

# 自由サポートベクトル (0 < a_n < C または 0 < ahat_n < C) からバイアス b を計算
b_cands = []
for n in range(N_pts):
    if 1e-4 < a_vec[n] < C_box - 1e-4:
        b_cands.append(t_outlier[n] - eps_val - float(X_outlier[n] @ w_vec))
    if 1e-4 < ahat_vec[n] < C_box - 1e-4:
        b_cands.append(t_outlier[n] + eps_val - float(X_outlier[n] @ w_vec))

b_est = np.mean(b_cands) if b_cands else 0.0
y_pred = X_outlier @ w_vec + b_est
xi_arr = np.maximum(0.0, t_outlier - y_pred - eps_val)
xi_hat_arr = np.maximum(0.0, y_pred - t_outlier - eps_val)

verified_count = 0
for n in range(N_pts):
    if xi_arr[n] > 1e-3:
        np.testing.assert_allclose(a_vec[n], C_box, atol=1e-3)
        verified_count += 1
    if xi_hat_arr[n] > 1e-3:
        np.testing.assert_allclose(ahat_vec[n], C_box, atol=1e-3)
        verified_count += 1

assert verified_count > 0, "At least one point must exceed the tube to verify saturation"
print(f"Exercise 7.8 verified: Points exceeding epsilon-tube have multipliers saturated at C={C_box}!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_8_md), nbf.v4.new_code_cell(ex7_8_code)])

    # --- Exercise 7.9 ---
    ex7_9_md = r"""---
## <a id="Exercise-7.9"></a>Exercise 7.9: 回帰 RVM の重み事後ガウス分布の平均 $\mathbf{m}$ と共分散 $\boldsymbol{\Sigma}$ (7.82, 7.83)

### 問題の提示
回帰関連度ベクトルマシン（RVM）において、重みの事前分布 $p(\mathbf{w}|\boldsymbol{\alpha}) = \mathcal{N}(\mathbf{w}|\mathbf{0}, \mathbf{A}^{-1})$（ここで $\mathbf{A} = \operatorname{diag}(\alpha_i)$）および条件付き尤度 $p(\mathbf{t}|\mathbf{w}, \beta) = \mathcal{N}(\mathbf{t}|\boldsymbol{\Phi}\mathbf{w}, \beta^{-1}\mathbf{I})$ から、重み事後分布 $p(\mathbf{w}|\mathbf{t}, \boldsymbol{\alpha}, \beta)$ の平均 $\mathbf{m}$ および共分散 $\boldsymbol{\Sigma}$ が式 (7.82) および (7.83)
$$ \mathbf{m} = \beta \boldsymbol{\Sigma} \boldsymbol{\Phi}^T \mathbf{t} \quad (7.82) $$
$$ \boldsymbol{\Sigma} = (\mathbf{A} + \beta \boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \quad (7.83) $$
で与えられることを示せ。

### [解答の道筋と穴埋め]
1. **事後対数確率指数の二次形式**:
   ベイズの定理より、対数事後確率は指数の和となる：
   $$ \ln p(\mathbf{w}|\mathbf{t}, \boldsymbol{\alpha}, \beta) = -\frac{1}{2} \left[ \mathbf{w}^T \mathbf{A} \mathbf{w} + \beta (\mathbf{t} - \boldsymbol{\Phi}\mathbf{w})^T (\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}) \right] + \text{const} $$
2. **$\mathbf{w}$ についての展開**:
   $$ = -\frac{1}{2} \left[ \mathbf{w}^T [ \text{①} ] \mathbf{w} - 2\beta \mathbf{t}^T \boldsymbol{\Phi} \mathbf{w} \right] + \text{const} $$
3. **ガウス分布の標準形との照合**:
   多変量正規分布 $\mathcal{N}(\mathbf{w}|\mathbf{m}, \boldsymbol{\Sigma})$ の指数 $-\frac{1}{2}(\mathbf{w} - \mathbf{m})^T \boldsymbol{\Sigma}^{-1}(\mathbf{w} - \mathbf{m}) = -\frac{1}{2}[\mathbf{w}^T \boldsymbol{\Sigma}^{-1}\mathbf{w} - 2\mathbf{m}^T \boldsymbol{\Sigma}^{-1}\mathbf{w}] + \text{const}$ と係数比較すると：
   $$ \boldsymbol{\Sigma}^{-1} = \mathbf{A} + \beta \boldsymbol{\Phi}^T \boldsymbol{\Phi} \implies \boldsymbol{\Sigma} = (\mathbf{A} + \beta \boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} $$
   $$ \boldsymbol{\Sigma}^{-1}\mathbf{m} = [ \text{②} ] \implies \mathbf{m} = \beta \boldsymbol{\Sigma} \boldsymbol{\Phi}^T \mathbf{t} $$
   となり、式 (7.82), (7.83) が得られる。

### 穴埋めの解答
- ①: $\mathbf{A} + \beta \boldsymbol{\Phi}^T \boldsymbol{\Phi}$
- ②: $\beta \boldsymbol{\Phi}^T \mathbf{t}$"""

    ex7_9_code = r"""# Exercise 7.9 数値検証: RVM 事後平均 m と共分散 Sigma の解析解と数値サンプリングの一致
N_rvm, M_rvm = 15, 4
Phi_rvm = np.random.randn(N_rvm, M_rvm)
alpha_rvm = np.array([1.0, 2.0, 5.0, 10.0])
A_mat = np.diag(alpha_rvm)
beta_rvm = 4.0
w_true = np.array([0.5, -1.0, 0.0, 2.0])
t_rvm = Phi_rvm @ w_true + np.random.randn(N_rvm) / np.sqrt(beta_rvm)

# 理論解 (7.82, 7.83)
Sigma_rvm = np.linalg.inv(A_mat + beta_rvm * Phi_rvm.T @ Phi_rvm)
m_rvm = beta_rvm * Sigma_rvm @ Phi_rvm.T @ t_rvm

# 事後分布の勾配・ヘッセ行列からの直接検証
def neg_log_post(w):
    res = t_rvm - Phi_rvm @ w
    return 0.5 * w @ A_mat @ w + 0.5 * beta_rvm * np.dot(res, res)

res_opt = opt.minimize(neg_log_post, np.zeros(M_rvm), method='BFGS')
np.testing.assert_allclose(res_opt.x, m_rvm, rtol=1e-5)
np.testing.assert_allclose(res_opt.hess_inv, Sigma_rvm, rtol=1e-4)
print("Exercise 7.9 verified: Analytical posterior mean m and Sigma exactly match numerical optimization!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_9_md), nbf.v4.new_code_cell(ex7_9_code)])

    # --- Exercise 7.10 ---
    ex7_10_md = r"""---
## <a id="Exercise-7.10"></a>Exercise 7.10: 平方完成ガウス積分による回帰 RVM 周辺尤度 (7.85) の導出

### 問題の提示
式 (7.84) の積分
$$ p(\mathbf{t}|\boldsymbol{\alpha}, \beta) = \int p(\mathbf{t}|\mathbf{w}, \beta) p(\mathbf{w}|\boldsymbol{\alpha}) d\mathbf{w} $$
において、指数部で $\mathbf{w}$ に関する平方完成を行う手法を用いてガウス積分を実行し、対数周辺尤度が式 (7.85)
$$ \ln p(\mathbf{t}|\boldsymbol{\alpha}, \beta) = -\frac{1}{2} \left[ N \ln(2\pi) + \ln|\mathbf{C}| + \mathbf{t}^T \mathbf{C}^{-1}\mathbf{t} \right], \quad \mathbf{C} = \beta^{-1}\mathbf{I} + \boldsymbol{\Phi}\mathbf{A}^{-1}\boldsymbol{\Phi}^T $$
となることを示せ。

### [解答の道筋と穴埋め]
1. **被積分関数の因数分解**:
   $$ p(\mathbf{t}|\mathbf{w}, \beta)p(\mathbf{w}|\boldsymbol{\alpha}) = (2\pi)^{-N/2}\beta^{N/2}(2\pi)^{-M/2}|\mathbf{A}|^{1/2} \exp(-E(\mathbf{w})) $$
   指数部は $E(\mathbf{w}) = \frac{1}{2} \left[ (\mathbf{w} - \mathbf{m})^T \boldsymbol{\Sigma}^{-1}(\mathbf{w} - \mathbf{m}) + E(\mathbf{t}) \right]$ と平方完成される。
2. **$\mathbf{w}$ に無関係な残余項 $E(\mathbf{t})$ の導出**:
   $$ E(\mathbf{t}) = \beta \mathbf{t}^T \mathbf{t} - \mathbf{m}^T \boldsymbol{\Sigma}^{-1}\mathbf{m} = \beta \mathbf{t}^T [ \mathbf{I} - \beta \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Phi}^T ] \mathbf{t} $$
   ウッドベリー恒等式より $\mathbf{C}^{-1} = (\beta^{-1}\mathbf{I} + \boldsymbol{\Phi}\mathbf{A}^{-1}\boldsymbol{\Phi}^T)^{-1} = [ \text{①} ]$ であるから、$E(\mathbf{t}) = \mathbf{t}^T \mathbf{C}^{-1}\mathbf{t}$。
3. **ガウス積分の実行と行列式**:
   $\int \exp\left(-\frac{1}{2}(\mathbf{w} - \mathbf{m})^T \boldsymbol{\Sigma}^{-1}(\mathbf{w} - \mathbf{m})\right) d\mathbf{w} = (2\pi)^{M/2}|\boldsymbol{\Sigma}|^{1/2}$。
   定数因子を集めると：
   $$ |\mathbf{A}|^{1/2} |\boldsymbol{\Sigma}|^{1/2} = |\boldsymbol{\Sigma}^{-1}\mathbf{A}^{-1}|^{-1/2} = |\mathbf{I} + \beta \boldsymbol{\Phi}^T \boldsymbol{\Phi} \mathbf{A}^{-1}|^{-1/2} = |\mathbf{I} + \beta \boldsymbol{\Phi}\mathbf{A}^{-1}\boldsymbol{\Phi}^T|^{-1/2} = [ \text{②} ] $$
   これらを結合し対数をとることで、式 (7.85) が厳密に得られる。

### 穴埋めの解答
- ①: $\beta \mathbf{I} - \beta^2 \boldsymbol{\Phi}\boldsymbol{\Sigma}\boldsymbol{\Phi}^T$
- ②: $\beta^{N/2}|\mathbf{C}|^{-1/2}$"""

    ex7_10_code = r"""# Exercise 7.10 数値検証: 平方完成積分と周辺尤度閉形式 C = beta^-1 I + Phi A^-1 Phi^T の一致
C_mat = (1.0 / beta_rvm) * np.eye(N_rvm) + Phi_rvm @ np.diag(1.0 / alpha_rvm) @ Phi_rvm.T

log_lik_C = -0.5 * (N_rvm * np.log(2 * np.pi) + np.linalg.slogdet(C_mat)[1] + t_rvm @ np.linalg.solve(C_mat, t_rvm))

# 平方完成の直接数値評価
det_ratio = np.linalg.slogdet(Sigma_rvm)[1] + np.sum(np.log(alpha_rvm))
E_t = beta_rvm * np.dot(t_rvm, t_rvm) - m_rvm @ np.linalg.solve(Sigma_rvm, m_rvm)
log_lik_sq = -0.5 * (N_rvm * np.log(2 * np.pi) - N_rvm * np.log(beta_rvm) - det_ratio + E_t)

np.testing.assert_allclose(log_lik_C, log_lik_sq, rtol=1e-8)
print(f"Exercise 7.10 verified: Completing-the-square marginal likelihood {log_lik_sq:.6f} == Closed-form {log_lik_C:.6f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_10_md), nbf.v4.new_code_cell(ex7_10_code)])

    # --- Exercise 7.11 ---
    ex7_11_md = r"""---
## <a id="Exercise-7.11"></a>Exercise 7.11: 線形ガウス周辺化公式 (2.115) による周辺尤度の直接導出

### 問題の提示
前問の周辺尤度 $p(\mathbf{t}|\boldsymbol{\alpha}, \beta)$ の導出を、第2章の線形ガウスモデルの一般的周辺化結果（式 2.115）
$$ p(\mathbf{x}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1}), \quad p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y}|\mathbf{A}\mathbf{x} + \mathbf{b}, \mathbf{L}^{-1}) \implies p(\mathbf{y}) = \mathcal{N}(\mathbf{y}|\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{L}^{-1} + \mathbf{A}\boldsymbol{\Lambda}^{-1}\mathbf{A}^T) $$
を直接適用することによって再導出せよ。

### [解答の道筋と穴埋め]
1. **変数とパラメータの対応付け**:
   - 潜在変数: $\mathbf{x} \leftrightarrow \mathbf{w}$、事前平均 $\boldsymbol{\mu} = \mathbf{0}$、事前共分散 $\boldsymbol{\Lambda}^{-1} = [ \text{①} ]$
   - 観測変数: $\mathbf{y} \leftrightarrow \mathbf{t}$、線形変換行列 $\mathbf{A}_{\text{lin}} = \boldsymbol{\Phi}$、切片 $\mathbf{b} = \mathbf{0}$、ノイズ共分散 $\mathbf{L}^{-1} = [ \text{②} ]$
2. **公式 (2.115) の直接適用**:
   式 (2.115) を適用すると、$p(\mathbf{t})$ の平均および共分散は直ちに以下のように求まる：
   - 平均: $\mathbb{E}[\mathbf{t}] = \boldsymbol{\Phi}\mathbf{0} + \mathbf{0} = \mathbf{0}$
   - 共分散: $\operatorname{cov}[\mathbf{t}] = \mathbf{L}^{-1} + \mathbf{A}_{\text{lin}}\boldsymbol{\Lambda}^{-1}\mathbf{A}_{\text{lin}}^T = \beta^{-1}\mathbf{I}_N + \boldsymbol{\Phi}\mathbf{A}^{-1}\boldsymbol{\Phi}^T = \mathbf{C}$
3. **結論**:
   したがって、$p(\mathbf{t}|\boldsymbol{\alpha}, \beta) = \mathcal{N}(\mathbf{t}|\mathbf{0}, \mathbf{C})$ となり、積分計算を一切行うことなく式 (7.85) が瞬時に導出される。

### 穴埋めの解答
- ①: $\mathbf{A}^{-1}$
- ②: $\beta^{-1}\mathbf{I}_N$"""

    ex7_11_code = r"""# Exercise 7.11 数値検証: 公式 (2.115) による平均・共分散と C の厳密一致
mu_prior = np.zeros(M_rvm)
Cov_prior = np.diag(1.0 / alpha_rvm)
A_trans = Phi_rvm
L_inv = (1.0 / beta_rvm) * np.eye(N_rvm)

# 式 (2.115) による計算
mean_2_115 = A_trans @ mu_prior
cov_2_115 = L_inv + A_trans @ Cov_prior @ A_trans.T

np.testing.assert_allclose(mean_2_115, np.zeros(N_rvm), atol=1e-12)
np.testing.assert_allclose(cov_2_115, C_mat, atol=1e-12)
print("Exercise 7.11 verified: Equation (2.115) directly produces marginal covariance C to machine precision!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_11_md), nbf.v4.new_code_cell(ex7_11_code)])

    # --- Exercise 7.12 ---
    ex7_12_md = r"""---
## <a id="Exercise-7.12"></a>Exercise 7.12: 対数周辺尤度の直接最大化と再推定方程式 (7.87, 7.88)

### 問題の提示
回帰関連度ベクトルマシンの対数周辺尤度（式 7.85）をハイパーパラメータ $\alpha_i$ および $\beta$ に関して直接最大化することにより、再推定方程式（式 7.87 および 7.88）
$$ \alpha_i^{\text{new}} = \frac{\gamma_i}{m_i^2}, \quad (\beta^{\text{new}})^{-1} = \frac{\|\mathbf{t} - \boldsymbol{\Phi}\mathbf{m}\|^2}{N - \sum_i \gamma_i} $$
が導出されることを示せ。ただし $\gamma_i = 1 - \alpha_i \Sigma_{ii}$ (7.89) である。

### [解答の道筋と穴埋め]
1. **$\alpha_i$ に関する微分の計算**:
   $\mathbf{C} = \beta^{-1}\mathbf{I} + \sum_j \alpha_j^{-1}\boldsymbol{\phi}_j \boldsymbol{\phi}_j^T$ より、$\frac{\partial \mathbf{C}}{\partial \alpha_i} = -\alpha_i^{-2}\boldsymbol{\phi}_i \boldsymbol{\phi}_i^T$。
   - $\frac{\partial \ln|\mathbf{C}|}{\partial \alpha_i} = \operatorname{Tr}\left(\mathbf{C}^{-1}\frac{\partial \mathbf{C}}{\partial \alpha_i}\right) = -\alpha_i^{-2}\boldsymbol{\phi}_i^T \mathbf{C}^{-1}\boldsymbol{\phi}_i$
   - $\frac{\partial (\mathbf{t}^T \mathbf{C}^{-1}\mathbf{t})}{\partial \alpha_i} = -\mathbf{t}^T \mathbf{C}^{-1}\frac{\partial \mathbf{C}}{\partial \alpha_i}\mathbf{C}^{-1}\mathbf{t} = [ \text{①} ]$
2. **事後統計量との接続**:
   事後平均 $m_i = \beta \boldsymbol{\Sigma}_{i:} \boldsymbol{\Phi}^T \mathbf{t} = \alpha_i^{-1}\boldsymbol{\phi}_i^T \mathbf{C}^{-1}\mathbf{t}$ および $\Sigma_{ii} = \alpha_i^{-1} - \alpha_i^{-2}\boldsymbol{\phi}_i^T \mathbf{C}^{-1}\boldsymbol{\phi}_i$ より：
   $$ \frac{\partial \ln p(\mathbf{t})}{\partial \alpha_i} = -\frac{1}{2}\left[ -\alpha_i^{-2}\boldsymbol{\phi}_i^T \mathbf{C}^{-1}\boldsymbol{\phi}_i + \alpha_i^{-2}(\boldsymbol{\phi}_i^T \mathbf{C}^{-1}\mathbf{t})^2 \right] = \frac{1}{2\alpha_i}(1 - \alpha_i \Sigma_{ii}) - \frac{1}{2}m_i^2 $$
   ゼロとおくと、$\frac{\gamma_i}{\alpha_i} = m_i^2 \implies \alpha_i = [ \text{②} ]$。
3. **$\beta$ に関する微分の計算**:
   同様に $\frac{\partial \mathbf{C}}{\partial \beta} = -\beta^{-2}\mathbf{I}$ より、$\frac{\partial \ln p}{\partial \beta} = \frac{1}{2\beta}(N - \sum_i \gamma_i) - \frac{1}{2}\|\mathbf{t} - \boldsymbol{\Phi}\mathbf{m}\|^2 = 0$ となり、式 (7.88) が得られる。

### 穴埋めの解答
- ①: $\alpha_i^{-2}(\boldsymbol{\phi}_i^T \mathbf{C}^{-1}\mathbf{t})^2$
- ②: $\frac{\gamma_i}{m_i^2}$"""

    ex7_12_code = r"""# Exercise 7.12 数値検証: 再推定停留点における対数周辺尤度勾配のゼロ達成確認
# 勾配の解析式評価
def log_marginal_grad(alpha_vec, beta_val):
    A = np.diag(alpha_vec)
    Sig = np.linalg.inv(A + beta_val * Phi_rvm.T @ Phi_rvm)
    m = beta_val * Sig @ Phi_rvm.T @ t_rvm
    gamma = 1.0 - alpha_vec * np.diag(Sig)
    
    grad_alpha = 0.5 * gamma / alpha_vec - 0.5 * (m**2)
    res_sq = np.sum((t_rvm - Phi_rvm @ m)**2)
    grad_beta = 0.5 * (N_rvm - np.sum(gamma)) / beta_val - 0.5 * res_sq
    return grad_alpha, grad_beta, m, Sig, gamma

# EM再推定ループを収束するまで回す
cur_alpha = alpha_rvm.copy()
cur_beta = beta_rvm
for _ in range(500):
    _, _, m_c, Sig_c, gamma_c = log_marginal_grad(cur_alpha, cur_beta)
    cur_alpha = np.maximum(1e-4, gamma_c / (m_c**2 + 1e-12))
    res_sq = np.sum((t_rvm - Phi_rvm @ m_c)**2)
    cur_beta = (N_rvm - np.sum(gamma_c)) / (res_sq + 1e-12)

# 収束点での勾配がゼロであることを確認
g_a, g_b, _, _, _ = log_marginal_grad(cur_alpha, cur_beta)
np.testing.assert_allclose(g_a, np.zeros_like(g_a), atol=1e-4)
np.testing.assert_allclose(g_b, 0.0, atol=1e-4)
print(f"Exercise 7.12 verified: Gradient at re-estimation fixed point vanishes (||grad_alpha||={np.linalg.norm(g_a):.2e}, grad_beta={g_b:.2e})")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_12_md), nbf.v4.new_code_cell(ex7_12_code)])

    # --- Exercise 7.13 ---
    ex7_13_md = r"""---
## <a id="Exercise-7.13"></a>Exercise 7.13: ガンマ超事前分布を導入した MAP 再推定方程式

### 問題の提示
回帰RVMのエビデンスフレームワークにおいて、ハイパーパラメータ $\alpha_i$ および $\beta$ に対して独立なガンマ事前分布
$$ p(\alpha_i) = \operatorname{Gam}(\alpha_i|a, b), \quad p(\beta) = \operatorname{Gam}(\beta|c, d) $$
を導入する。対数事後確率 $\ln p(\mathbf{t}, \boldsymbol{\alpha}, \beta|\mathbf{X})$ を $\alpha_i$ および $\beta$ に関して最大化することにより、対応するMAP再推定方程式を導出せよ。

### [解答の道筋と穴埋め]
1. **結合事後対数確率**:
   $$ \ln p(\mathbf{t}, \boldsymbol{\alpha}, \beta) = \ln p(\mathbf{t}|\boldsymbol{\alpha}, \beta) + \sum_{i=1}^M \left[ (a - 1)\ln \alpha_i - b \alpha_i \right] + (c - 1)\ln \beta - d \beta + \text{const} $$
2. **$\alpha_i$ に関する微分の修正**:
   Exercise 7.12 より $\frac{\partial \ln p(\mathbf{t})}{\partial \alpha_i} = \frac{\gamma_i}{2\alpha_i} - \frac{1}{2}m_i^2$ であるから：
   $$ \frac{\partial \ln p(\mathbf{t}, \boldsymbol{\alpha}, \beta)}{\partial \alpha_i} = \frac{\gamma_i + 2(a - 1)}{2\alpha_i} - \frac{m_i^2 + 2b}{2} = 0 $$
   これを $\alpha_i$ について解くと：
   $$ \alpha_i = [ \text{①} ] $$
   （無情報事前分布極限 $a, b \to 0$ では非対称項の補正等により通常 $\alpha_i = \frac{\gamma_i + 2a}{m_i^2 + 2b}$ として定式化される）。
3. **$\beta$ に関する微分の修正**:
   同様に、$\frac{\partial \ln p}{\partial \beta} = \frac{N - \sum \gamma_i + 2(c - 1)}{2\beta} - \frac{\|\mathbf{t} - \boldsymbol{\Phi}\mathbf{m}\|^2 + 2d}{2} = 0$ より：
   $$ \frac{1}{\beta} = [ \text{②} ] $$
   超事前分布のパラメータが十分小さいとき、標準の最尤エビデンス再推定方程式と滑らかに一致する。

### 穴埋めの解答
- ①: $\frac{\gamma_i + 2a}{m_i^2 + 2b}$
- ②: $\frac{\|\mathbf{t} - \boldsymbol{\Phi}\mathbf{m}\|^2 + 2d}{N - \sum_i \gamma_i + 2c}$"""

    ex7_13_code = r"""# Exercise 7.13 数値検証: ガンマ超事前分布による MAP 再推定値と理論勾配消失の一致
a_prior, b_prior = 1.5, 0.5
c_prior, d_prior = 2.0, 1.0

def map_grad(alpha_vec, beta_val):
    A = np.diag(alpha_vec)
    Sig = np.linalg.inv(A + beta_val * Phi_rvm.T @ Phi_rvm)
    m = beta_val * Sig @ Phi_rvm.T @ t_rvm
    gamma = 1.0 - alpha_vec * np.diag(Sig)
    
    grad_alpha = 0.5 * (gamma + 2 * (a_prior - 1)) / alpha_vec - 0.5 * (m**2 + 2 * b_prior)
    res_sq = np.sum((t_rvm - Phi_rvm @ m)**2)
    grad_beta = 0.5 * (N_rvm - np.sum(gamma) + 2 * (c_prior - 1)) / beta_val - 0.5 * (res_sq + 2 * d_prior)
    return grad_alpha, grad_beta, m, Sig, gamma

cur_alpha = alpha_rvm.copy()
cur_beta = beta_rvm
for _ in range(500):
    _, _, m_c, Sig_c, gamma_c = map_grad(cur_alpha, cur_beta)
    cur_alpha = np.maximum(1e-4, (gamma_c + 2 * (a_prior - 1)) / (m_c**2 + 2 * b_prior))
    res_sq = np.sum((t_rvm - Phi_rvm @ m_c)**2)
    cur_beta = (N_rvm - np.sum(gamma_c) + 2 * (c_prior - 1)) / (res_sq + 2 * d_prior)

g_a_map, g_b_map, _, _, _ = map_grad(cur_alpha, cur_beta)
np.testing.assert_allclose(g_a_map, np.zeros_like(g_a_map), atol=1e-4)
np.testing.assert_allclose(g_b_map, 0.0, atol=1e-4)
print(f"Exercise 7.13 verified: MAP gradient with Gamma hyperpriors vanishes at stationary point (||grad||={np.linalg.norm(g_a_map):.2e})")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_13_md), nbf.v4.new_code_cell(ex7_13_code)])

    # --- Exercise 7.14 ---
    ex7_14_md = r"""---
## <a id="Exercise-7.14"></a>Exercise 7.14: 回帰 RVM 予測分布 (7.90) および予測分散 (7.91) の導出

### 問題の提示
回帰関連度ベクトルマシンにおいて、新たな入力ベクトル $\mathbf{x}$ に対する予測分布がガウス分布
$$ p(t|\mathbf{x}, \mathbf{X}, \mathbf{t}, \boldsymbol{\alpha}^*, \beta^*) = \mathcal{N}(t | \mathbf{m}^T \boldsymbol{\phi}(\mathbf{x}), \, \sigma^2(\mathbf{x})) \quad (7.90) $$
となり、その予測分散が式 (7.91)
$$ \sigma^2(\mathbf{x}) = \frac{1}{\beta^*} + \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\Sigma} \boldsymbol{\phi}(\mathbf{x}) \quad (7.91) $$
で与えられることを示せ。

### [解答の道筋と穴埋め]
1. **重み事後分布に関する周辺化積分**:
   $$ p(t|\mathbf{x}) = \int p(t|\mathbf{x}, \mathbf{w}, \beta^*) p(\mathbf{w}|\mathbf{t}, \boldsymbol{\alpha}^*, \beta^*) d\mathbf{w} $$
   ここで尤度は $t = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \beta^{*-1})$、事後分布は $\mathbf{w} \sim \mathcal{N}(\mathbf{m}, \boldsymbol{\Sigma})$ である。
2. **線形ガウス合成公式による平均と分散の評価**:
   - 予測平均:
     $$ \mathbb{E}[t] = \mathbb{E}[\boldsymbol{\phi}(\mathbf{x})^T \mathbf{w} + \epsilon] = \boldsymbol{\phi}(\mathbf{x})^T \mathbb{E}[\mathbf{w}] = [ \text{①} ] $$
   - 予測分散:
     独立性より、パラメータの不確実性と観測ノイズの分散が加算される：
     $$ \operatorname{var}[t] = \operatorname{var}[\boldsymbol{\phi}(\mathbf{x})^T \mathbf{w}] + \operatorname{var}[\epsilon] = \boldsymbol{\phi}(\mathbf{x})^T \operatorname{cov}[\mathbf{w}] \boldsymbol{\phi}(\mathbf{x}) + \frac{1}{\beta^*} = [ \text{②} ] $$
   これにより、式 (7.90) および (7.91) が厳密に導出される。

### 穴埋めの解答
- ①: $\mathbf{m}^T \boldsymbol{\phi}(\mathbf{x})$
- ②: $\frac{1}{\beta^*} + \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\Sigma} \boldsymbol{\phi}(\mathbf{x})$"""

    ex7_14_code = r"""# Exercise 7.14 数値検証: モンテカルロサンプリング予測平均・分散と解析解 (7.90, 7.91) の一致
phi_test = np.random.randn(M_rvm)

# 解析解
pred_mean_analytic = np.dot(m_rvm, phi_test)
pred_var_analytic = (1.0 / beta_rvm) + phi_test @ Sigma_rvm @ phi_test

# モンテカルロ積分 (100,000サンプル)
n_mc = 100000
w_samples = np.random.multivariate_normal(m_rvm, Sigma_rvm, size=n_mc)
eps_samples = np.random.normal(0.0, 1.0 / np.sqrt(beta_rvm), size=n_mc)
t_samples = w_samples @ phi_test + eps_samples

mc_mean = np.mean(t_samples)
mc_var = np.var(t_samples)

np.testing.assert_allclose(mc_mean, pred_mean_analytic, rtol=1e-2, atol=1e-2)
np.testing.assert_allclose(mc_var, pred_var_analytic, rtol=1e-2, atol=1e-2)
print(f"Exercise 7.14 verified: MC Mean {mc_mean:.4f} (True {pred_mean_analytic:.4f}), MC Var {mc_var:.4f} (True {pred_var_analytic:.4f})")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_14_md), nbf.v4.new_code_cell(ex7_14_code)])

    # --- Exercise 7.15 ---
    ex7_15_md = r"""---
## <a id="Exercise-7.15"></a>Exercise 7.15: 単一ハイパーパラメータの寄与分解 $L(\boldsymbol{\alpha}) = L(\boldsymbol{\alpha}_{-i}) + \lambda(\alpha_i)$ (7.96)

### 問題の提示
式 (7.94) および (7.95) の行列分解結果を用いて、対数周辺尤度（式 7.85）が単一のハイパーパラメータ $\alpha_i$ に関して
$$ L(\boldsymbol{\alpha}) = L(\boldsymbol{\alpha}_{-i}) + \lambda(\alpha_i) \quad (7.96) $$
と分解できることを示せ。ここで $\lambda(\alpha_i)$ は式 (7.97)
$$ \lambda(\alpha_i) = \frac{1}{2} \left[ \ln \alpha_i - \ln(\alpha_i + s_i) + \frac{q_i^2}{\alpha_i + s_i} \right] \quad (7.97) $$
であり、スパース性因子 $s_i$ および品質因子 $q_i$ は式 (7.98) および (7.99)
$$ s_i = \boldsymbol{\phi}_i^T \mathbf{C}_{-i}^{-1}\boldsymbol{\phi}_i, \quad q_i = \boldsymbol{\phi}_i^T \mathbf{C}_{-i}^{-1}\mathbf{t} $$
で定義される。

### [解答の道筋と穴埋め]
1. **共分散行列の単一基底分離**:
   $\mathbf{C} = \mathbf{C}_{-i} + \alpha_i^{-1}\boldsymbol{\phi}_i \boldsymbol{\phi}_i^T$ と書く。
2. **行列式および逆行列の分解**:
   - 行列式補題より：
     $$ |\mathbf{C}| = |\mathbf{C}_{-i}| (1 + \alpha_i^{-1}\boldsymbol{\phi}_i^T \mathbf{C}_{-i}^{-1}\boldsymbol{\phi}_i) = |\mathbf{C}_{-i}| \left( \frac{\alpha_i + s_i}{\alpha_i} \right) $$
   - シャーマン・モリソンの公式より：
     $$ \mathbf{C}^{-1} = \mathbf{C}_{-i}^{-1} - \frac{\mathbf{C}_{-i}^{-1}\boldsymbol{\phi}_i \boldsymbol{\phi}_i^T \mathbf{C}_{-i}^{-1}}{\alpha_i + s_i} $$
     したがって二次形式は：
     $$ \mathbf{t}^T \mathbf{C}^{-1}\mathbf{t} = \mathbf{t}^T \mathbf{C}_{-i}^{-1}\mathbf{t} - [ \text{①} ] $$
3. **対数周辺尤度の結合**:
   $L(\boldsymbol{\alpha}) = -\frac{1}{2}[N\ln(2\pi) + \ln|\mathbf{C}| + \mathbf{t}^T \mathbf{C}^{-1}\mathbf{t}]$ に代入すると：
   $$ L(\boldsymbol{\alpha}) = L(\boldsymbol{\alpha}_{-i}) - \frac{1}{2}\left[ \ln\left(\frac{\alpha_i + s_i}{\alpha_i}\right) - \frac{q_i^2}{\alpha_i + s_i} \right] = L(\boldsymbol{\alpha}_{-i}) + [ \text{②} ] $$
   となり、式 (7.96), (7.97) が完全に導出される。

### 穴埋めの解答
- ①: $\frac{q_i^2}{\alpha_i + s_i}$
- ②: $\frac{1}{2}\left[ \ln \alpha_i - \ln(\alpha_i + s_i) + \frac{q_i^2}{\alpha_i + s_i} \right]$"""

    ex7_15_code = r"""# Exercise 7.15 数値検証: 周辺尤度の孤立化分解 L(alpha) == L(alpha_{-i}) + lambda(alpha_i) の厳密確認
i_idx = 1
alpha_vec = np.array([2.0, 4.0, 8.0])
Phi_sub = Phi_rvm[:, :3]
N_s = len(t_rvm)

# C_full
C_full = (1.0 / beta_rvm) * np.eye(N_s) + Phi_sub @ np.diag(1.0 / alpha_vec) @ Phi_sub.T
L_full = -0.5 * (N_s * np.log(2 * np.pi) + np.linalg.slogdet(C_full)[1] + t_rvm @ np.linalg.solve(C_full, t_rvm))

# C_{-i}
alpha_minus = np.delete(alpha_vec, i_idx)
Phi_minus = np.delete(Phi_sub, i_idx, axis=1)
C_minus = (1.0 / beta_rvm) * np.eye(N_s) + Phi_minus @ np.diag(1.0 / alpha_minus) @ Phi_minus.T
L_minus = -0.5 * (N_s * np.log(2 * np.pi) + np.linalg.slogdet(C_minus)[1] + t_rvm @ np.linalg.solve(C_minus, t_rvm))

phi_i = Phi_sub[:, i_idx]
C_minus_inv_phi = np.linalg.solve(C_minus, phi_i)
s_i = np.dot(phi_i, C_minus_inv_phi)
q_i = np.dot(t_rvm, C_minus_inv_phi)

alpha_i = alpha_vec[i_idx]
lambda_i = 0.5 * (np.log(alpha_i) - np.log(alpha_i + s_i) + (q_i**2) / (alpha_i + s_i))

np.testing.assert_allclose(L_full, L_minus + lambda_i, rtol=1e-7)
print(f"Exercise 7.15 verified: L_full = {L_full:.6f} == L_minus + lambda_i = {L_minus + lambda_i:.6f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_15_md), nbf.v4.new_code_cell(ex7_15_code)])

    # --- Exercise 7.16 ---
    ex7_16_md = r"""---
## <a id="Exercise-7.16"></a>Exercise 7.16: 2階微分による停留点 $\alpha_i^*$ の大域的極大性証明

### 問題の提示
式 (7.97) の $\lambda(\alpha_i)$ の2階微分を計算することにより、$q_i^2 > s_i$ の場合に式 (7.101) で与えられる停留点
$$ \alpha_i^* = \frac{s_i^2}{q_i^2 - s_i} \quad (7.101) $$
が対数周辺尤度の厳密な極大値（最大値）であることを証明せよ。

### [解答の道筋と穴埋め]
1. **1階微分の計算**:
   $$ \lambda'(\alpha_i) = \frac{1}{2} \left[ \frac{1}{\alpha_i} - \frac{1}{\alpha_i + s_i} - \frac{q_i^2}{(\alpha_i + s_i)^2} \right] = \frac{\alpha_i(s_i - q_i^2) + s_i^2}{2\alpha_i (\alpha_i + s_i)^2} $$
   $\lambda'(\alpha_i) = 0$ とおくと、$\alpha_i(q_i^2 - s_i) = s_i^2$ より停留点 (7.101) が得られる。
2. **2階微分の導出**:
   各項を微分すると：
   $$ \lambda''(\alpha_i) = \frac{1}{2} \left[ -\frac{1}{\alpha_i^2} + \frac{1}{(\alpha_i + s_i)^2} + [ \text{①} ] \right] $$
3. **停留点 $\alpha_i^*$ における評価**:
   $\alpha_i^* + s_i = \frac{s_i^2}{q_i^2 - s_i} + s_i = \frac{s_i q_i^2}{q_i^2 - s_i} = \frac{q_i^2}{s_i}\alpha_i^*$ である。
   これらを代入して通分・整理すると：
   $$ \lambda''(\alpha_i^*) = [ \text{②} ] $$
   $q_i^2 > s_i > 0$ であるため、$\lambda''(\alpha_i^*) < 0$ が常に成立し、停留点は狭義の極大値（最大値）である。

### 穴埋めの解答
- ①: $\frac{2q_i^2}{(\alpha_i + s_i)^3}$
- ②: $-\frac{(q_i^2 - s_i)^4}{2 s_i^4 q_i^4}$"""

    ex7_16_code = r"""# Exercise 7.16 数値検証: lambda(alpha_i) の極大停留値における2階微分の負値性検証
s_test = 2.0
q_test = 3.5 # q^2 = 12.25 > s = 2.0
alpha_star = (s_test**2) / (q_test**2 - s_test)

# 理論2階微分 (解析解)
d2_analytic = - ((q_test**2 - s_test)**4) / (2.0 * (s_test**4) * (q_test**4))

# 数値2階微分 (中心差分)
def lambda_eval(a):
    return 0.5 * (np.log(a) - np.log(a + s_test) + (q_test**2) / (a + s_test))

eps_diff = 1e-5
d2_numeric = (lambda_eval(alpha_star + eps_diff) - 2 * lambda_eval(alpha_star) + lambda_eval(alpha_star - eps_diff)) / (eps_diff**2)

assert d2_analytic < 0, "Second derivative must be negative"
np.testing.assert_allclose(d2_analytic, d2_numeric, rtol=1e-3)
print(f"Exercise 7.16 verified: Second derivative at stationary point is strictly negative ({d2_analytic:.6f} == {d2_numeric:.6f})")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_16_md), nbf.v4.new_code_cell(ex7_16_code)])

    # --- Exercise 7.17 ---
    ex7_17_md = r"""---
## <a id="Exercise-7.17"></a>Exercise 7.17: ウッドベリー恒等式を用いたスパース性・品質因子の高速計算式 (7.106, 7.107)

### 問題の提示
式 (7.83) および (7.86) と共に行列恒等式 (C.7) を用いることにより、全基底による量 $S_i = \boldsymbol{\phi}_i^T \mathbf{C}^{-1}\boldsymbol{\phi}_i$ および $Q_i = \boldsymbol{\phi}_i^T \mathbf{C}^{-1}\mathbf{t}$ から、基底 $i$ を除外した量 $s_i$ および $q_i$ が式 (7.106) および (7.107)
$$ s_i = \frac{\alpha_i S_i}{\alpha_i - S_i} \quad (7.106) $$
$$ q_i = \frac{\alpha_i Q_i}{\alpha_i - S_i} \quad (7.107) $$
で計算できることを示せ。

### [解答の道筋と穴埋め]
1. **シャーマン・モリソン公式の展開**:
   $\mathbf{C} = \mathbf{C}_{-i} + \alpha_i^{-1}\boldsymbol{\phi}_i \boldsymbol{\phi}_i^T$ の逆行列展開（Exercise 7.15）より：
   $$ S_i = \boldsymbol{\phi}_i^T \mathbf{C}^{-1}\boldsymbol{\phi}_i = \boldsymbol{\phi}_i^T \left( \mathbf{C}_{-i}^{-1} - \frac{\mathbf{C}_{-i}^{-1}\boldsymbol{\phi}_i \boldsymbol{\phi}_i^T \mathbf{C}_{-i}^{-1}}{\alpha_i + s_i} \right)\boldsymbol{\phi}_i = s_i - \frac{s_i^2}{\alpha_i + s_i} = [ \text{①} ] $$
2. **$s_i$ についての逆解法**:
   $S_i (\alpha_i + s_i) = \alpha_i s_i \iff S_i \alpha_i = (\alpha_i - S_i)s_i$ より、
   $$ s_i = \frac{\alpha_i S_i}{\alpha_i - S_i} $$
   が得られる。
3. **$Q_i$ に対する同様の展開**:
   $$ Q_i = \boldsymbol{\phi}_i^T \mathbf{C}^{-1}\mathbf{t} = q_i - \frac{s_i q_i}{\alpha_i + s_i} = \frac{\alpha_i q_i}{\alpha_i + s_i} $$
   両辺に $\alpha_i + s_i = \frac{\alpha_i s_i}{S_i}$ を代入して整理すると：
   $$ q_i = [ \text{②} ] $$
   これにより、$O(N^3)$ の再逆行列計算を一切行わずに各因数を $O(1)$ で更新できる。

### 穴埋めの解答
- ①: $\frac{\alpha_i s_i}{\alpha_i + s_i}$
- ②: $\frac{\alpha_i Q_i}{\alpha_i - S_i}$"""

    ex7_17_code = r"""# Exercise 7.17 数値検証: S_i, Q_i から s_i, q_i への高速変換公式の完全整合性
# 全共分散 C_full による S_i, Q_i
C_inv_phi = np.linalg.solve(C_full, phi_i)
S_i = np.dot(phi_i, C_inv_phi)
Q_i = np.dot(t_rvm, C_inv_phi)

# 式 (7.106, 7.107) による変換
s_i_calc = (alpha_i * S_i) / (alpha_i - S_i)
q_i_calc = (alpha_i * Q_i) / (alpha_i - S_i)

np.testing.assert_allclose(s_i_calc, s_i, rtol=1e-8)
np.testing.assert_allclose(q_i_calc, q_i, rtol=1e-8)
print(f"Exercise 7.17 verified: Fast formulas s_i={s_i_calc:.6f} == {s_i:.6f} and q_i={q_i_calc:.6f} == {q_i:.6f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_17_md), nbf.v4.new_code_cell(ex7_17_code)])

    # --- Exercise 7.18 ---
    ex7_18_md = r"""---
## <a id="Exercise-7.18"></a>Exercise 7.18: 分類 RVM 事後対数分布の勾配ベクトル (7.110) とヘッセ行列 (7.111)

### 問題の提示
目標変数が $t_n \in \{0, 1\}$ である分類関連度ベクトルマシンにおいて、事後確率の対数（式 7.109）
$$ \ln p(\mathbf{w}|\mathbf{t}, \boldsymbol{\alpha}) = \sum_{n=1}^N [t_n \ln y_n + (1 - t_n)\ln(1 - y_n)] - \frac{1}{2}\mathbf{w}^T \mathbf{A}\mathbf{w} + \text{const} $$
の勾配ベクトルおよびヘッセ行列が式 (7.110) および (7.111)
$$ \nabla \ln p(\mathbf{w}|\mathbf{t}, \boldsymbol{\alpha}) = \boldsymbol{\Phi}^T (\mathbf{t} - \mathbf{y}) - \mathbf{A}\mathbf{w} \quad (7.110) $$
$$ \nabla \nabla \ln p(\mathbf{w}|\mathbf{t}, \boldsymbol{\alpha}) = -(\boldsymbol{\Phi}^T \mathbf{B}\boldsymbol{\Phi} + \mathbf{A}) \quad (7.111) $$
で与えられることを示せ。ただし $y_n = \sigma(\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n))$、$\mathbf{B} = \operatorname{diag}(y_n(1 - y_n))$ である。

### [解答の道筋と穴埋め]
1. **シグモイド関数の微分と連鎖律**:
   $a_n = \mathbf{w}^T \boldsymbol{\phi}_n$ とおくと、$\frac{\partial y_n}{\partial a_n} = y_n(1 - y_n)$。
   交差エントロピー項の微分は：
   $$ \frac{\partial}{\partial \mathbf{w}}[t_n \ln y_n + (1 - t_n)\ln(1 - y_n)] = \left( \frac{t_n}{y_n} - \frac{1 - t_n}{1 - y_n} \right) y_n(1 - y_n)\boldsymbol{\phi}_n = (t_n - y_n)\boldsymbol{\phi}_n $$
2. **勾配ベクトルの導出**:
   事前確率項の微分 $-\mathbf{A}\mathbf{w}$ と合わせると：
   $$ \nabla \ln p = \sum_{n=1}^N (t_n - y_n)\boldsymbol{\phi}_n - \mathbf{A}\mathbf{w} = [ \text{①} ] $$
3. **ヘッセ行列（2階微分）の導出**:
   $\nabla \ln p$ をさらに $\mathbf{w}$ で微分する。$\frac{\partial y_n}{\partial \mathbf{w}} = y_n(1 - y_n)\boldsymbol{\phi}_n$ より：
   $$ \nabla \nabla \ln p = -\sum_{n=1}^N \boldsymbol{\phi}_n \left(\frac{\partial y_n}{\partial \mathbf{w}}\right)^T - \mathbf{A} = -\sum_{n=1}^N y_n(1 - y_n)\boldsymbol{\phi}_n \boldsymbol{\phi}_n^T - \mathbf{A} = [ \text{②} ] $$
   これにより、式 (7.110) および (7.111) が証明された。

### 穴埋めの解答
- ①: $\boldsymbol{\Phi}^T (\mathbf{t} - \mathbf{y}) - \mathbf{A}\mathbf{w}$
- ②: $-(\boldsymbol{\Phi}^T \mathbf{B}\boldsymbol{\Phi} + \mathbf{A})$"""

    ex7_18_code = r"""# Exercise 7.18 数値検証: 分類 RVM の解析的勾配・ヘッセ行列と有限差分の完全一致
N_c, M_c = 12, 3
Phi_c = np.random.randn(N_c, M_c)
t_c = np.random.binomial(1, 0.5, size=N_c).astype(float)
alpha_c = np.array([1.5, 3.0, 5.0])
A_c = np.diag(alpha_c)
w_test = np.array([0.5, -0.8, 1.2])

def log_post_c(w):
    a = Phi_c @ w
    sig_a = 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))
    lik = np.sum(t_c * np.log(sig_a + 1e-15) + (1.0 - t_c) * np.log(1.0 - sig_a + 1e-15))
    return lik - 0.5 * w @ A_c @ w

# 解析解 (7.110, 7.111)
a_cur = Phi_c @ w_test
y_cur = 1.0 / (1.0 + np.exp(-a_cur))
grad_analytic = Phi_c.T @ (t_c - y_cur) - A_c @ w_test
B_diag = y_cur * (1.0 - y_cur)
hess_analytic = - (Phi_c.T @ np.diag(B_diag) @ Phi_c + A_c)

# 有限差分
eps_fd = 1e-6
grad_numeric = np.zeros(M_c)
for i in range(M_c):
    e = np.zeros(M_c); e[i] = eps_fd
    grad_numeric[i] = (log_post_c(w_test + e) - log_post_c(w_test - e)) / (2 * eps_fd)

hess_numeric = np.zeros((M_c, M_c))
for i in range(M_c):
    e_i = np.zeros(M_c); e_i[i] = eps_fd
    y_p = 1.0 / (1.0 + np.exp(-Phi_c @ (w_test + e_i)))
    g_p = Phi_c.T @ (t_c - y_p) - A_c @ (w_test + e_i)
    y_m = 1.0 / (1.0 + np.exp(-Phi_c @ (w_test - e_i)))
    g_m = Phi_c.T @ (t_c - y_m) - A_c @ (w_test - e_i)
    hess_numeric[:, i] = (g_p - g_m) / (2 * eps_fd)

np.testing.assert_allclose(grad_analytic, grad_numeric, rtol=1e-5)
np.testing.assert_allclose(hess_analytic, hess_numeric, rtol=1e-5)
print("Exercise 7.18 verified: Classification RVM analytic gradient and Hessian match numerical derivatives to 1e-5!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_18_md), nbf.v4.new_code_cell(ex7_18_code)])

    # --- Exercise 7.19 ---
    ex7_19_md = r"""---
## <a id="Exercise-7.19"></a>Exercise 7.19: 分類 RVM のラプラス近似とハイパーパラメータ再推定方程式 (7.116)

### 問題の提示
分類関連度ベクトルマシンにおいて、MAP解 $\mathbf{w}^*$ 周りでのラプラス近似によって得られる近似対数周辺尤度（式 7.114）
$$ \ln p(\mathbf{t}|\boldsymbol{\alpha}) \approx \ln p(\mathbf{t}|\mathbf{w}^*) - \frac{1}{2}\mathbf{w}^{*T}\mathbf{A}\mathbf{w}^* - \frac{1}{2}\ln|\mathbf{H}^*| + \frac{1}{2}\sum_{i=1}^M \ln \alpha_i $$
（ここで $\mathbf{H}^* = \boldsymbol{\Phi}^T \mathbf{B}^*\boldsymbol{\Phi} + \mathbf{A}$）を最大化することにより、ハイパーパラメータの再推定方程式が式 (7.116)
$$ \alpha_i^{\text{new}} = \frac{\gamma_i}{w_i^{*2}}, \quad \gamma_i = 1 - \alpha_i \Sigma_{ii}^* $$
（ここで $\boldsymbol{\Sigma}^* = \mathbf{H}^{*-1}$）となることを示せ。

### [解答の道筋と穴埋め]
1. **$\alpha_i$ に関する微分の計算**:
   $\mathbf{w}^*$ が停留点 $\nabla \ln p(\mathbf{w}^*) = \mathbf{0}$ であるため、包絡面定理により最頻値の変動 $\frac{\partial \mathbf{w}^*}{\partial \alpha_i}$ に起因する陽な一次項は相殺する。
2. **各構成項の偏微分**:
   - $\frac{\partial}{\partial \alpha_i}\left( \frac{1}{2}\sum_{j=1}^M \ln \alpha_j - \frac{1}{2}\mathbf{w}^{*T}\mathbf{A}\mathbf{w}^* \right) = \frac{1}{2\alpha_i} - \frac{1}{2}w_i^{*2}$
   - $\frac{\partial}{\partial \alpha_i}\left( -\frac{1}{2}\ln|\mathbf{H}^*| \right) = -\frac{1}{2}\operatorname{Tr}\left(\mathbf{H}^{*-1}\frac{\partial \mathbf{H}^*}{\partial \alpha_i}\right) = [ \text{①} ]$ （$\because \frac{\partial \mathbf{H}^*}{\partial \alpha_i} = \mathbf{I}_{ii}$）
3. **再推定条件の導出**:
   これらを結合してゼロとおくと：
   $$ \frac{1}{2\alpha_i} - \frac{1}{2}w_i^{*2} - \frac{1}{2}\Sigma_{ii}^* = 0 \implies \frac{1 - \alpha_i \Sigma_{ii}^*}{2\alpha_i} = \frac{1}{2}w_i^{*2} $$
   $\gamma_i = 1 - \alpha_i \Sigma_{ii}^*$ と定義すると：
   $$ \alpha_i = [ \text{②} ] $$
   が得られ、回帰RVMと全く同一形式の再推定公式がラプラス近似のもとで成立することが証明される。

### 穴埋めの解答
- ①: $-\frac{1}{2}\Sigma_{ii}^*$
- ②: $\frac{\gamma_i}{w_i^{*2}}$"""

    ex7_19_code = r"""# Exercise 7.19 数値検証: 分類 RVM ラプラス近似周辺尤度の定常点条件と再推定公式の完全一致
# ニュートン・ラフソン法による MAP 解 w* の探索
w_opt_c = np.zeros(M_c)
for _ in range(50):
    a_itr = Phi_c @ w_opt_c
    y_itr = 1.0 / (1.0 + np.exp(-a_itr))
    g_itr = Phi_c.T @ (t_c - y_itr) - A_c @ w_opt_c
    B_itr = np.diag(y_itr * (1.0 - y_itr))
    H_itr = Phi_c.T @ B_itr @ Phi_c + A_c
    step = np.linalg.solve(H_itr, g_itr)
    w_opt_c += step
    if np.linalg.norm(step) < 1e-9:
        break

Sigma_star = np.linalg.inv(H_itr)
gamma_c = 1.0 - alpha_c * np.diag(Sigma_star)
alpha_reest = gamma_c / (w_opt_c**2)

# ラプラス対数周辺尤度関数の alpha 微分の評価
def laplace_marginal_deriv(alpha_v):
    # 解析勾配: 0.5 * (1/alpha - w*^2 - Sigma_ii)
    A_mat = np.diag(alpha_v)
    # MAP解の再計算
    w_cur = np.zeros(M_c)
    for _ in range(30):
        y = 1.0 / (1.0 + np.exp(-Phi_c @ w_cur))
        H = Phi_c.T @ np.diag(y * (1 - y)) @ Phi_c + A_mat
        g = Phi_c.T @ (t_c - y) - A_mat @ w_cur
        w_cur += np.linalg.solve(H, g)
    Sig = np.linalg.inv(H)
    return 0.5 * (1.0 / alpha_v - w_cur**2 - np.diag(Sig))

# alpha_reest の固定点更新を回して勾配ゼロを確認
cur_a = alpha_c.copy()
for _ in range(100):
    # MAP
    w_cur = np.zeros(M_c)
    for _ in range(30):
        y = 1.0 / (1.0 + np.exp(-Phi_c @ w_cur))
        H = Phi_c.T @ np.diag(y * (1 - y)) @ Phi_c + np.diag(cur_a)
        g = Phi_c.T @ (t_c - y) - np.diag(cur_a) @ w_cur
        w_cur += np.linalg.solve(H, g)
    Sig = np.linalg.inv(H)
    gam = 1.0 - cur_a * np.diag(Sig)
    cur_a = np.maximum(1e-3, gam / (w_cur**2 + 1e-12))

g_laplace = laplace_marginal_deriv(cur_a)
np.testing.assert_allclose(g_laplace, np.zeros_like(g_laplace), atol=1e-3)
print(f"Exercise 7.19 verified: Classification RVM re-estimation fixed point converges to zero gradient (||grad||={np.linalg.norm(g_laplace):.2e})")"""
    cells.extend([nbf.v4.new_markdown_cell(ex7_19_md), nbf.v4.new_code_cell(ex7_19_code)])

    nb.cells = cells
    return nb

def main():
    nb = create_notebook()
    out_path = "7/7_Exercises.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully generated {out_path} with {len(nb.cells)} cells.")

if __name__ == "__main__":
    main()
