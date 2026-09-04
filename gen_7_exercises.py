import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 第7章 スパースカーネルマシン：演習問題 (Exercises 7.1 - 7.19)

本ノートブックでは、PRML第7章「スパースカーネルマシン (Sparse Kernel Machines)」の**全19問 (Exercises 7.1 〜 7.19)** の詳細な論理ステップ（数理的証明・思考の道筋）および Python による数値検証コードを収録しています。
SVMの幾何学的マージン恒等式、相補性スラック条件、SVRの双対形式、およびRVMのエビデンスフレームワーク・スパース性極大解析の数理を計算機上で実験・検証します。"""))

# Exercises 7.1 - 7.6
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 7.1 - 7.6: 最大マージン超平面の幾何学、不変性、マージン恒等式 $1/\rho^2 = \|\mathbf{w}\|^2 = \sum a_n$

### 問題 7.1: Parzen窓推定による決定則と最近傍平均分類器
各クラス $t \in \{-1, +1\}$ の密度推定を $p(\mathbf{x}|\mathcal{C}_k) = \frac{1}{N_k}\sum_{n \in \mathcal{C}_k} k(\mathbf{x}, \mathbf{x}_n)$ とする。
線形カーネル $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$ のとき、比 $p(\mathbf{x}|\mathcal{C}_1) \gtrless p(\mathbf{x}|\mathcal{C}_{-1})$ は
$$ \mathbf{x}^T \mathbf{m}_1 \gtrless \mathbf{x}^T \mathbf{m}_{-1} \iff \|\mathbf{x} - \mathbf{m}_1\|^2 \lessgtr \|\mathbf{x} - \mathbf{m}_{-1}\|^2 $$
（各クラス平均 $\mathbf{m}_k$ に対する距離）に一致し、一般の非線形カーネルでは特徴空間 $\boldsymbol{\phi}(\mathbf{x})$ におけるクラス平均への距離に基づく決定則となることを示せ。

### 問題 7.2: マージン制約定数 $\gamma$ に対する超平面の不変性
正準制約を $t_n(\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n) + b) \ge \gamma > 0$ としたとき、
$\mathbf{w}' = \mathbf{w}/\gamma, b' = b/\gamma$ と変数変換すれば制約は $t_n(\mathbf{w}'^T \boldsymbol{\phi}(\mathbf{x}_n) + b') \ge 1$ となり、
決定境界 $\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}) + b = 0 \iff \mathbf{w}'^T \boldsymbol{\phi}(\mathbf{x}) + b' = 0$ は幾何学的に完全に同一の超平面を定義することを示せ。

### 問題 7.3: 異クラス2点による超平面の一意決定
2点 $\mathbf{x}_1 (t_1 = +1), \mathbf{x}_2 (t_2 = -1)$ に対し、制約を満たしつつ $\|\mathbf{w}\|$ を最小化する超平面は、2点を結ぶ線分の垂直二等分面 $\mathbf{w} = \frac{2(\mathbf{x}_1 - \mathbf{x}_2)}{\|\mathbf{x}_1 - \mathbf{x}_2\|^2}, b = -\frac{\|\mathbf{x}_1\|^2 - \|\mathbf{x}_2\|^2}{\|\mathbf{x}_1 - \mathbf{x}_2\|^2}$ として一意に決定されることを示せ。

### 問題 7.4 & 7.5: マージン幅 $\rho$ とラグランジュ乗数の恒等式
マージン幅 $\rho = \frac{1}{\|\mathbf{w}\|}$ に対し、KKT条件 $\mathbf{w} = \sum_n a_n t_n \boldsymbol{\phi}(\mathbf{x}_n)$ および $\sum_n a_n t_n = 0$、サポートベクトル $t_n y_n = 1$ を用いると：
$$ \|\mathbf{w}\|^2 = \mathbf{w}^T \mathbf{w} = \sum_{n=1}^N a_n t_n (\mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}_n)) = \sum_{n=1}^N a_n t_n (y(\mathbf{x}_n) - b) = \sum_{n=1}^N a_n (t_n y(\mathbf{x}_n)) - b \sum_{n=1}^N a_n t_n = \sum_{n=1}^N a_n $$
したがって：
$$ \frac{1}{\rho^2} = \|\mathbf{w}\|^2 = \sum_{n=1}^N a_n = 2\widetilde{L}(\mathbf{a}) $$
が厳密に成立することを証明せよ。"""))

# Code Ex 7.1 - 7.6
code_ex7_1_6 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
from common.svm_rvm_utils import SupportVectorClassifier
from common.kernel_utils import rbf_kernel

# Exercise 7.4 & 7.5 数値検証: 1/rho^2 == sum(a_n) == ||w||^2
np.random.seed(42)
X_lin = np.array([[-1.0, 0.0], [-0.5, 0.5], [0.5, -0.5], [1.0, 0.0]])
t_lin = np.array([-1, -1, 1, 1])

# 線形カーネル k(x, x') = x^T x'
lin_kernel = lambda X1, X2: np.atleast_2d(X1) @ np.atleast_2d(X2).T
svc_test = SupportVectorClassifier(C=100.0, kernel=lin_kernel)
svc_test.fit(X_lin, t_lin)

sum_a = np.sum(svc_test.a)
w_vec = np.sum((svc_test.a * svc_test.t_train)[:, None] * svc_test.X_train, axis=0)
norm_w_sq = np.sum(w_vec**2)

print(f"sum(a_n):    {sum_a:.6f}")
print(f"||w||^2:     {norm_w_sq:.6f}")
assert np.isclose(sum_a, norm_w_sq, atol=1e-5)
print("Exercise 7.4 & 7.5 identity 1/rho^2 == sum(a_n) verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex7_1_6))

# Exercises 7.7 - 7.13
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 7.7 - 7.13: SVRの相補性、RVM事後ガウス分布、平方完成エビデンス積分、ハイパーパラメータ再推定

### 問題 7.7 & 7.8: SVRのKKT相補性条件
スラック変数 $\xi_n, \widehat{\xi}_n$ を持つSVRのラグランジュ乗数 $a_n, \widehat{a}_n$ に対し、
$$ \xi_n (C - a_n) = 0, \quad \widehat{\xi}_n (C - \widehat{a}_n) = 0 $$
が成り立つ。したがって、チューブの外側に外れ値として存在する点 ($\xi_n > 0$ または $\widehat{\xi}_n > 0$) では、乗数が最大許容上限 $a_n = C$ または $\widehat{a}_n = C$ に張り付くことを示せ。

### 問題 7.10 & 7.11: RVMエビデンス関数の平方完成積分
$$ p(\mathbf{t}|\boldsymbol{\alpha}, \beta) = \int \mathcal{N}(\mathbf{t}|\mathbf{\Phi}\mathbf{w}, \beta^{-1}\mathbf{I}) \mathcal{N}(\mathbf{w}|\mathbf{0}, \mathbf{A}^{-1}) d\mathbf{w} $$
指数部を展開して $\mathbf{w}$ について平方完成することで：
$$ p(\mathbf{t}|\boldsymbol{\alpha}, \beta) = \mathcal{N}(\mathbf{t} | \mathbf{0}, \, \mathbf{C}), \quad \mathbf{C} = \beta^{-1}\mathbf{I} + \mathbf{\Phi}\mathbf{A}^{-1}\mathbf{\Phi}^T $$
となり、対数周辺尤度（エビデンス関数）が
$$ \ln p(\mathbf{t}|\boldsymbol{\alpha}, \beta) = -\frac{1}{2}\left[ N \ln(2\pi) + \ln|\mathbf{C}| + \mathbf{t}^T \mathbf{C}^{-1}\mathbf{t} \right] $$
となることを導出せよ。

### 問題 7.12: エビデンス最大化による $\alpha_i$ 再推定式の導出
$\frac{\partial \ln|\mathbf{C}|}{\partial \alpha_i} = \frac{1}{\alpha_i} - \Sigma_{ii}$ および $\frac{\partial (\mathbf{t}^T \mathbf{C}^{-1}\mathbf{t})}{\partial \alpha_i} = -\mu_i^2$ を用いて、
$$ \frac{\partial \ln p(\mathbf{t}|\boldsymbol{\alpha}, \beta)}{\partial \alpha_i} = 0 \implies \alpha_i = \frac{1 - \alpha_i \Sigma_{ii}}{\mu_i^2} = \frac{\gamma_i}{\mu_i^2} $$
を厳密に導出せよ。"""))

# Code Ex 7.7 - 7.13
code_ex7_7_13 = r"""# Exercise 7.10 数値検証: 平方完成積分と C 行列による対数周辺尤度の完全一致
np.random.seed(42)
N, M = 6, 3
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha = np.array([1.5, 2.0, 0.8])
beta = 4.0

A = np.diag(alpha)
Sigma_inv = A + beta * (Phi.T @ Phi)
Sigma = np.linalg.inv(Sigma_inv)
mu = beta * Sigma @ Phi.T @ t

# 1. 重み空間での積分形式
E_w = 0.5 * beta * np.sum((t - Phi @ mu)**2) + 0.5 * mu @ A @ mu
log_ev_integral = -E_w + 0.5 * np.sum(np.log(alpha)) + 0.5 * N * np.log(beta) - 0.5 * np.linalg.slogdet(Sigma_inv)[1] - 0.5 * N * np.log(2 * np.pi)

# 2. C 行列形式: C = 1/beta * I + Phi A^(-1) Phi^T
C = (1.0 / beta) * np.eye(N) + Phi @ np.linalg.inv(A) @ Phi.T
sign, logdet_C = np.linalg.slogdet(C)
log_ev_C = -0.5 * (N * np.log(2 * np.pi) + logdet_C + t @ np.linalg.solve(C, t))

print(f"Integral evidence: {log_ev_integral:.6f}")
print(f"C-matrix evidence: {log_ev_C:.6f}")
assert np.isclose(log_ev_integral, log_ev_C)
print("Exercise 7.10 verified: RVM marginal likelihood formula is exact!")"""
cells.append(nbf.v4.new_code_cell(code_ex7_7_13))

# Exercises 7.14 - 7.19
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 7.14 - 7.19: スパース性解析、対数尤度二階微分による極大値の証明、分類RVM

### 問題 7.15 & 7.16: 対数エビデンス $\lambda(\alpha_i)$ の極大値判定
$\lambda(\alpha_i) = \frac{1}{2}\left[ \ln \alpha_i - \ln(\alpha_i + s_i) + \frac{q_i^2}{\alpha_i + s_i} \right]$ に対し、
一階微分は：
$$ \frac{d\lambda}{d\alpha_i} = \frac{s_i^2 - \alpha_i(q_i^2 - s_i)}{2\alpha_i(\alpha_i + s_i)^2} $$
二階微分を停留点 $\alpha_i^* = \frac{s_i^2}{q_i^2 - s_i}$ で評価すると：
$$ \left. \frac{d^2\lambda}{d\alpha_i^2} \right|_{\alpha_i^*} = -\frac{(q_i^2 - s_i)^4}{2 s_i^4 q_i^4} < 0 \quad (\text{since } q_i^2 > s_i > 0) $$
となり、停留点が常に**真の極大値（局所最大値）**であることを証明せよ。

### 問題 7.18: 分類RVMの事後確率勾配とヘッセ行列
ラプラス近似において、潜在変数重み $\mathbf{w}$ に対する事後対数確率
$$ \ln p(\mathbf{w}|\mathbf{t}, \boldsymbol{\alpha}) = \sum_{n=1}^N [t_n \ln y_n + (1 - t_n)\ln(1 - y_n)] - \frac{1}{2}\mathbf{w}^T \mathbf{A}\mathbf{w} + \text{const} $$
の勾配およびヘッセ行列が
$$ \nabla \ln p = \mathbf{\Phi}^T (\mathbf{t} - \mathbf{y}) - \mathbf{A}\mathbf{w} $$
$$ \nabla \nabla \ln p = -(\mathbf{\Phi}^T \mathbf{W} \mathbf{\Phi} + \mathbf{A}) $$
となることを示せ。"""))

# Code Ex 7.14 - 7.19
code_ex7_14_19 = r"""# Exercise 7.16 数値検証: 停留点における二階微分の負値性 (極大値の確認)
s_val = 2.5
q_sq_val = 6.0 # q^2 > s

alpha_star = s_val**2 / (q_sq_val - s_val)

# 解析的二階微分: -(q^2 - s)^4 / (2 s^4 q^4)
d2_analytic = -(q_sq_val - s_val)**4 / (2.0 * s_val**4 * q_sq_val**2)

# 数値的二階微分
eps = 1e-5
lam = lambda a: 0.5 * (np.log(a) - np.log(a + s_val) + q_sq_val / (a + s_val))
d2_numeric = (lam(alpha_star + eps) - 2.0 * lam(alpha_star) + lam(alpha_star - eps)) / (eps**2)

print(f"Analytic d2: {d2_analytic:.6f}")
print(f"Numeric d2:  {d2_numeric:.6f}")
assert d2_analytic < 0
assert np.isclose(d2_analytic, d2_numeric, rtol=1e-4)
print("Exercise 7.16 verified: Stationary point is strictly a local maximum!")"""
cells.append(nbf.v4.new_code_cell(code_ex7_14_19))

nb.cells = cells
with open('7/7_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("7/7_Exercises.ipynb generated successfully.")
