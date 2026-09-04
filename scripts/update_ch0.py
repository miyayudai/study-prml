import nbformat as nbf
import json

with open('0/0_Foundations_of_Probability.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Markdown for 0.9 and 0.10
md_0_9_10 = r"""## 0.9 線形ガウスモデルの周辺化と事後分布の一般公式 (Linear Gaussian Relations)

PRML 第2章 (2.3.3節) および第3章（ベイズ線形回帰）の予測分布やエビデンス関数の導出で最も頻繁に用いられる根本的な定理です。

### 設定
事前分布と条件付き分布が以下のように与えられているとします：
$$ p(\mathbf{x}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \mathbf{\Lambda}^{-1}) $$
$$ p(\mathbf{y} | \mathbf{x}) = \mathcal{N}(\mathbf{y} | \mathbf{A}\mathbf{x} + \mathbf{b}, \mathbf{L}^{-1}) $$

### 1. 周辺分布 $p(\mathbf{y})$
$\mathbf{y}$ の期待値と共分散を行列計算により求めることで、直接積分を行わずに周辺ガウス分布が得られます：
$$ \mathbb{E}[\mathbf{y}] = \mathbb{E}[\mathbf{A}\mathbf{x} + \mathbf{b}] = \mathbf{A}\boldsymbol{\mu} + \mathbf{b} $$
$$ \mathrm{cov}[\mathbf{y}] = \mathrm{cov}[\mathbf{A}\mathbf{x}] + \mathrm{cov}[\mathbf{y}|\mathbf{x}] = \mathbf{A} \mathbf{\Lambda}^{-1} \mathbf{A}^T + \mathbf{L}^{-1} $$
したがって、
$$ p(\mathbf{y}) = \mathcal{N}\left(\mathbf{y} \middle| \mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \, \mathbf{L}^{-1} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T \right) $$

### 2. 事後条件付き分布 $p(\mathbf{x} | \mathbf{y})$
結合分布 $p(\mathbf{x}, \mathbf{y})$ の指数部を展開して平方完成することで、事後分布もガウス分布となります：
$$ p(\mathbf{x} | \mathbf{y}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\Sigma}(\mathbf{A}^T \mathbf{L}(\mathbf{y} - \mathbf{b}) + \mathbf{\Lambda}\boldsymbol{\mu}), \boldsymbol{\Sigma}) $$
ここで
$$ \boldsymbol{\Sigma} = (\mathbf{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A})^{-1} $$
第3章のベイズ線形回帰では、$\mathbf{x} \to \mathbf{w}, \boldsymbol{\mu} \to \mathbf{m}_0, \mathbf{\Lambda} \to \mathbf{S}_0^{-1}, \mathbf{y} \to \mathbf{t}, \mathbf{A} \to \mathbf{\Phi}, \mathbf{b} \to \mathbf{0}, \mathbf{L} \to \beta \mathbf{I}$ と対応付けることで、直ちに事後パラメータが得られます。

---

## 0.10 行列式とトレースの微分公式 (Determinant Derivatives and Matrix Calculus)

第3章のエビデンス近似（3.5節）や第4章以降のラプラス近似・フィッシャー情報量の計算では、行列の対数行列式の微分が決定的な役割を果たします。

### 基本恒等式
正則な対称正定値行列 $\mathbf{A}(\alpha)$ に対し、
$$ \frac{\partial}{\partial \alpha} \ln |\mathbf{A}| = \mathrm{Tr}\left( \mathbf{A}^{-1} \frac{\partial \mathbf{A}}{\partial \alpha} \right) $$

### 証明のステップ
行列 $\mathbf{A}$ の固有値を $\mu_i$ とすると、行列式は固有値の積 $|\mathbf{A}| = \prod_{i=1}^M \mu_i$ です。対数をとると和になります：
$$ \ln |\mathbf{A}| = \sum_{i=1}^M \ln \mu_i $$
$\alpha$ で微分すると：
$$ \frac{\partial}{\partial \alpha} \ln |\mathbf{A}| = \sum_{i=1}^M \frac{1}{\mu_i} \frac{\partial \mu_i}{\partial \alpha} $$
固有ベクトル行列 $\mathbf{U}$ による対角化 $\mathbf{A} = \mathbf{U} \mathrm{diag}(\mu_i) \mathbf{U}^T$ を用いると、トレースの循環不変性より
$$ \sum_{i=1}^M \frac{1}{\mu_i} \frac{\partial \mu_i}{\partial \alpha} = \mathrm{Tr}\left( \mathbf{A}^{-1} \frac{\partial \mathbf{A}}{\partial \alpha} \right) $$
が成立します。特に $\mathbf{A} = \alpha \mathbf{I} + \mathbf{B}$ の場合、$\frac{\partial \mathbf{A}}{\partial \alpha} = \mathbf{I}$ となり、
$$ \frac{\partial}{\partial \alpha} \ln |\alpha \mathbf{I} + \mathbf{B}| = \mathrm{Tr}((\alpha \mathbf{I} + \mathbf{B})^{-1}) = \sum_{i=1}^M \frac{1}{\alpha + \lambda_i} $$
と極めて簡潔に計算できます。"""

code_0_9_10 = r"""# 0.9 & 0.10 数値検証
import numpy as np

# 1. 線形ガウス関係式の検証
np.random.seed(42)
dx, dy = 3, 2
mu_x = np.array([1.0, -0.5, 0.2])
Lambda_x = np.diag([2.0, 1.5, 3.0])
cov_x = np.linalg.inv(Lambda_x)

A_mat = np.array([[0.5, -1.0, 2.0], [1.5, 0.2, -0.8]])
b_vec = np.array([0.1, -0.3])
L_mat = np.diag([4.0, 5.0])
cov_y_given_x = np.linalg.inv(L_mat)

# 周辺分布 p(y) の平均と共分散
mean_y = A_mat @ mu_x + b_vec
cov_y = cov_y_given_x + A_mat @ cov_x @ A_mat.T

# モンテカルロサンプリングによる一致検証
x_samples = np.random.multivariate_normal(mu_x, cov_x, size=50000)
noise_y = np.random.multivariate_normal(np.zeros(dy), cov_y_given_x, size=50000)
y_samples = (A_mat @ x_samples.T).T + b_vec + noise_y

mc_mean_y = np.mean(y_samples, axis=0)
mc_cov_y = np.cov(y_samples, rowvar=False)

print("Theoretical mean_y:", np.round(mean_y, 4))
print("MC mean_y:         ", np.round(mc_mean_y, 4))
assert np.allclose(mean_y, mc_mean_y, atol=0.03), "Mean mismatch"
assert np.allclose(cov_y, mc_cov_y, atol=0.05), "Covariance mismatch"

# 2. 行列式微分の検証
alpha_val = 1.5
B_mat = np.array([[2.0, 0.5], [0.5, 1.0]])
A_alpha = alpha_val * np.eye(2) + B_mat
eps = 1e-6
d_ln_det_num = (np.linalg.slogdet(A_alpha + eps * np.eye(2))[1] - np.linalg.slogdet(A_alpha - eps * np.eye(2))[1]) / (2 * eps)
d_ln_det_theory = np.trace(np.linalg.inv(A_alpha))

print(f"Numerical d/d_alpha ln|A|:   {d_ln_det_num:.6f}")
print(f"Theoretical Tr(A^-1):        {d_ln_det_theory:.6f}")
assert np.isclose(d_ln_det_num, d_ln_det_theory, rtol=1e-5)
print("Sections 0.9 and 0.10 verified successfully!")"""

nb.cells.append(nbf.v4.new_markdown_cell(md_0_9_10))
nb.cells.append(nbf.v4.new_code_cell(code_0_9_10))

with open('0/0_Foundations_of_Probability.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("0/0_Foundations_of_Probability.ipynb updated with sections 0.9 and 0.10.")
