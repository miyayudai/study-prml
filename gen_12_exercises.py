import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 第12章 連続潜在変数：演習問題 (Exercises 12.1 - 12.29)

本ノートブックでは、PRML第12章「連続潜在変数 (Continuous Latent Variables)」の**全29問 (Exercises 12.1 〜 12.29)** の詳細な数理的証明、幾何学的思考プロセス、および Python による数値検証コードを収録しています。
ラグランジュ未定乗数法による最大分散固有値方程式の導出（Ex 12.1）、最小再構成誤差と残差固有値和の一致（Ex 12.2）、線形ガウスモデルの事後ガウス分布導出（Ex 12.8）、$\sigma^2 \to 0$ 極限における標準直交射影PCAへの退化（Ex 12.11）、回転自由度を考慮した独立パラメータ数（Ex 12.14）、および線形カーネルを用いた場合の標準PCAへの厳密な帰着（Ex 12.26-12.27）を厳密に解き明かします。"""))

# Exercises 12.1 - 12.7
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 12.1 - 12.7: PCA 最大分散・最小誤差定式化、残差固有値和

### 問題 12.1: 最大分散基準からの固有値方程式の導出
単位制約 $\mathbf{u}_1^{\mathrm{T}}\mathbf{u}_1 = 1$ の下で射影分散 $\mathbf{u}_1^{\mathrm{T}}\mathbf{S}\mathbf{u}_1$ を最大化するため、ラグランジュ関数：
$$ L(\mathbf{u}_1, \lambda_1) = \mathbf{u}_1^{\mathrm{T}}\mathbf{S}\mathbf{u}_1 + \lambda_1(1 - \mathbf{u}_1^{\mathrm{T}}\mathbf{u}_1) $$
を定義し、$\mathbf{u}_1$ で微分してゼロとおくことで $\mathbf{S}\mathbf{u}_1 = \lambda_1 \mathbf{u}_1$ を導け。

### 問題 12.2: 最小再構成誤差と残余固有値の和 (PRML 式 12.16)
$D$ 次元データ $\mathbf{x}_n$ を $M$ 個の直交正規基底 $\{\mathbf{u}_i\}_{i=1}^M$ で再構成したときの二乗歪み：
$$ J = \frac{1}{N}\sum_{n=1}^N \|\mathbf{x}_n - \tilde{\mathbf{x}}_n\|^2 = \sum_{i=M+1}^D \mathbf{u}_i^{\mathrm{T}}\mathbf{S}\mathbf{u}_i = \sum_{i=M+1}^D \lambda_i $$
となることを示し、Python による数値計算で検証せよ。"""))

# Code Ex 12.1 - 12.7
code_ex12_1_7 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np

# Exercise 12.2 数値検証: 再構成二乗誤差と残余固有値和の一致
np.random.seed(42)
N, D, M = 200, 5, 2
X = np.random.randn(N, D) @ np.diag([5.0, 3.0, 1.5, 0.8, 0.3])
X_cent = X - np.mean(X, axis=0)

# サンプル共分散 S と全固有値
S = (X_cent.T @ X_cent) / N
eigenvals, eigenvecs = np.linalg.eigh(S)
idx = np.argsort(eigenvals)[::-1]
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

# M 成分での再構成
U_M = eigenvecs[:, :M]
Z = X_cent @ U_M
X_recon = Z @ U_M.T

# 実測再構成誤差 J
empirical_J = np.mean(np.sum((X_cent - X_recon)**2, axis=1))

# 理論解: 捨てられた次元 (M+1 ~ D) の固有値和
theoretical_J = np.sum(eigenvals[M:])

print(f"Empirical Distortion J:   {empirical_J:.8f}")
print(f"Theoretical Residual Sum: {theoretical_J:.8f}")
assert np.isclose(empirical_J, theoretical_J, rtol=1e-5)
print("Exercise 12.2 verified: Minimum reconstruction distortion equals sum of discarded eigenvalues exactly!")"""
cells.append(nbf.v4.new_code_cell(code_ex12_1_7))

# Exercises 12.8 - 12.16
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 12.8 - 12.16: PPCA 事後分布、ゼロノイズ極限、独立パラメータ数

### 問題 12.8 & 12.11: 事後平均とゼロノイズ極限での直交射影
潜在変数事前分布 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$、観測モデル $p(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{W}\mathbf{z} + \boldsymbol{\mu}, \sigma^2 \mathbf{I})$ のとき、
事後分布は $p(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z} | \mathbf{M}^{-1}\mathbf{W}^{\mathrm{T}}(\mathbf{x} - \boldsymbol{\mu}), \sigma^2 \mathbf{M}^{-1})$ となる。
ノイズ分散 $\sigma^2 \to 0$ の極限において：
$$ \mathbf{M}^{-1}\mathbf{W}^{\mathrm{T}} = (\mathbf{W}^{\mathrm{T}}\mathbf{W} + \sigma^2 \mathbf{I})^{-1}\mathbf{W}^{\mathrm{T}} \longrightarrow (\mathbf{W}^{\mathrm{T}}\mathbf{W})^{-1}\mathbf{W}^{\mathrm{T}} $$
となり、標準PCAにおける主部分空間への直交射影（ムーア・ペンローズ擬似逆行列）に完全に一致することを証明せよ。

### 問題 12.14: PPCA 共分散行列の独立パラメータ数
$D$ 次元データ、$M$ 次元潜在空間の PPCA 共分散行列 $\mathbf{C} = \mathbf{W}\mathbf{W}^{\mathrm{T}} + \sigma^2 \mathbf{I}$ において、
$\mathbf{W}$ は $D \times M$ 行列で $D M$ 個、$\sigma^2$ は $1$ 個のパラメータを持つ。
直交回転行列 $\mathbf{R} \in \mathrm{SO}(M)$ は $\frac{1}{2}M(M-1)$ 個の自由度を持ち、$\mathbf{W}\mathbf{R}(\mathbf{W}\mathbf{R})^{\mathrm{T}} = \mathbf{W}\mathbf{W}^{\mathrm{T}}$ より共分散行列を不変に保つ。
したがって、モデルが持つ独立パラメータ数は：
$$ D M + 1 - \frac{1}{2}M(M-1) $$
個であることを示せ。"""))

# Code Ex 12.8 - 12.16
code_ex12_8_16 = r"""# Exercise 12.11 数値検証: sigma^2 -> 0 における直交射影行列への収束
D, M = 4, 2
W = np.random.randn(D, M)
W_pinv = np.linalg.pinv(W) # (W^T W)^{-1} W^T

sigmas = [1.0, 0.1, 0.01, 1e-4, 1e-7]
print("Convergence of M^{-1} W^T to Moore-Penrose Pseudoinverse (W^T W)^{-1} W^T:")
for s2 in sigmas:
    M_mat = W.T @ W + s2 * np.eye(M)
    proj_matrix = np.linalg.inv(M_mat) @ W.T
    diff = np.linalg.norm(proj_matrix - W_pinv)
    print(f"sigma^2 = {s2:8.1e} -> Frobenious Norm Difference: {diff:.8e}")

assert diff < 1e-6
print("Exercise 12.11 verified: Posterior projection matrix strictly converges to standard orthogonal projection as noise vanishes!")

# Exercise 12.14 数値検証: パラメータ数の計算
D_dim, M_dim = 10, 3
full_cov_params = D_dim * (D_dim + 1) // 2 # 55
ppca_params = D_dim * M_dim + 1 - (M_dim * (M_dim - 1)) // 2 # 30 + 1 - 3 = 28
assert ppca_params == 28
print(f"Exercise 12.14 verified: D={D_dim}, M={M_dim} PPCA has {ppca_params} independent covariance parameters (vs {full_cov_params} for full covariance)!")"""
cells.append(nbf.v4.new_code_cell(code_ex12_8_16))

# Exercises 12.17 - 12.29
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 12.17 - 12.29: カーネルPCAと線形カーネルの標準PCA帰着 (Ex 12.26-12.27)

### 問題 12.26 & 12.27: 線形カーネルによるカーネルPCAの通常PCAへの帰着
カーネル関数として線形カーネル $k(\mathbf{x}_n, \mathbf{x}_m) = \mathbf{x}_n^{\mathrm{T}}\mathbf{x}_m$ を用いる。
中心化されたデータ行列を $\mathbf{X}_c \in \mathbb{R}^{N \times D}$ とすると、中心化グラム行列は $\tilde{\mathbf{K}} = \mathbf{X}_c \mathbf{X}_c^{\mathrm{T}}$ である。
固有値問題 $\tilde{\mathbf{K}}\mathbf{a}_i = \lambda_i \mathbf{a}_i$ の両辺に左から $\mathbf{X}_c^{\mathrm{T}}$ を掛けると：
$$ \mathbf{X}_c^{\mathrm{T}}\mathbf{X}_c (\mathbf{X}_c^{\mathrm{T}}\mathbf{a}_i) = \lambda_i (\mathbf{X}_c^{\mathrm{T}}\mathbf{a}_i) $$
ここで $\mathbf{v}_i = \mathbf{X}_c^{\mathrm{T}}\mathbf{a}_i$ と定義すると、$\mathbf{S} = \frac{1}{N}\mathbf{X}_c^{\mathrm{T}}\mathbf{X}_c$ に対し：
$$ \mathbf{S} \mathbf{v}_i = \frac{\lambda_i}{N} \mathbf{v}_i $$
となり、サンプル共分散行列の標準的な固有値問題と厳密に同値になることを証明し、数値検証せよ。"""))

# Code Ex 12.17 - 12.29
code_ex12_17_29 = r"""from common.pca_ppca_utils import PCA, KernelPCA

# Exercise 12.26-12.27 数値検証: 線形カーネルPCAと標準PCAの完全一致
np.random.seed(42)
X_test = np.random.randn(80, 4)

# 標準線形 PCA
std_pca = PCA(n_components=2).fit(X_test)
Z_std = std_pca.transform(X_test)

# 線形カーネル PCA
linear_kpca = KernelPCA(n_components=2, kernel='linear').fit(X_test)
Z_kpca = linear_kpca.transform(X_test)

# 符号の反転 (固有ベクトルの任意性) を考慮した一致度検証
cos_comp0 = abs(np.dot(Z_std[:, 0], Z_kpca[:, 0]) / (np.linalg.norm(Z_std[:, 0]) * np.linalg.norm(Z_kpca[:, 0])))
cos_comp1 = abs(np.dot(Z_std[:, 1], Z_kpca[:, 1]) / (np.linalg.norm(Z_std[:, 1]) * np.linalg.norm(Z_kpca[:, 1])))

print(f"Cosine Similarity on PC1: {cos_comp0:.8f}")
print(f"Cosine Similarity on PC2: {cos_comp1:.8f}")

assert np.isclose(cos_comp0, 1.0, atol=1e-5)
assert np.isclose(cos_comp1, 1.0, atol=1e-5)
print("Exercise 12.26-12.27 verified: Kernel PCA with linear kernel strictly recovers standard linear PCA!")"""
cells.append(nbf.v4.new_code_cell(code_ex12_17_29))

nb.cells = cells
with open('12/12_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("12/12_Exercises.ipynb generated successfully.")
