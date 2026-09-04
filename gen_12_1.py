import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 12.1 主成分分析 (Principal Component Analysis: PCA)

本ノートブックでは、非教師あり次元削減および特徴抽出の最重要古典手法である **主成分分析 (PCA)** を学びます。
射影後の最大分散定式化および直交補空間への最小二乗再構成誤差定式化の導出、手書き数字データに対する **平均画像と主成分固有ベクトル（固有数字: Eigen-digits, PRML Figure 12.3）**、成分数 $M$ の増加に伴う **画像の段階的再構成（PRML Figure 12.4）**、および Old Faithful 間欠泉データを用いた **主成分白色化（Whitening, PRML Figure 12.6）** を完全実装・再現します。"""))

# Theory
cells.append(nbf.v4.new_markdown_cell(r"""## 12.1.1 最大分散と最小誤差の定式化

中心化されたデータ行ベクトル $\mathbf{x}_n - \bar{\mathbf{x}}$ に対し、単位ベクトル $\mathbf{u}_1$（$\mathbf{u}_1^{\mathrm{T}}\mathbf{u}_1 = 1$）への射影分散を最大化します：
$$ \frac{1}{N} \sum_{n=1}^N (\mathbf{u}_1^{\mathrm{T}}\mathbf{x}_n - \mathbf{u}_1^{\mathrm{T}}\bar{\mathbf{x}})^2 = \mathbf{u}_1^{\mathrm{T}}\mathbf{S}\mathbf{u}_1 $$
ここで $\mathbf{S} = \frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n - \bar{\mathbf{x}})(\mathbf{x}_n - \bar{\mathbf{x}})^{\mathrm{T}}$ はサンプル共分散行列です。
ラグランジュ未定乗数法により $\mathbf{S}\mathbf{u}_1 = \lambda_1 \mathbf{u}_1$ が導かれ、最大固有値に対応する固有ベクトルが第1主成分となります。"""))

# Code: PRML Figure 12.3 & 12.4 Reproduction (Eigen-digits and Reconstruction)
code_fig12_3_4 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from common.plot_utils import save_plot, setup_style
from common.pca_ppca_utils import PCA
setup_style()

# 手書き数字データの読み込み (8x8 = 64次元)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# 数字 '3' のサブセットを選択 (PRML Figure 12.3 に準拠)
mask_3 = (y_digits == 3)
X_threes = X_digits[mask_3]

# PCA モデルの適合
pca = PCA(n_components=64).fit(X_threes)

# PRML Figure 12.3: 平均画像と最初の4つの固有ベクトル (Eigen-digits)
fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))

axes[0].imshow(pca.mean_.reshape(8, 8), cmap='gray')
axes[0].set_title(r'Mean Image $\bar{\mathbf{x}}$', fontsize=11)
axes[0].axis('off')

for i in range(4):
    axes[i+1].imshow(pca.components_[i].reshape(8, 8), cmap='coolwarm')
    axes[i+1].set_title(rf'Eigenvector $\mathbf{{u}}_{i+1}$' + '\n' + rf'($\lambda_{i+1}={pca.explained_variance_[i]:.1f}$)', fontsize=10)
    axes[i+1].axis('off')

plt.suptitle('Mean Digit and First Four PCA Eigenvectors (PRML Figure 12.3)', fontsize=13, y=1.05)
plt.tight_layout()
save_plot(fig, 'result', 'fig12_3_pca_eigen_digits.png')
plt.show()

# PRML Figure 12.4: 主成分数 M の増加に伴う画像の段階的再構成
sample_img = X_threes[0:1]
M_candidates = [1, 2, 5, 10, 20, 64]

fig, axes = plt.subplots(1, len(M_candidates) + 1, figsize=(16, 3.2))

# 元画像
axes[0].imshow(sample_img.reshape(8, 8), cmap='gray')
axes[0].set_title('Original Digit', fontsize=11)
axes[0].axis('off')

for idx, M in enumerate(M_candidates):
    # M 成分での射影と逆射影
    Z_m = (sample_img - pca.mean_) @ pca.components_[:M].T
    recon_img = Z_m @ pca.components_[:M] + pca.mean_
    
    axes[idx+1].imshow(recon_img.reshape(8, 8), cmap='gray')
    axes[idx+1].set_title(f'$M = {M}$ Components', fontsize=11)
    axes[idx+1].axis('off')

plt.suptitle('PCA Reconstruction of Handwritten Digit for Various M (PRML Figure 12.4)', fontsize=13, y=1.05)
plt.tight_layout()
save_plot(fig, 'result', 'fig12_4_pca_reconstruction_steps.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig12_3_4))

# 12.1.3 Whitening & Figure 12.6
cells.append(nbf.v4.new_markdown_cell(r"""## 12.1.3 主成分白色化 (Whitening, PRML Figure 12.6)

主成分分析により直交座標系へと回転したデータ $\mathbf{y}_n = \mathbf{U}^{\mathrm{T}}(\mathbf{x}_n - \bar{\mathbf{x}})$ に対し、各主成分の標準偏差 $\sqrt{\lambda_i}$ で割る変換（スケーリング）：
$$ z_{ni} = \frac{y_{ni}}{\sqrt{\lambda_i}} $$
を **白色化 (Whitening / Sphering)** と呼びます。
変換後のデータ共分散行列は単位行列 $\mathbf{I}$ となり、各方向の相関が完全に除去され分散が均一化されます。"""))

# Code: PRML Figure 12.6 Reproduction
code_fig12_6 = r"""import pandas as pd

# Old Faithful データセットの読み込み
df = pd.read_csv('../common/faithful.csv')
X_faithful = df[['duration', 'waiting']].values

# 1. 中心化
mean_f = np.mean(X_faithful, axis=0)
X_cent = X_faithful - mean_f

# 2. 主成分回転 (PCA Transform)
pca_f = PCA(n_components=2).fit(X_faithful)
Y_rot = pca_f.transform(X_faithful)

# 3. 白色化 (Whitening: 各軸を固有値の平方根で除算)
Z_white = Y_rot / np.sqrt(pca_f.explained_variance_)

# PRML Figure 12.6 のプロット
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

# (a) 元のデータ (中心化前)
axes[0].scatter(X_faithful[:, 0], X_faithful[:, 1], color='royalblue', s=25, alpha=0.8, edgecolors='k', lw=0.5)
axes[0].set_title('Original Old Faithful Data', fontsize=11)
axes[0].set_xlabel('Eruptions (mins)', fontsize=10); axes[0].set_ylabel('Waiting (mins)', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.3)

# (b) 主成分軸への回転 (無相関化)
axes[1].scatter(Y_rot[:, 0], Y_rot[:, 1], color='crimson', s=25, alpha=0.8, edgecolors='k', lw=0.5)
axes[1].set_title('PCA Projected (Decorrelated)', fontsize=11)
axes[1].set_xlabel('$y_1$', fontsize=10); axes[1].set_ylabel('$y_2$', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.3)

# (c) 白色化 (球状化: 共分散行列 = I)
axes[2].scatter(Z_white[:, 0], Z_white[:, 1], color='forestgreen', s=25, alpha=0.8, edgecolors='k', lw=0.5)
axes[2].set_title('Whitened Data (Covariance = I)', fontsize=11)
axes[2].set_xlabel('$z_1$', fontsize=10); axes[2].set_ylabel('$z_2$', fontsize=10)
axes[2].set_xlim(-3, 3); axes[2].set_ylim(-3, 3)
axes[2].set_aspect('equal', 'box')
axes[2].grid(True, linestyle='--', alpha=0.3)

plt.suptitle('PCA Data Pre-processing and Whitening (PRML Figure 12.6)', fontsize=13, y=1.02)
plt.tight_layout()
save_plot(fig, 'result', 'fig12_6_pca_whitening_faithful.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig12_6))

nb.cells = cells
with open('12/12.1_Principal_Component_Analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("12/12.1_Principal_Component_Analysis.ipynb generated successfully.")
