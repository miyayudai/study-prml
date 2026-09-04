import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 9.1 K-means クラスタリング (K-means Clustering)

本ノートブックでは、非教師あり学習の代表的アルゴリズムである **K-means クラスタリング** を学びます。
歪み尺度（コスト関数）$J$ の定式化、座標降下法による2段階の反復最適化、イエローストーン国立公園の **Old Faithful 間欠泉データ** に対するクラスタ形成の反復過程（**PRML Figure 9.1**）と歪み尺度の単調減少（**PRML Figure 9.2**）を完全再現します。
さらに、応用例としてカラー画像の **画像セグメンテーションと色量子化（PRML Figure 9.3）** を実装し、クラスタ数 $K$ による圧縮効果を可視化します。"""))

# 9.1 Theory & Figure 9.1, 9.2
cells.append(nbf.v4.new_markdown_cell(r"""## 9.1 K-means アルゴリズムと Old Faithful データのクラスタリング (PRML Figure 9.1, 9.2)

### 歪み尺度 (Distortion Measure)
$N$ 個のデータ点 $\{\mathbf{x}_n\}$ を $K$ 個のクラスタに分割するため、二値指示変数 $r_{nk} \in \{0, 1\}$ （$\sum_{k=1}^K r_{nk} = 1$）および各クラスタの代表ベクトル $\boldsymbol{\mu}_k$ を導入します。目的関数は各データ点と所属クラスタ中心との二乗距離の総和です（PRML 式 9.1）：
$$ J = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2 $$

### 2ステップ反復最適化
1. **割り当てステップ ($r_{nk}$ の更新)**: $\boldsymbol{\mu}_k$ を固定し、各点を最も近い中心に割り当てる：
   $$ r_{nk} = \begin{cases} 1 & \text{if } k = \arg\min_j \|\mathbf{x}_n - \boldsymbol{\mu}_j\|^2 \\ 0 & \text{otherwise} \end{cases} $$
2. **中心更新ステップ ($\boldsymbol{\mu}_k$ の更新)**: $r_{nk}$ を固定し、$J$ を $\boldsymbol{\mu}_k$ で微分して $0$ とおくことで、クラスタ内の平均値に更新：
   $$ \boldsymbol{\mu}_k = \frac{\sum_{n=1}^N r_{nk} \mathbf{x}_n}{\sum_{n=1}^N r_{nk}} $$
各ステップで $J$ は厳密に単調減少し、有限回の反復で必ず局所最小解に収束します。"""))

# Code: PRML Figure 9.1 & 9.2
code_fig9_1_2 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.plot_utils import save_plot, setup_style
from common.mixture_em_utils import KMeans
setup_style()

# Old Faithful データの読み込みと標準化 (PRML 9.1)
df = pd.read_csv('../common/faithful.csv')
X_raw = df[['duration', 'waiting']].values
# 平均0, 標準偏差1にスケーリング
X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)

# PRML Figure 9.1 の初期値設定 (手動初期化による反復過程の忠実な再現)
K = 2
mu_init = np.array([[-1.5, 1.0], [1.5, -1.0]])

# ステップごとの状態を記録
history_centers = [mu_init.copy()]
history_labels = []
cost_history = []

centers = mu_init.copy()
for step in range(5):
    # 割り当て
    diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    labels = np.argmin(dist_sq, axis=1)
    cost = np.sum(np.min(dist_sq, axis=1))
    
    history_labels.append(labels.copy())
    cost_history.append(cost)
    
    # 中心更新
    new_centers = np.array([np.mean(X[labels == k], axis=0) for k in range(K)])
    centers = new_centers.copy()
    history_centers.append(centers.copy())

# PRML Figure 9.1 のプロット: 4つの反復ステージ
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

cluster_colors = ['crimson', 'royalblue']
step_indices = [0, 1, 2, 4]

for idx, s in enumerate(step_indices):
    ax = axes[idx]
    lbl = history_labels[s]
    mu_curr = history_centers[s]
    
    # データ点のプロット
    for k in range(K):
        ax.scatter(X[lbl == k, 0], X[lbl == k, 1], c=cluster_colors[k], alpha=0.6, s=35, edgecolors='none')
        # クラスタ中心 (クロスマーカー)
        ax.scatter(mu_curr[k, 0], mu_curr[k, 1], c=cluster_colors[k], marker='x', s=160, lw=3.5, zorder=5)
        
    ax.set_title(f'Iteration {s+1} (Distortion $J={cost_history[s]:.1f}$)', fontsize=12)
    ax.set_xlabel('Eruption Duration (standardized)', fontsize=10)
    ax.set_ylabel('Waiting Time (standardized)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

plt.suptitle('Illustration of K-means Algorithm on Old Faithful Data (PRML Figure 9.1)', fontsize=14, y=1.02)
plt.tight_layout()
save_plot(fig, 'result', 'fig9_1_kmeans_faithful_steps.png')
plt.show()

# PRML Figure 9.2 のプロット: 歪み尺度 J の単調減少曲線
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, len(cost_history) + 1), cost_history, 'ro-', lw=2, markersize=8)
ax.set_xlabel('Iteration Step', fontsize=12)
ax.set_ylabel('Distortion Measure $J$', fontsize=12)
ax.set_title('Monotonic Decrease of Distortion Measure $J$ (PRML Figure 9.2)', fontsize=13)
ax.set_xticks(range(1, len(cost_history) + 1))
ax.grid(True, linestyle='--', alpha=0.4)

save_plot(fig, 'result', 'fig9_2_distortion_measure_decrease.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig9_1_2))

# 9.1.1 Image Segmentation & Figure 9.3
cells.append(nbf.v4.new_markdown_cell(r"""## 9.1.1 画像セグメンテーションと色量子化 (PRML Figure 9.3)

各ピクセルのRGBカラー値（3次元ベクトル）をデータ点と見なし、K-means クラスタリングを適用します。
各ピクセルを所属クラスタの中心色 $\boldsymbol{\mu}_k$ に置き換えることで、画像をわずか $K$ 色に色量子化（圧縮）します。
$K=2, 3, 10$ のときの画質と圧縮表現の推移を可視化します。"""))

# Code: PRML Figure 9.3 Reproduction
code_fig9_3 = r"""# PRML Figure 9.3: 画像セグメンテーションと色量子化
# 豊かな色彩を持つテストパターンの生成 (グラデーションと幾何学図形)
H, W = 100, 100
np.random.seed(42)
img_rgb = np.zeros((H, W, 3))

# 背景: 空の青から夕焼けのオレンジへのグラデーション
for y in range(H):
    t = y / H
    img_rgb[y, :, 0] = 0.2 + 0.7 * t      # Red
    img_rgb[y, :, 1] = 0.4 + 0.2 * (1-t)  # Green
    img_rgb[y, :, 2] = 0.8 * (1 - t)      # Blue

# 前景: 緑の丘と黄色の太陽
Y, X_grid = np.ogrid[:H, :W]
sun_mask = (X_grid - 70)**2 + (Y - 30)**2 < 18**2
img_rgb[sun_mask] = [0.95, 0.85, 0.1] # 太陽 (Yellow)

hill_mask = Y > (70 - 15 * np.sin(X_grid / 15.0))
img_rgb[hill_mask] = [0.15, 0.65, 0.2] # 丘 (Green)

# ノイズを少し加えて実画像に近づける
img_rgb = np.clip(img_rgb + np.random.normal(0, 0.02, img_rgb.shape), 0.0, 1.0)

# ピクセル配列の展開 (H*W, 3)
pixels = img_rgb.reshape(-1, 3)

# K = 2, 3, 10 によるセグメンテーション
K_values = [2, 3, 10]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

axes[0].imshow(img_rgb)
axes[0].set_title('Original Synthetic Image\n(Continuous Colors)', fontsize=11)
axes[0].axis('off')

for i, K_val in enumerate(K_values):
    km = KMeans(n_clusters=K_val, max_iter=30, random_state=42)
    km.fit(pixels)
    # 各ピクセルをクラスタ中心の色に置換
    quantized_pixels = km.cluster_centers_[km.labels_]
    quantized_img = quantized_pixels.reshape(H, W, 3)
    
    axes[i+1].imshow(quantized_img)
    axes[i+1].set_title(f'$K = {K_val}$ Clusters\n({K_val} Colors Segmented)', fontsize=11)
    axes[i+1].axis('off')

plt.suptitle('Image Segmentation and Color Quantization via K-means (PRML Figure 9.3)', fontsize=13, y=1.02)
plt.tight_layout()
save_plot(fig, 'result', 'fig9_3_image_segmentation_kmeans.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig9_3))

nb.cells = cells
with open('9/9.1_K_means_Clustering.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("9/9.1_K_means_Clustering.ipynb generated successfully.")
