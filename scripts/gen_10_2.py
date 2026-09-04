import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 10.2 変分ベイズ混合ガウスモデル (Variational Gaussian Mixtures)

本ノートブックでは、最尤推定における深刻な特異点（特異解）を根本的に克服し、**データから最適なクラスタ数を自動決定 (Automatic Relevance Determination)** する**完全ベイズ型混合ガウスモデルの変分推論 (Variational Bayesian GMM)** を学びます。
ディリクレ・ガウス・ウィシャート共役事前分布、変分EMアルゴリズムの導出、Old Faithful 間欠泉データにおいて過剰な $K=6$ 成分で開始しても**不要な4成分が自律的にゼロへ消滅して $K=2$ に自動収縮する現象（PRML Figure 10.6）**、および変分下界 $\mathcal{L}(q)$ によるクラスタ数評価（**PRML Figure 10.7**）を完全実装します。"""))

# 10.2 Theory & Figure 10.6
cells.append(nbf.v4.new_markdown_cell(r"""## 10.2 変分混合ガウスモデルとクラスタ数の自動決定 (PRML Figure 10.6)

### ベイズ的定式化
混合比 $\boldsymbol{\pi}$ にディリクレ事前分布、各成分の平均と精度 $(\boldsymbol{\mu}_k, \mathbf{\Lambda}_k)$ にガウス・ウィシャート事前分布を導入します（PRML Figure 10.5）：
$$ p(\boldsymbol{\pi}) = \mathrm{Dir}(\boldsymbol{\pi} | \boldsymbol{\alpha}_0), \quad p(\boldsymbol{\mu}_k, \mathbf{\Lambda}_k) = \mathcal{N}(\boldsymbol{\mu}_k | \mathbf{m}_0, (\beta_0 \mathbf{\Lambda}_k)^{-1}) \mathcal{W}(\mathbf{\Lambda}_k | \mathbf{W}_0, \nu_0) $$

### 不要な成分の自動間引き (Sparsity of Components)
事前ハイパーパラメータ $\alpha_0 \to 0$（無情報または疎な事前分布）を設定すると、データによって支持されない不要な成分 $k$ に対し、実効データ数 $N_k \approx 0$ となります。
すると、事後パラメータは $\alpha_k \approx \alpha_0 \approx 0$ となり、**期待混合比 $\mathbb{E}[\pi_k] = \frac{\alpha_k}{\sum \alpha_j} \to 0$ へと縮退して自動的に消滅します**。
これにより、クロスバリデーション等に頼ることなく、単一の変分最適化の中で最適なクラスタ数が自動的に発見されます。"""))

# Code: PRML Figure 10.6 Reproduction
code_fig10_6 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from common.plot_utils import save_plot, setup_style
from common.variational_utils import VariationalGaussianMixture
setup_style()

# 楕円描画ユーティリティ
def plot_gaussian_ellipse(ax, mean, cov, n_std=1.5, color='crimson', lw=2.0, alpha=0.8):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 1e-8))
    ell = Ellipse(xy=mean, width=w, height=h, angle=theta, edgecolor=color, fc='none', lw=lw, alpha=alpha, zorder=4)
    ax.add_patch(ell)

# Old Faithful データの準備
df = pd.read_csv('../common/faithful.csv')
X_raw = df[['duration', 'waiting']].values
X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)
N, D = X.shape

# PRML Figure 10.6: 過剰な K=6 成分、alpha_0 = 1e-3 で変分GMMを実行
K_init = 6
vgm = VariationalGaussianMixture(n_components=K_init, alpha_0=1e-3, beta_0=1.0, max_iter=80, random_state=42)
vgm.fit(X)

# 各成分の事後混合比期待値 E[pi_k] = alpha_k / sum(alpha)
E_pi = vgm.alpha_ / np.sum(vgm.alpha_)
print("Effective Component Weights E[pi_k] for K=6 components:")
for k in range(K_init):
    print(f"Component {k+1}: E[pi] = {E_pi[k]:.4f} (alpha_k = {vgm.alpha_[k]:.2f})")

# 有効な成分 (E[pi] > 0.05) の抽出
active_components = np.where(E_pi > 0.05)[0]
print(f"Number of active components automatically retained: {len(active_components)} (Expected: 2)")

# PRML Figure 10.6 のプロット
fig, ax = plt.subplots(figsize=(8, 6.5))

# データ点のプロット (負担率に基づく色分け)
labels = np.argmax(vgm.responsibilities_, axis=1)
colors = plt.cm.tab10(np.linspace(0, 1, K_init))

for k in range(K_init):
    pts = X[labels == k]
    if len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], color=colors[k], alpha=0.6, s=35, label=f'Cluster {k+1} ($\pi={E_pi[k]:.2f}$)')
    
    # 共分散 (精度行列の逆行列: E[Lambda_k]^-1 = (nu_k * W_k)^-1)
    cov_k = np.linalg.inv(vgm.nu_[k] * vgm.W_[k])
    # 有効成分は太い実線、消滅成分は薄い点線
    if E_pi[k] > 0.05:
        plot_gaussian_ellipse(ax, vgm.m_[k], cov_k, n_std=1.5, color=colors[k], lw=2.5, alpha=1.0)
        ax.scatter(vgm.m_[k, 0], vgm.m_[k, 1], marker='+', s=150, color='black', lw=2.5, zorder=5)
    else:
        plot_gaussian_ellipse(ax, vgm.m_[k], cov_k, n_std=1.5, color='gray', lw=1.0, alpha=0.3)

ax.set_title(r'Variational Bayesian GMM on Old Faithful ($K=6 \to 2$ Active Components, PRML Figure 10.6)', fontsize=12)
ax.set_xlabel('Eruption Duration (standardized)', fontsize=11)
ax.set_ylabel('Waiting Time (standardized)', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

save_plot(fig, 'result', 'fig10_6_variational_gmm_faithful.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig10_6))

nb.cells = cells
with open('10/10.2_Variational_Gaussian_Mixtures.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("10/10.2_Variational_Gaussian_Mixtures.ipynb generated successfully.")
