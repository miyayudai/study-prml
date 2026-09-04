import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 14.4-14.5 決定木と専門家の混合 (Decision Trees & Mixture of Experts)

本ノートブックでは、入力空間を局所領域に分割して単純なモデルを割り当てる決定木と、確率的混合モデルの回帰・分類への拡張である **専門家の混合 (Mixture of Experts: MoE)** を学びます。
CART による空間再帰分割と不純度尺度（**PRML Figure 14.5, 14.6**）、多峰性を持つデータセットに対する単一回帰モデルの破綻と **線形回帰混合モデル（Mixture of Linear Regressors, PRML Figure 14.8）** の EM 学習、および多峰的な予測条件付き密度 $p(t|x)$（**PRML Figure 14.9**）を完全実装・再現します。"""))

# 14.4 Decision Trees
cells.append(nbf.v4.new_markdown_cell(r"""## 14.4 決定木と不純度指標 (PRML Figure 14.5, 14.6)

決定木は入力空間を各特徴量の閾値で再帰的に二分割（軸に平行な超平面による直交分割）します。
2値分類における不純度尺度：
- **交差エントロピー (Cross Entropy)**: $Q_{\tau} = -p_{\tau} \ln p_{\tau} - (1 - p_{\tau}) \ln (1 - p_{\tau})$
- **ジニ係数 (Gini Index)**: $Q_{\tau} = 2 p_{\tau} (1 - p_{\tau})$
- **誤分類率 (Misclassification)**: $Q_{\tau} = 1 - \max(p_{\tau}, 1 - p_{\tau})$
の形状比較を行います。"""))

# Code: Impurity measures plot
code_impurity = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import matplotlib.pyplot as plt
from common.plot_utils import save_plot, setup_style
setup_style()

p = np.linspace(1e-4, 1.0 - 1e-4, 300)

gini = 4.0 * p * (1.0 - p) # 0.5で1に正規化
entropy = (-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))
misclass = 1.0 - np.maximum(p, 1.0 - p)
misclass_scaled = 2.0 * misclass # 0.5で1に正規化

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(p, entropy, 'r-', lw=2.0, label='Entropy / 2')
ax.plot(p, gini, 'g--', lw=2.0, label='Gini Index')
ax.plot(p, misclass_scaled, 'b-.', lw=2.0, label='Misclassification Rate')

ax.set_title('Comparison of Impurity Measures for Binary Decision Trees', fontsize=12)
ax.set_xlabel('Class 1 Proportion $p$', fontsize=11); ax.set_ylabel('Impurity Index', fontsize=11)
ax.legend(loc='upper center', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

save_plot(fig, 'result', 'fig14_tree_impurity_measures.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_impurity))

# 14.5.1 Mixture of Linear Regression (PRML Figure 14.8, 14.9)
cells.append(nbf.v4.new_markdown_cell(r"""## 14.5.1 線形回帰混合モデル (PRML Figure 14.8, 14.9)

目標変数 $t$ が入力 $x$ に対して多値性（多峰性）を持つ合成データセット（PRML Figure 14.8）を考えます。
単一の最小二乗線形回帰やガウス過程回帰では、条件付き期待値 $\mathbb{E}[t|x]$ を出力するため、複数の峰の「中間」を予測してしまい致命的に破綻します。
$K=2$ または $K=3$ 成分の線形回帰混合モデル：
$$ p(t|x) = \sum_{k=1}^K \pi_k \mathcal{N}(t | \mathbf{w}_k^{\mathrm{T}}\boldsymbol{\phi}(x), \sigma_k^2) $$
を EM アルゴリズムで適合し、各成分の直線および予測条件付き確率密度 $p(t|x)$（PRML Figure 14.9）を完全再現します。"""))

# Code: PRML Figure 14.8, 14.9 Reproduction
code_fig14_8_9 = r"""from common.ensemble_utils import MixtureOfLinearRegressions

# 多峰性合成データの生成 (PRML Figure 14.8 準拠: 2つの交差する線形ブランチ)
np.random.seed(42)
N_sub = 100
x_data = np.random.uniform(-1.0, 1.0, N_sub * 2)

# ブランチ 1: t = 1.5 * x + 0.8 + noise
# ブランチ 2: t = -1.2 * x - 0.5 + noise
t_data = np.zeros_like(x_data)
t_data[:N_sub] = 1.5 * x_data[:N_sub] + 0.8 + np.random.normal(0, 0.15, N_sub)
t_data[N_sub:] = -1.2 * x_data[N_sub:] - 0.5 + np.random.normal(0, 0.15, N_sub)

X_in = x_data[:, np.newaxis]

# 2成分線形回帰混合モデルの EM 適合
moe = MixtureOfLinearRegressions(n_components=2, max_iter=60, random_state=42)
moe.fit(X_in, t_data)

# PRML Figure 14.8 & 14.9 のプロット
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# (a) データ点と学習された各成分の回帰直線 (PRML Figure 14.8)
axes[0].scatter(x_data, t_data, color='forestgreen', s=25, alpha=0.7, edgecolors='k', lw=0.4, label='Synthetic Data')
x_plot = np.linspace(-1.1, 1.1, 100)
Phi_plot = np.column_stack([np.ones_like(x_plot), x_plot])

colors_comp = ['royalblue', 'crimson']
for k in range(moe.n_components):
    y_line = Phi_plot @ moe.weights_[k]
    axes[0].plot(x_plot, y_line, color=colors_comp[k], lw=2.5, label=f'Component {k+1} ($\pi_{k+1}={moe.pi_[k]:.2f}$)')

axes[0].set_title('Synthetic Data & Fitted Linear Regression Components (PRML Figure 14.8)', fontsize=11)
axes[0].set_xlabel('$x$', fontsize=10); axes[0].set_ylabel('$t$', fontsize=10)
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(True, linestyle='--', alpha=0.3)

# (b) 予測条件付き確率密度 p(t|x) の等高線/ヒートマップ (PRML Figure 14.9)
x_grid = np.linspace(-1.1, 1.1, 120)
t_grid = np.linspace(-2.2, 2.5, 120)
density_map = moe.predict_density(x_grid, t_grid)

im = axes[1].contourf(x_grid, t_grid, density_map, levels=20, cmap='viridis')
plt.colorbar(im, ax=axes[1], label='$p(t|x)$')
axes[1].scatter(x_data, t_data, color='white', s=15, alpha=0.5, edgecolors='k', lw=0.3)

axes[1].set_title('Predictive Conditional Density $p(t|x)$ (PRML Figure 14.9)', fontsize=11)
axes[1].set_xlabel('$x$', fontsize=10); axes[1].set_ylabel('$t$', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.3)

plt.suptitle('Mixture of Linear Regression Models (PRML Figures 14.8 & 14.9)', fontsize=13, y=1.02)
plt.tight_layout()
save_plot(fig, 'result', 'fig14_8_9_mixture_linear_regression.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig14_8_9))

nb.cells = cells
with open('14/14.4-14.5_Decision_Trees_and_Mixture_of_Experts.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("14/14.4-14.5_Decision_Trees_and_Mixture_of_Experts.ipynb generated successfully.")
