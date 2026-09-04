import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 14.1-14.3 バギングと AdaBoost (Bagging & AdaBoost)

本ノートブックでは、複数の予測器を組み合わせて単一モデルよりも汎化性能を向上させるアンサンブル学習の基本、**バギング (Bootstrap Aggregating / Committees)** と **AdaBoost (Adaptive Boosting)** を学びます。
コミッティの平均化による分散低減定理（PRML 式 14.10）、指数損失の逐次最適化としての AdaBoost アルゴリズムの導出、**データ点の重み拡大と複合決定境界の学習過程（PRML Figure 14.2）**、および分類における **各種サロゲート損失関数（指数・二乗・ロジスティック・0-1損失）の比較（PRML Figure 14.3）** を完全実装・再現します。"""))

# 14.2 Committees & Bagging Variance Reduction
cells.append(nbf.v4.new_markdown_cell(r"""## 14.2 コミッティモデルとバギングの分散低減定理 (PRML 式 14.10)

$M$ 個の独立なモデルの予測値 $y_m(\mathbf{x}) = h(\mathbf{x}) + \epsilon_m(\mathbf{x})$（真の関数 $h(\mathbf{x})$、平均ゼロの誤差 $\epsilon_m$）を平均化したコミッティモデル $y_{\mathrm{COM}}(\mathbf{x}) = \frac{1}{M}\sum_{m=1}^M y_m(\mathbf{x})$ の期待二乗誤差は：
$$ E_{\mathrm{COM}} = \frac{1}{M} E_{\mathrm{AV}} $$
を満たし、個々のモデルの平均二乗誤差 $E_{\mathrm{AV}}$ の $\frac{1}{M}$ に激減します。
ブートストラップ・サンプリングを用いたバギングにより、モデル数を増やすほど二乗誤差が低減する様子を数値検証します。"""))

# Code: Bagging Simulation
code_bagging = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from common.plot_utils import save_plot, setup_style
setup_style()

# 真の関数とノイズ付きデータ
np.random.seed(42)
N_train = 80
x_train = np.sort(np.random.uniform(0, 1, N_train))
y_true = np.sin(2 * np.pi * x_train)
y_train = y_true + np.random.normal(0, 0.25, N_train)

# テストデータ
x_test = np.linspace(0, 1, 200)
y_test_true = np.sin(2 * np.pi * x_test)

# M = 1 から 50 までのコミッティ (バギング) の構築
M_max = 40
committee_preds = []
indiv_errors = []

for m in range(M_max):
    # ブートストラップ・リサンプリング
    boot_idx = np.random.choice(N_train, N_train, replace=True)
    tree = DecisionTreeRegressor(max_depth=4, random_state=m)
    tree.fit(x_train[boot_idx, np.newaxis], y_train[boot_idx])
    pred = tree.predict(x_test[:, np.newaxis])
    committee_preds.append(pred)
    indiv_errors.append(np.mean((pred - y_test_true)**2))

committee_preds = np.array(committee_preds) # (M, 200)

# M 個の平均予測の誤差推移
com_errors = []
for m in range(1, M_max + 1):
    ens_pred = np.mean(committee_preds[:m], axis=0)
    com_errors.append(np.mean((ens_pred - y_test_true)**2))

E_AV = np.mean(indiv_errors)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(range(1, M_max + 1), com_errors, 'r-o', lw=2, markersize=4, label='Committee Error $E_{COM}$')
ax.axhline(E_AV, color='b', linestyle='--', lw=1.8, label=f'Single Model Average Error $E_{{AV}} = {E_AV:.4f}$')
ax.set_title('Bagging Ensemble: Reduction of Expected Squared Error (PRML 14.2)', fontsize=12)
ax.set_xlabel('Number of Models $M$', fontsize=11); ax.set_ylabel('Mean Squared Error', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

save_plot(fig, 'result', 'fig14_bagging_variance_reduction.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_bagging))

# 14.3 AdaBoost Reproduction (PRML Figure 14.2)
cells.append(nbf.v4.new_markdown_cell(r"""## 14.3 AdaBoost アルゴリズムの視覚化 (PRML Figure 14.2)

AdaBoost は、各反復 $m$ において直前の弱分類器 $y_m(\mathbf{x})$ が誤分類したデータ点の重み $w_n$ を指数関数的に増大させ、
次の弱分類器が難しいデータ点に集中するように学習を誘導します。
PRML Figure 14.2 に倣い、
- 各弱分類器の決定境界
- データ点の重み $w_n$ に比例した散布図マーカーのサイズ
- 弱分類器が結合された複合分類器 $F_m(\mathbf{x}) = \sum_{l=1}^m \alpha_l y_l(\mathbf{x})$ の境界
をイテレーション $m=1, 2, 3$ および最終決定境界で視覚化します。"""))

# Code: PRML Figure 14.2 Reproduction
code_fig14_2 = r"""from common.ensemble_utils import AdaBoostClassifier

# 非線形な2クラス2次元トイデータセット (PRML Figure 14.2 準拠)
np.random.seed(14)
N_pts = 30
X_pos = np.random.randn(N_pts // 2, 2) * 0.4 + np.array([0.3, 0.3])
X_neg = np.random.randn(N_pts // 2, 2) * 0.5 + np.array([-0.3, -0.3])
X_ada = np.vstack([X_pos, X_neg])
y_ada = np.array([1]*(N_pts // 2) + [-1]*(N_pts // 2))

# 4反復の AdaBoost
ada = AdaBoostClassifier(n_estimators=4).fit(X_ada, y_ada)

# グリッドの作成
x_min, x_max = X_ada[:, 0].min() - 0.5, X_ada[:, 0].max() + 0.5
y_min, y_max = X_ada[:, 1].min() - 0.5, X_ada[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# PRML Figure 14.2 のプロット (イテレーション 1, 2, 3 と複合境界)
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.ravel()

for m in range(4):
    ax = axes[m]
    w = ada.weights_history[m] # 現在のイテレーションでのデータ点重み
    # 点の大きさを重みに比例させる
    point_sizes = (w / np.mean(w)) * 50.0 + 15.0
    
    # 弱分類器の決定境界
    stump = ada.models[m]
    pred_grid = stump.predict(grid_points).reshape(xx.shape)
    
    ax.contourf(xx, yy, pred_grid, levels=[-1.5, 0, 1.5], colors=['#ffcccc', '#ccccff'], alpha=0.35)
    ax.contour(xx, yy, pred_grid, levels=[0], colors='k', linestyles='--', linewidths=1.8)
    
    # データ点のプロット
    ax.scatter(X_ada[y_ada == 1, 0], X_ada[y_ada == 1, 1], color='royalblue',
               s=point_sizes[y_ada == 1], edgecolors='k', lw=0.8, label='Class +1')
    ax.scatter(X_ada[y_ada == -1, 0], X_ada[y_ada == -1, 1], color='crimson',
               s=point_sizes[y_ada == -1], edgecolors='k', lw=0.8, label='Class -1')
    
    ax.set_title(f'Iteration m = {m+1} (Stump Weight $\\alpha_{m+1} = {ada.alphas[m]:.2f}$)', fontsize=11)
    ax.set_xlabel('$x_1$', fontsize=10); ax.set_ylabel('$x_2$', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    if m == 0:
        ax.legend(loc='upper left', fontsize=9)

plt.suptitle('AdaBoost Sequential Training and Adaptive Sample Weights (PRML Figure 14.2)', fontsize=13, y=1.02)
plt.tight_layout()
save_plot(fig, 'result', 'fig14_2_adaboost_iterations.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig14_2))

# 14.3.2 Comparison of Error Functions (PRML Figure 14.3)
cells.append(nbf.v4.new_markdown_cell(r"""## 14.3.2 分類における損失関数の比較 (PRML Figure 14.3)

マージン $z = y f(\mathbf{x})$ に対する各種損失関数の振る舞いを比較します：
- **誤分類損失 (0-1 損失)**: $I(z < 0)$
- **指数損失 (AdaBoost)**: $E_{\mathrm{EXP}}(z) = \exp(-z)$
- **二乗誤差損失**: $E_{\mathrm{SQR}}(z) = (1 - z)^2$
- **ロジスティック損失 (交差エントロピー)**: $E_{\mathrm{LOG}}(z) = \frac{1}{\ln 2} \ln(1 + \exp(-z))$

指数損失は $z \ll 0$（大きな外れ値・誤分類）に対して指数関数的なペナルティを課すため、外れ値に敏感になる性質が分かります。"""))

# Code: PRML Figure 14.3 Reproduction
code_fig14_3 = r"""# マージン z のグリッド
z = np.linspace(-2.5, 2.5, 400)

loss_zero_one = np.where(z < 0, 1.0, 0.0)
loss_exp = np.exp(-z)
loss_sqr = (1.0 - z)**2
loss_log = np.log2(1.0 + np.exp(-z))

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(z, loss_zero_one, 'k-', lw=2.0, label='0-1 Misclassification')
ax.plot(z, loss_exp, 'g-', lw=2.2, label=r'Exponential: $\exp(-z)$')
ax.plot(z, loss_sqr, 'r--', lw=2.0, label=r'Squared Error: $(1-z)^2$')
ax.plot(z, loss_log, 'b-.', lw=2.0, label=r'Logistic: $\log_2(1 + \exp(-z))$')

ax.set_ylim(-0.2, 5.0)
ax.set_xlim(-2.5, 2.5)
ax.set_title('Comparison of Error Functions for Classification (PRML Figure 14.3)', fontsize=12)
ax.set_xlabel('Margin $z = y f(x)$', fontsize=11); ax.set_ylabel('Loss $E(z)$', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

save_plot(fig, 'result', 'fig14_3_error_functions_comparison.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig14_3))

nb.cells = cells
with open('14/14.1-14.3_Bagging_and_AdaBoost.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("14/14.1-14.3_Bagging_and_AdaBoost.ipynb generated successfully.")
