import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 13.3 線形動的システムとカルマンフィルタ (Linear Dynamical Systems & Kalman Filter)

本ノートブックでは、潜在変数が連続実数ベクトルであるマルコフ動的システムである **線形動的システム (LDS)** を学びます。
状態方程式と観測方程式の定式化、逐次オンライン推定を行う **カルマンフィルタ (Kalman Filter)**、状態拡散による不確実性の拡大と新規データ観測による分散収縮の幾何学的過程（**PRML Figure 13.21**）、および系列全体を後ろ向きに再帰修正して精度を最大化する **カルマンスムーザ (Rauch-Tung-Striebel: RTS)** を完全実装・再現します。"""))

# 13.3 Theory & Figure 13.21
cells.append(nbf.v4.new_markdown_cell(r"""## 13.3.1 カルマンフィルタと不確実性のダイナミクス (PRML Figure 13.21)

線形ガウス動的システムにおいて：
1. **予測ステップ (Prior)**:
   $$ \mathbf{P}_n = \mathbf{A}\mathbf{V}_{n-1}\mathbf{A}^{\mathrm{T}} + \mathbf{\Gamma} $$
   システムノイズ $\mathbf{\Gamma}$ の加算により、状態の分散楕円は拡散して拡大します。
2. **観測更新ステップ (Posterior)**:
   $$ \mathbf{K}_n = \mathbf{P}_n \mathbf{C}^{\mathrm{T}}(\mathbf{C}\mathbf{P}_n\mathbf{C}^{\mathrm{T}} + \mathbf{\Sigma})^{-1} $$
   $$ \mathbf{V}_n = (\mathbf{I} - \mathbf{K}_n \mathbf{C})\mathbf{P}_n $$
   新たな観測データ $\mathbf{x}_n$ が得られることで、不確実性は補正されて分散楕円は収縮します（PRML Figure 13.21）。"""))

# Code: PRML Figure 13.21 Reproduction
code_fig13_21 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from common.plot_utils import save_plot, setup_style
from common.sequential_utils import KalmanFilter
setup_style()

def plot_cov_ellipse(ax, mean, cov, color='blue', alpha=0.5, linestyle='-', label=None):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * np.sqrt(np.maximum(vals, 1e-8)) * 1.5 # 1.5-sigma
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor=color, facecolor='none', linestyle=linestyle, lw=2.0, alpha=alpha, label=label)
    ax.add_patch(ell)

# 2次元平面上での物体追跡 LDS
# 状態 z = [pos_x, pos_y], 観測 x = [meas_x, meas_y]
A_mat = np.array([[1.0, 0.0], [0.0, 1.0]]) # ランダムウォークドリフト
C_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
Gamma_mat = np.array([[0.3, 0.05], [0.05, 0.2]]) # システム拡散ノイズ
Sigma_mat = np.array([[0.4, 0.0], [0.0, 0.4]])   # 観測ノイズ

mu_0 = np.array([0.0, 0.0])
V_0 = np.array([[0.1, 0.0], [0.0, 0.1]])

# 3ステップの追跡過程 (PRML Figure 13.21)
# ステップ 1: 事後 -> ステップ 2: 予測 (拡散拡大) -> 観測到着 -> 事後 (収縮)
np.random.seed(42)
z_true = [mu_0]
x_obs = [z_true[0] + np.random.multivariate_normal([0, 0], Sigma_mat)]

for _ in range(4):
    z_next = A_mat @ z_true[-1] + np.random.multivariate_normal([0, 0], Gamma_mat)
    x_next = C_mat @ z_next + np.random.multivariate_normal([0, 0], Sigma_mat)
    z_true.append(z_next)
    x_obs.append(x_next)

z_true = np.array(z_true)
x_obs = np.array(x_obs)

# カルマンフィルタの実行
kf = KalmanFilter(A_mat, C_mat, Gamma_mat, Sigma_mat, mu_0, V_0)
mu_filt, V_filt = kf.filter(x_obs)

# PRML Figure 13.21 のプロット: 予測楕円 (拡散) と 更新楕円 (収縮)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

# (a) 予測ステップでの拡散による不確実性の増大
axes[0].scatter(mu_filt[0, 0], mu_filt[0, 1], color='forestgreen', s=60, label=r'Step $n-1$ Posterior $\mu_{n-1}$')
plot_cov_ellipse(axes[0], mu_filt[0], V_filt[0], color='forestgreen', linestyle='-', label=r'Posterior Variance $\mathbf{V}_{n-1}$')

# 予測分散 P_n = A V A^T + Gamma
P_pred = A_mat @ V_filt[0] @ A_mat.T + Gamma_mat
mu_pred = A_mat @ mu_filt[0]
axes[0].scatter(mu_pred[0], mu_pred[1], color='crimson', marker='^', s=70, label=r'Predicted Mean $\mathbf{A}\mu_{n-1}$')
plot_cov_ellipse(axes[0], mu_pred, P_pred, color='crimson', linestyle='--', label=r'Predicted Variance $\mathbf{P}_n$ (Expanded)')

axes[0].set_title('Diffusion increases uncertainty (PRML Figure 13.21 Left)', fontsize=11)
axes[0].set_xlim(-1.5, 2.5); axes[0].set_ylim(-1.5, 2.5)
axes[0].legend(loc='upper left', fontsize=9)
axes[0].grid(True, linestyle='--', alpha=0.3)

# (b) 観測データ到着による不確実性の収縮
axes[1].scatter(mu_pred[0], mu_pred[1], color='crimson', marker='^', s=70, label=r'Prior $\mathbf{P}_n$')
plot_cov_ellipse(axes[1], mu_pred, P_pred, color='crimson', linestyle='--', label='Prior Variance')

axes[1].scatter(x_obs[1, 0], x_obs[1, 1], color='black', marker='x', s=100, lw=2, label=r'Observation $\mathbf{x}_n$')
plot_cov_ellipse(axes[1], x_obs[1], Sigma_mat, color='black', linestyle=':', label=r'Observation Noise $\mathbf{\Sigma}$')

axes[1].scatter(mu_filt[1, 0], mu_filt[1, 1], color='royalblue', s=70, label=r'Updated Posterior $\mu_n$')
plot_cov_ellipse(axes[1], mu_filt[1], V_filt[1], color='royalblue', linestyle='-', label=r'Posterior Variance $\mathbf{V}_n$ (Shrunk)')

axes[1].set_title('Data arrival reduces uncertainty (PRML Figure 13.21 Right)', fontsize=11)
axes[1].set_xlim(-1.5, 2.5); axes[1].set_ylim(-1.5, 2.5)
axes[1].legend(loc='upper left', fontsize=9)
axes[1].grid(True, linestyle='--', alpha=0.3)

plt.suptitle('Dynamics of Uncertainty in Kalman Filtering (PRML Figure 13.21)', fontsize=13, y=1.02)
plt.tight_layout()
save_plot(fig, 'result', 'fig13_21_kalman_uncertainty_dynamics.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig13_21))

# Kalman Filter vs Kalman Smoother Comparison
cells.append(nbf.v4.new_markdown_cell(r"""## カルマンフィルタ vs カルマンスムーザ (RTS) の軌跡平滑化比較

全時刻系列 $N=30$ において、
1. **真の状態系列 (True Trajectory)**
2. **ノイズの乗った観測点 (Noisy Observations)**
3. **カルマンフィルタ推定値 (Kalman Filter)**: 過去の観測のみを利用
4. **カルマンスムーザ推定値 (Kalman Smoother / RTS)**: 未来も含めた全観測を利用
を直接比較し、スムーザ推定値がより真の軌跡に近く、分散が極小化されることを検証します。"""))

# Code: Filter vs Smoother Plot
code_kf_vs_ks = r"""# 長い時系列の生成
N_long = 35
np.random.seed(42)
z_sim = [np.array([0.0, 0.0])]
x_sim = []

for _ in range(N_long):
    z_sim.append(A_mat @ z_sim[-1] + np.random.multivariate_normal([0, 0], Gamma_mat))
    x_sim.append(C_mat @ z_sim[-1] + np.random.multivariate_normal([0, 0], Sigma_mat))
z_sim = np.array(z_sim[1:])
x_sim = np.array(x_sim)

# フィルタとスムーザの実行
mu_f, V_f = kf.filter(x_sim)
mu_s, V_s = kf.smooth(x_sim)

# 二乗誤差 (MSE) の比較
mse_obs = np.mean(np.sum((x_sim - z_sim)**2, axis=1))
mse_filt = np.mean(np.sum((mu_f - z_sim)**2, axis=1))
mse_smooth = np.mean(np.sum((mu_s - z_sim)**2, axis=1))

print(f"Observation MSE:     {mse_obs:.4f}")
print(f"Kalman Filter MSE:   {mse_filt:.4f}")
print(f"Kalman Smoother MSE: {mse_smooth:.4f}")
assert mse_smooth < mse_filt < mse_obs # スムーザが最も誤差小

# プロット
fig, ax = plt.subplots(figsize=(9, 6.5))

ax.plot(z_sim[:, 0], z_sim[:, 1], 'g-', lw=2.5, label='True Hidden Trajectory')
ax.scatter(x_sim[:, 0], x_sim[:, 1], color='gray', marker='x', s=45, alpha=0.7, label=f'Observations (MSE={mse_obs:.2f})')
ax.plot(mu_f[:, 0], mu_f[:, 1], 'b--', lw=1.8, label=f'Kalman Filter (MSE={mse_filt:.2f})')
ax.plot(mu_s[:, 0], mu_s[:, 1], 'r-', lw=2.2, label=f'Kalman Smoother (MSE={mse_smooth:.2f})')

ax.set_title('Trajectory Reconstruction: Filter vs Smoother (RTS)', fontsize=12)
ax.set_xlabel('$z_1$', fontsize=11); ax.set_ylabel('$z_2$', fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

save_plot(fig, 'result', 'fig13_kalman_filter_vs_smoother.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_kf_vs_ks))

nb.cells = cells
with open('13/13.3_Linear_Dynamical_Systems_Kalman_Filter.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("13/13.3_Linear_Dynamical_Systems_Kalman_Filter.ipynb generated successfully.")
