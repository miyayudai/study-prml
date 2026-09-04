import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 10.1 変分推論の基礎と平均場近似 (Variational Inference Foundations)

本ノートブックでは、解析的に厳密解が得られない複雑な事後分布に対する強力な決定論的近似手法である**変分推論 (Variational Inference)** を学びます。
平均場近似（因数分解変分分布）の一般解、**KLダイバージェンスの方向性による極めて対照的な挙動（ゼロ回避 vs ゼロ強制、PRML Figure 10.2, 10.3）**、および1変量ガウス分布の未知平均 $\mu$ と未知精度 $\tau$ に対する変分ベイズ反復学習の幾何学的収束過程（**PRML Figure 10.4**）を完全再現します。"""))

# 10.1 KL divergence orientation & Figure 10.2
cells.append(nbf.v4.new_markdown_cell(r"""## 10.1.2 因数分解近似の性質と KL ダイバージェンスの方向性 (PRML Figure 10.2)

真の分布 $p(\mathbf{z})$ を因数分解分布 $q(\mathbf{z}) = \prod_i q_i(z_i)$ で近似するとき、最小化するKLの向きによって全く異なる結果を生じます：

1. **$\mathrm{KL}(q \,||\, p)$ の最小化 (変分ベイズ)**:
   $$ \mathrm{KL}(q \,||\, p) = -\int q(\mathbf{z}) \ln \left\{ \frac{p(\mathbf{z})}{q(\mathbf{z})} \right\} d\mathbf{z} $$
   $p(\mathbf{z}) = 0$ の領域で $q(\mathbf{z}) > 0$ となると被積分関数が $+\infty$ に発散するため、**$q(\mathbf{z})$ は $p(\mathbf{z}) = 0$ の領域を絶対に回避します（ゼロ回避 / Zero-avoiding）**。
   多峰性分布に対しては単一の鋭いピーク（モード）に強く集中し、**分散を過小評価**します（PRML Figure 10.2 左）。

2. **$\mathrm{KL}(p \,||\, p)$ の最小化 (期待値伝播法 / モーメント整合)**:
   $$ \mathrm{KL}(p \,||\, q) = -\int p(\mathbf{z}) \ln \left\{ \frac{q(\mathbf{z})}{p(\mathbf{z})} \right\} d\mathbf{z} $$
   $p(\mathbf{z}) > 0$ の領域で $q(\mathbf{z}) = 0$ となると被積分関数が発散するため、**$p(\mathbf{z}) > 0$ の全領域を $q(\mathbf{z})$ が覆い尽くす必要があります（ゼロ強制 / Zero-forcing）**。
   その結果、分布全体の平均と分散に一致するよう広がり、**分散を過大評価**します（PRML Figure 10.2 右）。"""))

# Code: PRML Figure 10.2 Reproduction
code_fig10_2 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from common.plot_utils import save_plot, setup_style
setup_style()

# 強い相関を持つ二変量ガウス真分布 p(z)
mean_p = np.array([0.0, 0.0])
cov_p = np.array([[1.0, 0.9], [0.9, 1.0]])
p_dist = multivariate_normal(mean=mean_p, cov=cov_p)
Lambda_p = np.linalg.inv(cov_p)

# 1. KL(q || p) の最適解 (PRML 式 10.13, 10.15)
# 平均は真の平均と一致: m1 = 0, m2 = 0
# 分散は精度行列の対角成分の逆数: var_i = 1 / Lambda_ii
var_q_rev = 1.0 / np.diag(Lambda_p) # 過小評価
cov_q_rev = np.diag(var_q_rev)
q_rev_dist = multivariate_normal(mean=mean_p, cov=cov_q_rev)

# 2. KL(p || q) の最適解
# モーメント整合: 平均と分散が真の周辺平均・周辺分散と完全一致
var_q_fwd = np.diag(cov_p) # [1.0, 1.0] (過大評価)
cov_q_fwd = np.diag(var_q_fwd)
q_fwd_dist = multivariate_normal(mean=mean_p, cov=cov_q_fwd)

# グリッドの作成
z1 = np.linspace(-3.0, 3.0, 200)
z2 = np.linspace(-3.0, 3.0, 200)
Z1, Z2 = np.meshgrid(z1, z2)
pos = np.dstack((Z1, Z2))

P_vals = p_dist.pdf(pos)
Q_rev_vals = q_rev_dist.pdf(pos)
Q_fwd_vals = q_fwd_dist.pdf(pos)

# PRML Figure 10.2 プロット
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# (a) KL(q || p) - Variational Inference (Mode-seeking / Underestimation)
axes[0].contour(Z1, Z2, P_vals, levels=4, colors='crimson', linestyles='-', linewidths=2.0)
axes[0].contour(Z1, Z2, Q_rev_vals, levels=4, colors='royalblue', linestyles='--', linewidths=2.0)
axes[0].set_title(r'(a) $\mathrm{KL}(q || p)$ Variational (Zero-avoiding)', fontsize=12)
axes[0].set_xlabel('$z_1$', fontsize=11); axes[0].set_ylabel('$z_2$', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.3)
axes[0].plot([], [], 'r-', lw=2, label=r'True correlated $p(\mathbf{z})$')
axes[0].plot([], [], 'b--', lw=2, label=r'Factorized $q(\mathbf{z})$')
axes[0].legend(loc='upper left', fontsize=10)

# (b) KL(p || q) - Expectation Propagation (Moment-matching / Overestimation)
axes[1].contour(Z1, Z2, P_vals, levels=4, colors='crimson', linestyles='-', linewidths=2.0)
axes[1].contour(Z1, Z2, Q_fwd_vals, levels=4, colors='royalblue', linestyles='--', linewidths=2.0)
axes[1].set_title(r'(b) $\mathrm{KL}(p || q)$ Moment Matching (Zero-forcing)', fontsize=12)
axes[1].set_xlabel('$z_1$', fontsize=11); axes[1].set_ylabel('$z_2$', fontsize=11)
axes[1].grid(True, linestyle='--', alpha=0.3)
axes[1].plot([], [], 'r-', lw=2, label=r'True correlated $p(\mathbf{z})$')
axes[1].plot([], [], 'b--', lw=2, label=r'Factorized $q(\mathbf{z})$')
axes[1].legend(loc='upper left', fontsize=10)

plt.suptitle('Comparison of Factorized Approximations for Two KL Directions (PRML Figure 10.2)', fontsize=13, y=1.02)
plt.tight_layout()
save_plot(fig, 'result', 'fig10_2_kl_divergence_directions.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig10_2))

# 10.1.3 Univariate Gaussian & Figure 10.4
cells.append(nbf.v4.new_markdown_cell(r"""## 10.1.3 1変量ガウス分布の変分ベイズ推論 (PRML Figure 10.4)

未知の平均 $\mu$ と未知の精度 $\tau$ を持つガウス観測データ $\mathcal{D} = \{x_1, \dots, x_N\}$ に対し、事後分布 $p(\mu, \tau | \mathcal{D})$ を因数分解分布 $q(\mu, \tau) = q_\mu(\mu) q_\tau(\tau)$ で近似します。
- $q_\mu(\mu) = \mathcal{N}(\mu | \mu_N, \lambda_N^{-1})$
- $q_\tau(\tau) = \mathrm{Gam}(\tau | a_N, b_N)$
交互更新により、$q_\mu$ と $q_\tau$ が真の事後分布の等高線（緑）に対して最適に収束していく幾何学的様子（**PRML Figure 10.4**）を完全再現します。"""))

# Code: PRML Figure 10.4 Reproduction
code_fig10_4 = r"""from common.variational_utils import variational_gaussian_1d
from scipy.stats import norm, gamma as gamma_scipy

# 人工データ生成 (真値: mu = 1.0, tau = 1.5)
np.random.seed(42)
N_pts = 10
mu_true = 1.0
tau_true = 1.5
X_data = np.random.normal(mu_true, 1.0 / np.sqrt(tau_true), N_pts)

# 変分推論の実行
history = variational_gaussian_1d(X_data, mu_0=0.0, lambda_0=0.0, a_0=0.0, b_0=0.0, max_iter=6)

# グリッド
mu_vals = np.linspace(-0.5, 2.5, 150)
tau_vals = np.linspace(0.1, 4.0, 150)
M_grid, T_grid = np.meshgrid(mu_vals, tau_vals)

# 真の結合事後分布 p(mu, tau | D) (ガウス・ガンマ分布)
# p(mu, tau | D) = N(mu | x_bar, (N*tau)^-1) * Gam(tau | a_N, b_N)
x_bar = np.mean(X_data)
s_sq = np.sum((X_data - x_bar)**2)
a_true = (N_pts) / 2.0
b_true = 0.5 * s_sq

P_true = np.zeros_like(M_grid)
for i in range(len(tau_vals)):
    t = tau_vals[i]
    p_tau = gamma_scipy.pdf(t, a=a_true, scale=1.0/b_true)
    p_mu = norm.pdf(mu_vals, loc=x_bar, scale=1.0/np.sqrt(N_pts * t))
    P_true[i, :] = p_tau * p_mu

# PRML Figure 10.4 のプロット (イテレーション 0, 1, 2, 5)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
stages = [0, 1, 2, 4]

for idx, s in enumerate(stages):
    ax = axes[idx]
    mu_N, lambda_N, a_N, b_N = history[s]
    
    # 近似因数分解事後分布 Q(mu, tau) = q(mu) * q(tau)
    Q_approx = np.zeros_like(M_grid)
    q_mu = norm.pdf(mu_vals, loc=mu_N, scale=1.0/np.sqrt(lambda_N))
    q_tau = gamma_scipy.pdf(tau_vals, a=a_N, scale=1.0/b_N)
    for i in range(len(tau_vals)):
        Q_approx[i, :] = q_tau[i] * q_mu
        
    ax.contour(M_grid, T_grid, P_true, levels=5, colors='forestgreen', linewidths=2.0)
    ax.contour(M_grid, T_grid, Q_approx, levels=5, colors='royalblue', linestyles='--', linewidths=2.0)
    
    ax.set_title(f'Iteration {s+1}', fontsize=12)
    ax.set_xlabel(r'$\mu$', fontsize=12); ax.set_ylabel(r'$\tau$', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

axes[0].plot([], [], color='forestgreen', lw=2, label=r'True Posterior $p(\mu, \tau|\mathcal{D})$')
axes[0].plot([], [], color='royalblue', linestyle='--', lw=2, label=r'Variational $q_\mu(\mu)q_\tau(\tau)$')
axes[0].legend(loc='upper right', fontsize=9)

plt.suptitle('Variational Inference for Mean and Precision of Univariate Gaussian (PRML Figure 10.4)', fontsize=13, y=1.02)
plt.tight_layout()
save_plot(fig, 'result', 'fig10_4_variational_gaussian_steps.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig10_4))

nb.cells = cells
with open('10/10.1_Variational_Inference_Foundations.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("10/10.1_Variational_Inference_Foundations.ipynb generated successfully.")
