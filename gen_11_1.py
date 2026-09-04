import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 11.1 基本的サンプリング法 (Basic Sampling Algorithms)

本ノートブックでは、確率モデルからの乱数生成および期待値計算の基礎となる非MCMCサンプリング技法を学びます。
一様乱数から任意分布を生成する逆関数法と **Box-Muller 法**、包絡線提案分布を用いた **棄却サンプリング（PRML Figure 11.4, 11.5）**、および直接サンプリングが困難な分布における期待値評価のための **重点サンプリング (Importance Sampling)** と **重点リサンプリング (SIR)** を完全実装・検証します。"""))

# 11.1.2 Rejection Sampling & Figure 11.4, 11.5
cells.append(nbf.v4.new_markdown_cell(r"""## 11.1.2 棄却サンプリング (PRML Figure 11.4, 11.5)

正規化定数が未知の目標分布 $\tilde{p}(z)$ に対し、容易にサンプリング可能で常に $\tilde{p}(z) \le k q(z)$ を満たす提案分布 $q(z)$ と定数 $k$ を用意します。
1. $z^* \sim q(z)$ を生成
2. $u \sim \mathrm{Uniform}(0, k q(z^*))$ を生成
3. $u \le \tilde{p}(z^*)$ ならば $z^*$ を受容、そうでなければ棄却

### ガンマ分布のサンプリング (PRML Figure 11.5)
形状パラメータ $a > 1$ のガンマ分布 $p(z) \propto z^{a-1} e^{-bz}$ に対し、適切な尺度を持つコーシー分布や指数分布を提案分布 $q(z)$ として包絡線を設定する様子を可視化します。"""))

# Code: PRML Figure 11.4 & 11.5 Reproduction
code_fig11_4_5 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_scipy
from common.plot_utils import save_plot, setup_style
from common.sampling_utils import rejection_sample
setup_style()

# 目標分布: 二峰性混合ガウス分布 p_tilde(z) (PRML Figure 11.4)
def target_pdf(z):
    return 0.4 * np.exp(-0.5 * (z - 2.0)**2 / 0.8**2) + 0.6 * np.exp(-0.5 * (z - 5.0)**2 / 1.2**2)

# 提案分布 q(z): 幅広いガウス分布 N(3.5, 2.5^2)
mu_q, sigma_q = 3.5, 2.5
def proposal_pdf(z):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma_q)) * np.exp(-0.5 * (z - mu_q)**2 / sigma_q**2)

# 包絡係数 k (k*q(z) >= target(z))
z_grid = np.linspace(-3.0, 10.0, 500)
k = np.max(target_pdf(z_grid) / proposal_pdf(z_grid)) * 1.15

# 棄却サンプリングの実行
np.random.seed(42)
N_trials = 1500
z_cand = np.random.normal(mu_q, sigma_q, N_trials)
u = np.random.uniform(0, k * proposal_pdf(z_cand))
accepted = u <= target_pdf(z_cand)

# PRML Figure 11.4 プロット
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(z_grid, target_pdf(z_grid), 'r-', lw=2.5, label=r'Target Distribution $\tilde{p}(z)$')
ax.plot(z_grid, k * proposal_pdf(z_grid), 'b-', lw=2.0, label=r'Comparison Distribution $k q(z)$')

# 乱数点のプロット (受容点は緑、棄却点は灰色)
ax.scatter(z_cand[accepted], u[accepted], color='forestgreen', s=12, alpha=0.7, label=f'Accepted ({np.sum(accepted)} pts)')
ax.scatter(z_cand[~accepted], u[~accepted], color='gray', s=8, alpha=0.3, label=f'Rejected ({np.sum(~accepted)} pts)')

ax.set_title(f'Rejection Sampling (PRML Figure 11.4, Acceptance Rate: {np.mean(accepted):.2%})', fontsize=12)
ax.set_xlabel('$z$', fontsize=11); ax.set_ylabel('Density / Value', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

save_plot(fig, 'result', 'fig11_4_rejection_sampling.png')
plt.show()

# PRML Figure 11.5: ガンマ分布と線形包絡線
z_gam = np.linspace(0.01, 8.0, 300)
a_param, b_param = 3.0, 1.0
gamma_pdf = (b_param**a_param / 2.0) * z_gam**(a_param - 1) * np.exp(-b_param * z_gam)

# 線形包絡線の例
tangent_z = a_param - 1.0 # モード
envelope = np.where(z_gam < tangent_z, 
                    gamma_pdf[np.argmin(np.abs(z_gam - tangent_z))] * (z_gam / tangent_z),
                    gamma_pdf[np.argmin(np.abs(z_gam - tangent_z))] * np.exp(-0.4 * (z_gam - tangent_z)))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(z_gam, gamma_pdf, 'g-', lw=2.5, label=r'Gamma Distribution $\mathrm{Gam}(z|3, 1)$')
ax.plot(z_gam, envelope, 'b--', lw=2.0, label='Envelope Distribution')
ax.set_title('Envelope for Gamma Distribution (PRML Figure 11.5)', fontsize=12)
ax.set_xlabel('$z$', fontsize=11); ax.set_ylabel('Density', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)

save_plot(fig, 'result', 'fig11_5_gamma_envelope.png')
plt.show()"""
cells.append(nbf.v4.new_code_cell(code_fig11_4_5))

# 11.1.4 Importance Sampling & Box-Muller
cells.append(nbf.v4.new_markdown_cell(r"""## 11.1.4 重点サンプリング (Importance Sampling) と Box-Muller 法

### Box-Muller 法
独立な一様乱数 $u_1, u_2 \sim \mathrm{Uniform}(0, 1)$ から、2つの独立な標準正規乱数を正確に生成：
$$ z_1 = \sqrt{-2 \ln u_1} \cos(2\pi u_2), \quad z_2 = \sqrt{-2 \ln u_1} \sin(2\pi u_2) $$

### 重点サンプリング
期待値 $\mathbb{E}[f] = \int f(z) p(z) dz$ を、サンプリング容易な $q(z)$ からのサンプル $\{z_l\}_{l=1}^L$ と重点重み $w_l = \frac{\tilde{p}(z_l)}{q(z_l)}$ を用いて不偏推定：
$$ \mathbb{E}[f] \approx \frac{\sum_{l=1}^L w_l f(z_l)}{\sum_{l=1}^L w_l} $$"""))

# Code: Box-Muller and Importance Sampling
code_box_muller_is = r"""# Box-Muller 法の検証
np.random.seed(42)
N_bm = 10000
u1 = np.random.uniform(0, 1, N_bm)
u2 = np.random.uniform(0, 1, N_bm)
z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

print(f"Box-Muller Sample 1: mean = {np.mean(z1):.4f}, std = {np.std(z1):.4f}")
print(f"Box-Muller Sample 2: mean = {np.mean(z2):.4f}, std = {np.std(z2):.4f}")
assert np.isclose(np.mean(z1), 0.0, atol=0.03)
assert np.isclose(np.std(z1), 1.0, atol=0.03)
print("Box-Muller standard Gaussian generation verified successfully!")

# 重点サンプリングの検証: コーシー分布目標 p(z) に対する E[z^2 * I(|z|<3)] の推定
# 提案分布 q(z): ガウス分布 N(0, 2^2)
p_cauchy = lambda z: 1.0 / (np.pi * (1.0 + z**2))
q_gauss = lambda z: (1.0 / (np.sqrt(2 * np.pi) * 2.0)) * np.exp(-0.5 * (z / 2.0)**2)
f_func = lambda z: (z**2) * (np.abs(z) < 3.0)

z_samples = np.random.normal(0, 2.0, 100000)
weights = p_cauchy(z_samples) / q_gauss(z_samples)
est_expectation = np.sum(weights * f_func(z_samples)) / np.sum(weights)

# 数値積分による真値
from scipy.integrate import quad
true_val, _ = quad(lambda z: f_func(z) * p_cauchy(z), -3.0, 3.0)
print(f"Importance Sampling Estimate: {est_expectation:.4f}, Numerical Integral True: {true_val:.4f}")
assert np.isclose(est_expectation, true_val, rtol=0.08)
print("Importance sampling expectation estimation verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_box_muller_is))

nb.cells = cells
with open('11/11.1_Basic_Sampling_Algorithms.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("11/11.1_Basic_Sampling_Algorithms.ipynb generated successfully.")
