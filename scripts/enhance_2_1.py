"""
Script to enhance 2/2.1_Binary_Variables.ipynb with deep theory, interactive code cells,
Laplace smoothing, Bayesian vs MLE convergence, and executed visualization outputs.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Ensure result directory exists
os.makedirs("2/result", exist_ok=True)

# Generate fig2_2_beta_priors
mu_vals = np.linspace(0.001, 0.999, 500)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
params = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (2.0, 8.0)]
titles = ["Beta(0.5, 0.5) [Jeffreys Prior]", "Beta(1, 1) [Uniform Prior]",
          "Beta(2, 2) [Informative Bell]", "Beta(2, 8) [Skewed to Tails]"]

for ax, (a, b), title in zip(axes.ravel(), params, titles):
    pdf = stats.beta.pdf(mu_vals, a, b)
    ax.plot(mu_vals, pdf, 'r-', lw=2.5, label=f'a={a}, b={b}')
    ax.fill_between(mu_vals, 0, pdf, color='red', alpha=0.15)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\mu$', fontsize=11)
    ax.set_ylabel(r'$p(\mu)$', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

plt.tight_layout()
fig2_path = "2/result/fig2_2_beta_priors.png"
plt.savefig(fig2_path, dpi=150)
plt.close()

# Generate fig2_3_bayesian_sequential_learning
np.random.seed(42)
true_mu = 0.7
flips = np.random.binomial(1, true_mu, size=30)
a_init, b_init = 2.0, 2.0

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
checkpoints = [0, 1, 5, 20]

for ax, n in zip(axes.ravel(), checkpoints):
    m = np.sum(flips[:n]) if n > 0 else 0
    l = n - m
    a_post = a_init + m
    b_post = b_init + l
    pdf = stats.beta.pdf(mu_vals, a_post, b_post)
    ax.plot(mu_vals, pdf, 'b-', lw=2.5, label=f'Posterior Beta({a_post:.0f}, {b_post:.0f})')
    ax.fill_between(mu_vals, 0, pdf, color='blue', alpha=0.15)
    ax.axvline(true_mu, color='darkgreen', linestyle='--', lw=2, label=f'True $\\mu={true_mu}$')
    if n > 0:
        ax.axvline(m / n, color='magenta', linestyle=':', lw=2, label=f'MLE $\\mu_{{ML}}={m/n:.2f}$')
    ax.set_title(f'N = {n} (Heads={m}, Tails={l})', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$\mu$', fontsize=11)
    ax.set_ylabel(r'$p(\mu|\mathcal{D})$', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
fig3_path = "2/result/fig2_3_bayesian_sequential_learning.png"
plt.savefig(fig3_path, dpi=150)
plt.close()

# Generate fig2_1_mle_vs_bayes_convergence
N_steps = np.arange(1, 101)
flips_100 = np.random.binomial(1, true_mu, size=100)
heads_cum = np.cumsum(flips_100)
mle_estimates = heads_cum / N_steps
bayes_estimates = (2.0 + heads_cum) / (2.0 + 2.0 + N_steps)

plt.figure(figsize=(9, 5))
plt.plot(N_steps, mle_estimates, 'm-', lw=2, label=r'MLE $\mu_{\mathrm{ML}} = m/N$')
plt.plot(N_steps, bayes_estimates, 'b-', lw=2.5, label=r'Bayes Posterior Mean $\mathbb{E}[\mu] = \frac{a+m}{a+b+N}$')
plt.axhline(true_mu, color='darkgreen', linestyle='--', lw=2, label=f'True $\\mu = {true_mu}$')
plt.fill_between(N_steps, true_mu - 0.05, true_mu + 0.05, color='green', alpha=0.1, label=r'$\pm 5\%$ Tolerance Band')
plt.title(r'PRML Fig 2.1: Convergence of MLE vs. Bayesian Estimator ($\mu^*=0.7$)', fontsize=13, fontweight='bold')
plt.xlabel('Sample Size $N$', fontsize=12)
plt.ylabel(r'Estimated $\mu$', fontsize=12)
plt.ylim(0.3, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
fig_conv_path = "2/result/fig2_1_mle_vs_bayes_convergence.png"
plt.savefig(fig_conv_path, dpi=150)
plt.close()

# Generate fig2_laplace_smoothing (Zero-frequency problem)
small_N = np.array([1, 2, 3, 5, 8])
all_heads_mle = np.ones_like(small_N, dtype=float)
all_heads_laplace = (small_N + 1.0) / (small_N + 2.0)

plt.figure(figsize=(8, 4.5))
plt.plot(small_N, all_heads_mle, 'ro--', lw=2.5, markersize=8, label='MLE (Overfitting: $\mu=1.0$)')
plt.plot(small_N, all_heads_laplace, 'bs-', lw=2.5, markersize=8, label='Laplace Succession Rule $\\frac{N+1}{N+2}$')
plt.axhline(1.0, color='gray', linestyle=':', alpha=0.7)
plt.title('PRML 2.1.1 Zero-Frequency Catastrophe & Laplace Smoothing', fontsize=12, fontweight='bold')
plt.xlabel('Consecutive Head Tosses $N$', fontsize=11)
plt.ylabel(r'Predicted Probability $p(x=1)$', fontsize=11)
plt.ylim(0.5, 1.1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
fig_laplace_path = "2/result/fig2_laplace_smoothing.png"
plt.savefig(fig_laplace_path, dpi=150)
plt.close()

print("Figures generated successfully for 2.1.")
