import nbformat as nbf
import json
import os
import sys

with open('0/0_Foundations_of_Probability.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

md_0_11_12 = r"""---

## 0.11 イェンセンの不等式 (Jensen's Inequality) と情報幾何・ELBO (第9章・第10章への補講)

PRML 第1章 (1.6節 情報理論)、第9章 (EMアルゴリズム)、第10章 (変分推論法) の理論的基盤をなすのが**イェンセンの不等式 (Jensen's Inequality)** です。

### 1. イェンセンの不等式
関数 $f(x)$ が凸関数（convex function, 下に凸: 二階微分 $f''(x) \ge 0$）であるとき、任意の確率変数 $x$ に対して
$$ f(\mathbb{E}[x]) \le \mathbb{E}[f(x)] $$
が成立します。
逆に対数関数 $f(x) = \ln x$ は狭義凹関数（concave function, 上に凸: $(\ln x)'' = -1/x^2 < 0$）であるため、不等号が逆転します：
$$ \ln \mathbb{E}[x] \ge \mathbb{E}[\ln x] $$

### 2. カルバック・ライブラー情報量 (KLダイバージェンス) の非負性
2つの確率分布 $p(x), q(x)$ に対し、KLダイバージェンスは次のように定義されます：
$$ \mathrm{KL}(q \parallel p) = - \int q(x) \ln \left( \frac{p(x)}{q(x)} \right) dx $$
ここで凹関数に対するイェンセンの不等式を適用すると：
$$ -\int q(x) \ln \left( \frac{p(x)}{q(x)} \right) dx \ge -\ln \left( \int q(x) \frac{p(x)}{q(x)} dx \right) = -\ln \left( \int p(x) dx \right) = -\ln 1 = 0 $$
したがって、常に $\mathrm{KL}(q \parallel p) \ge 0$ であり、等号成立は $q(x) = p(x)$ (ほぼ至る所) の場合に限られます。

### 3. エビデンス下界 (ELBO: Evidence Lower Bound) の幾何学
観測データ $\mathbf{X}$ と潜在変数 $\mathbf{Z}$ をもつモデルにおいて、対数周辺尤度 $\ln p(\mathbf{X})$ は次のように展開されます：
$$ \ln p(\mathbf{X}) = \ln \int p(\mathbf{X}, \mathbf{Z}) d\mathbf{Z} = \ln \int q(\mathbf{Z}) \frac{p(\mathbf{X}, \mathbf{Z})}{q(\mathbf{Z})} d\mathbf{Z} $$
イェンセンの不等式を適用することで、変分下界 $\mathcal{L}(q)$ が導かれます：
$$ \ln p(\mathbf{X}) \ge \int q(\mathbf{Z}) \ln \left( \frac{p(\mathbf{X}, \mathbf{Z})}{q(\mathbf{Z})} \right) d\mathbf{Z} \equiv \mathcal{L}(q) $$
さらに恒等的に
$$ \ln p(\mathbf{X}) = \mathcal{L}(q) + \mathrm{KL}(q(\mathbf{Z}) \parallel p(\mathbf{Z}|\mathbf{X})) $$
が成立します。$\mathrm{KL} \ge 0$ であるため、$\mathcal{L}(q)$ は常に $\ln p(\mathbf{X})$ の厳密な下界となります。これが第9章のEMアルゴリズムおよび第10章の変分推論法の全理論を支配する基本原理です。

---

## 0.12 マルコフ連鎖の詳細釣り合い条件 (Detailed Balance) と定常分布 (第11章・第13章への補講)

PRML 第11章（サンプリング法、MCMC、メトロポリス・ヘイスティングス法）および第13章（隠れマルコフモデル）の基盤となる確率過程の基礎性質です。

### 1. 定常分布 (Stationary Distribution)
状態空間上の遷移核（遷移確率密度）を $T(\mathbf{z} \to \mathbf{z}') = p(\mathbf{z}'|\mathbf{z})$ とします。
分布 $p^*(\mathbf{z})$ が遷移核 $T$ に関して**不変（定常）**であるとは、1ステップ遷移させた後の分布が元の分布と同一であることを意味します：
$$ \int p^*(\mathbf{z}) T(\mathbf{z} \to \mathbf{z}') d\mathbf{z} = p^*(\mathbf{z}') $$

### 2. 詳細釣り合い条件 (Detailed Balance)
定常性の十分条件として、**詳細釣り合い条件**（可逆性: Reversibility）が極めて重要です：
$$ p^*(\mathbf{z}) T(\mathbf{z} \to \mathbf{z}') = p^*(\mathbf{z}') T(\mathbf{z}' \to \mathbf{z}) $$
この式の両辺を $\mathbf{z}$ について積分すると：
$$ \int p^*(\mathbf{z}) T(\mathbf{z} \to \mathbf{z}') d\mathbf{z} = \int p^*(\mathbf{z}') T(\mathbf{z}' \to \mathbf{z}) d\mathbf{z} = p^*(\mathbf{z}') \int T(\mathbf{z}' \to \mathbf{z}) d\mathbf{z} = p^*(\mathbf{z}') \cdot 1 = p^*(\mathbf{z}') $$
となり、詳細釣り合いが満たされれば直ちに定常条件が満たされることが証明されます。
MCMC（マルコフ連鎖モンテカルロ法）における Metropolis-Hastings アルゴリズムの受容確率
$$ A(\mathbf{z}, \mathbf{z}') = \min\left(1, \frac{p^*(\mathbf{z}') q(\mathbf{z}|\mathbf{z}')}{p^*(\mathbf{z}) q(\mathbf{z}'|\mathbf{z})}\right) $$
は、まさにこの詳細釣り合い条件を厳密に成立させるように設計されています。"""

code_0_11_12 = r"""# 0.11 & 0.12 数値検証と可視化
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('0/result', exist_ok=True)

# 1. Jensen の不等式の幾何学的検証
# f(x) = -ln(x) は凸関数
x_vals = np.linspace(0.2, 5.0, 500)
f_vals = -np.log(x_vals)

x1, x2 = 0.8, 3.5
lambda_param = 0.4
x_bar = lambda_param * x1 + (1 - lambda_param) * x2
f_x_bar = -np.log(x_bar)
expected_f = lambda_param * (-np.log(x1)) + (1 - lambda_param) * (-np.log(x2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Jensen's inequality
axes[0].plot(x_vals, f_vals, 'b-', lw=2, label=r'$f(x) = -\ln(x)$ (Convex)')
axes[0].plot([x1, x2], [-np.log(x1), -np.log(x2)], 'r--', lw=1.5, label='Secant chord')
axes[0].plot(x_bar, f_x_bar, 'go', markersize=9, label=r'$f(\mathbb{E}[x]) = f(\bar{x})$')
axes[0].plot(x_bar, expected_f, 'ro', markersize=9, label=r'$\mathbb{E}[f(x)]$')
axes[0].vlines(x_bar, f_x_bar, expected_f, colors='purple', linestyles='dotted', lw=2,
               label=r'Jensen Gap $\ge 0$')
axes[0].set_title("Jensen's Inequality: $f(\mathbb{E}[x]) \leq \mathbb{E}[f(x)]$", fontsize=13)
axes[0].set_xlabel("x", fontsize=11)
axes[0].set_ylabel("f(x)", fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# 2. マルコフ連鎖の詳細釣り合いと定常分布への収束検証
# 3状態可逆マルコフ連鎖
# 状態空間 {1, 2, 3}, 目標定常分布 p* = [0.2, 0.5, 0.3]
p_target = np.array([0.2, 0.5, 0.3])
T = np.zeros((3, 3))

# 詳細釣り合いを満たす遷移核をメトロポリス法で構築
Q = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) # 提案行列
for i in range(3):
    for j in range(3):
        if i != j:
            acc = min(1.0, (p_target[j] * Q[j, i]) / (p_target[i] * Q[i, j]))
            T[i, j] = Q[i, j] * acc
    T[i, i] = 1.0 - np.sum(T[i, :])

# 詳細釣り合い p_i T_ij = p_j T_ji の数値検証
for i in range(3):
    for j in range(3):
        assert np.isclose(p_target[i] * T[i, j], p_target[j] * T[j, i]), f"Detailed balance violated at {i},{j}"

# 初期分布からのステップ推移
p_curr = np.array([0.9, 0.05, 0.05])
history = [p_curr]
for step in range(15):
    p_curr = p_curr @ T
    history.append(p_curr)
history = np.array(history)

for s in range(3):
    axes[1].plot(range(16), history[:, s], marker='o', label=f'State {s+1} (Target: {p_target[s]:.1f})')
    axes[1].axhline(p_target[s], color='gray', linestyle=':', alpha=0.7)

axes[1].set_title("MCMC Convergence via Detailed Balance", fontsize=13)
axes[1].set_xlabel("Transition Steps", fontsize=11)
axes[1].set_ylabel("Probability Distribution", fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
fig_path = '0/result/fig0_jensen_kl_detailed_balance.png'
plt.savefig(fig_path, dpi=150)
plt.close()

# KLダイバージェンスの非負性と解析解 vs モンテカルロ検証
# 2つの1次元ガウス分布 N(mu1, s1^2) と N(mu2, s2^2)
mu1, s1 = 0.0, 1.0
mu2, s2 = 1.0, 1.5
# KL(N1 || N2) = ln(s2/s1) + (s1^2 + (mu1 - mu2)^2)/(2 s2^2) - 1/2
kl_analytic = np.log(s2 / s1) + (s1**2 + (mu1 - mu2)**2) / (2 * s2**2) - 0.5
samples1 = np.random.normal(mu1, s1, size=100000)
# MC推定: E_p [ ln p(x) - ln q(x) ]
log_p = -0.5 * np.log(2 * np.pi * s1**2) - 0.5 * ((samples1 - mu1) / s1)**2
log_q = -0.5 * np.log(2 * np.pi * s2**2) - 0.5 * ((samples1 - mu2) / s2)**2
kl_mc = np.mean(log_p - log_q)

print(f"KL Analytic: {kl_analytic:.5f}")
print(f"KL Monte Carlo: {kl_mc:.5f}")
assert kl_analytic >= 0, "KL must be non-negative"
assert np.isclose(kl_analytic, kl_mc, atol=0.01), "KL analytic and MC mismatch"
print(f"Saved visualization to {fig_path}")
print("Section 0.11 and 0.12 verified and passed!")"""

nb.cells.append(nbf.v4.new_markdown_cell(md_0_11_12))
nb.cells.append(nbf.v4.new_code_cell(code_0_11_12))

with open('0/0_Foundations_of_Probability.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Appended sections 0.11 and 0.12 successfully.")
