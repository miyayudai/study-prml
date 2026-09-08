# scripts/build_ch13_part2.py
"""
Build Chapter 13 Exercises: Part 2 (Exercises 13.5 - 13.10)
HMM Baum-Welch M-step updates and Forward-Backward recursions
"""

import nbformat as nbf

def build_part2():
    """Exercises 13.5 - 13.10: HMM Baum-Welch M-step & Forward-Backward"""
    cells = []

    # Exercise 13.5
    ex13_5_md = """---
## Exercise 13.5: Baum-Welch M-step における遷移確率行列 $A_{jk}$ のラグランジュ乗数法による更新式導出

### 問題の背景と数学的証明
Baum-Welch (EM) アルゴリズムの E ステップにおいて、現在のパラメータ $\\boldsymbol{\\theta}^{\\text{old}}$ の下での完全データ対数尤度の期待値 $Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{\\text{old}})$ は次式で与えられる（(13.17)式）：
$$ Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{\\text{old}}) = \\sum_{k=1}^K \\gamma(z_{1k}) \\ln \\pi_k + \\sum_{n=2}^N \\sum_{j=1}^K \\sum_{k=1}^K \\xi(z_{n-1,j}, z_{nk}) \\ln A_{jk} + \\sum_{n=1}^N \\sum_{k=1}^K \\gamma(z_{nk}) \\ln p(\\mathbf{x}_n | \\boldsymbol{\\phi}_k) $$
ここで $\\xi(z_{n-1,j}, z_{nk}) = p(z_{n-1,j}=1, z_{nk}=1 | \\mathbf{X}, \\boldsymbol{\\theta}^{\\text{old}})$ である。
M ステップでは、各行の確率の和が1であるという制約：
$$ \\sum_{k=1}^K A_{jk} = 1 \\quad (j = 1, \\dots, K) $$
の下で、$Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{\\text{old}})$ を $A_{jk}$ に関して最大化する。

**ラグランジュ未定乗数法による導出**:
1. 各 $j$ に対する制約条件を取り入れたラグランジュ関数 $L(\\{A_{jk}\\}, \\{\\lambda_j\\})$ を定義する：
$$ L = \\sum_{n=2}^N \\sum_{j=1}^K \\sum_{k=1}^K \\xi(z_{n-1,j}, z_{nk}) \\ln A_{jk} + \\sum_{j=1}^K \\lambda_j \\left( 1 - \\sum_{k=1}^K A_{jk} \\right) $$
2. $A_{jk}$ に関して偏微分してゼロとおく：
$$ \\frac{\\partial L}{\\partial A_{jk}} = \\frac{\\sum_{n=2}^N \\xi(z_{n-1,j}, z_{nk})}{A_{jk}} - \\lambda_j = 0 $$
3. したがって：
$$ A_{jk} = \\frac{1}{\\lambda_j} \\sum_{n=2}^N \\xi(z_{n-1,j}, z_{nk}) $$
4. 制約 $\\sum_{k=1}^K A_{jk} = 1$ に代入して未定乗数 $\\lambda_j$ を決定する：
$$ \\sum_{k=1}^K A_{jk} = \\frac{1}{\\lambda_j} \\sum_{n=2}^N \\sum_{k=1}^K \\xi(z_{n-1,j}, z_{nk}) = 1 \\implies \\lambda_j = \\sum_{n=2}^N \\sum_{k=1}^K \\xi(z_{n-1,j}, z_{nk}) $$
5. ここで、事後結合確率の定義から $\\sum_{k=1}^K \\xi(z_{n-1,j}, z_{nk}) = \\gamma(z_{n-1,j})$ であるため：
$$ \\lambda_j = \\sum_{n=2}^N \\gamma(z_{n-1,j}) $$
6. これを代入すると、求める Baum-Welch M-step 更新公式が得られる：
$$ A_{jk} = \\frac{\\sum_{n=2}^N \\xi(z_{n-1,j}, z_{nk})}{\\sum_{n=2}^N \\gamma(z_{n-1,j})} $$

#### 穴埋め問題
1. 遷移確率行列の各行は確率分布であるため、制約条件は $\\text{[ (A) ]} = 1$ である。
2. ラグランジュ関数を $A_{jk}$ で偏微分すると、$A_{jk} \\propto \\text{[ (B) ]}$ となる。
3. 周辺化関係 $\\sum_{k=1}^K \\xi(z_{n-1,j}, z_{nk}) = \\text{[ (C) ]}$ を用いることで、分母が簡単化される。
*(解: A: $\\sum_{k=1}^K A_{jk}$, B: $\\sum_{n=2}^N \\xi(z_{n-1,j}, z_{nk})$, C: $\\gamma(z_{n-1,j})$)*
"""
    ex13_5_code = """# Exercise 13.5 数値検証: Baum-Welch Mステップ更新公式と数値的制約最適化の一致
import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
N = 10
K = 3

# モックの事後統計量 xi (N-1, K, K)
xi = np.random.dirichlet(np.ones(K*K), size=N-1).reshape(N-1, K, K)
gamma = np.sum(xi, axis=2)  # (N-1, K)

# 解析公式による A_jk の計算
xi_sum = np.sum(xi, axis=0)  # (K, K)
gamma_sum = np.sum(gamma, axis=0)  # (K,)
A_analytical = xi_sum / gamma_sum[:, None]

# 数値的最適化 (SLSQP) による解
A_numerical = np.zeros((K, K))
for j in range(K):
    def objective(a_row):
        # 負の対数期待尤度
        return -np.sum(xi_sum[j, :] * np.log(a_row + 1e-15))
    
    cons = ({'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0})
    bounds = [(1e-6, 1.0) for _ in range(K)]
    init = np.ones(K) / K
    res = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-10, 'maxiter': 200})
    A_numerical[j, :] = res.x

np.testing.assert_allclose(A_analytical, A_numerical, atol=1e-3)
np.testing.assert_allclose(np.sum(A_analytical, axis=1), np.ones(K), atol=1e-12)
print("Exercise 13.5 verified: Analytical Baum-Welch M-step matches constrained numerical optimization perfectly!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_5_md), nbf.v4.new_code_cell(ex13_5_code)])

    # Exercise 13.6
    ex13_6_md = """---
## Exercise 13.6: 初期状態確率ベクトル $\\pi_k$ の制約付き最尤推定公式の導出

### 問題の背景と数学的証明
初期潜在状態 $z_1$ の事前確率ベクトル $\\boldsymbol{\\pi} = (\\pi_1, \\dots, \\pi_K)^{\\mathrm{T}}$ は $\\sum_{k=1}^K \\pi_k = 1$ を満たす。
$Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{\\text{old}})$ の中で $\\boldsymbol{\\pi}$ に依存する項は第1項のみである：
$$ Q_\\pi = \\sum_{k=1}^K \\gamma(z_{1k}) \\ln \\pi_k $$
ここで $\\gamma(z_{1k}) = p(z_{1k}=1 | \\mathbf{X}, \\boldsymbol{\\theta}^{\\text{old}})$ である。

**ラグランジュ乗数法による導出**:
1. ラグランジュ関数：
$$ L(\\boldsymbol{\\pi}, \\lambda) = \\sum_{k=1}^K \\gamma(z_{1k}) \\ln \\pi_k + \\lambda \\left( 1 - \\sum_{k=1}^K \\pi_k \\right) $$
2. $\\pi_k$ で微分してゼロとおく：
$$ \\frac{\\partial L}{\\partial \\pi_k} = \\frac{\\gamma(z_{1k})}{\\pi_k} - \\lambda = 0 \\implies \\pi_k = \\frac{\\gamma(z_{1k})}{\\lambda} $$
3. 制約 $\\sum_k \\pi_k = 1$ より $\\lambda = \\sum_{k=1}^K \\gamma(z_{1k})$。
4. 事後周辺確率の正規化条件 $\\sum_{k=1}^K \\gamma(z_{1k}) = 1$ より、$\\lambda = 1$ となる。
5. したがって、更新公式は：
$$ \\pi_k = \\gamma(z_{1k}) $$
となる。

#### 穴埋め問題
1. $\\pi_k$ に関するラグランジュ乗数は、事後確率の総和より $\\lambda = \\text{[ (A) ]}$ となる。
2. したがって、M ステップにおける最適初期確率は、時刻 $1$ における $\\text{[ (B) ]}$ にそのまま一致する。
*(解: A: $1$, B: 事後周辺確率 $\\gamma(z_{1k})$)*
"""
    ex13_6_code = """# Exercise 13.6 数値検証: 初期確率 pi_k の更新式
gamma_1 = np.array([0.2, 0.5, 0.3])
# 解析解
pi_opt = gamma_1.copy()

# 数値最適化による確認
def obj_pi(pi):
    return -np.sum(gamma_1 * np.log(pi + 1e-15))

cons_pi = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0})
res_pi = minimize(obj_pi, np.ones(3)/3, method='SLSQP', bounds=[(1e-6, 1.0)]*3, constraints=cons_pi, options={'ftol': 1e-10, 'maxiter': 200})

np.testing.assert_allclose(pi_opt, res_pi.x, atol=1e-3)
print(f"Exercise 13.6 verified: pi_k = gamma(z_1k) correctly maximizes Q_pi: {pi_opt}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_6_md), nbf.v4.new_code_cell(ex13_6_code)])

    # Exercise 13.7
    ex13_7_md = """---
## Exercise 13.7: 多変量ガウス放出モデルにおける平均 $\\boldsymbol{\\mu}_k$ と共分散 $\\mathbf{\\Sigma}_k$ の M-step 更新式の導出

### 問題の背景と数学的証明
連続観測変数 $\\mathbf{x}_n \\in \\mathbb{R}^D$ に対する放出分布が多変量正規分布 $p(\\mathbf{x}_n | z_k) = \\mathcal{N}(\\mathbf{x}_n | \\boldsymbol{\\mu}_k, \\mathbf{\\Sigma}_k)$ の場合、
$Q$ 関数の放出確率項は：
$$ Q_\\phi = -\\frac{1}{2} \\sum_{n=1}^N \\sum_{k=1}^K \\gamma(z_{nk}) \\left( \\ln |\\mathbf{\\Sigma}_k| + (\\mathbf{x}_n - \\boldsymbol{\\mu}_k)^{\\mathrm{T}} \\mathbf{\\Sigma}_k^{-1} (\\mathbf{x}_n - \\boldsymbol{\\mu}_k) + D \\ln(2\\pi) \\right) $$
となる。

**パラメータ導出**:
1. **平均 $\\boldsymbol{\\mu}_k$ の最大化**:
$$ \\frac{\\partial Q_\\phi}{\\partial \\boldsymbol{\\mu}_k} = \\sum_{n=1}^N \\gamma(z_{nk}) \\mathbf{\\Sigma}_k^{-1} (\\mathbf{x}_n - \\boldsymbol{\\mu}_k) = \\mathbf{0} $$
両辺に左から $\\mathbf{\\Sigma}_k$ を掛けると：
$$ \\sum_{n=1}^N \\gamma(z_{nk}) (\\mathbf{x}_n - \\boldsymbol{\\mu}_k) = \\mathbf{0} \\implies \\boldsymbol{\\mu}_k = \\frac{\\sum_{n=1}^N \\gamma(z_{nk}) \\mathbf{x}_n}{\\sum_{n=1}^N \\gamma(z_{nk})} $$
2. **共分散 $\\mathbf{\\Sigma}_k$ の最大化**:
精度行列 $\\mathbf{\\Lambda}_k = \\mathbf{\\Sigma}_k^{-1}$ に関して微分する：
$$ \\frac{\\partial Q_\\phi}{\\partial \\mathbf{\\Lambda}_k} = \\frac{1}{2} \\sum_{n=1}^N \\gamma(z_{nk}) \\left( \\mathbf{\\Lambda}_k^{-1} - (\\mathbf{x}_n - \\boldsymbol{\\mu}_k)(\\mathbf{x}_n - \\boldsymbol{\\mu}_k)^{\\mathrm{T}} \\right) = \\mathbf{0} $$
したがって：
$$ \\mathbf{\\Sigma}_k = \\frac{\\sum_{n=1}^N \\gamma(z_{nk}) (\\mathbf{x}_n - \\boldsymbol{\\mu}_k)(\\mathbf{x}_n - \\boldsymbol{\\mu}_k)^{\\mathrm{T}}}{\\sum_{n=1}^N \\gamma(z_{nk})} $$
これは負担率 $\\gamma(z_{nk})$ を重みとする重み付き最尤推定量に一致する。

#### 穴埋め問題
1. $\\boldsymbol{\\mu}_k$ の推定量は、観測データ $\\mathbf{x}_n$ を負担率 $\\gamma(z_{nk})$ で重み付けした $\\text{[ (A) ]}$ である。
2. 共分散行列 $\\mathbf{\\Sigma}_k$ の推定量は、中心化二乗誤差の $\\text{[ (B) ]}$ となる。
*(解: A: 重み付き平均, B: 重み付き標本共分散)*
"""
    ex13_7_code = """# Exercise 13.7 数値検証: ガウス放出モデルの重み付き最尤推定解
D = 2
N = 20
X_data = np.random.randn(N, D)
gamma_k = np.random.uniform(0.1, 0.9, size=N)

# 解析公式による計算
N_k = np.sum(gamma_k)
mu_k_analytical = np.sum(gamma_k[:, None] * X_data, axis=0) / N_k
diff = X_data - mu_k_analytical
Sigma_k_analytical = (gamma_k[:, None, None] * (diff[:, :, None] @ diff[:, None, :])).sum(axis=0) / N_k

# 勾配が厳密にゼロであることを確認
grad_mu = np.sum(gamma_k[:, None] * np.linalg.solve(Sigma_k_analytical, (X_data - mu_k_analytical).T).T, axis=0)
np.testing.assert_allclose(grad_mu, np.zeros(D), atol=1e-10)

print(f"Exercise 13.7 verified: Gradient at analytical mu_k is zero: {grad_mu}")
print(f"Sigma_k condition number: {np.linalg.cond(Sigma_k_analytical):.4f}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_7_md), nbf.v4.new_code_cell(ex13_7_code)])

    # Exercise 13.8
    ex13_8_md = """---
## Exercise 13.8: 離散多項放出モデルにおける放出確率行列 $\\phi_{km}$ の M-step 更新式の導出

### 問題の背景と数学的証明
観測変数 $x_n$ が $M$ 種類の離散シンボル $x_n \\in \\{1, \\dots, M\\}$ をとる場合、放出確率は行列 $\\phi_{km} = p(x_n = m | z_{nk} = 1)$ で表現され、各状態 $k$ について $\\sum_{m=1}^M \\phi_{km} = 1$ を満たす。
$Q$ 関数の放出項は：
$$ Q_\\phi = \\sum_{n=1}^N \\sum_{k=1}^K \\sum_{m=1}^M \\gamma(z_{nk}) \\mathbb{I}(x_n = m) \\ln \\phi_{km} $$
である。

**ラグランジュ乗数法による導出**:
1. 各 $k$ の制約に対するラグランジュ関数：
$$ L = \\sum_{n=1}^N \\sum_{k=1}^K \\sum_{m=1}^M \\gamma(z_{nk}) \\mathbb{I}(x_n = m) \\ln \\phi_{km} + \\sum_{k=1}^K \\lambda_k \\left( 1 - \\sum_{m=1}^M \\phi_{km} \\right) $$
2. $\\phi_{km}$ で微分してゼロとおく：
$$ \\frac{\\partial L}{\\partial \\phi_{km}} = \\frac{\\sum_{n=1}^N \\gamma(z_{nk}) \\mathbb{I}(x_n = m)}{\\phi_{km}} - \\lambda_k = 0 \\implies \\phi_{km} = \\frac{\\sum_{n=1}^N \\gamma(z_{nk}) \\mathbb{I}(x_n = m)}{\\lambda_k} $$
3. 制約 $\\sum_{m=1}^M \\phi_{km} = 1$ より：
$$ \\lambda_k = \\sum_{m=1}^M \\sum_{n=1}^N \\gamma(z_{nk}) \\mathbb{I}(x_n = m) = \\sum_{n=1}^N \\gamma(z_{nk}) $$
4. したがって、求める M-step 更新式は：
$$ \\phi_{km} = \\frac{\\sum_{n=1}^N \\gamma(z_{nk}) \\mathbb{I}(x_n = m)}{\\sum_{n=1}^N \\gamma(z_{nk})} $$
となる。

#### 穴埋め問題
1. 離散放出モデルでは、シンボル $m$ が観測された時刻の負担率の和を、全時刻の負担率の和で $\\text{[ (A) ]}$ する。
2. これにより得られた推定量は、各状態における離散分布の $\\text{[ (B) ]}$ に一致する。
*(解: A: 除算 (正規化), B: 最尤推定量)*
"""
    ex13_8_code = """# Exercise 13.8 数値検証: 離散多項放出モデルの M ステップ更新
N_samples = 30
M_symbols = 4
K_states = 2
obs = np.random.choice(M_symbols, size=N_samples)
gamma_mat = np.random.dirichlet(np.ones(K_states), size=N_samples) # (N, K)

phi_analytical = np.zeros((K_states, M_symbols))
for k in range(K_states):
    for m in range(M_symbols):
        phi_analytical[k, m] = np.sum(gamma_mat[obs == m, k]) / np.sum(gamma_mat[:, k])

# 行和が 1 であることを検証
np.testing.assert_allclose(np.sum(phi_analytical, axis=1), np.ones(K_states), atol=1e-12)
print("Exercise 13.8 verified: Discrete emission probabilities sum strictly to 1 across all states:")
print(phi_analytical)
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_8_md), nbf.v4.new_code_cell(ex13_8_code)])

    # Exercise 13.9
    ex13_9_md = """---
## Exercise 13.9: Forward-Backward アルゴリズム：前向き変数 $\\alpha(z_n)$ の再帰漸化式の導出

### 問題の背景と数学的証明
前向き変数 $\\alpha(z_n)$ は、時刻 $n$ までの観測系列と時刻 $n$ の潜在状態の結合確率として定義される（(13.34)式）：
$$ \\alpha(z_n) \\equiv p(\\mathbf{x}_1, \\dots, \\mathbf{x}_n, z_n) $$
本問では、条件付き独立性を利用して、前向き再帰漸化式（(13.36)式）：
$$ \\alpha(z_n) = p(\\mathbf{x}_n | z_n) \\sum_{z_{n-1}} \\alpha(z_{n-1}) p(z_n | z_{n-1}) $$
を代数的に導出する。

**代数的導出**:
1. 定義より、結合確率を積の規則で展開する：
$$ p(\\mathbf{x}_1, \\dots, \\mathbf{x}_n, z_n) = p(\\mathbf{x}_n | \\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_n) p(\\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_n) $$
2. HMM の条件付き独立性より、$\\mathbf{x}_n$ は $z_n$ のみが与えられれば過去の観測値とは独立であるため：
$$ p(\\mathbf{x}_n | \\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_n) = p(\\mathbf{x}_n | z_n) $$
3. 第2項 $p(\\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_n)$ に直前の潜在状態 $z_{n-1}$ を導入し、周辺化の和をとる：
$$ p(\\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_n) = \\sum_{z_{n-1}} p(\\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_{n-1}, z_n) $$
4. ここで結合確率を条件付き確率に分解する：
$$ p(\\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_{n-1}, z_n) = p(z_n | \\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_{n-1}) p(\\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_{n-1}) $$
5. マルコフ性より $p(z_n | \\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_{n-1}) = p(z_n | z_{n-1})$、また定義より $p(\\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}, z_{n-1}) = \\alpha(z_{n-1})$ である。
6. 以上をまとめると：
$$ \\alpha(z_n) = p(\\mathbf{x}_n | z_n) \\sum_{z_{n-1}} \\alpha(z_{n-1}) p(z_n | z_{n-1}) $$
初期条件は $\\alpha(z_1) = p(\\mathbf{x}_1, z_1) = p(z_1) p(\\mathbf{x}_1 | z_1) = \\pi_k p(\\mathbf{x}_1 | \\boldsymbol{\\phi}_k)$ である。

#### 穴埋め問題
1. $\\mathbf{x}_n$ の放出確率は、時刻 $n$ の潜在状態 $z_n$ のみに依存するため $\\text{[ (A) ]}$ と簡略化される。
2. 直前状態の同時確率は、前向き変数 $\\alpha(z_{n-1})$ と遷移行列 $A_{j k} = \\text{[ (B) ]}$ の積の和となる。
3. したがって、各ステップの計算量は状態数 $K$ に対し $\\text{[ (C) ]}$ で実行できる。
*(解: A: $p(\\mathbf{x}_n | z_n)$, B: $p(z_n | z_{n-1})$, C: $\\mathcal{O}(K^2)$)*
"""
    ex13_9_code = """# Exercise 13.9 数値検証: 前向き再帰法と直接結合確率周辺化の一致
N_steps = 4
K_states = 2
pi_vec = np.array([0.6, 0.4])
A_mat = np.array([[0.7, 0.3],
                  [0.4, 0.6]])
B_mat = np.array([[0.8, 0.2],
                  [0.1, 0.9]])
obs_seq = [0, 1, 0, 1]

# 1. 前向きアルゴリズムによる alpha の計算
alpha = np.zeros((N_steps, K_states))
alpha[0, :] = pi_vec * B_mat[:, obs_seq[0]]
for n in range(1, N_steps):
    alpha[n, :] = B_mat[:, obs_seq[n]] * (alpha[n-1, :] @ A_mat)

# 2. ブルートフォース全状態同時確率計算による alpha(z_N) の算出
# p(x_1, ..., x_N, z_N) = sum_{z_1...z_{N-1}} p(x_1...x_N, z_1...z_N)
alpha_brute = np.zeros(K_states)
for z1 in range(K_states):
    for z2 in range(K_states):
        for z3 in range(K_states):
            for z4 in range(K_states):
                p_joint = (pi_vec[z1] * B_mat[z1, obs_seq[0]] *
                           A_mat[z1, z2] * B_mat[z2, obs_seq[1]] *
                           A_mat[z2, z3] * B_mat[z3, obs_seq[2]] *
                           A_mat[z3, z4] * B_mat[z4, obs_seq[3]])
                alpha_brute[z4] += p_joint

np.testing.assert_allclose(alpha[-1, :], alpha_brute, atol=1e-12)
print("Exercise 13.9 verified: Forward recursion alpha strictly matches full brute force marginalization!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_9_md), nbf.v4.new_code_cell(ex13_9_code)])

    # Exercise 13.10
    ex13_10_md = """---
## Exercise 13.10: Forward-Backward アルゴリズム：後ろ向き変数 $\\beta(z_n)$ の再帰漸化式の導出

### 問題の背景と数学的証明
後ろ向き変数 $\\beta(z_n)$ は、現在の潜在状態 $z_n$ が与えられたときの未来の観測系列の条件付き確率として定義される（(13.44)式）：
$$ \\beta(z_n) \\equiv p(\\mathbf{x}_{n+1}, \\dots, \\mathbf{x}_N | z_n) $$
本問では、後ろ向き再帰漸化式（(13.47)式）：
$$ \\beta(z_n) = \\sum_{z_{n+1}} \\beta(z_{n+1}) p(\\mathbf{x}_{n+1} | z_{n+1}) p(z_{n+1} | z_n) $$
および終端境界条件 $\\beta(z_N) = 1$ を代数的に証明する。

**代数的導出**:
1. 条件付き確率の定義より、未来の潜在状態 $z_{n+1}$ を導入して周辺化する：
$$ p(\\mathbf{x}_{n+1}, \\dots, \\mathbf{x}_N | z_n) = \\sum_{z_{n+1}} p(\\mathbf{x}_{n+1}, \\dots, \\mathbf{x}_N, z_{n+1} | z_n) $$
2. 被和数を積の規則で展開する：
$$ p(\\mathbf{x}_{n+1}, \\dots, \\mathbf{x}_N, z_{n+1} | z_n) = p(\\mathbf{x}_{n+2}, \\dots, \\mathbf{x}_N | \\mathbf{x}_{n+1}, z_{n+1}, z_n) p(\\mathbf{x}_{n+1} | z_{n+1}, z_n) p(z_{n+1} | z_n) $$
3. d-分離と条件付き独立性より：
   - $z_{n+1}$ が与えられれば、未来の観測 $\\mathbf{x}_{n+2}, \\dots, \\mathbf{x}_N$ は $\\mathbf{x}_{n+1}$ および $z_n$ と独立である：$p(\\mathbf{x}_{n+2}, \\dots, \\mathbf{x}_N | z_{n+1}) = \\beta(z_{n+1})$
   - $z_{n+1}$ が与えられれば、$\\mathbf{x}_{n+1}$ は $z_n$ と独立である：$p(\\mathbf{x}_{n+1} | z_{n+1}, z_n) = p(\\mathbf{x}_{n+1} | z_{n+1})$
4. これらを代入すると：
$$ \\beta(z_n) = \\sum_{z_{n+1}} \\beta(z_{n+1}) p(\\mathbf{x}_{n+1} | z_{n+1}) p(z_{n+1} | z_n) $$
5. 終端時刻 $N$ においては、未来の観測値が存在しない空集合であるため：
$$ p(\\emptyset | z_N) = 1 \\implies \\beta(z_N) = 1 $$
が自然に満たされる。

#### 穴埋め問題
1. 後ろ向き変数 $\\beta(z_n)$ は、未来の観測値が与えられた条件付き確率ではなく、状態 $z_n$ が与えられたときの未来の観測値の $\\text{[ (A) ]}$ である。
2. 漸化式において、未来側からのメッセージに $\\text{[ (B) ]}$ と遷移確率を掛けて状態 $z_{n+1}$ について和をとる。
3. 終端境界条件は $\\beta(z_N) = \\text{[ (C) ]}$ である。
*(解: A: 条件付き尤度, B: 放出確率 $p(\\mathbf{x}_{n+1}|z_{n+1})$, C: $1$)*
"""
    ex13_10_code = """# Exercise 13.10 数値検証: 後ろ向き再帰法と直接条件付き確率の一致
# 後ろ向きアルゴリズムによる beta の計算
beta = np.zeros((N_steps, K_states))
beta[-1, :] = 1.0  # 終端境界条件

for n in range(N_steps - 2, -1, -1):
    beta[n, :] = A_mat @ (beta[n+1, :] * B_mat[:, obs_seq[n+1]])

# ブルートフォースによる直接条件付き確率 p(x_{n+1}...x_N | z_n) の計算 (n=0: p(x_2, x_3, x_4 | z_1))
beta_brute_0 = np.zeros(K_states)
for z1 in range(K_states):
    denom = pi_vec[z1] * B_mat[z1, obs_seq[0]]
    num = 0.0
    for z2 in range(K_states):
        for z3 in range(K_states):
            for z4 in range(K_states):
                num += (pi_vec[z1] * B_mat[z1, obs_seq[0]] *
                        A_mat[z1, z2] * B_mat[z2, obs_seq[1]] *
                        A_mat[z2, z3] * B_mat[z3, obs_seq[2]] *
                        A_mat[z3, z4] * B_mat[z4, obs_seq[3]])
    beta_brute_0[z1] = num / denom

np.testing.assert_allclose(beta[0, :], beta_brute_0, atol=1e-12)
print(f"Exercise 13.10 verified: Backward beta[0] {beta[0, :]} matches brute force condition: {beta_brute_0}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_10_md), nbf.v4.new_code_cell(ex13_10_code)])

    return cells
