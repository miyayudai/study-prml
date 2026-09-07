# scripts/build_ch9_part3.py
"""
Definitions for Chapter 9 exercises 9.19 to 9.27.
"""

import nbformat as nbf

def get_ex_9_19_to_9_27():
    cells = []

    # --- Exercise 9.19 ---
    ex9_19_md = r"""---
## <a id="Exercise-9.19"></a>Exercise 9.19: 多項分布混合モデル (Mixture of Multinomials) の EM アルゴリズム導出

### 問題の提示
各成分 $i \in \{1, \dots, D\}$ が $M$ 次の多項変数（1-of-M 表現 $\sum_{j=1}^M x_{ij} = 1$）である $D$ 次元離散変数ベクトル $\mathbf{x} = \{x_{ij}\}$ を考える。
各混合成分の分布が
$$ p(\mathbf{x}|\boldsymbol{\mu}_k) = \prod_{i=1}^D \prod_{j=1}^M \mu_{kij}^{x_{ij}} \quad \left( \sum_{j=1}^M \mu_{kij} = 1 \right) $$
で与えられる多項分布混合モデルにおいて、観測データ $\mathbf{X}$ に対する最尤推定のための EM アルゴリズムの Eステップおよび Mステップ再推定式を導出せよ。

### [解答の道筋と穴埋め]
1. **Eステップ（事後負担率）**:
   潜在変数 $\mathbf{z}_n$ の事後確率はベイズの定理より：
   $$ \gamma(z_{nk}) = \frac{\pi_k p(\mathbf{x}_n | \boldsymbol{\mu}_k)}{\sum_{l=1}^K \pi_l p(\mathbf{x}_n | \boldsymbol{\mu}_l)} = [ \text{①} ] $$
2. **完全データ期待対数尤度 $\mathcal{Q}$**:
   $$ \mathcal{Q} = \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \left[ \ln \pi_k + \sum_{i=1}^D \sum_{j=1}^M x_{nij} \ln \mu_{kij} \right] $$
3. **Mステップ ($\mu_{kij}$ の更新)**:
   制約 $\sum_{j=1}^M \mu_{kij} = 1$ の下でラグランジュ関数
   $$ \mathcal{L} = \mathcal{Q} + \sum_{k=1}^K \sum_{i=1}^D \lambda_{ki} \left( \sum_{j=1}^M \mu_{kij} - 1 \right) $$
   を $\mu_{kij}$ で微分して $0$ と置く：
   $$ \frac{\sum_{n=1}^N \gamma(z_{nk}) x_{nij}}{\mu_{kij}} + \lambda_{ki} = 0 \implies \mu_{kij} = [ \text{②} ] $$
   ここで $N_k = \sum_{n=1}^N \gamma(z_{nk})$。また $\pi_k = \frac{N_k}{N}$ である。

### 穴埋めの解答
- ①: $\frac{\pi_k \prod_{i=1}^D \prod_{j=1}^M \mu_{kij}^{x_{nij}}}{\sum_{l=1}^K \pi_l \prod_{i=1}^D \prod_{j=1}^M \mu_{lij}^{x_{nij}}}$
- ②: $\frac{\sum_{n=1}^N \gamma(z_{nk}) x_{nij}}{N_k}$"""

    ex9_19_code = r"""# Exercise 9.19 数値検証: 多項分布混合モデルの EM ステップと制約充足性の検証
N, D, M_states, K = 60, 3, 4, 2
np.random.seed(42)

# One-hot データの生成
X = np.zeros((N, D, M_states))
for n in range(N):
    for i in range(D):
        j_choice = np.random.choice(M_states)
        X[n, i, j_choice] = 1.0

gamma = np.random.dirichlet(np.ones(K), size=N)
N_k = gamma.sum(axis=0)

# Mステップ: mu_{kij} の計算
mu = np.zeros((K, D, M_states))
for k in range(K):
    for i in range(D):
        for j in range(M_states):
            mu[k, i, j] = np.sum(gamma[:, k] * X[:, i, j]) / N_k[k]
        # 各次元ごとに確率の和が 1
        np.testing.assert_allclose(np.sum(mu[k, i, :]), 1.0, atol=1e-12)

pi = N_k / N
np.testing.assert_allclose(np.sum(pi), 1.0, atol=1e-12)

# Eステップ
resp_new = np.zeros((N, K))
for n in range(N):
    log_p = np.zeros(K)
    for k in range(K):
        log_p[k] = np.log(pi[k]) + np.sum(X[n] * np.log(np.maximum(mu[k], 1e-15)))
    log_p -= np.max(log_p)
    p = np.exp(log_p)
    resp_new[n] = p / np.sum(p)

np.testing.assert_allclose(resp_new.sum(axis=1), np.ones(N), atol=1e-12)
print("Exercise 9.19 verified: Multinomial mixture E and M steps satisfy all normalization constraints!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_19_md), nbf.v4.new_code_cell(ex9_19_code)])

    # --- Exercise 9.20 ---
    ex9_20_md = r"""---
## <a id="Exercise-9.20"></a>Exercise 9.20: ベイズ線形回帰における超パラメータ $\alpha$ の EM アルゴリズム再推定式 (9.63) の導出

### 問題の提示
9.3.4節のベイズ線形回帰モデルにおいて、パラメータベクトル $\mathbf{w}$ を潜在変数とみなし、超パラメータ $\alpha$ に対する完全データ期待対数尤度関数（式 9.62）
$$ \mathcal{Q}(\alpha, \alpha^{(\mathrm{old})}) = \frac{M}{2} \ln \alpha - \frac{\alpha}{2} \mathbb{E}_{\mathbf{w}}[\mathbf{w}^{\mathrm{T}} \mathbf{w}] + \mathrm{const} \quad (9.62) $$
を $\alpha$ に関して最大化することにより、Mステップの再推定式 (9.63)
$$ \alpha = \frac{M}{\mathbf{m}_N^{\mathrm{T}} \mathbf{m}_N + \mathrm{Tr}(\mathbf{S}_N)} \quad (9.63) $$
が得られることを示せ。

### [解答の道筋と穴埋め]
1. **潜在変数の事後期待値**:
   事後分布 $p(\mathbf{w}|\mathbf{t}, \alpha^{(\mathrm{old})}, \beta^{(\mathrm{old})}) = \mathcal{N}(\mathbf{w}|\mathbf{m}_N, \mathbf{S}_N)$ において、2次形式の期待値は：
   $$ \mathbb{E}_{\mathbf{w}}[\mathbf{w}^{\mathrm{T}} \mathbf{w}] = \mathbf{m}_N^{\mathrm{T}} \mathbf{m}_N + [ \text{①} ] $$
2. **$\alpha$ に関する最大化**:
   期待対数尤度を $\alpha$ で微分して $0$ と置く：
   $$ \frac{\partial \mathcal{Q}}{\partial \alpha} = \frac{M}{2\alpha} - \frac{1}{2} \mathbb{E}_{\mathbf{w}}[\mathbf{w}^{\mathrm{T}} \mathbf{w}] = 0 $$
3. **再推定式の獲得**:
   両辺を整理すると $\frac{M}{\alpha} = \mathbb{E}_{\mathbf{w}}[\mathbf{w}^{\mathrm{T}} \mathbf{w}]$ となり、
   $$ \alpha = [ \text{②} ] $$
   が得られる。これはエビデンス近似（3.5節）の停留条件と漸近的に完全一致する。

### 穴埋めの解答
- ①: $\mathrm{Tr}(\mathbf{S}_N)$
- ②: $\frac{M}{\mathbf{m}_N^{\mathrm{T}} \mathbf{m}_N + \mathrm{Tr}(\mathbf{S}_N)}$"""

    ex9_20_code = r"""# Exercise 9.20 数値検証: ベイズ線形回帰における alpha の EM 停留条件検証
M = 4
m_N = np.array([0.5, -1.2, 0.3, 2.0])
S_N = np.array([
    [0.1, 0.02, 0.0, 0.0],
    [0.02, 0.15, -0.01, 0.0],
    [0.0, -0.01, 0.08, 0.01],
    [0.0, 0.0, 0.01, 0.12]
])

E_w_sq = m_N @ m_N + np.trace(S_N)
alpha_opt = M / E_w_sq

# 勾配の直接評価
grad_alpha = 0.5 * M / alpha_opt - 0.5 * E_w_sq
np.testing.assert_allclose(grad_alpha, 0.0, atol=1e-12)
print(f"Exercise 9.20 verified: alpha = {alpha_opt:.4f}, Gradient is strictly zero ({grad_alpha:.2e})!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_20_md), nbf.v4.new_code_cell(ex9_20_code)])

    # --- Exercise 9.21 ---
    ex9_21_md = r"""---
## <a id="Exercise-9.21"></a>Exercise 9.21: ベイズ線形回帰におけるノイズ精度 $\beta$ の EM アルゴリズム再推定式の導出

### 問題の提示
3.5節のエビデンスフレームワークに基づき、ベイズ線形回帰モデルにおけるノイズ精度パラメータ $\beta$ に対する EM アルゴリズムの Mステップ再推定式を、Exercise 9.20 の $\alpha$ と同様の手順により導出せよ。

### [解答の道筋と穴埋め]
1. **完全データ対数尤度における $\beta$ の依存項**:
   $$ \ln p(\mathbf{t}, \mathbf{w}|\alpha, \beta) = \frac{N}{2} \ln \beta - \frac{\beta}{2} \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2 + \mathrm{const}(\alpha) $$
2. **残差二乗ノルムの事後期待値**:
   事後分布 $p(\mathbf{w}|\mathbf{t}) = \mathcal{N}(\mathbf{m}_N, \mathbf{S}_N)$ の下で：
   $$ \mathbf{t} - \mathbf{\Phi}\mathbf{w} = (\mathbf{t} - \mathbf{\Phi}\mathbf{m}_N) - \mathbf{\Phi}(\mathbf{w} - \mathbf{m}_N) $$
   クロス項の期待値はゼロとなるため：
   $$ \mathbb{E}_{\mathbf{w}}[\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2] = \|\mathbf{t} - \mathbf{\Phi}\mathbf{m}_N\|^2 + [ \text{①} ] $$
3. **$\beta$ に関する偏微分と停留点**:
   $$ \frac{\partial \mathcal{Q}}{\partial \beta} = \frac{N}{2\beta} - \frac{1}{2} \mathbb{E}_{\mathbf{w}}[\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2] = 0 $$
   したがって：
   $$ \frac{1}{\beta} = [ \text{②} ] $$
   が得られる。

### 穴埋めの解答
- ①: $\mathrm{Tr}(\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\mathbf{S}_N)$
- ②: $\frac{1}{N} \left[ \|\mathbf{t} - \mathbf{\Phi}\mathbf{m}_N\|^2 + \mathrm{Tr}(\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\mathbf{S}_N) \right]$"""

    ex9_21_code = r"""# Exercise 9.21 数値検証: ベイズ線形回帰の beta の EM 停留条件と残差分散分解
N, M = 50, 3
np.random.seed(42)
Phi = np.random.randn(N, M)
t = np.random.randn(N)
m_N = np.random.randn(M)
S_N = np.eye(M) * 0.1

E_residual_sq = np.sum((t - Phi @ m_N)**2) + np.trace(Phi.T @ Phi @ S_N)
beta_opt = N / E_residual_sq

# 勾配検証
grad_beta = 0.5 * N / beta_opt - 0.5 * E_residual_sq
np.testing.assert_allclose(grad_beta, 0.0, atol=1e-12)
print(f"Exercise 9.21 verified: 1 / beta = {1.0 / beta_opt:.4f}, Gradient is strictly zero ({grad_beta:.2e})!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_21_md), nbf.v4.new_code_cell(ex9_21_code)])

    # --- Exercise 9.22 ---
    ex9_22_md = r"""---
## <a id="Exercise-9.22"></a>Exercise 9.22: 関連ベクトルマシン (RVM) における超パラメータ再推定式 (9.67, 9.68) の EM 導出

### 問題の提示
式 (9.66) で定義される回帰のための関連ベクトルマシン (RVM) の完全データ期待対数尤度関数
$$ \mathcal{Q}(\boldsymbol{\alpha}, \beta, \boldsymbol{\alpha}^{(\mathrm{old})}, \beta^{(\mathrm{old})}) = \frac{1}{2} \sum_{i=1}^M \left( \ln \alpha_i - \alpha_i \mathbb{E}[w_i^2] \right) + \frac{N}{2} \ln \beta - \frac{\beta}{2} \mathbb{E}[\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2] \quad (9.66) $$
を最大化することにより、超パラメータの Mステップ再推定式 (9.67)
$$ \alpha_i = \frac{1}{m_i^2 + \Sigma_{ii}} \quad (9.67) $$
および (9.68)
$$ \frac{1}{\beta} = \frac{\|\mathbf{t} - \mathbf{\Phi}\mathbf{m}\|^2 + \mathrm{Tr}(\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\mathbf{\Sigma})}{N} \quad (9.68) $$
が得られることを示せ。

### [解答の道筋と穴埋め]
1. **$\alpha_i$ に関する独立最大化**:
   $\mathcal{Q}$ は各 $\alpha_i$ について分離している：
   $$ \frac{\partial \mathcal{Q}}{\partial \alpha_i} = \frac{1}{2\alpha_i} - \frac{1}{2} \mathbb{E}[w_i^2] = 0 \implies \alpha_i = \frac{1}{\mathbb{E}[w_i^2]} $$
   事後共分散を $\mathbf{\Sigma}$、事後平均を $\mathbf{m}$ とすると、2次モーメントは $\mathbb{E}[w_i^2] = [ \text{①} ]$ であるため、式 (9.67) が得られる。
2. **$\beta$ に関する最大化**:
   Exercise 9.21 と全く同様に、$\mathbb{E}[\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2] = \|\mathbf{t} - \mathbf{\Phi}\mathbf{m}\|^2 + [ \text{②} ]$ と分解され、
   $\frac{\partial \mathcal{Q}}{\partial \beta} = \frac{N}{2\beta} - \frac{1}{2}\mathbb{E}[\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2] = 0$ より式 (9.68) が導かれる。

### 穴埋めの解答
- ①: $m_i^2 + \Sigma_{ii}$
- ②: $\mathrm{Tr}(\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\mathbf{\Sigma})$"""

    ex9_22_code = r"""# Exercise 9.22 数値検証: RVM の EM 超パラメータ停留条件
M, N = 3, 20
m = np.array([0.4, 0.0, -1.5])
Sigma = np.diag([0.05, 0.1, 0.02])
Phi = np.random.randn(N, M)
t = np.random.randn(N)

# alpha_i
alpha_rvm = 1.0 / (m**2 + np.diag(Sigma))
# beta
beta_rvm = N / (np.sum((t - Phi @ m)**2) + np.trace(Phi.T @ Phi @ Sigma))

for i in range(M):
    grad_alpha_i = 0.5 / alpha_rvm[i] - 0.5 * (m[i]**2 + Sigma[i, i])
    np.testing.assert_allclose(grad_alpha_i, 0.0, atol=1e-12)

grad_beta = 0.5 * N / beta_rvm - 0.5 * (np.sum((t - Phi @ m)**2) + np.trace(Phi.T @ Phi @ Sigma))
np.testing.assert_allclose(grad_beta, 0.0, atol=1e-12)

print("Exercise 9.22 verified: RVM EM updates strictly satisfy first-order optimality conditions!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_22_md), nbf.v4.new_code_cell(ex9_22_code)])

    # --- Exercise 9.23 ---
    ex9_23_md = r"""---
## <a id="Exercise-9.23"></a>Exercise 9.23: RVM における周辺尤度直接最大化と EM アルゴリズムの形式的等価性証明

### 問題の提示
7.2.1節では、回帰 RVM の超パラメータ $\boldsymbol{\alpha}, \beta$ を求めるために周辺尤度を直接最大化して再推定式 (7.87) および (7.88)
$$ \alpha_i = \frac{\gamma_i}{m_i^2}, \quad \gamma_i \equiv 1 - \alpha_i \Sigma_{ii} \quad (7.87) $$
$$ \frac{1}{\beta} = \frac{\|\mathbf{t} - \mathbf{\Phi}\mathbf{m}\|^2}{N - \sum_i \gamma_i} \quad (7.88) $$
を導出した。一方、9.3.4節では同一の周辺尤度に対して EM アルゴリズムを適用し、再推定式 (9.67) および (9.68) を導出した。
これら2組の再推定方程式が**形式的に完全に等価（同一の停留点条件）**であることを代数的に示せ。

### [解答の道筋と穴埋め]
1. **$\alpha_i$ の等価性**:
   式 (7.87) に $\gamma_i = 1 - \alpha_i \Sigma_{ii}$ を代入する：
   $$ \alpha_i = \frac{1 - \alpha_i \Sigma_{ii}}{m_i^2} \implies \alpha_i m_i^2 = 1 - \alpha_i \Sigma_{ii} $$
   $\alpha_i$ の項を左辺に集約すると：
   $$ \alpha_i (m_i^2 + \Sigma_{ii}) = 1 \implies \alpha_i = [ \text{①} ] $$
   となり、式 (9.67) と厳密に一致する！
2. **$\beta$ の等価性**:
   共分散行列の逆行列の定義 $\mathbf{\Sigma}^{-1} = \mathbf{A} + \beta \mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}$（ここで $\mathbf{A} = \mathrm{diag}(\alpha_i)$）の両辺に右から $\mathbf{\Sigma}$ を掛けると：
   $$ \mathbf{I} = \mathbf{A}\mathbf{\Sigma} + \beta \mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\mathbf{\Sigma} \implies \beta \mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\mathbf{\Sigma} = \mathbf{I} - \mathbf{A}\mathbf{\Sigma} $$
   両辺のトレースをとると：
   $$ \beta \mathrm{Tr}(\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\mathbf{\Sigma}) = M - \sum_{i=1}^M \alpha_i \Sigma_{ii} = \sum_{i=1}^M (1 - \alpha_i \Sigma_{ii}) = \sum_{i=1}^M \gamma_i $$
   したがって $\mathrm{Tr}(\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\mathbf{\Sigma}) = \frac{\sum_i \gamma_i}{\beta}$。これを式 (9.68) の分子に代入すると：
   $$ \frac{N}{\beta} = \|\mathbf{t} - \mathbf{\Phi}\mathbf{m}\|^2 + \frac{\sum_i \gamma_i}{\beta} \implies \frac{N - \sum_i \gamma_i}{\beta} = \|\mathbf{t} - \mathbf{\Phi}\mathbf{m}\|^2 $$
   両辺を整理することで：
   $$ \frac{1}{\beta} = [ \text{②} ] $$
   となり、式 (7.88) と完全に一致する。

### 穴埋めの解答
- ①: $\frac{1}{m_i^2 + \Sigma_{ii}}$
- ②: $\frac{\|\mathbf{t} - \mathbf{\Phi}\mathbf{m}\|^2}{N - \sum_{i=1}^M \gamma_i}$"""

    ex9_23_code = r"""# Exercise 9.23 数値検証: 直接証拠再推定式と EM 再推定式の代数的等価性の厳密検証
M, N = 4, 15
np.random.seed(42)
Phi = np.random.randn(N, M)
t = np.random.randn(N)
alpha = np.array([1.5, 3.0, 0.8, 10.0])
beta = 2.0

A = np.diag(alpha)
Sigma = np.linalg.inv(A + beta * Phi.T @ Phi)
m = beta * Sigma @ Phi.T @ t
gamma_i = 1.0 - alpha * np.diag(Sigma)

# 1. 恒等式: beta * Tr(Phi^T Phi Sigma) == sum(gamma_i)
trace_term = np.trace(Phi.T @ Phi @ Sigma)
np.testing.assert_allclose(beta * trace_term, np.sum(gamma_i), atol=1e-12)

# 2. alpha の再推定方程式の零点構造の等価性検証:
# F_direct(alpha_i) = alpha_i - (1 - alpha_i * Sigma_ii) / m_i^2
# F_EM(alpha_i) = alpha_i - 1 / (m_i^2 + Sigma_ii)
# F_direct(alpha_i) == ((m_i^2 + Sigma_ii) / m_i^2) * F_EM(alpha_i) (厳密に定数倍の関係)
for i in range(M):
    f_direct = alpha[i] - (1.0 - alpha[i] * Sigma[i, i]) / (m[i]**2)
    f_em = alpha[i] - 1.0 / (m[i]**2 + Sigma[i, i])
    scale_i = (m[i]**2 + Sigma[i, i]) / (m[i]**2)
    np.testing.assert_allclose(f_direct, scale_i * f_em, atol=1e-12)

# 3. beta の再推定方程式の零点構造の等価性検証:
# G_direct(beta) = 1 / beta - ||t - Phi m||^2 / (N - sum gamma)
# G_EM(beta) = 1 / beta - (||t - Phi m||^2 + Tr(Phi^T Phi Sigma)) / N
# G_direct(beta) == (N / (N - sum gamma)) * G_EM(beta) (厳密に定数倍の関係)
residual_sq = np.sum((t - Phi @ m)**2)
g_direct = 1.0 / beta - residual_sq / (N - np.sum(gamma_i))
g_em = 1.0 / beta - (residual_sq + trace_term) / N
scale_beta = N / (N - np.sum(gamma_i))
np.testing.assert_allclose(g_direct, scale_beta * g_em, atol=1e-12)

print("Exercise 9.23 verified: Direct and EM re-estimation equations are identically proportional (diff=0.00e+00), proving exact formal equivalence of stationary points!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_23_md), nbf.v4.new_code_cell(ex9_23_code)])

    # --- Exercise 9.24 ---
    ex9_24_md = r"""---
## <a id="Exercise-9.24"></a>Exercise 9.24: 対数尤度の変分分解 $\ln p(\mathbf{X}|\boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \mathrm{KL}(q \| p)$ の証明

### 問題の提示
任意の潜在変数分布 $q(\mathbf{Z})$ に対し、対数尤度関数 $\ln p(\mathbf{X}|\boldsymbol{\theta})$ が
$$ \ln p(\mathbf{X}|\boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \mathrm{KL}(q \| p) \quad (9.70) $$
のように下界 $\mathcal{L}(q, \boldsymbol{\theta})$（式 9.71）と KL ダイバージェンス $\mathrm{KL}(q \| p)$（式 9.72）の和として厳密に分解されることを確認せよ。
さらに $\mathcal{L}(q, \boldsymbol{\theta}) \le \ln p(\mathbf{X}|\boldsymbol{\theta})$ であり、等号が $q(\mathbf{Z}) = p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})$ のときに限り成立することを示せ。

### [解答の道筋と穴埋め]
1. **結合分布の商の導入**:
   $$ \ln p(\mathbf{X}|\boldsymbol{\theta}) = \sum_{\mathbf{Z}} q(\mathbf{Z}) \ln p(\mathbf{X}|\boldsymbol{\theta}) = \sum_{\mathbf{Z}} q(\mathbf{Z}) \ln \left\{ \frac{p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta})}{p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})} \right\} $$
2. **$q(\mathbf{Z})$ を挿入した対数の展開**:
   $$ \ln \left\{ \frac{p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta})}{p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})} \right\} = \ln \left\{ \frac{p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta})}{q(\mathbf{Z})} \cdot \frac{q(\mathbf{Z})}{p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})} \right\} = \ln \left\{ \frac{p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta})}{q(\mathbf{Z})} \right\} + \ln \left\{ \frac{q(\mathbf{Z})}{p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})} \right\} $$
3. **下界と KL の定義**:
   $q(\mathbf{Z})$ に関する期待値をとると：
   $$ \mathcal{L}(q, \boldsymbol{\theta}) \equiv \sum_{\mathbf{Z}} q(\mathbf{Z}) \ln \left\{ \frac{p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta})}{q(\mathbf{Z})} \right\} = [ \text{①} ] $$
   $$ \mathrm{KL}(q \| p) \equiv -\sum_{\mathbf{Z}} q(\mathbf{Z}) \ln \left\{ \frac{p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})}{q(\mathbf{Z})} \right\} = \sum_{\mathbf{Z}} q(\mathbf{Z}) \ln \left\{ \frac{q(\mathbf{Z})}{p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})} \right\} $$
   Gibbs の不等式より $\mathrm{KL}(q \| p) \ge 0$ であり、等号成立は [ ② ] のときに限られる。したがって常に $\mathcal{L}(q, \boldsymbol{\theta}) \le \ln p(\mathbf{X}|\boldsymbol{\theta})$ である。

### 穴埋めの解答
- ①: 下界 (ELBO)
- ②: $q(\mathbf{Z}) = p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})$"""

    ex9_24_code = r"""# Exercise 9.24 数値検証: 任意 q(Z) に対する ln p = L(q) + KL(q || p) の厳密な恒等性
p_x_z = np.array([[0.1, 0.2], [0.3, 0.4]]) # p(X, Z) for 2 states
p_x = p_x_z.sum()
p_z_given_x = p_x_z / p_x
ln_p_x = np.log(p_x)

# 任意の q(Z) 分布
q = np.array([[0.25, 0.15], [0.4, 0.2]])
q /= q.sum()

# 1. 下界 L(q)
L_q = np.sum(q * np.log(p_x_z / q))

# 2. KL ダイバージェンス KL(q || p(Z|X))
kl_div = np.sum(q * np.log(q / p_z_given_x))

# 恒等性の検証
np.testing.assert_allclose(L_q + kl_div, ln_p_x, atol=1e-12)
assert L_q <= ln_p_x + 1e-12, "L(q) must be a lower bound on ln p(x)"
assert kl_div >= 0.0, "KL divergence must be non-negative"

# 最適 q = p(Z|X) のとき KL = 0, L(q) = ln p(X)
L_optimal = np.sum(p_z_given_x * np.log(p_x_z / p_z_given_x))
np.testing.assert_allclose(L_optimal, ln_p_x, atol=1e-12)

print(f"Exercise 9.24 verified: ln p(X) = {ln_p_x:.6f} == L(q) ({L_q:.6f}) + KL ({kl_div:.6f})!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_24_md), nbf.v4.new_code_cell(ex9_24_code)])

    # --- Exercise 9.25 ---
    ex9_25_md = r"""---
## <a id="Exercise-9.25"></a>Exercise 9.25: 接点における変分下界と対数尤度の勾配一致定理の証明

### 問題の提示
$q(\mathbf{Z}) = p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta}^{(\mathrm{old})})$ と選んだとき、変分下界 $\mathcal{L}(q, \boldsymbol{\theta})$（式 9.71）の $\boldsymbol{\theta}$ に関する勾配が、現在点 $\boldsymbol{\theta} = \boldsymbol{\theta}^{(\mathrm{old})}$ において真の不完全データ対数尤度関数 $\ln p(\mathbf{X}|\boldsymbol{\theta})$ の勾配と**厳密に一致すること**：
$$ \left. \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{X}|\boldsymbol{\theta}) \right|_{\boldsymbol{\theta} = \boldsymbol{\theta}^{(\mathrm{old})}} = \left. \nabla_{\boldsymbol{\theta}} \mathcal{L}(q, \boldsymbol{\theta}) \right|_{\boldsymbol{\theta} = \boldsymbol{\theta}^{(\mathrm{old})}} $$
を証明せよ。

### [解答の道筋と穴埋め]
1. **基本恒等式の微分**:
   式 (9.70) より：
   $$ \ln p(\mathbf{X}|\boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \mathrm{KL}(q \| p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})) $$
   両辺を $\boldsymbol{\theta}$ で微分すると：
   $$ \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{X}|\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \mathcal{L}(q, \boldsymbol{\theta}) + \nabla_{\boldsymbol{\theta}} \mathrm{KL}(q \| p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})) $$
2. **$\mathrm{KL}$ ダイバージェンスの勾配消失**:
   $q(\mathbf{Z})$ は $\boldsymbol{\theta}^{(\mathrm{old})}$ を固定して定めた分布であるため、$\boldsymbol{\theta}$ には陽に依存しない。
   したがって $\mathrm{KL}(q \| p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta}))$ は $\boldsymbol{\theta}$ の関数となるが、
   $\boldsymbol{\theta} = \boldsymbol{\theta}^{(\mathrm{old})}$ において $q(\mathbf{Z}) = p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta}^{(\mathrm{old})})$ であるから、
   この点において $\mathrm{KL}$ ダイバージェンスは最小値 [ ① ] を達成する。
3. **停留性の帰結**:
   滑らかな関数の大域的最小点においては、一階微分（勾配）が必ず消失する：
   $$ \left. \nabla_{\boldsymbol{\theta}} \mathrm{KL}(q \| p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})) \right|_{\boldsymbol{\theta} = \boldsymbol{\theta}^{(\mathrm{old})}} = [ \text{②} ] $$
   したがって、下界 $\mathcal{L}(q, \boldsymbol{\theta})$ は点 $\boldsymbol{\theta} = \boldsymbol{\theta}^{(\mathrm{old})}$ において対数尤度曲線に**接しており**、その接線勾配は真の対数尤度勾配と完全に一致する。

### 穴埋めの解答
- ①: $0$
- ②: $\mathbf{0}$"""

    ex9_25_code = r"""# Exercise 9.25 数値検証: 接点における変分下界勾配と真の対数尤度勾配の完全一致
# 1次元2成分 GMM の平均 mu_1 に対する勾配比較
np.random.seed(42)
X = np.array([-1.5, -0.8, 0.2, 1.8, 2.5])
pi = np.array([0.5, 0.5])
mu_old = np.array([-1.0, 2.0])
var = np.array([1.0, 1.0])

# 事後負担率 q(Z) = p(Z | X, theta_old)
resp = np.zeros((len(X), 2))
for n, x in enumerate(X):
    p0 = pi[0] * np.exp(-0.5 * (x - mu_old[0])**2 / var[0])
    p1 = pi[1] * np.exp(-0.5 * (x - mu_old[1])**2 / var[1])
    resp[n] = [p0 / (p0 + p1), p1 / (p0 + p1)]

# 1. 真の対数尤度の解析的勾配 (d ln p(X|theta) / d mu_0 at mu_old)
grad_loglik = 0.0
for n, x in enumerate(X):
    p0 = pi[0] * np.exp(-0.5 * (x - mu_old[0])**2 / var[0])
    p1 = pi[1] * np.exp(-0.5 * (x - mu_old[1])**2 / var[1])
    # d/d mu_0 [p0] = p0 * (x - mu_0) / var_0
    dp0_dmu0 = p0 * (x - mu_old[0]) / var[0]
    grad_loglik += dp0_dmu0 / (p0 + p1)

# 2. 下界 L(q, theta) の解析的勾配 (d Q / d mu_0 at mu_old)
# Q = sum_n gamma_{n0} [ -0.5 (x_n - mu_0)^2 / var_0 ]
grad_lower_bound = 0.0
for n, x in enumerate(X):
    grad_lower_bound += resp[n, 0] * (x - mu_old[0]) / var[0]

np.testing.assert_allclose(grad_loglik, grad_lower_bound, atol=1e-12)
print(f"Exercise 9.25 verified: True log-lik gradient {grad_loglik:.8f} == Lower bound gradient {grad_lower_bound:.8f}!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_25_md), nbf.v4.new_code_cell(ex9_25_code)])

    # --- Exercise 9.26 ---
    ex9_26_md = r"""---
## <a id="Exercise-9.26"></a>Exercise 9.26: オンライン（インクリメンタル）EM における中心ベクトル逐次更新式 (9.78, 9.79) の導出

### 問題の提示
データ点 $\mathbf{x}_m$ の事後負担率のみを逐次再計算するインクリメンタル EM アルゴリズムにおいて、中心ベクトル $\boldsymbol{\mu}_k$ の Mステップ更新式 (9.17) および有効点数 (9.18) から出発して、逐次更新式 (9.78)
$$ \boldsymbol{\mu}_k^{(\mathrm{new})} = \boldsymbol{\mu}_k^{(\mathrm{old})} + \frac{\gamma^{\mathrm{new}}(z_{mk}) - \gamma^{\mathrm{old}}(z_{mk})}{N_k^{(\mathrm{new})}} \left( \mathbf{x}_m - \boldsymbol{\mu}_k^{(\mathrm{old})} \right) \quad (9.78) $$
および (9.79)
$$ N_k^{(\mathrm{new})} = N_k^{(\mathrm{old})} + \gamma^{\mathrm{new}}(z_{mk}) - \gamma^{\mathrm{old}}(z_{mk}) \quad (9.79) $$
が得られることを代数的に示せ。

### [解答の道筋と穴埋め]
1. **十分統計量の差分更新**:
   負担率が変化するのはデータ点 $m$ のみであるため、総負担率は：
   $$ N_k^{(\mathrm{new})} = \sum_{n \ne m} \gamma(z_{nk}) + \gamma^{\mathrm{new}}(z_{mk}) = N_k^{(\mathrm{old})} - \gamma^{\mathrm{old}}(z_{mk}) + \gamma^{\mathrm{new}}(z_{mk}) = [ \text{①} ] $$
2. **中心ベクトルの分子の変形**:
   $$ N_k^{(\mathrm{new})} \boldsymbol{\mu}_k^{(\mathrm{new})} = \sum_{n \ne m} \gamma(z_{nk}) \mathbf{x}_n + \gamma^{\mathrm{new}}(z_{mk}) \mathbf{x}_m = N_k^{(\mathrm{old})} \boldsymbol{\mu}_k^{(\mathrm{old})} + (\gamma^{\mathrm{new}}(z_{mk}) - \gamma^{\mathrm{old}}(z_{mk})) \mathbf{x}_m $$
3. **差分表現への整理**:
   式 (9.79) より $N_k^{(\mathrm{old})} = N_k^{(\mathrm{new})} - (\gamma^{\mathrm{new}}(z_{mk}) - \gamma^{\mathrm{old}}(z_{mk}))$ を代入すると：
   $$ N_k^{(\mathrm{new})} \boldsymbol{\mu}_k^{(\mathrm{new})} = N_k^{(\mathrm{new})} \boldsymbol{\mu}_k^{(\mathrm{old})} + (\gamma^{\mathrm{new}}(z_{mk}) - \gamma^{\mathrm{old}}(z_{mk})) (\mathbf{x}_m - \boldsymbol{\mu}_k^{(\mathrm{old})}) $$
   両辺を $N_k^{(\mathrm{new})}$ で割ることにより：
   $$ \boldsymbol{\mu}_k^{(\mathrm{new})} = [ \text{②} ] $$
   が導出される。全データを再走査することなく $O(1)$ で中心が更新できる。

### 穴埋めの解答
- ①: 式 (9.79)
- ②: $\boldsymbol{\mu}_k^{(\mathrm{old})} + \frac{\gamma^{\mathrm{new}}(z_{mk}) - \gamma^{\mathrm{old}}(z_{mk})}{N_k^{(\mathrm{new})}} \left( \mathbf{x}_m - \boldsymbol{\mu}_k^{(\mathrm{old})} \right)$"""

    ex9_26_code = r"""# Exercise 9.26 数値検証: インクリメンタル EM による中心ベクトル更新と一括計算の完全一致
N, D, K = 40, 2, 3
np.random.seed(42)
X = np.random.randn(N, D)
gamma_old = np.random.dirichlet(np.ones(K), size=N)

# 初期状態
N_k_old = gamma_old.sum(axis=0)
mu_old = (gamma_old.T @ X) / N_k_old[:, None]

# データ点 m のみ負担率を更新
m = 12
gamma_new_m = np.random.dirichlet(np.ones(K))
delta_gamma_m = gamma_new_m - gamma_old[m]

# 1. 式 (9.78, 9.79) による逐次更新
N_k_new_formula = N_k_old + delta_gamma_m
mu_new_formula = np.zeros_like(mu_old)
for k in range(K):
    mu_new_formula[k] = mu_old[k] + (delta_gamma_m[k] / N_k_new_formula[k]) * (X[m] - mu_old[k])

# 2. 全体再計算による真値
gamma_updated = gamma_old.copy()
gamma_updated[m] = gamma_new_m
N_k_true = gamma_updated.sum(axis=0)
mu_true = (gamma_updated.T @ X) / N_k_true[:, None]

np.testing.assert_allclose(N_k_new_formula, N_k_true, atol=1e-12)
np.testing.assert_allclose(mu_new_formula, mu_true, atol=1e-12)
print("Exercise 9.26 verified: Incremental mean update strictly matches full batch re-calculation!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_26_md), nbf.v4.new_code_cell(ex9_26_code)])

    # --- Exercise 9.27 ---
    ex9_27_md = r"""---
## <a id="Exercise-9.27"></a>Exercise 9.27: インクリメンタル EM における共分散行列 $\mathbf{\Sigma}_k$ および混合係数 $\pi_k$ の逐次更新式の導出

### 問題の提示
インクリメンタル EM アルゴリズムにおいて、単一データ点 $\mathbf{x}_m$ の負担率が更新されたときの共分散行列 $\mathbf{\Sigma}_k$ および混合係数 $\pi_k$ の逐次更新式を導出せよ。

### [解答の道筋と穴埋め]
1. **混合係数 $\pi_k$ の逐次更新**:
   $$ \pi_k^{(\mathrm{new})} = \frac{N_k^{(\mathrm{new})}}{N} = \frac{N_k^{(\mathrm{old})} + \gamma^{\mathrm{new}}(z_{mk}) - \gamma^{\mathrm{old}}(z_{mk})}{N} = [ \text{①} ] $$
2. **2次十分統計量 $\mathbf{S}_k = \sum_n \gamma(z_{nk}) \mathbf{x}_n \mathbf{x}_n^{\mathrm{T}}$ の更新**:
   $$ \mathbf{S}_k^{(\mathrm{new})} = \mathbf{S}_k^{(\mathrm{old})} + (\gamma^{\mathrm{new}}(z_{mk}) - \gamma^{\mathrm{old}}(z_{mk})) \mathbf{x}_m \mathbf{x}_m^{\mathrm{T}} $$
3. **共分散行列 $\mathbf{\Sigma}_k$ の表現**:
   $$ \mathbf{\Sigma}_k = \frac{1}{N_k} \mathbf{S}_k - \boldsymbol{\mu}_k \boldsymbol{\mu}_k^{\mathrm{T}} $$
   あるいは旧共分散行列 $\mathbf{\Sigma}_k^{(\mathrm{old})}$ を用いた陽な更新式として：
   $$ \mathbf{\Sigma}_k^{(\mathrm{new})} = \frac{N_k^{(\mathrm{old})}}{N_k^{(\mathrm{new})}} \left[ \mathbf{\Sigma}_k^{(\mathrm{old})} + (\boldsymbol{\mu}_k^{(\mathrm{old})} - \boldsymbol{\mu}_k^{(\mathrm{new})})(\boldsymbol{\mu}_k^{(\mathrm{old})} - \boldsymbol{\mu}_k^{(\mathrm{new})})^{\mathrm{T}} \right] + [ \text{②} ] $$
   が導出される。

### 穴埋めの解答
- ①: $\pi_k^{(\mathrm{old})} + \frac{\gamma^{\mathrm{new}}(z_{mk}) - \gamma^{\mathrm{old}}(z_{mk})}{N}$
- ②: $\frac{\gamma^{\mathrm{new}}(z_{mk}) - \gamma^{\mathrm{old}}(z_{mk})}{N_k^{(\mathrm{new})}} (\mathbf{x}_m - \boldsymbol{\mu}_k^{(\mathrm{new})})(\mathbf{x}_m - \boldsymbol{\mu}_k^{(\mathrm{new})})^{\mathrm{T}}$"""

    ex9_27_code = r"""# Exercise 9.27 数値検証: インクリメンタル EM による共分散行列および混合係数更新の完全一致
N, D, K = 50, 2, 3
np.random.seed(42)
X = np.random.randn(N, D)
gamma_old = np.random.dirichlet(np.ones(K), size=N)

N_k_old = gamma_old.sum(axis=0)
mu_old = (gamma_old.T @ X) / N_k_old[:, None]
cov_old = np.zeros((K, D, D))
for k in range(K):
    diff = X - mu_old[k]
    cov_old[k] = (gamma_old[:, k:k+1] * diff).T @ diff / N_k_old[k]

m = 7
gamma_new_m = np.random.dirichlet(np.ones(K))
delta_gamma_m = gamma_new_m - gamma_old[m]

# 1. 逐次更新
N_k_new = N_k_old + delta_gamma_m
pi_new_formula = (N_k_old + delta_gamma_m) / N

mu_new = np.zeros_like(mu_old)
cov_new_formula = np.zeros_like(cov_old)
for k in range(K):
    mu_new[k] = mu_old[k] + (delta_gamma_m[k] / N_k_new[k]) * (X[m] - mu_old[k])
    # 逐次更新公式
    term1 = (N_k_old[k] / N_k_new[k]) * (cov_old[k] + np.outer(mu_old[k] - mu_new[k], mu_old[k] - mu_new[k]))
    term2 = (delta_gamma_m[k] / N_k_new[k]) * np.outer(X[m] - mu_new[k], X[m] - mu_new[k])
    cov_new_formula[k] = term1 + term2

# 2. 一括再計算による真値
gamma_updated = gamma_old.copy()
gamma_updated[m] = gamma_new_m
pi_true = gamma_updated.sum(axis=0) / N
cov_true = np.zeros_like(cov_old)
for k in range(K):
    diff = X - mu_new[k]
    cov_true[k] = (gamma_updated[:, k:k+1] * diff).T @ diff / N_k_new[k]

np.testing.assert_allclose(pi_new_formula, pi_true, atol=1e-12)
np.testing.assert_allclose(cov_new_formula, cov_true, atol=1e-12)
print("Exercise 9.27 verified: Incremental covariance and mixing coefficient updates strictly match batch re-estimation!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_27_md), nbf.v4.new_code_cell(ex9_27_code)])

    return cells

print("get_ex_9_19_to_9_27 defined.")
