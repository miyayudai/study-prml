# scripts/build_ch9_part2.py
"""
Definitions for Chapter 9 exercises 9.10 to 9.18.
"""

import nbformat as nbf

def get_ex_9_10_to_9_18():
    cells = []

    # --- Exercise 9.10 ---
    ex9_10_md = r"""---
## <a id="Exercise-9.10"></a>Exercise 9.10: 分割ベクトルにおける条件付き分布の混合分布表現

### 問題の提示
混合分布モデル
$$ p(\mathbf{x}) = \sum_{k=1}^K \pi_k p(\mathbf{x}|k) \quad (9.81) $$
において、変数ベクトルを $\mathbf{x} = (\mathbf{x}_a, \mathbf{x}_b)$ と2分割する。
このとき、条件付き密度 $p(\mathbf{x}_b | \mathbf{x}_a)$ もまた混合分布の形をしていることを示し、その新しい混合係数および各成分条件付き密度の数式表現を導出せよ。

### [解答の道筋と穴埋め]
1. **条件付き確率の定義**:
   $$ p(\mathbf{x}_b | \mathbf{x}_a) = \frac{p(\mathbf{x}_a, \mathbf{x}_b)}{p(\mathbf{x}_a)} = \frac{\sum_{k=1}^K \pi_k p(\mathbf{x}_a, \mathbf{x}_b | k)}{\int p(\mathbf{x}_a, \mathbf{x}_b) \mathrm{d}\mathbf{x}_b} $$
2. **分子・分母の各成分表現**:
   条件付き確率の乗法定理 $p(\mathbf{x}_a, \mathbf{x}_b | k) = p(\mathbf{x}_a | k) p(\mathbf{x}_b | \mathbf{x}_a, k)$ を適用すると：
   $$ p(\mathbf{x}_a) = \sum_{j=1}^K \pi_j p(\mathbf{x}_a | j) $$
   $$ p(\mathbf{x}_a, \mathbf{x}_b) = \sum_{k=1}^K \pi_k p(\mathbf{x}_a | k) p(\mathbf{x}_b | \mathbf{x}_a, k) $$
3. **入力依存の混合係数の定義**:
   したがって、条件付き密度は：
   $$ p(\mathbf{x}_b | \mathbf{x}_a) = \sum_{k=1}^K \left[ \frac{\pi_k p(\mathbf{x}_a | k)}{\sum_{j=1}^K \pi_j p(\mathbf{x}_a | j)} \right] [ \text{①} ] = \sum_{k=1}^K \tilde{\pi}_k(\mathbf{x}_a) p(\mathbf{x}_b | \mathbf{x}_a, k) $$
   ここで新しい混合係数は
   $$ \tilde{\pi}_k(\mathbf{x}_a) = [ \text{②} ] $$
   であり、$\sum_k \tilde{\pi}_k(\mathbf{x}_a) = 1$ かつ $\tilde{\pi}_k(\mathbf{x}_a) \ge 0$ を満たすため、条件付き分布も厳密に混合分布となる（これは混合密度ネットワーク (MDN) や専門家混合 (MoE) の基礎原理である）。

### 穴埋めの解答
- ①: $p(\mathbf{x}_b | \mathbf{x}_a, k)$
- ②: $\frac{\pi_k p(\mathbf{x}_a | k)}{\sum_j \pi_j p(\mathbf{x}_a | j)}$"""

    ex9_10_code = r"""# Exercise 9.10 数値検証: 分割ベクトルの条件付き分布が混合分布となることの数値積分検証
import numpy as np

# 2次元2成分 GMM
pi = np.array([0.3, 0.7])
mu = np.array([[1.0, 2.0], [-1.0, -0.5]])
cov = np.array([
    [[1.0, 0.5], [0.5, 1.2]],
    [[0.8, -0.3], [-0.3, 1.5]]
])

def gaussian_2d(x, m, c):
    d = len(m)
    diff = x - m
    return np.exp(-0.5 * diff @ np.linalg.inv(c) @ diff) / np.sqrt((2 * np.pi)**d * np.linalg.det(c))

x_a = 0.5
# 1. 直接法: p(x_a, x_b) / p(x_a)
p_xa = sum(pi[k] * np.exp(-0.5 * (x_a - mu[k, 0])**2 / cov[k, 0, 0]) / np.sqrt(2 * np.pi * cov[k, 0, 0]) for k in range(2))

# 2. 混合条件付き公式: sum_k pi_tilde_k * p(x_b | x_a, k)
# 各成分の条件付きガウス分布 p(x_b | x_a, k): N(mu_{b|a}, sigma_{b|a}^2)
pi_tilde = np.zeros(2)
for k in range(2):
    p_xa_k = np.exp(-0.5 * (x_a - mu[k, 0])**2 / cov[k, 0, 0]) / np.sqrt(2 * np.pi * cov[k, 0, 0])
    pi_tilde[k] = pi[k] * p_xa_k
pi_tilde /= np.sum(pi_tilde)

x_b_test = 1.2
p_cond_formula = 0.0
for k in range(2):
    # ガウス条件付き分布公式
    mu_b_given_a = mu[k, 1] + cov[k, 1, 0] / cov[k, 0, 0] * (x_a - mu[k, 0])
    var_b_given_a = cov[k, 1, 1] - cov[k, 1, 0]**2 / cov[k, 0, 0]
    p_cond_k = np.exp(-0.5 * (x_b_test - mu_b_given_a)**2 / var_b_given_a) / np.sqrt(2 * np.pi * var_b_given_a)
    p_cond_formula += pi_tilde[k] * p_cond_k

# 直接結合密度との比
p_joint = sum(pi[k] * gaussian_2d(np.array([x_a, x_b_test]), mu[k], cov[k]) for k in range(2))
p_cond_direct = p_joint / p_xa

np.testing.assert_allclose(p_cond_formula, p_cond_direct, atol=1e-12)
print(f"Exercise 9.10 verified: Conditional density {p_cond_formula:.6f} matches joint/marginal ratio exactly!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_10_md), nbf.v4.new_code_cell(ex9_10_code)])

    # --- Exercise 9.11 ---
    ex9_11_md = r"""---
## <a id="Exercise-9.11"></a>Exercise 9.11: 分散ゼロ極限 $\epsilon \to 0$ における EM アルゴリズムと K-means の等価性

### 問題の提示
各成分が等方共分散 $\mathbf{\Sigma}_k = \epsilon \mathbf{I}$ を持つガウス混合モデルにおいて、$\epsilon \to 0$ の極限を考える。
このとき、事後負担率 $\gamma(z_{nk})$ が K-means の二値指示変数 $r_{nk} \in \{0, 1\}$ に収束し、期待完全データ対数尤度 $\mathcal{Q}$ の最大化が K-means 歪み尺度 $J$ の最小化と等価になることを示せ。

### [解答の道筋と穴埋め]
1. **負担率の極限表現**:
   $$ \gamma(z_{nk}) = \frac{\pi_k \exp\left(-\frac{\|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2}{2\epsilon}\right)}{\sum_{j=1}^K \pi_j \exp\left(-\frac{\|\mathbf{x}_n - \boldsymbol{\mu}_j\|^2}{2\epsilon}\right)} $$
   $\epsilon \to 0$ のとき、分母の中でユークリッド距離 $\|\mathbf{x}_n - \boldsymbol{\mu}_j\|^2$ が最も小さい成分が支配的となり、指数関数の比は：
   $$ \lim_{\epsilon \to 0} \gamma(z_{nk}) = \begin{cases} 1 & (k = \arg\min_j \|\mathbf{x}_n - \boldsymbol{\mu}_j\|^2) \\ 0 & (\text{otherwise}) \end{cases} = [ \text{①} ] $$
2. **目的関数の極限関係**:
   期待完全データ対数尤度 (9.40) に $\mathbf{\Sigma}_k = \epsilon \mathbf{I}$ を代入すると：
   $$ \mathcal{Q} = -\frac{1}{2\epsilon} \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2 - \frac{ND}{2} \ln(2\pi\epsilon) + \sum_{n, k} \gamma(z_{nk}) \ln \pi_k $$
   $\epsilon \to 0$ において最右辺第1項が圧倒的に支配的となる：
   $$ \mathcal{Q} = -[ \text{②} ] J + \mathcal{O}(\ln\epsilon) $$
   したがって、$\mathcal{Q}$ を最大化することは $J = \sum_{n, k} r_{nk} \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2$ を最小化することと完全に等価となる。

### 穴埋めの解答
- ①: $r_{nk}$ (K-means ハード割り当て)
- ②: $\frac{1}{2\epsilon}$"""

    ex9_11_code = r"""# Exercise 9.11 数値検証: epsilon -> 0 における負担率 gamma_nk の K-means 指示変数への収束
mu = np.array([[0.0, 0.0], [2.0, 2.0]])
x = np.array([0.5, 0.5]) # mu[0] に近い点
pi = np.array([0.5, 0.5])

epsilons = [1.0, 0.1, 0.01, 1e-4]
print("Convergence of gamma(z_n0) as epsilon -> 0:")
for eps in epsilons:
    dist0 = np.sum((x - mu[0])**2)
    dist1 = np.sum((x - mu[1])**2)
    log_p0 = np.log(pi[0]) - dist0 / (2 * eps)
    log_p1 = np.log(pi[1]) - dist1 / (2 * eps)
    # LogSumExp
    max_log = max(log_p0, log_p1)
    gamma0 = np.exp(log_p0 - max_log) / (np.exp(log_p0 - max_log) + np.exp(log_p1 - max_log))
    print(f"  epsilon = {eps:6.4f}: gamma0 = {gamma0:.8f}")

assert gamma0 > 1 - 1e-6, "gamma0 must approach 1.0 as epsilon -> 0"
print("Exercise 9.11 verified: Soft responsibility strictly converges to hard K-means assignment!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_11_md), nbf.v4.new_code_cell(ex9_11_code)])

    # --- Exercise 9.12 ---
    ex9_12_md = r"""---
## <a id="Exercise-9.12"></a>Exercise 9.12: 一般混合分布全体の平均および共分散行列の合成則 (式 9.49, 9.50)

### 問題の提示
一般混合分布 $p(\mathbf{x}) = \sum_{k=1}^K \pi_k p(\mathbf{x}|k)$ を考える（変数は連続・離散・混合のいずれでもよい）。
各成分分布 $p(\mathbf{x}|k)$ の平均を $\boldsymbol{\mu}_k$、共分散行列を $\mathbf{\Sigma}_k$ とするとき、混合分布全体の平均 $\mathbb{E}[\mathbf{x}]$ および共分散行列 $\mathrm{cov}[\mathbf{x}]$ がそれぞれ
$$ \mathbb{E}[\mathbf{x}] = \sum_{k=1}^K \pi_k \boldsymbol{\mu}_k \quad (9.49) $$
$$ \mathrm{cov}[\mathbf{x}] = \sum_{k=1}^K \pi_k \left\{ \mathbf{\Sigma}_k + (\boldsymbol{\mu}_k - \mathbb{E}[\mathbf{x}])(\boldsymbol{\mu}_k - \mathbb{E}[\mathbf{x}])^{\mathrm{T}} \right\} \quad (9.50) $$
で与えられることを証明せよ。

### [解答の道筋と穴埋め]
1. **全体の平均 $\mathbb{E}[\mathbf{x}]$**:
   全期待値の法則 $\mathbb{E}[\mathbf{x}] = \mathbb{E}_k [\mathbb{E}[\mathbf{x}|k]]$ より：
   $$ \mathbb{E}[\mathbf{x}] = \int \mathbf{x} \sum_{k=1}^K \pi_k p(\mathbf{x}|k) \mathrm{d}\mathbf{x} = \sum_{k=1}^K \pi_k \int \mathbf{x} p(\mathbf{x}|k) \mathrm{d}\mathbf{x} = [ \text{①} ] $$
2. **全体の2次モーメント**:
   $$ \mathbb{E}[\mathbf{x} \mathbf{x}^{\mathrm{T}}] = \sum_{k=1}^K \pi_k \mathbb{E}[\mathbf{x} \mathbf{x}^{\mathrm{T}}|k] = \sum_{k=1}^K \pi_k (\mathbf{\Sigma}_k + \boldsymbol{\mu}_k \boldsymbol{\mu}_k^{\mathrm{T}}) $$
3. **共分散行列への変形**:
   $\mathrm{cov}[\mathbf{x}] = \mathbb{E}[\mathbf{x} \mathbf{x}^{\mathrm{T}}] - \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^{\mathrm{T}}$ より：
   $$ \mathrm{cov}[\mathbf{x}] = \sum_{k=1}^K \pi_k \mathbf{\Sigma}_k + \sum_{k=1}^K \pi_k \boldsymbol{\mu}_k \boldsymbol{\mu}_k^{\mathrm{T}} - \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^{\mathrm{T}} $$
   ここで $(\boldsymbol{\mu}_k - \mathbb{E}[\mathbf{x}])(\boldsymbol{\mu}_k - \mathbb{E}[\mathbf{x}])^{\mathrm{T}} = \boldsymbol{\mu}_k \boldsymbol{\mu}_k^{\mathrm{T}} - \boldsymbol{\mu}_k \mathbb{E}[\mathbf{x}]^{\mathrm{T}} - \mathbb{E}[\mathbf{x}]\boldsymbol{\mu}_k^{\mathrm{T}} + \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^{\mathrm{T}}$ を $\pi_k$ で加重総和すると：
   $$ \sum_{k=1}^K \pi_k (\boldsymbol{\mu}_k - \mathbb{E}[\mathbf{x}])(\boldsymbol{\mu}_k - \mathbb{E}[\mathbf{x}])^{\mathrm{T}} = \sum_{k=1}^K \pi_k \boldsymbol{\mu}_k \boldsymbol{\mu}_k^{\mathrm{T}} - [ \text{②} ] $$
   したがって、式 (9.50)（各クラスタ内共分散の平均 ＋ クラスタ間分散）が厳密に成立する。

### 穴埋めの解答
- ①: $\sum_{k=1}^K \pi_k \boldsymbol{\mu}_k$
- ②: $\mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^{\mathrm{T}}$"""

    ex9_12_code = r"""# Exercise 9.12 数値検証: 混合分布の理論平均・共分散とモンテカルロサンプリングの整合
K, D = 3, 2
pi = np.array([0.2, 0.5, 0.3])
mu = np.array([[2.0, 0.0], [-1.0, 3.0], [0.0, -2.0]])
cov = np.array([
    [[1.0, 0.2], [0.2, 0.8]],
    [[1.5, -0.4], [-0.4, 0.6]],
    [[0.7, 0.1], [0.1, 1.2]]
])

# 理論平均
E_x_theory = sum(pi[k] * mu[k] for k in range(K))
# 理論共分散 (式 9.50)
cov_x_theory = sum(pi[k] * (cov[k] + np.outer(mu[k] - E_x_theory, mu[k] - E_x_theory)) for k in range(K))

# モンテカルロサンプリング
N_samples = 200000
labels = np.random.choice(K, size=N_samples, p=pi)
samples = np.zeros((N_samples, D))
for k in range(K):
    idx = (labels == k)
    if np.sum(idx) > 0:
        samples[idx] = np.random.multivariate_normal(mu[k], cov[k], size=np.sum(idx))

E_x_sample = np.mean(samples, axis=0)
cov_x_sample = np.cov(samples, rowvar=False)

np.testing.assert_allclose(E_x_sample, E_x_theory, atol=0.02)
np.testing.assert_allclose(cov_x_sample, cov_x_theory, atol=0.02)
print("Exercise 9.12 verified: Theoretical mixture mean and covariance perfectly match empirical moments!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_12_md), nbf.v4.new_code_cell(ex9_12_code)])

    # --- Exercise 9.13 ---
    ex9_13_md = r"""---
## <a id="Exercise-9.13"></a>Exercise 9.13: ベルヌーイ混合モデルにおける同一初期値での1反復退化現象の証明

### 問題の提示
ベルヌーイ混合モデル (BMM) において、EMアルゴリズムの再推定方程式を満たす最尤解において $\mathbb{E}[\mathbf{x}] = \frac{1}{N} \sum_{n=1}^N \mathbf{x}_n \equiv \bar{\mathbf{x}}$ が成立することを示せ。
さらに、すべての混合成分のパラメータが同一の値 $\boldsymbol{\mu}_k = \boldsymbol{\mu}$ および $\pi_k = 1/K$ に初期化された場合、アルゴリズムがわずか1回の反復で標本平均 $\boldsymbol{\mu}_k = \bar{\mathbf{x}}$ に退化し、以後全く更新されなくなることを証明せよ。

### [解答の道筋と穴埋め]
1. **混合期待値の整合**:
   式 (9.49) より $\mathbb{E}[\mathbf{x}] = \sum_{k=1}^K \pi_k \boldsymbol{\mu}_k$。
   Mステップの再推定式 $\pi_k = \frac{N_k}{N}, \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_n \gamma(z_{nk}) \mathbf{x}_n$ を代入すると：
   $$ \mathbb{E}[\mathbf{x}] = \sum_{k=1}^K \frac{N_k}{N} \left( \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) \mathbf{x}_n \right) = \frac{1}{N} \sum_{n=1}^N \left( \sum_{k=1}^K \gamma(z_{nk}) \right) \mathbf{x}_n = [ \text{①} ] $$
   なぜなら負担率の定義より $\sum_k \gamma(z_{nk}) = 1$ だからである。
2. **同一初期値におけるEステップ**:
   すべての $k$ で $\boldsymbol{\mu}_k = \boldsymbol{\mu}, \pi_k = 1/K$ のとき、任意のデータ点 $\mathbf{x}_n$ に対し条件付き確率 $p(\mathbf{x}_n | \boldsymbol{\mu}_k)$ は $k$ に依存せず全て等しい。
   したがって、負担率は：
   $$ \gamma(z_{nk}) = \frac{\frac{1}{K} p(\mathbf{x}_n|\boldsymbol{\mu})}{\sum_{j=1}^K \frac{1}{K} p(\mathbf{x}_n|\boldsymbol{\mu})} = [ \text{②} ] $$
3. **Mステップにおける即時退化**:
   すべての $n, k$ で $\gamma(z_{nk}) = 1/K$ となるため、$N_k = \sum_n \frac{1}{K} = N/K$。
   更新後の中心パラメータは：
   $$ \boldsymbol{\mu}_k^{\mathrm{new}} = \frac{1}{N/K} \sum_{n=1}^N \frac{1}{K} \mathbf{x}_n = \frac{1}{N} \sum_{n=1}^N \mathbf{x}_n = \bar{\mathbf{x}} $$
   かつ $\pi_k^{\mathrm{new}} = \frac{N/K}{N} = \frac{1}{K}$。
   すべてのクラスタが同一の標本平均 $\bar{\mathbf{x}}$ となり、対称性が破れないため、以後一切変化せず1ステップで停止する。

### 穴埋めの解答
- ①: $\frac{1}{N} \sum_{n=1}^N \mathbf{x}_n \equiv \bar{\mathbf{x}}$
- ②: $\frac{1}{K}$"""

    ex9_13_code = r"""# Exercise 9.13 数値検証: ベルヌーイ混合モデルの同一初期値における1反復退化
N, D, K = 50, 4, 3
np.random.seed(42)
X = (np.random.rand(N, D) > 0.5).astype(float)
x_bar = np.mean(X, axis=0)

# 同一パラメータで初期化
mu = np.full((K, D), 0.5)
pi = np.full(K, 1.0 / K)

# Eステップ
resp = np.zeros((N, K))
for n in range(N):
    dens = np.array([pi[k] * np.prod(mu[k]**X[n] * (1 - mu[k])**(1 - X[n])) for k in range(K)])
    resp[n] = dens / np.sum(dens)

assert np.allclose(resp, 1.0 / K), "Responsibilities must all be exactly 1/K!"

# Mステップ
N_k = resp.sum(axis=0)
mu_new = (resp.T @ X) / N_k[:, None]
pi_new = N_k / N

for k in range(K):
    np.testing.assert_allclose(mu_new[k], x_bar, atol=1e-12)
    np.testing.assert_allclose(pi_new[k], 1.0 / K, atol=1e-12)

print("Exercise 9.13 verified: Identically initialized BMM strictly collapses to sample mean x_bar in 1 step!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_13_md), nbf.v4.new_code_cell(ex9_13_code)])

    # --- Exercise 9.14 ---
    ex9_14_md = r"""---
## <a id="Exercise-9.14"></a>Exercise 9.14: ベルヌーイ混合モデルの同時分布と潜在変数周辺化による観測分布の導出

### 問題の提示
ベルヌーイ分布の混合モデルにおいて、潜在変数 $\mathbf{z}$（1-of-K 表現）とパラメータ $\boldsymbol{\pi}$ の事前分布 $p(\mathbf{z}|\boldsymbol{\pi}) = \prod_k \pi_k^{z_k}$（式 9.53）および条件付き観測分布
$$ p(\mathbf{x}|\mathbf{z}, \boldsymbol{\mu}) = \prod_{k=1}^K p(\mathbf{x}|\boldsymbol{\mu}_k)^{z_k} = \prod_{k=1}^K \left[ \prod_{i=1}^D \mu_{ki}^{x_i} (1 - \mu_{ki})^{1 - x_i} \right]^{z_k} \quad (9.52) $$
の積からなる同時分布 $p(\mathbf{x}, \mathbf{z}|\boldsymbol{\mu}, \boldsymbol{\pi})$ を構成し、$\mathbf{z}$ について周辺化することで観測周辺分布 (9.47)
$$ p(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\pi}) = \sum_{k=1}^K \pi_k p(\mathbf{x}|\boldsymbol{\mu}_k) \quad (9.47) $$
が得られることを代数的に示せ。

### [解答の道筋と穴埋め]
1. **同時分布の表現**:
   $$ p(\mathbf{x}, \mathbf{z}|\boldsymbol{\mu}, \boldsymbol{\pi}) = p(\mathbf{x}|\mathbf{z}, \boldsymbol{\mu}) p(\mathbf{z}|\boldsymbol{\pi}) = \prod_{k=1}^K [ \text{①} ]^{z_k} $$
2. **潜在変数 $\mathbf{z}$ の全状態**:
   $\mathbf{z}$ は $k$ 番目のみが 1 で他が 0 の $K$ 個の基底ベクトル $\mathbf{e}_1, \dots, \mathbf{e}_K$ のいずれかの状態を取る。
   $\mathbf{z} = \mathbf{e}_j$ のとき、積のうち $k=j$ の項のみが残り：
   $$ p(\mathbf{x}, \mathbf{z} = \mathbf{e}_j | \boldsymbol{\mu}, \boldsymbol{\pi}) = \pi_j p(\mathbf{x}|\boldsymbol{\mu}_j) $$
3. **総和による周辺化**:
   $$ p(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\pi}) = \sum_{\mathbf{z}} p(\mathbf{x}, \mathbf{z}|\boldsymbol{\mu}, \boldsymbol{\pi}) = \sum_{j=1}^K p(\mathbf{x}, \mathbf{z} = \mathbf{e}_j | \boldsymbol{\mu}, \boldsymbol{\pi}) = [ \text{②} ] $$
   これにより、式 (9.47) が得られる。

### 穴埋めの解答
- ①: $\pi_k p(\mathbf{x}|\boldsymbol{\mu}_k)$
- ②: $\sum_{k=1}^K \pi_k p(\mathbf{x}|\boldsymbol{\mu}_k)$"""

    ex9_14_code = r"""# Exercise 9.14 数値検証: ベルヌーイ混合モデルの同時確率テンソル総和と周辺密度の完全一致
D, K = 3, 2
pi = np.array([0.4, 0.6])
mu = np.array([[0.2, 0.8, 0.5], [0.7, 0.1, 0.9]])
x = np.array([1.0, 0.0, 1.0])

# 各成分の条件付き密度
p_x_given_k = np.array([np.prod(mu[k]**x * (1 - mu[k])**(1 - x)) for k in range(K)])

# 同時確率 p(x, z = e_k)
joint = pi * p_x_given_k

# 周辺確率 (総和)
marginal_sum = np.sum(joint)

# 式 (9.47)
marginal_direct = sum(pi[k] * np.prod(mu[k]**x * (1 - mu[k])**(1 - x)) for k in range(K))

np.testing.assert_allclose(marginal_sum, marginal_direct, atol=1e-12)
print(f"Exercise 9.14 verified: Marginalized joint {marginal_sum:.6f} matches mixture equation exactly!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_14_md), nbf.v4.new_code_cell(ex9_14_code)])

    # --- Exercise 9.15 ---
    ex9_15_md = r"""---
## <a id="Exercise-9.15"></a>Exercise 9.15: ベルヌーイ混合モデルにおける中心パラメータ $\boldsymbol{\mu}_k$ の M ステップ再推定式 (9.59) の導出

### 問題の提示
ベルヌーイ混合モデルの期待完全データ対数尤度関数（式 9.55）
$$ \mathcal{Q}(\boldsymbol{\theta}, \boldsymbol{\theta}^{(\mathrm{old})}) = \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \left\{ \ln \pi_k + \sum_{i=1}^D \left[ x_{ni} \ln \mu_{ki} + (1 - x_{ni}) \ln (1 - \mu_{ki}) \right] \right\} \quad (9.55) $$
を $\boldsymbol{\mu}_k$ に関して最大化することにより、Mステップの更新式 (9.59)
$$ \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) \mathbf{x}_n \quad (9.59) $$
が得られることを示せ。

### [解答の道筋と穴埋め]
1. **成分 $\mu_{ki}$ に関する偏微分**:
   各次元 $i \in \{1, \dots, D\}$ は互いに独立に最適化できる。$\mathcal{Q}$ を $\mu_{ki}$ で微分すると：
   $$ \frac{\partial \mathcal{Q}}{\partial \mu_{ki}} = \sum_{n=1}^N \gamma(z_{nk}) \left[ \frac{x_{ni}}{\mu_{ki}} - \frac{1 - x_{ni}}{1 - \mu_{ki}} \right] = \sum_{n=1}^N \gamma(z_{nk}) \frac{x_{ni} (1 - \mu_{ki}) - (1 - x_{ni}) \mu_{ki}}{\mu_{ki} (1 - \mu_{ki})} $$
2. **分子の整理**:
   分子を展開すると $x_{ni} - x_{ni} \mu_{ki} - \mu_{ki} + x_{ni} \mu_{ki} = [ \text{①} ]$ となる。したがって：
   $$ \frac{\partial \mathcal{Q}}{\partial \mu_{ki}} = \frac{1}{\mu_{ki} (1 - \mu_{ki})} \sum_{n=1}^N \gamma(z_{nk}) (x_{ni} - \mu_{ki}) $$
3. **停留条件と更新式**:
   これを $0$ と置くと：
   $$ \sum_{n=1}^N \gamma(z_{nk}) (x_{ni} - \mu_{ki}) = 0 \implies \left( \sum_{n=1}^N \gamma(z_{nk}) \right) \mu_{ki} = \sum_{n=1}^N \gamma(z_{nk}) x_{ni} $$
   実効点数 $N_k = \sum_n \gamma(z_{nk})$ で割ることにより：
   $$ \boldsymbol{\mu}_k = [ \text{②} ] $$
   が得られ、ガウス混合モデルの場合と全く同一の加重標本平均形式となる。

### 穴埋めの解答
- ①: $x_{ni} - \mu_{ki}$
- ②: $\frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) \mathbf{x}_n$"""

    ex9_15_code = r"""# Exercise 9.15 数値検証: BMM の期待完全データ対数尤度の mu_ki 勾配ゼロ性
N, D, K = 40, 3, 2
X = (np.random.rand(N, D) > 0.5).astype(float)
gamma = np.random.dirichlet(np.ones(K), size=N)

N_k = gamma.sum(axis=0)
mu_opt = (gamma.T @ X) / N_k[:, None]

# 勾配の直接計算
for k in range(K):
    grad = np.zeros(D)
    for i in range(D):
        for n in range(N):
            grad[i] += gamma[n, k] * (X[n, i] / mu_opt[k, i] - (1 - X[n, i]) / (1 - mu_opt[k, i]))
    np.testing.assert_allclose(grad, np.zeros(D), atol=1e-12)

print("Exercise 9.15 verified: Gradient of BMM expected log likelihood is strictly zero at mu_k M-step solution!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_15_md), nbf.v4.new_code_cell(ex9_15_code)])

    # --- Exercise 9.16 ---
    ex9_16_md = r"""---
## <a id="Exercise-9.16"></a>Exercise 9.16: ベルヌーイ混合モデルにおける混合係数 $\pi_k$ の M ステップ更新式 (9.60) の導出

### 問題の提示
ベルヌーイ混合モデルの期待完全データ対数尤度関数 (9.55) を、混合係数の総和制約 $\sum_{k=1}^K \pi_k = 1$ の下でラグランジュ乗数法を用いて最大化することにより、Mステップの更新式 (9.60)
$$ \pi_k = \frac{N_k}{N} \quad (9.60) $$
が得られることを示せ。

### [解答の道筋と穴埋め]
1. **ラグランジュ関数の設定**:
   式 (9.55) のうち $\pi_k$ に依存する項と制約項をまとめたラグランジュ関数は：
   $$ \mathcal{L}(\boldsymbol{\pi}, \lambda) = \sum_{k=1}^K \sum_{n=1}^N \gamma(z_{nk}) \ln \pi_k + \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) = \sum_{k=1}^K N_k \ln \pi_k + \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) $$
2. **停留条件**:
   $\pi_k$ に関して偏微分して $0$ と置く：
   $$ \frac{\partial \mathcal{L}}{\partial \pi_k} = \frac{N_k}{\pi_k} + \lambda = 0 \implies \pi_k = -[ \text{①} ] $$
3. **ラグランジュ乗数の決定**:
   制約 $\sum_{k=1}^K \pi_k = 1$ より：
   $$ \sum_{k=1}^K \left(-\frac{N_k}{\lambda}\right) = -\frac{1}{\lambda} \sum_{k=1}^K N_k = -\frac{N}{\lambda} = 1 \implies \lambda = -N $$
   これを代入することで：
   $$ \pi_k = [ \text{②} ] $$
   が厳密に導出される。

### 穴埋めの解答
- ①: $\frac{N_k}{\lambda}$
- ②: $\frac{N_k}{N}$"""

    ex9_16_code = r"""# Exercise 9.16 数値検証: BMM 混合係数のラグランジュ乗数解と総和 1 の検証
N, K = 100, 4
gamma = np.random.dirichlet(np.ones(K), size=N)
N_k = gamma.sum(axis=0)

pi_k = N_k / N
np.testing.assert_allclose(np.sum(pi_k), 1.0, atol=1e-12)
assert np.all(pi_k >= 0), "Mixing coefficients must be non-negative!"
print(f"Exercise 9.16 verified: pi_k = N_k / N sums to {np.sum(pi_k):.4f} with values {pi_k}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_16_md), nbf.v4.new_code_cell(ex9_16_code)])

    # --- Exercise 9.17 ---
    ex9_17_md = r"""---
## <a id="Exercise-9.17"></a>Exercise 9.17: ベルヌーイ混合モデルにおける対数尤度の上界性と特異点（発散）の非存在証明

### 問題の提示
離散確率変数ベクトル $\mathbf{x}_n$ に対するベルヌーイ分布の確率値が常に $0 \le p(\mathbf{x}_n | \boldsymbol{\mu}_k) \le 1$ を満たすことから、ベルヌーイ混合モデルの不完全データ対数尤度関数
$$ \ln p(\mathbf{X}|\boldsymbol{\mu}, \boldsymbol{\pi}) = \sum_{n=1}^N \ln \left\{ \sum_{k=1}^K \pi_k p(\mathbf{x}_n | \boldsymbol{\mu}_k) \right\} $$
が上に有界（$\le 0$）であり、したがってガウス混合モデルに見られるような尤度が正の無限大へ発散する**特異点（singularity）が存在しない**ことを証明せよ。

### [解答の道筋と穴埋め]
1. **単一成分の確率の有界性**:
   各成分の確率質量関数は $0 \le \mu_{ki} \le 1$ に対し：
   $$ p(\mathbf{x}_n | \boldsymbol{\mu}_k) = \prod_{i=1}^D \mu_{ki}^{x_{ni}} (1 - \mu_{ki})^{1 - x_{ni}} $$
   各因子は $0$ 以上 $1$ 以下の値を取るため、その積も必ず $[ \text{①} ]$ を満たす。
2. **混合確率の有界性**:
   $\pi_k \ge 0$ かつ $\sum_{k=1}^K \pi_k = 1$ であるため、凸結合の性質より：
   $$ 0 \le p(\mathbf{x}_n) = \sum_{k=1}^K \pi_k p(\mathbf{x}_n | \boldsymbol{\mu}_k) \le \sum_{k=1}^K \pi_k \cdot 1 = 1 $$
3. **対数尤度の上界**:
   対数関数は単調増加であり、$\ln(1) = 0$ であるため：
   $$ \ln p(\mathbf{x}_n) \le 0 \implies \ln p(\mathbf{X}) = \sum_{n=1}^N \ln p(\mathbf{x}_n) \le [ \text{②} ] $$
   連続変数のガウス分布では共分散行列の分散 $\sigma_k^2 \to 0$ のとき確率密度が $+\infty$ に発散し得るが、離散確率分布の確率質量は決して $1$ を超えないため、特異点が生じることは原理的にあり得ない。

### 穴埋めの解答
- ①: $0 \le p(\mathbf{x}_n | \boldsymbol{\mu}_k) \le 1$
- ②: $0$"""

    ex9_17_code = r"""# Exercise 9.17 数値検証: BMM の対数尤度が常に <= 0 であることの実験的検証
N, D, K = 30, 5, 3
np.random.seed(42)
X = (np.random.rand(N, D) > 0.5).astype(float)

# 多様なパラメータ（境界付近を含む）で評価
for trial in range(5):
    mu_rand = np.random.uniform(0.01, 0.99, size=(K, D))
    pi_rand = np.random.dirichlet(np.ones(K))
    
    log_lik = 0.0
    for n in range(N):
        p_xn = sum(pi_rand[k] * np.prod(mu_rand[k]**X[n] * (1 - mu_rand[k])**(1 - X[n])) for k in range(K))
        assert p_xn <= 1.0 + 1e-12, "Discrete probability cannot exceed 1!"
        log_lik += np.log(p_xn)
    
    assert log_lik <= 0.0, "Log likelihood of discrete mixture must be non-positive!"

print("Exercise 9.17 verified: BMM log likelihood is strictly bounded above by 0 with no singularities!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_17_md), nbf.v4.new_code_cell(ex9_17_code)])

    # --- Exercise 9.18 ---
    ex9_18_md = r"""---
## <a id="Exercise-9.18"></a>Exercise 9.18: Beta-Dirichlet 事前分布を持つベルヌーイ混合モデルの MAP-EM アルゴリズム導出

### 問題の提示
9.3.3節のベルヌーイ混合モデルにおいて、各成分パラメータ $\mu_{ki}$ に対する独立なベータ事前分布
$$ p(\mu_{ki} | a_{ki}, b_{ki}) = \frac{\Gamma(a_{ki} + b_{ki})}{\Gamma(a_{ki})\Gamma(b_{ki})} \mu_{ki}^{a_{ki} - 1} (1 - \mu_{ki})^{b_{ki} - 1} $$
および混合係数 $\boldsymbol{\pi}$ に対するディリクレ事前分布
$$ p(\boldsymbol{\pi} | \boldsymbol{\alpha}) = \frac{\Gamma\left(\sum_k \alpha_k\right)}{\prod_k \Gamma(\alpha_k)} \prod_{k=1}^K \pi_k^{\alpha_k - 1} $$
を導入する。パラメータの事後分布 $p(\boldsymbol{\mu}, \boldsymbol{\pi}|\mathbf{X})$ を最大化する MAP-EM アルゴリズムの Eステップおよび Mステップ更新式を導出せよ。

### [解答の道筋と穴埋め]
1. **Eステップ**:
   Exercise 9.4 で示した通り、事前分布は潜在変数 $\mathbf{Z}$ に依存しないため、事後負担率 $\gamma(z_{nk})$ の計算式は最尤推定の場合と完全に同一である：
   $$ \gamma(z_{nk}) = \frac{\pi_k p(\mathbf{x}_n|\boldsymbol{\mu}_k)}{\sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j)} $$
2. **Mステップ ($\mu_{ki}$ の更新)**:
   最大化すべき目的関数は $\mathcal{Q} + \ln p(\boldsymbol{\mu}) + \ln p(\boldsymbol{\pi})$ である。$\mu_{ki}$ に依存する項を取り出すと：
   $$ \sum_{n=1}^N \gamma(z_{nk}) [x_{ni} \ln \mu_{ki} + (1 - x_{ni}) \ln(1 - \mu_{ki})] + (a_{ki} - 1)\ln \mu_{ki} + (b_{ki} - 1)\ln(1 - \mu_{ki}) $$
   これを $\mu_{ki}$ で微分して $0$ と置くと：
   $$ \frac{\sum_n \gamma(z_{nk}) x_{ni} + a_{ki} - 1}{\mu_{ki}} - \frac{\sum_n \gamma(z_{nk}) (1 - x_{ni}) + b_{ki} - 1}{1 - \mu_{ki}} = 0 $$
   したがって：
   $$ \mu_{ki} = [ \text{①} ] $$
3. **Mステップ ($\pi_k$ の更新)**:
   ラグランジュ未定乗数法により $\sum_k \pi_k = 1$ を課して最大化すると：
   $$ \frac{N_k + \alpha_k - 1}{\pi_k} + \lambda = 0 \implies \pi_k = [ \text{②} ] $$
   ここで $\alpha_0 = \sum_{k=1}^K \alpha_k$。事前分布が擬似カウントとして滑らかに加算されることが確認できる。

### 穴埋めの解答
- ①: $\frac{\sum_{n=1}^N \gamma(z_{nk}) x_{ni} + a_{ki} - 1}{N_k + a_{ki} + b_{ki} - 2}$
- ②: $\frac{N_k + \alpha_k - 1}{N + \sum_{j=1}^K \alpha_j - K}$"""

    ex9_18_code = r"""# Exercise 9.18 数値検証: MAP-EM による事前分布擬似カウントの反映と目的関数単調増加
N, D, K = 30, 3, 2
X = (np.random.rand(N, D) > 0.6).astype(float)
# 事前分布パラメータ (Beta, Dirichlet)
a = np.full((K, D), 2.0)
b = np.full((K, D), 2.0)
alpha = np.full(K, 3.0)

mu = np.random.uniform(0.3, 0.7, size=(K, D))
pi = np.full(K, 1.0 / K)

def map_obj(X, mu, pi, a, b, alpha):
    log_post = 0.0
    for n in range(N):
        p_n = sum(pi[k] * np.prod(mu[k]**X[n] * (1 - mu[k])**(1 - X[n])) for k in range(K))
        log_post += np.log(max(p_n, 1e-15))
    # Prior on mu
    for k in range(K):
        for i in range(D):
            log_post += (a[k, i] - 1) * np.log(mu[k, i]) + (b[k, i] - 1) * np.log(1 - mu[k, i])
    # Prior on pi
    for k in range(K):
        log_post += (alpha[k] - 1) * np.log(pi[k])
    return log_post

history = [map_obj(X, mu, pi, a, b, alpha)]
for it in range(10):
    # Eステップ
    resp = np.zeros((N, K))
    for n in range(N):
        dens = np.array([pi[k] * np.prod(mu[k]**X[n] * (1 - mu[k])**(1 - X[n])) for k in range(K)])
        resp[n] = dens / np.sum(dens)
    
    # Mステップ
    N_k = resp.sum(axis=0)
    for k in range(K):
        for i in range(D):
            mu[k, i] = (np.sum(resp[:, k] * X[:, i]) + a[k, i] - 1.0) / (N_k[k] + a[k, i] + b[k, i] - 2.0)
    pi = (N_k + alpha - 1.0) / (N + np.sum(alpha) - K)
    
    history.append(map_obj(X, mu, pi, a, b, alpha))

diffs = np.diff(history)
assert np.all(diffs >= -1e-10), "MAP objective must be non-decreasing!"
print(f"Exercise 9.18 verified: MAP-EM monotonically increased objective from {history[0]:.2f} to {history[-1]:.2f}!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_18_md), nbf.v4.new_code_cell(ex9_18_code)])

    return cells

print("get_ex_9_10_to_9_18 defined.")
