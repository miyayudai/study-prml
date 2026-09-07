# scripts/build_ch9_part1.py
"""
Definitions for Chapter 9 exercises 9.1 to 9.9.
"""

import nbformat as nbf

def get_ex_9_1_to_9_9():
    cells = []

    # --- Exercise 9.1 ---
    ex9_1_md = r"""---
## <a id="Exercise-9.1"></a>Exercise 9.1: K-means アルゴリズムの有限回反復収束証明

### 問題の提示
9.1節で論じられた K-means アルゴリズムを考える。$N$ 個のデータ点を $K$ 個のクラスタに割り当てる場合の数が高々 $K^N$ 通り（有限集合）であること、および歪み尺度
$$ J = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2 \quad (9.1) $$
が各ステップで単調非増加であることを用いて、K-means アルゴリズムが必ず有限回の反復で厳密に停止（収束）することを示せ。

### [解答の道筋と穴埋め]
1. **有限状態空間**:
   $N$ 点の各々に $1$ から $K$ のクラスタラベルを割り当てる全配置の集合は、高々 [ ① ] 通りの有限集合である。
2. **中心ベクトルの最適性**:
   クラスタ割り当て $\{r_{nk}\}$ が固定されたとき、歪み尺度 $J$ は各 $\boldsymbol{\mu}_k$ に関して凸な2次形式であり、
   $$ \boldsymbol{\mu}_k = \frac{\sum_n r_{nk} \mathbf{x}_n}{\sum_n r_{nk}} \quad (9.4) $$
   において大域的かつ [ ② ] に最小化される。
3. **歪み尺度の厳密な単調減少性**:
   - $r_{nk}$ の更新ステップ（割り当てステップ）：各点を最も近い中心 $\boldsymbol{\mu}_k$ に割り当てるため、$J$ は非増加（減少または不変）となる。
   - $\boldsymbol{\mu}_k$ の更新ステップ（中心更新ステップ）：重心を再計算するため、$J$ は非増加となる。
   割り当て $\{r_{nk}\}$ に変化が生じる限り、$J$ は前回の値より厳密に減少する。
4. **有限ステップ停止**:
   $J$ が厳密に減少するため、アルゴリズムが過去に訪れた割り当て配置に再び戻る（サイクルする）ことはあり得ない。
   可能な割り当ての総数は有限であるため、高々有限回のステップで割り当てが一切変化しない状態（$\Delta r_{nk} = 0$）に到達し、アルゴリズムは確実に停止する。

### 穴埋めの解答
- ①: $K^N$
- ②: 一意"""

    ex9_1_code = r"""# Exercise 9.1 数値検証: K-means の有限ステップ停止性と歪み尺度 J の単調非増加性
import numpy as np

np.random.seed(42)
N, D, K = 60, 2, 3
X = np.vstack([
    np.random.randn(20, D) + np.array([0, 0]),
    np.random.randn(20, D) + np.array([4, 4]),
    np.random.randn(20, D) + np.array([-4, 4]),
])

# 初期クラスタ中心
centers = X[np.random.choice(N, K, replace=False)].copy()
J_history = []
assignment_history = []

for step in range(100):
    # 1. 割り当てステップ (r_nk)
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)**2
    assignments = np.argmin(dists, axis=1)
    J = np.sum(np.min(dists, axis=1))
    J_history.append(J)
    assignment_history.append(tuple(assignments))

    # 収束判定
    if step > 0 and assignment_history[-1] == assignment_history[-2]:
        print(f"K-means terminated strictly at step {step} with {len(set(assignment_history))} visited states.")
        break

    # 2. 中心更新ステップ
    for k in range(K):
        pts = X[assignments == k]
        if len(pts) > 0:
            centers[k] = np.mean(pts, axis=0)

# J が単調減少していることを検証
diffs = np.diff(J_history)
assert np.all(diffs <= 1e-10), "Distortion J must be monotonically non-increasing!"
print(f"Exercise 9.1 verified: J decreased from {J_history[0]:.2f} to {J_history[-1]:.2f} without cycling.")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_1_md), nbf.v4.new_code_cell(ex9_1_code)])

    # --- Exercise 9.2 ---
    ex9_2_md = r"""---
## <a id="Exercise-9.2"></a>Exercise 9.2: Robbins-Monro 逐次推定法によるオンライン K-means 更新則の導出

### 問題の提示
2.3.5節で論じられた Robbins-Monro 逐次根探索手順を、回帰関数
$$ g(\boldsymbol{\mu}_k) = \mathbb{E}_{\mathbf{x}} [2 r_{nk} (\boldsymbol{\mu}_k - \mathbf{x})] $$
の根（すなわち歪み尺度 $J$ の勾配期待値がゼロとなる点）の探索に適用せよ。
これにより、逐次 K-means 更新式 (9.5)
$$ \boldsymbol{\mu}_k^{(\tau)} = \boldsymbol{\mu}_k^{(\tau-1)} + \eta_\tau (\mathbf{x}_n - \boldsymbol{\mu}_k^{(\tau-1)}) \quad (9.5) $$
（ここでデータ点 $\mathbf{x}_n$ がクラスタ $k$ に割り当てられた場合）が自然に導出されることを示せ。

### [解答の道筋と穴埋め]
1. **Robbins-Monro の一般形式**:
   目的とする回帰関数 $g(\boldsymbol{\mu}) = 0$ の根を求めるため、観測された確率的サンプル $z_\tau$ を用いて：
   $$ \boldsymbol{\mu}^{(\tau)} = \boldsymbol{\mu}^{(\tau-1)} - a_\tau [ \text{①} ] $$
   の形で更新を行う。ここで学習率 $a_\tau$ は $\sum_\tau a_\tau = \infty, \sum_\tau a_\tau^2 < \infty$ を満たす。
2. **K-means 歪み関数の勾配観測**:
   単一データ点 $\mathbf{x}_n$ に対する歪み $J_n = \sum_k r_{nk} \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2$ の $\boldsymbol{\mu}_k$ に関する勾配は：
   $$ \nabla_{\boldsymbol{\mu}_k} J_n = 2 r_{nk} (\boldsymbol{\mu}_k - \mathbf{x}_n) $$
   データ点 $\mathbf{x}_n$ がクラスタ $k$ に割り当てられている（$r_{nk} = 1$）とき、不偏勾配観測値は $z_\tau = 2(\boldsymbol{\mu}_k^{(\tau-1)} - \mathbf{x}_n)$ となる。
3. **逐次更新式の整合**:
   これを Robbins-Monro 更新則に代入すると：
   $$ \boldsymbol{\mu}_k^{(\tau)} = \boldsymbol{\mu}_k^{(\tau-1)} - a_\tau \cdot 2(\boldsymbol{\mu}_k^{(\tau-1)} - \mathbf{x}_n) = \boldsymbol{\mu}_k^{(\tau-1)} + [ \text{②} ] (\mathbf{x}_n - \boldsymbol{\mu}_k^{(\tau-1)}) $$
   学習率 $\eta_\tau = 2 a_\tau$ と置くことで、式 (9.5) が厳密に得られる。

### 穴埋めの解答
- ①: $z_\tau$ (勾配の観測サンプル)
- ②: $2 a_\tau$"""

    ex9_2_code = r"""# Exercise 9.2 数値検証: Robbins-Monro 逐次 K-means と標本平均の漸近一致
N = 1000
true_mean = np.array([3.0, -2.0])
X_samples = true_mean + np.random.randn(N, 2)

# 逐次推定 (Robbins-Monro: eta_tau = 1 / tau)
mu_seq = np.array([0.0, 0.0])
for tau in range(1, N + 1):
    x_n = X_samples[tau - 1]
    eta_tau = 1.0 / tau
    mu_seq = mu_seq + eta_tau * (x_n - mu_seq)

batch_mean = np.mean(X_samples, axis=0)
np.testing.assert_allclose(mu_seq, batch_mean, atol=1e-12)
print(f"Exercise 9.2 verified: Robbins-Monro sequential estimate {mu_seq} exactly equals batch mean {batch_mean}!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_2_md), nbf.v4.new_code_cell(ex9_2_code)])

    # --- Exercise 9.3 ---
    ex9_3_md = r"""---
## <a id="Exercise-9.3"></a>Exercise 9.3: ガウス混合モデルにおける周辺分布と事後負担率（ベイズの定理）の導出

### 問題の提示
潜在変数 $\mathbf{z} \in \{0, 1\}^K$（1-of-K 表現）の周辺分布が式 (9.10)
$$ p(\mathbf{z}) = \prod_{k=1}^K \pi_k^{z_k} \quad (9.10) $$
で与えられ、$\mathbf{z}$ が与えられたときの観測変数 $\mathbf{x}$ の条件付き分布が式 (9.11)
$$ p(\mathbf{x}|\mathbf{z}) = \prod_{k=1}^K \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \mathbf{\Sigma}_k)^{z_k} \quad (9.11) $$
で与えられるガウス混合モデル (GMM) を考える。
1. $\mathbf{z}$ を周辺化することにより、周辺分布 $p(\mathbf{x})$ が式 (9.7) の標準的な混合ガウス分布になることを示せ。
2. ベイズの定理を用いて、事後負担率 $\gamma(z_k) \equiv p(z_k=1|\mathbf{x})$ が式 (9.13) で与えられることを示せ。

### [解答の道筋と穴埋め]
1. **潜在変数の周辺化**:
   $p(\mathbf{x}) = \sum_{\mathbf{z}} p(\mathbf{z}) p(\mathbf{x}|\mathbf{z})$ を計算する。
   $\mathbf{z}$ の取り得る状態は、$j$ 番目の要素のみが 1 で他が 0 である $K$ 個の単位ベクトル $\mathbf{e}_j$ である。
   $\mathbf{z} = \mathbf{e}_j$ のとき、$p(\mathbf{z}) = \pi_j$ かつ $p(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_j, \mathbf{\Sigma}_j)$ となるため：
   $$ p(\mathbf{x}) = \sum_{j=1}^K \pi_j [ \text{①} ] $$
2. **事後確率（負担率）の導出**:
   ベイズの定理より：
   $$ \gamma(z_k) = p(z_k = 1 | \mathbf{x}) = \frac{p(z_k = 1) p(\mathbf{x} | z_k = 1)}{\sum_{j=1}^K p(z_j = 1) p(\mathbf{x} | z_j = 1)} = \frac{\pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \mathbf{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_j, \mathbf{\Sigma}_j)} = [ \text{②} ] $$
   これにより、Eステップで計算される負担率の確率論的意味が明確になる。

### 穴埋めの解答
- ①: $\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_j, \mathbf{\Sigma}_j)$
- ②: $\gamma(z_k)$"""

    ex9_3_code = r"""# Exercise 9.3 数値検証: 1-of-K 潜在変数周辺化とベイズ負担率の厳密性
K = 3
pi = np.array([0.2, 0.5, 0.3])
mu = np.array([-2.0, 0.0, 3.0])
sigma = np.array([0.5, 1.0, 0.8])

def norm_pdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s)**2) / (np.sqrt(2 * np.pi) * s)

# テスト点
x_test = 0.5

# 1. 周辺分布 p(x)
p_x = sum(pi[k] * norm_pdf(x_test, mu[k], sigma[k]) for k in range(K))

# 2. 事後負担率 gamma(z_k)
gamma = np.array([pi[k] * norm_pdf(x_test, mu[k], sigma[k]) for k in range(K)]) / p_x
np.testing.assert_allclose(np.sum(gamma), 1.0, atol=1e-12)

# 周辺密度の数値積分による規格化確認
grid = np.linspace(-10, 10, 1000)
p_grid = sum(pi[k] * norm_pdf(grid, mu[k], sigma[k]) for k in range(K))
integral = np.trapezoid(p_grid, grid)
np.testing.assert_allclose(integral, 1.0, atol=1e-3)

print(f"Exercise 9.3 verified: Marginal p(x={x_test}) = {p_x:.4f}, Responsibilities = {gamma}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_3_md), nbf.v4.new_code_cell(ex9_3_code)])

    # --- Exercise 9.4 ---
    ex9_4_md = r"""---
## <a id="Exercise-9.4"></a>Exercise 9.4: パラメータ事前分布を持つモデルにおける MAP 推定 EM アルゴリズム

### 問題の提示
観測データ $\mathbf{X}$ と潜在変数 $\mathbf{Z}$ を含む確率モデルにおいて、パラメータ $\boldsymbol{\theta}$ に対する事前分布 $p(\boldsymbol{\theta})$ が与えられているとする。
パラメータの事後確率 $p(\boldsymbol{\theta}|\mathbf{X})$ を最大化（MAP 推定）するために EM アルゴリズムを適用する場合：
1. Eステップは最尤推定の場合と全く同一であること
2. Mステップにおいて最大化すべき目的関数が
   $$ \mathcal{Q}(\boldsymbol{\theta}, \boldsymbol{\theta}^{(\mathrm{old})}) + \ln p(\boldsymbol{\theta}) $$
   となることを示せ。

### [解答の道筋と穴埋め]
1. **パラメータ事後分布の対数分解**:
   $$ \ln p(\boldsymbol{\theta}|\mathbf{X}) = \ln p(\mathbf{X}, \boldsymbol{\theta}) - \ln p(\mathbf{X}) = \ln p(\mathbf{X}|\boldsymbol{\theta}) + \ln p(\boldsymbol{\theta}) - \ln p(\mathbf{X}) $$
   ここで $\ln p(\mathbf{X})$ は $\boldsymbol{\theta}$ に依存しない定数である。
2. **対数尤度の変分下界の導入**:
   任意の潜在変数分布 $q(\mathbf{Z})$ に対し：
   $$ \ln p(\mathbf{X}|\boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \mathrm{KL}(q \| p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})) $$
   両辺に $\ln p(\boldsymbol{\theta})$ を加えると：
   $$ \ln p(\mathbf{X}, \boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \ln p(\boldsymbol{\theta}) + \mathrm{KL}(q \| p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})) $$
3. **Eステップ**:
   現在のパラメータ $\boldsymbol{\theta}^{(\mathrm{old})}$ において下界を最大化するには、$\mathrm{KL}$ ダイバージェンスをゼロにすればよい。
   $\boldsymbol{\theta}$ の事前分布は $\mathbf{Z}$ に依存しないため、最適な $q(\mathbf{Z})$ は：
   $$ q(\mathbf{Z}) = [ \text{①} ] $$
   となり、最尤推定の場合と完全に同一である。
4. **Mステップ**:
   固定された $q(\mathbf{Z}) = p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta}^{(\mathrm{old})})$ の下で、下界 $\mathcal{L}(q, \boldsymbol{\theta}) + \ln p(\boldsymbol{\theta})$ を $\boldsymbol{\theta}$ に関して最大化する。
   期待完全データ対数尤度 $\mathcal{Q}(\boldsymbol{\theta}, \boldsymbol{\theta}^{(\mathrm{old})}) = \mathbb{E}_{\mathbf{Z}}[\ln p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta})]$ を用いると、
   最大化対象は：
   $$ [ \text{②} ] $$
   となる。

### 穴埋めの解答
- ①: $p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta}^{(\mathrm{old})})$
- ②: $\mathcal{Q}(\boldsymbol{\theta}, \boldsymbol{\theta}^{(\mathrm{old})}) + \ln p(\boldsymbol{\theta})$"""

    ex9_4_code = r"""# Exercise 9.4 数値検証: MAP-EM における Eステップの同一性と Mステップ下界単調増加
# 1次元2成分 GMM with Gaussian prior on means: p(mu_k) = N(mu_k | 0, sigma0^2)
np.random.seed(42)
N = 50
X = np.concatenate([np.random.randn(25) * 0.5 - 2, np.random.randn(25) * 0.5 + 2])
sigma0_sq = 4.0 # Prior variance on mu

# 初期値
mu = np.array([-0.5, 0.5])
pi = np.array([0.5, 0.5])
var = np.array([1.0, 1.0])

def compute_map_objective(X, mu, pi, var, sigma0_sq):
    # ln p(X | theta) + ln p(theta)
    log_lik = 0.0
    for x in X:
        p_x = sum(pi[k] * np.exp(-0.5 * (x - mu[k])**2 / var[k]) / np.sqrt(2 * np.pi * var[k]) for k in range(2))
        log_lik += np.log(p_x)
    log_prior = -0.5 * np.sum(mu**2) / sigma0_sq
    return log_lik + log_prior

map_history = [compute_map_objective(X, mu, pi, var, sigma0_sq)]

for it in range(10):
    # Eステップ: 最尤と全く同一
    resp = np.zeros((N, 2))
    for n, x in enumerate(X):
        dens = np.array([pi[k] * np.exp(-0.5 * (x - mu[k])**2 / var[k]) / np.sqrt(2 * np.pi * var[k]) for k in range(2)])
        resp[n] = dens / np.sum(dens)

    # Mステップ: MAP 更新 (mu_k の解)
    # d/d mu_k [ -0.5 * sum_n gamma_{nk} (x_n - mu_k)^2 / var_k - 0.5 * mu_k^2 / sigma0_sq ] = 0
    # => mu_k * (N_k / var_k + 1 / sigma0_sq) = sum_n gamma_{nk} x_n / var_k
    N_k = resp.sum(axis=0)
    for k in range(2):
        mu[k] = (np.sum(resp[:, k] * X) / var[k]) / (N_k[k] / var[k] + 1.0 / sigma0_sq)
    pi = N_k / N
    
    map_history.append(compute_map_objective(X, mu, pi, var, sigma0_sq))

diffs = np.diff(map_history)
assert np.all(diffs >= -1e-10), "MAP objective must monotonically increase!"
print(f"Exercise 9.4 verified: MAP objective monotonically increased from {map_history[0]:.2f} to {map_history[-1]:.2f}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_4_md), nbf.v4.new_code_cell(ex9_4_code)])

    # --- Exercise 9.5 ---
    ex9_5_md = r"""---
## <a id="Exercise-9.5"></a>Exercise 9.5: d分離による潜在変数事後分布のデータ点間因数分解証明

### 問題の提示
図 9.6 に示されるガウス混合モデルの有向グラフを考える。
8.2節で論じられた d分離基準（d-separation criterion）を用いて、所与のパラメータ $(\boldsymbol{\mu}, \mathbf{\Sigma}, \boldsymbol{\pi})$ および全観測データ $\mathbf{X}$ の下で、潜在変数 $\mathbf{Z} = \{\mathbf{z}_1, \dots, \mathbf{z}_N\}$ の事後分布が各データ点について完全に因数分解すること：
$$ p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\mu}, \mathbf{\Sigma}, \boldsymbol{\pi}) = \prod_{n=1}^N p(\mathbf{z}_n | \mathbf{x}_n, \boldsymbol{\mu}, \mathbf{\Sigma}, \boldsymbol{\pi}) \quad (9.28) $$
を示せ。

### [解答の道筋と穴埋め]
1. **有向グラフの構造**:
   パラメータノード $\{\boldsymbol{\mu}, \mathbf{\Sigma}, \boldsymbol{\pi}\}$ から各潜在変数 $\mathbf{z}_n$ へエッジが伸び、各潜在変数から対応する観測変数 $\mathbf{x}_n$ へエッジ $\mathbf{z}_n \rightarrow \mathbf{x}_n$ が伸びている。
2. **異なるデータ点間の経路**:
   任意の異なる2つの潜在変数 $\mathbf{z}_m$ と $\mathbf{z}_n$ ($m \ne n$) の間の唯一の経路は、共通の親である [ ① ] ノードを経由する分岐経路（tail-to-tail）
   $$ \mathbf{z}_m \leftarrow \{\boldsymbol{\mu}, \mathbf{\Sigma}, \boldsymbol{\pi}\} \rightarrow \mathbf{z}_n $$
   である。
3. **d分離の判定**:
   パラメータ $\{\boldsymbol{\mu}, \mathbf{\Sigma}, \boldsymbol{\pi}\}$ は [ ② ] されているため、分岐経路（tail-to-tail）は完全に遮断（ブロック）される。
   また、観測ノード $\mathbf{x}_m, \mathbf{x}_n$ は合流点（head-to-head）を持たない。
   したがって：
   $$ \mathbf{z}_m \perp\!\!\!\perp \mathbf{z}_n \mid \mathbf{X}, \boldsymbol{\mu}, \mathbf{\Sigma}, \boldsymbol{\pi} $$
   が任意の $m \ne n$ について成立し、全潜在変数の事後確率は各データ点ごとの積へと因数分解する。

### 穴埋めの解答
- ①: パラメータ
- ②: 観測（所与と）"""

    ex9_5_code = r"""# Exercise 9.5 数値検証: 2点 GMM における結合事後分布と独立積の完全一致
from prml.graphical import check_d_separation

# 1. d分離のグラフ検証
# adj: 'params' -> 'z1', 'params' -> 'z2', 'z1' -> 'x1', 'z2' -> 'x2'
adj = {
    'params': ['z1', 'z2'],
    'z1': ['x1'],
    'z2': ['x2'],
    'x1': [],
    'x2': []
}
# params と x1, x2 を所与としたとき、z1 と z2 は d分離される
is_separated = check_d_separation(adj, ['z1'], ['z2'], ['params', 'x1', 'x2'])
assert is_separated, "z1 and z2 must be d-separated given params and observations!"

# 2. 確率テンソルの直接計算
pi = np.array([0.4, 0.6])
mu = np.array([1.0, -1.0])
x1, x2 = 0.8, -1.2

# p(z1, z2 | x1, x2)
joint_p = np.zeros((2, 2))
for z1 in range(2):
    for z2 in range(2):
        joint_p[z1, z2] = (pi[z1] * np.exp(-0.5 * (x1 - mu[z1])**2)) * (pi[z2] * np.exp(-0.5 * (x2 - mu[z2])**2))
joint_p /= joint_p.sum()

# 各点独立の p(z1 | x1) * p(z2 | x2)
p_z1 = np.array([pi[k] * np.exp(-0.5 * (x1 - mu[k])**2) for k in range(2)])
p_z1 /= p_z1.sum()
p_z2 = np.array([pi[k] * np.exp(-0.5 * (x2 - mu[k])**2) for k in range(2)])
p_z2 /= p_z2.sum()

np.testing.assert_allclose(joint_p, np.outer(p_z1, p_z2), atol=1e-12)
print("Exercise 9.5 verified: p(z1, z2 | x1, x2, theta) strictly factorizes into p(z1|x1) * p(z2|x2)!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_5_md), nbf.v4.new_code_cell(ex9_5_code)])

    # --- Exercise 9.6 ---
    ex9_6_md = r"""---
## <a id="Exercise-9.6"></a>Exercise 9.6: 共通共分散行列 $\mathbf{\Sigma}_k = \mathbf{\Sigma}$ を持つ GMM の EM アルゴリズム導出

### 問題の提示
すべての混合成分が共通の共分散行列 $\mathbf{\Sigma}_k = \mathbf{\Sigma}$ を共有するガウス混合モデルを考える。
この制約の下で尤度関数を最大化する EM アルゴリズムの更新方程式を導出せよ。

### [解答の道筋と穴埋め]
1. **完全データ期待対数尤度 $\mathcal{Q}$**:
   $$ \mathcal{Q} = -\frac{1}{2} \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \left[ D \ln(2\pi) + \ln|\mathbf{\Sigma}| + (\mathbf{x}_n - \boldsymbol{\mu}_k)^{\mathrm{T}} \mathbf{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) \right] + \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \ln \pi_k $$
2. **中心ベクトル $\boldsymbol{\mu}_k$ および混合係数 $\pi_k$ の更新**:
   $\boldsymbol{\mu}_k$ および $\pi_k$ に関する項は共通共分散の場合も通常モデルと全く同一の形をしているため、
   $$ \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) \mathbf{x}_n, \quad \pi_k = [ \text{①} ] $$
   となる。ここで $N_k = \sum_{n=1}^N \gamma(z_{nk})$。
3. **共通共分散行列 $\mathbf{\Sigma}$ の最大化**:
   トレースの性質 $\mathbf{v}^{\mathrm{T}} \mathbf{\Sigma}^{-1} \mathbf{v} = \mathrm{Tr}(\mathbf{\Sigma}^{-1} \mathbf{v} \mathbf{v}^{\mathrm{T}})$ を用いて $\mathbf{\Sigma}^{-1}$ で微分する：
   $$ \frac{\partial \mathcal{Q}}{\partial \mathbf{\Sigma}^{-1}} = \frac{1}{2} \left[ \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \right] \mathbf{\Sigma} - \frac{1}{2} \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^{\mathrm{T}} $$
   $\sum_{k=1}^K \gamma(z_{nk}) = 1$ より、第1項の係数は $\sum_{n=1}^N 1 = N$ となる。これをゼロと置くことで：
   $$ \mathbf{\Sigma} = [ \text{②} ] \sum_{k=1}^K \sum_{n=1}^N \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^{\mathrm{T}} = \sum_{k=1}^K \frac{N_k}{N} \mathbf{\Sigma}_k^{\mathrm{ind}} $$
   すなわち、各クラスタの個別共分散の加重平均となる。

### 穴埋めの解答
- ①: $\frac{N_k}{N}$
- ②: $\frac{1}{N}$"""

    ex9_6_code = r"""# Exercise 9.6 数値検証: 共通共分散 GMM の M ステップ加重平均解
N, D, K = 80, 2, 3
np.random.seed(42)
X = np.random.randn(N, D)
gamma = np.random.dirichlet(np.ones(K), size=N)
mu = np.random.randn(K, D)

# 各クラスタ個別共分散
covs_ind = []
for k in range(K):
    diff = X - mu[k]
    cov_k = (gamma[:, k:k+1] * diff).T @ diff / np.sum(gamma[:, k])
    covs_ind.append(cov_k)

# 共通共分散公式: 1/N sum_k sum_n gamma_nk (x_n - mu_k)(x_n - mu_k)^T
Sigma_common = np.zeros((D, D))
for k in range(K):
    diff = X - mu[k]
    Sigma_common += (gamma[:, k:k+1] * diff).T @ diff
Sigma_common /= N

# N_k / N による個別共分散の加重和と完全一致
Sigma_weighted = sum((np.sum(gamma[:, k]) / N) * covs_ind[k] for k in range(K))
np.testing.assert_allclose(Sigma_common, Sigma_weighted, atol=1e-12)
print("Exercise 9.6 verified: Common covariance equals the exact weighted sum of separate covariances!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_6_md), nbf.v4.new_code_cell(ex9_6_code)])

    # --- Exercise 9.7 ---
    ex9_7_md = r"""---
## <a id="Exercise-9.7"></a>Exercise 9.7: 完全データ対数尤度最大化による各成分パラメータの独立最尤推定

### 問題の提示
ガウス混合モデルにおける完全データ対数尤度（式 9.36）
$$ \ln p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\mu}, \mathbf{\Sigma}, \boldsymbol{\pi}) = \sum_{n=1}^N \sum_{k=1}^K z_{nk} \left\{ \ln \pi_k + \ln \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \mathbf{\Sigma}_k) \right\} \quad (9.36) $$
の最大化が、各成分 $k$ の平均 $\boldsymbol{\mu}_k$ および共分散 $\mathbf{\Sigma}_k$ をそのクラスタに属するデータ点集合のみから独立に適合すること、および混合係数 $\pi_k$ が各クラスタの点数割合によって与えられることを代数的に確認せよ。

### [解答の道筋と穴埋め]
1. **クラスタごとの完全分離**:
   式 (9.36) は $k$ に関する和の形をしており、相異なる成分 $j \ne k$ のパラメータ同士の結合項を一切持たない：
   $$ \ln p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta}) = \sum_{k=1}^K \left[ \sum_{n=1}^N z_{nk} \ln \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \mathbf{\Sigma}_k) \right] + \sum_{k=1}^K \left( \sum_{n=1}^N z_{nk} \right) \ln \pi_k $$
2. **中心と共分散の閉形式解**:
   データ点集合 $\mathcal{C}_k = \{n : z_{nk} = 1\}$ の点数を $N_k = \sum_n z_{nk}$ とすると、
   $$ \boldsymbol{\mu}_k = \frac{\sum_{n=1}^N z_{nk} \mathbf{x}_n}{\sum_{n=1}^N z_{nk}} = \frac{1}{N_k} \sum_{n \in \mathcal{C}_k} \mathbf{x}_n = [ \text{①} ] $$
   $$ \mathbf{\Sigma}_k = \frac{1}{N_k} \sum_{n \in \mathcal{C}_k} (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^{\mathrm{T}} $$
   となり、通常の単一ガウス分布の標本平均・標本共分散と完全に一致する。
3. **混合係数の解**:
   ラグランジュ未定乗数 $\lambda (\sum_k \pi_k - 1)$ を用いて最大化すると、
   $$ \pi_k = \frac{N_k}{N} = [ \text{②} ] $$
   となり、全データに対する所属割合となる。

### 穴埋めの解答
- ①: $\bar{\mathbf{x}}_k$ (クラスタ $k$ の標本平均)
- ②: $\frac{1}{N} \sum_{n=1}^N z_{nk}$"""

    ex9_7_code = r"""# Exercise 9.7 数値検証: 完全データ対数尤度最大化とクラスタ標本統計量の一致
N, D, K = 100, 2, 3
X = np.random.randn(N, D)
# 既知の潜在変数 (ハード割り当て)
labels = np.random.choice(K, size=N)
Z = np.zeros((N, K))
for n in range(N):
    Z[n, labels[n]] = 1.0

# 1. 式 (9.36) に基づく重み付き計算
N_k = Z.sum(axis=0)
mu_weighted = (Z.T @ X) / N_k[:, None]
pi_weighted = N_k / N

# 2. 各クラスタデータ点のみを抽出した通常標本統計量
for k in range(K):
    X_k = X[labels == k]
    mu_k_sample = np.mean(X_k, axis=0)
    pi_k_sample = len(X_k) / N
    np.testing.assert_allclose(mu_weighted[k], mu_k_sample, atol=1e-12)
    np.testing.assert_allclose(pi_weighted[k], pi_k_sample, atol=1e-12)

print("Exercise 9.7 verified: Complete-data MLE exactly matches individual cluster sample statistics!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_7_md), nbf.v4.new_code_cell(ex9_7_code)])

    # --- Exercise 9.8 ---
    ex9_8_md = r"""---
## <a id="Exercise-9.8"></a>Exercise 9.8: 期待完全データ対数尤度の中心ベクトル $\boldsymbol{\mu}_k$ に関する閉形式最大化

### 問題の提示
負担率 $\gamma(z_{nk})$ を固定したとき、式 (9.40) で定義される期待完全データ対数尤度
$$ \mathcal{Q}(\boldsymbol{\theta}, \boldsymbol{\theta}^{(\mathrm{old})}) = \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \left\{ \ln \pi_k + \ln \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \mathbf{\Sigma}_k) \right\} \quad (9.40) $$
を $\boldsymbol{\mu}_k$ に関して最大化することにより、Mステップの閉形式解 (9.17)
$$ \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) \mathbf{x}_n \quad (9.17) $$
が得られることを示せ。

### [解答の道筋と穴埋め]
1. **$\boldsymbol{\mu}_k$ に依存する項の抽出**:
   $$ \mathcal{Q}(\boldsymbol{\mu}_k) = -\frac{1}{2} \sum_{n=1}^N \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k)^{\mathrm{T}} \mathbf{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) + \mathrm{const} $$
2. **勾配ベクトルの計算**:
   ベクトル微分の公式 $\nabla_{\boldsymbol{\mu}} (\mathbf{x} - \boldsymbol{\mu})^{\mathrm{T}} \mathbf{A} (\mathbf{x} - \boldsymbol{\mu}) = -2 \mathbf{A} (\mathbf{x} - \boldsymbol{\mu})$ より：
   $$ \nabla_{\boldsymbol{\mu}_k} \mathcal{Q} = [ \text{①} ] \sum_{n=1}^N \gamma(z_{nk}) \mathbf{\Sigma}_k^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_k) $$
3. **停留条件と解の導出**:
   $\nabla_{\boldsymbol{\mu}_k} \mathcal{Q} = \mathbf{0}$ と置き、正定値行列 $\mathbf{\Sigma}_k^{-1}$ を左から乗じて相殺すると：
   $$ \sum_{n=1}^N \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k) = \mathbf{0} \implies \left( \sum_{n=1}^N \gamma(z_{nk}) \right) \boldsymbol{\mu}_k = \sum_{n=1}^N \gamma(z_{nk}) \mathbf{x}_n $$
   有効データ点数を $N_k = \sum_{n=1}^N \gamma(z_{nk})$ と置くことで、
   $$ \boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) \mathbf{x}_n = [ \text{②} ] $$
   が導出される。

### 穴埋めの解答
- ①: $\mathbf{\Sigma}_k^{-1}$
- ②: 式 (9.17)"""

    ex9_8_code = r"""# Exercise 9.8 数値検証: Q の中心ベクトル勾配が M ステップ解で厳密にゼロとなることの検証
N, D = 40, 2
X = np.random.randn(N, D)
gamma_k = np.random.rand(N)
Sigma_k = np.array([[2.0, 0.5], [0.5, 1.5]])
Sigma_k_inv = np.linalg.inv(Sigma_k)

# M ステップ公式による mu_k
N_k = np.sum(gamma_k)
mu_k_opt = np.sum(gamma_k[:, None] * X, axis=0) / N_k

# 勾配の直接評価
grad = np.zeros(D)
for n in range(N):
    grad += gamma_k[n] * Sigma_k_inv @ (X[n] - mu_k_opt)

np.testing.assert_allclose(grad, np.zeros(D), atol=1e-12)
print("Exercise 9.8 verified: Gradient of expected log likelihood is strictly zero at analytic mu_k!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_8_md), nbf.v4.new_code_cell(ex9_8_code)])

    # --- Exercise 9.9 ---
    ex9_9_md = r"""---
## <a id="Exercise-9.9"></a>Exercise 9.9: 期待完全データ対数尤度の共分散 $\mathbf{\Sigma}_k$ および混合係数 $\pi_k$ に関する閉形式最大化

### 問題の提示
負担率 $\gamma(z_{nk})$ を固定したとき、式 (9.40) の $\mathcal{Q}(\boldsymbol{\theta}, \boldsymbol{\theta}^{(\mathrm{old})})$ を $\mathbf{\Sigma}_k$ および $\pi_k$ に関して最大化することにより、Mステップの閉形式解 (9.19)
$$ \mathbf{\Sigma}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^{\mathrm{T}} \quad (9.19) $$
および (9.22)
$$ \pi_k = \frac{N_k}{N} \quad (9.22) $$
が得られることを示せ。

### [解答の道筋と穴埋め]
1. **共分散行列 $\mathbf{\Sigma}_k$ の最大化**:
   精度行列 $\mathbf{\Lambda}_k = \mathbf{\Sigma}_k^{-1}$ を用いると：
   $$ \mathcal{Q}(\mathbf{\Lambda}_k) = \frac{1}{2} \sum_{n=1}^N \gamma(z_{nk}) \left[ \ln|\mathbf{\Lambda}_k| - \mathrm{Tr}\left( \mathbf{\Lambda}_k (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^{\mathrm{T}} \right) \right] + \mathrm{const} $$
   $\frac{\partial \ln|\mathbf{\Lambda}|}{\partial \mathbf{\Lambda}} = \mathbf{\Lambda}^{-1} = \mathbf{\Sigma}_k$ より：
   $$ \frac{\partial \mathcal{Q}}{\partial \mathbf{\Lambda}_k} = \frac{1}{2} N_k \mathbf{\Sigma}_k - \frac{1}{2} \sum_{n=1}^N \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^{\mathrm{T}} = \mathbf{0} $$
   これを解くことで式 (9.19) が得られる。
2. **混合係数 $\pi_k$ の制約付き最大化**:
   和が 1 となる制約 $\sum_k \pi_k = 1$ の下でラグランジュ関数
   $$ \mathcal{L}(\boldsymbol{\pi}, \lambda) = \sum_{k=1}^K \sum_{n=1}^N \gamma(z_{nk}) \ln \pi_k + \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) = \sum_{k=1}^K N_k \ln \pi_k + \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) $$
   を定義する。$\pi_k$ で微分してゼロと置くと：
   $$ \frac{N_k}{\pi_k} + \lambda = 0 \implies \pi_k = -\frac{N_k}{\lambda} $$
   両辺を $k$ について総和をとると $1 = -\frac{\sum_k N_k}{\lambda} = -\frac{N}{\lambda} \implies \lambda = [ \text{①} ]$。
   したがって、
   $$ \pi_k = [ \text{②} ] $$
   が導出される。

### 穴埋めの解答
- ①: $-N$
- ②: $\frac{N_k}{N}$"""

    ex9_9_code = r"""# Exercise 9.9 数値検証: 期待完全データ対数尤度の Sigma_k, pi_k 停留条件の厳密検証
N, D, K = 50, 2, 3
X = np.random.randn(N, D)
gamma = np.random.dirichlet(np.ones(K), size=N)
mu = np.random.randn(K, D)

N_k = gamma.sum(axis=0)

# 1. pi_k = N_k / N の検証
pi_opt = N_k / N
np.testing.assert_allclose(np.sum(pi_opt), 1.0, atol=1e-12)

# 2. Sigma_k の勾配ゼロ性検証
for k in range(K):
    diff = X - mu[k]
    Sigma_k_opt = (gamma[:, k:k+1] * diff).T @ diff / N_k[k]
    Lambda_k = np.linalg.inv(Sigma_k_opt)
    
    # dQ / d Lambda_k = 0.5 * (N_k * Sigma_k - sum_n gamma_nk diff diff^T)
    grad_Lambda = 0.5 * (N_k[k] * Sigma_k_opt - (gamma[:, k:k+1] * diff).T @ diff)
    np.testing.assert_allclose(grad_Lambda, np.zeros((D, D)), atol=1e-12)

print("Exercise 9.9 verified: Closed form Sigma_k and pi_k satisfy stationary conditions exactly!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex9_9_md), nbf.v4.new_code_cell(ex9_9_code)])

    return cells

print("get_ex_9_1_to_9_9 defined.")
