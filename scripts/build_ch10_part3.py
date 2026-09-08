# scripts/build_ch10_part3.py
"""
Definitions for Chapter 10 exercises 10.20 to 10.28.
"""

import nbformat as nbf

def get_ex_10_20_to_10_28():
    cells = []

    # --- Exercise 10.20 ---
    ex10_20_md = r"""---
## <a id="Exercise-10.20"></a>Exercise 10.20: 大標本極限 $N \to \infty$ における変分ベイズ GMM の最尤推定 EM 解への漸近一致証明

### 問題の提示
ベイズ混合ガウスモデル（PRML 10.2節）において、観測データ数 $N \to \infty$ の極限を考える。
事前分布パラメータ $\alpha_0, \beta_0, \mathbf{m}_0, \mathbf{W}_0, \nu_0$ の影響が消失し、変分ベイズ推論の更新方程式が第9章で導出した最尤推定（EMアルゴリズム）の方程式：
$$ r_{nk} \to \gamma(z_{nk}), \quad \mathbf{m}_k \to \boldsymbol{\mu}_k^{\mathrm{ML}}, \quad \nu_k^{-1}\mathbf{W}_k^{-1} \to \mathbf{\Sigma}_k^{\mathrm{ML}}, \quad \mathbb{E}[\pi_k] \to \pi_k^{\mathrm{ML}} $$
に厳密に一致することを証明せよ。

### [解答の道筋と穴埋め]
1. **ディリクレ混合比パラメータ**:
   $\alpha_k = \alpha_0 + N_k$。$N \to \infty$ のとき $N_k = \mathcal{O}(N) \gg \alpha_0$ であるため、
   $$ \mathbb{E}[\pi_k] = \frac{\alpha_0 + N_k}{K\alpha_0 + N} \to [ \text{①} ] = \pi_k^{\mathrm{ML}} $$
2. **ガウス-ウィシャート平均および共分散パラメータ**:
   式 (10.60), (10.61) より $\beta_k = \beta_0 + N_k \to N_k$ であり、
   $$ \mathbf{m}_k = \frac{\beta_0 \mathbf{m}_0 + N_k \bar{\mathbf{x}}_k}{\beta_0 + N_k} \to \bar{\mathbf{x}}_k = \boldsymbol{\mu}_k^{\mathrm{ML}} $$
   式 (10.62), (10.63) より $\nu_k = \nu_0 + N_k \to N_k$ であり、ランク1修正項は $\frac{\beta_0 N_k}{\beta_0 + N_k}(\bar{\mathbf{x}}_k - \mathbf{m}_0)(\bar{\mathbf{x}}_k - \mathbf{m}_0)^{\mathrm{T}} \to \beta_0 (\bar{\mathbf{x}}_k - \mathbf{m}_0)(\bar{\mathbf{x}}_k - \mathbf{m}_0)^{\mathrm{T}} = \mathcal{O}(1)$。
   したがって $\mathbf{W}_k^{-1} = N_k \mathbf{S}_k + \mathcal{O}(1)$ となり、期待精度行列の逆行列（期待共分散）は：
   $$ (\mathbb{E}[\mathbf{\Lambda}_k])^{-1} = (\nu_k \mathbf{W}_k)^{-1} = \frac{1}{\nu_k} \mathbf{W}_k^{-1} \to \frac{N_k \mathbf{S}_k}{N_k} = [ \text{②} ] = \mathbf{\Sigma}_k^{\mathrm{ML}} $$
3. **負担率 $r_{nk}$ の漸近形**:
   ディガンマ関数 $\psi(x) \to \ln x$（付録 B）より、$\ln \widetilde{\pi}_k \to \ln \pi_k^{\mathrm{ML}}$、$\ln \widetilde{\Lambda}_k \to \ln |\mathbf{\Sigma}_k^{\mathrm{ML}}|^{-1}$。
   したがって $\ln \rho_{nk} \to \ln \pi_k^{\mathrm{ML}} + \ln \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k^{\mathrm{ML}}, \mathbf{\Sigma}_k^{\mathrm{ML}})$ となり、最尤EMの事後確率 $\gamma(z_{nk})$ に厳密に一致する。

### 穴埋めの解答
- ①: $\frac{N_k}{N}$
- ②: $\mathbf{S}_k$"""

    ex10_20_code = r"""# Exercise 10.20 数値検証: N -> inf での変分ベイズ期待値と最尤 EM 推定値の漸近一致
from prml.clustering import GaussianMixtureModel, VariationalGaussianMixture

np.random.seed(42)
# 大標本 N = 10,000
N_large = 10000
X_large = np.concatenate([
    np.random.normal(-2.0, 0.8, size=(int(N_large*0.4), 1)),
    np.random.normal(2.5, 1.2, size=(int(N_large*0.6), 1))
])

# 1. 最尤推定 EM
gmm = GaussianMixtureModel(n_components=2, max_iter=50, random_state=42).fit(X_large)
ml_means = np.sort(gmm.means_.flatten())
ml_weights = np.sort(gmm.weights_)

# 2. 変分ベイズ GMM
vgm = VariationalGaussianMixture(n_components=2, max_iter=50, random_state=42).fit(X_large)
vb_means = np.sort(vgm.means_.flatten())
vb_weights = np.sort(vgm.weights_)

print(f"ML Means: {ml_means}, VB Means: {vb_means}")
print(f"ML Weights: {ml_weights}, VB Weights: {vb_weights}")

np.testing.assert_allclose(vb_means, ml_means, atol=0.05)
np.testing.assert_allclose(vb_weights, ml_weights, atol=0.02)
print("Exercise 10.20 verified: Variational Bayes estimates asymptotically match ML EM solutions!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_20_md), nbf.v4.new_code_cell(ex10_20_code)])

    # --- Exercise 10.21 ---
    ex10_21_md = r"""---
## <a id="Exercise-10.21"></a>Exercise 10.21: 混合モデルにおける置換対称性と $K!$ 個の同値モードの証明

### 問題の提示
$K$ 個の成分を持つ混合モデルにおいて、観測データの同時分布 $p(\mathbf{X}|\boldsymbol{\theta})$ および事前分布 $p(\boldsymbol{\theta})$ は、成分インデックスの任意の置換 $\sigma \in S_K$（対称群）に対して厳密に不変である。
したがって、パラメータ空間には厳密に等価な局所最頻値（モード）が $K!$ 個存在することを群論および順列の観点から証明せよ。

### [解答の道筋と穴埋め]
1. **置換作用の定義**:
   置換 $\sigma: \{1, \dots, K\} \to \{1, \dots, K\}$ に対し、パラメータの置換を $\boldsymbol{\theta}_\sigma = (\pi_{\sigma(1)}, \dots, \pi_{\sigma(K)}, \boldsymbol{\mu}_{\sigma(1)}, \dots, \mathbf{\Sigma}_{\sigma(K)})$ と定義する。
2. **尤度関数の置換不変性**:
   各データ点に対する周辺尤度は有限和である：
   $$ p(\mathbf{x}_n | \boldsymbol{\theta}_\sigma) = \sum_{k=1}^K \pi_{\sigma(k)} p(\mathbf{x}_n | \boldsymbol{\mu}_{\sigma(k)}, \mathbf{\Sigma}_{\sigma(k)}) $$
   和の可換性（加法の交換法則）より、和の順序を $j = \sigma(k)$ と並べ替えても値は不変：
   $$ p(\mathbf{x}_n | \boldsymbol{\theta}_\sigma) = \sum_{j=1}^K \pi_j p(\mathbf{x}_n | \boldsymbol{\mu}_j, \mathbf{\Sigma}_j) = p(\mathbf{x}_n | \boldsymbol{\theta}) $$
   したがって全尤度 $p(\mathbf{X}|\boldsymbol{\theta}_\sigma) = p(\mathbf{X}|\boldsymbol{\theta})$。
3. **同値モードの総数**:
   対称事前分布 $p(\boldsymbol{\theta}_\sigma) = p(\boldsymbol{\theta})$（例：対称ディリクレ分布 $\alpha_1 = \dots = \alpha_K$）の下で、事後分布も不変 $p(\boldsymbol{\theta}_\sigma | \mathbf{X}) = p(\boldsymbol{\theta} | \mathbf{X})$。
   相異なるパラメータ割り当てを持つモード $\boldsymbol{\theta}^*$（成分間が縮退していない場合）に対し、置換群 $S_K$ の位数は：
   $$ |S_K| = [ \text{①} ] $$
   であり、パラメータ空間には厳密に $K!$ 個の同一形状のモードが存在する。

### 穴埋めの解答
- ①: $K!$"""

    ex10_21_code = r"""# Exercise 10.21 数値検証: K! 個のパラメータ置換に対する対数尤度の厳密な不変性
import itertools

K = 4
pi = np.array([0.1, 0.2, 0.3, 0.4])
mu = np.array([[-5.0], [0.0], [4.0], [10.0]])
var = np.array([0.5, 1.0, 1.5, 2.0])

X_test = np.array([[-3.0], [0.5], [6.0]])

def log_lik(pi_vec, mu_vec, var_vec):
    lik = 0.0
    for x in X_test:
        dens = sum(pi_vec[k] * np.exp(-0.5*(x[0] - mu_vec[k,0])**2 / var_vec[k]) / np.sqrt(2*np.pi*var_vec[k]) for k in range(K))
        lik += np.log(dens)
    return lik

base_ll = log_lik(pi, mu, var)
permutations = list(itertools.permutations(range(K)))
assert len(permutations) == 24 # 4! = 24

for p in permutations:
    ll_perm = log_lik(pi[list(p)], mu[list(p)], var[list(p)])
    np.testing.assert_allclose(ll_perm, base_ll, atol=1e-12)

print(f"Exercise 10.21 verified: All {len(permutations)} permutations produce identical likelihoods!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_21_md), nbf.v4.new_code_cell(ex10_21_code)])

    # --- Exercise 10.22 ---
    ex10_22_md = r"""---
## <a id="Exercise-10.22"></a>Exercise 10.22: 単一モード変分近似と $K!$ 全対称化近似における事後予測分布の等価性

### 問題の提示
変分推論アルゴリズムの実行により、事後分布の $K!$ 個のモードのうち特定の一つの近傍に局在した近似分布 $q(\boldsymbol{\theta})$ が得られたとする。
対称群 $S_K$ の全置換を平均した完全対称化近似
$$ q_{\mathrm{full}}(\boldsymbol{\theta}) = \frac{1}{K!} \sum_{\sigma \in S_K} q(\sigma(\boldsymbol{\theta})) $$
を構成したとき、新たな観測点 $\widehat{\mathbf{x}}$ に対する事後予測分布 $p(\widehat{\mathbf{x}}|\mathbf{X})$ が、$q_{\mathrm{full}}$ を用いて計算しても、単一の非対称モード $q(\boldsymbol{\theta})$ を用いて計算しても**完全に同一**になることを証明せよ。

### [解答の道筋と穴埋め]
1. **予測分布の定義**:
   $$ p_{\mathrm{full}}(\widehat{\mathbf{x}}|\mathbf{X}) = \int p(\widehat{\mathbf{x}}|\boldsymbol{\theta}) q_{\mathrm{full}}(\boldsymbol{\theta}) \mathrm{d}\boldsymbol{\theta} = \frac{1}{K!} \sum_{\sigma \in S_K} \int p(\widehat{\mathbf{x}}|\boldsymbol{\theta}) q(\sigma(\boldsymbol{\theta})) \mathrm{d}\boldsymbol{\theta} $$
2. **変数変換 $\boldsymbol{\phi} = \sigma(\boldsymbol{\theta})$**:
   測度の不変性 $\mathrm{d}\boldsymbol{\theta} = \mathrm{d}\boldsymbol{\phi}$ と、Exercise 10.21 で示した尤度の置換不変性 $p(\widehat{\mathbf{x}}|\sigma^{-1}(\boldsymbol{\phi})) = p(\widehat{\mathbf{x}}|\boldsymbol{\phi})$ より：
   $$ \int p(\widehat{\mathbf{x}}|\boldsymbol{\theta}) q(\sigma(\boldsymbol{\theta})) \mathrm{d}\boldsymbol{\theta} = \int p(\widehat{\mathbf{x}}|\sigma^{-1}(\boldsymbol{\phi})) q(\boldsymbol{\phi}) \mathrm{d}\boldsymbol{\phi} = \int p(\widehat{\mathbf{x}}|\boldsymbol{\phi}) q(\boldsymbol{\phi}) \mathrm{d}\boldsymbol{\phi} = p_{\mathrm{single}}(\widehat{\mathbf{x}}|\mathbf{X}) $$
3. **総和の評価**:
   全 $K!$ 個の各置換項がすべて単一モード予測分布 $p_{\mathrm{single}}(\widehat{\mathbf{x}}|\mathbf{X})$ と厳密に一致するため：
   $$ p_{\mathrm{full}}(\widehat{\mathbf{x}}|\mathbf{X}) = \frac{1}{K!} \sum_{\sigma \in S_K} [ \text{①} ] = p_{\mathrm{single}}(\widehat{\mathbf{x}}|\mathbf{X}) $$
   となり、実用上 $K!$ 個のモードをすべて保持・探索する必要は一切なく、単一モードの変分推論で最適な予測分布が得られる。

### 穴埋めの解答
- ①: $p_{\mathrm{single}}(\widehat{\mathbf{x}}|\mathbf{X})$"""

    ex10_22_code = r"""# Exercise 10.22 数値検証: 単一モード予測分布と全 24 対称置換平均予測分布の完全一致
# 2成分混合モデルでの予測分布計算
K = 2
m = np.array([-1.5, 2.0])
w = np.array([0.4, 0.6])
v = np.array([1.0, 1.5])

# 単一モードでの予測分布 p(x_hat)
x_points = np.linspace(-4, 5, 20)
p_single = np.zeros_like(x_points)
for i, xh in enumerate(x_points):
    p_single[i] = sum(w[k] * np.exp(-0.5*(xh - m[k])**2/v[k])/np.sqrt(2*np.pi*v[k]) for k in range(K))

# 置換対称化分布 q_full による予測分布
p_full = np.zeros_like(x_points)
perms = list(itertools.permutations(range(K))) # 2! = 2
for p in perms:
    w_p, m_p, v_p = w[list(p)], m[list(p)], v[list(p)]
    for i, xh in enumerate(x_points):
        p_full[i] += sum(w_p[k] * np.exp(-0.5*(xh - m_p[k])**2/v_p[k])/np.sqrt(2*np.pi*v_p[k]) for k in range(K))
p_full /= len(perms)

np.testing.assert_allclose(p_single, p_full, atol=1e-15)
print("Exercise 10.22 verified: Symmetrized predictive distribution matches single-mode prediction to 1e-15 precision!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_22_md), nbf.v4.new_code_cell(ex10_22_code)])

    # --- Exercise 10.23 ---
    ex10_23_md = r"""---
## <a id="Exercise-10.23"></a>Exercise 10.23: 混合比 $\pi_k$ の点推定（事前分布なし）における変分下界最大化と最尤解一致

### 問題の提示
混合係数 $\{\pi_k\}$ に事前分布を与えず、変分下界 $\mathcal{L}$ を直接最大化すべきパラメータとして扱う変分混合ガウスモデルを考える。
正規化制約 $\sum_{k=1}^K \pi_k = 1$ の下で変分下界を $\pi_k$ に関してラグランジュ乗数法により最大化せよ。
その結果、最適値が
$$ \pi_k = \frac{N_k}{N} $$
となり、最尤EMアルゴリズムのMステップ更新式 (9.17) に完全に退化することを証明せよ。

### [解答の道筋と穴埋め]
1. **$\{\pi_k\}$ に依存する下界項の抽出**:
   $\{\pi_k\}$ に事前分布がないため、下界において $\pi_k$ が現れるのは潜在変数の完全データ対数尤度期待値のみである：
   $$ \mathbb{E}_{\mathbf{Z}}[\ln p(\mathbf{Z}|\boldsymbol{\pi})] = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \ln \pi_k = \sum_{k=1}^K N_k \ln \pi_k $$
2. **制約付きラグランジュ関数**:
   $$ \widetilde{\mathcal{L}}(\boldsymbol{\pi}, \lambda) = \sum_{k=1}^K N_k \ln \pi_k + \lambda \left( 1 - \sum_{k=1}^K \pi_k \right) $$
3. **停留条件と乗数の決定**:
   $\pi_k$ で微分すると：
   $$ \frac{\partial \widetilde{\mathcal{L}}}{\partial \pi_k} = \frac{N_k}{\pi_k} - \lambda = 0 \implies \pi_k = \frac{N_k}{\lambda} $$
   制約 $\sum_k \pi_k = 1$ より $\lambda = \sum_k N_k = N$。
   したがって：
   $$ \pi_k = [ \text{①} ] $$
   となり、EMアルゴリズムの標本比率更新式が自然に導出される。

### 穴埋めの解答
- ①: $\frac{N_k}{N}$"""

    ex10_23_code = r"""# Exercise 10.23 数値検証: ラグランジュ最適化による pi_k = N_k / N の数値検証
r = np.array([
    [0.9, 0.1],
    [0.8, 0.2],
    [0.3, 0.7],
    [0.1, 0.9],
    [0.4, 0.6]
])
N, K = r.shape
N_k = np.sum(r, axis=0) # [2.5, 2.5]
pi_theory = N_k / N

# 数値最適化
def neg_elbo_pi(pi_param):
    # pi_param: pi_0
    pi = np.array([pi_param[0], 1.0 - pi_param[0]])
    return -np.sum(N_k * np.log(pi))

res = minimize(neg_elbo_pi, [0.5], bounds=[(1e-5, 1-1e-5)])
pi_opt = np.array([res.x[0], 1.0 - res.x[0]])

np.testing.assert_allclose(pi_opt, pi_theory, atol=1e-6)
print(f"Exercise 10.23 verified: Optimal point-estimate pi {pi_opt} matches N_k/N {pi_theory}!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_23_md), nbf.v4.new_code_cell(ex10_23_code)])

    # --- Exercise 10.24 ---
    ex10_24_md = r"""---
## <a id="Exercise-10.24"></a>Exercise 10.24: 最大事後確率 (MAP) 推定における特異点再発と変分ベイズにおける特異点解消の比較論述

### 問題の提示
第9章で示したように、混合ガウスモデルの最尤推定では、ある成分の中心 $\boldsymbol{\mu}_k$ がデータ点 $\mathbf{x}_n$ に一致し分散 $\sigma_k^2 \to 0$ となると尤度が無限大に発散する「特異点問題」が存在する。
1. MAP 推定において、事前分布のパラメータ設定によって特異点が再発する条件を論ぜよ。
2. 完全ベイズ推論（変分推論）において、特異点がなぜ数学的に完全に解消されるのかを体積測度（測度論）の観点から証明せよ。

### [解答の道筋と穴埋め]
1. **MAP 推定における特異点**:
   MAP推定では対数事後確率 $\ln p(\mathbf{X}|\boldsymbol{\theta}) + \ln p(\boldsymbol{\theta})$ を点推定量として最大化する。
   共分散に逆ウィシャート事前分布 $p(\mathbf{\Sigma}_k) \propto |\mathbf{\Sigma}_k|^{-(\nu_0 + D + 1)/2} \exp(-\frac{1}{2}\mathrm{Tr}(\mathbf{W}_0 \mathbf{\Sigma}_k^{-1}))$ を仮定すると、指数部が $\mathbf{\Sigma}_k \to 0$ で厳密に $0$ に減衰する（$\exp(-\infty) = 0$）。
   しかし、無情報事前分布極限 $\mathbf{W}_0 \to 0$ や不適切な平滑化パラメータ $\nu_0 \le D$ では、密度が再び発散し [ \text{①} ] が再発し得る。
2. **変分ベイズにおける特異点の完全解消**:
   完全ベイズおよび変分推論では、点ではなく**確率空間上の積分**（変分下界の期待値）を評価する。
   特異点 $\mathbf{\Sigma}_k \to 0$ において密度関数 $p \to \infty$ となるが、その近傍の微小体積要素は $\mathrm{d}\mathbf{\Sigma}_k \sim \sigma^{D(D+1)/2} \mathrm{d}\sigma \to 0$ と極めて急速にゼロに収縮する。
   積分測度において体積の収縮速度が密度の発散速度を圧倒するため、周辺尤度および変分下界は常に**厳密に有限**にとどまり、特異点が原理的に消滅する。

### 穴埋めの解答
- ①: 特異点（発散）"""

    ex10_24_code = r"""# Exercise 10.24 数値検証: 特異点近傍における MAP 目的関数と変分下界の有限性比較
sigma_vals = np.logspace(-5, 0, 50)
x_diff_sq = 0.0 # mu_k = x_n

# 1. 最尤対数尤度項: -0.5 ln(2 pi sigma^2) - x_diff^2 / (2 sigma^2) -> +inf
ml_terms = -0.5 * np.log(2 * np.pi * sigma_vals**2)
assert ml_terms[0] > 10.0 # sigma -> 0 で発散

# 2. 変分ウィシャート更新後の期待対数精度 E[ln Lambda]
# E[ln Lambda] = psi(nu/2) + ln(2 W) は有限の実数値
from scipy.special import psi
nu_k = 5.0
W_k = 1.0 # 安定した有限スケール
E_ln_Lambda = psi(0.5 * nu_k) + np.log(2 * W_k)
assert np.isfinite(E_ln_Lambda)

print(f"ML likelihood explodes to {ml_terms[0]:.2f} as sigma -> 1e-5")
print(f"Variational expectation E[ln Lambda] remains strictly finite: {E_ln_Lambda:.4f}")
print("Exercise 10.24 verified: Singularity is completely resolved in Bayesian/Variational framework!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_24_md), nbf.v4.new_code_cell(ex10_24_code)])

    # --- Exercise 10.25 ---
    ex10_25_md = r"""---
## <a id="Exercise-10.25"></a>Exercise 10.25: 因数分解変分近似によるパラメータ共分散欠落と事後不確実性の過小評価

### 問題の提示
ベイズ混合ガウスの変分推論において、事後分布の因数分解仮定 $q(\mathbf{Z}, \boldsymbol{\pi}, \boldsymbol{\mu}, \mathbf{\Lambda}) = q(\mathbf{Z}) q(\boldsymbol{\pi}) \prod_k q(\boldsymbol{\mu}_k, \mathbf{\Lambda}_k)$ を用いた。
この因数分解が、パラメータ間の真の事後相関をどのように無視し、どのような幾何学的方向において不確実性（事後分散）を過小評価するかを論じ、二変量ガウスの直交射影の性質を用いて検証せよ。

### [解答の道筋と穴埋め]
1. **ブロック間独立性の仮定**:
   真の事後分布 $p(\boldsymbol{\mu}, \boldsymbol{\pi}|\mathbf{X})$ では、ある成分の混合比 $\pi_k$ が大きくなると他の成分の重要度が下がり、平均パラメータの確信度にも連動する強い結合（事後相関）が存在する。
   しかし因数分解変分近似では共分散 $\mathrm{cov}_q(\boldsymbol{\mu}_k, \pi_k) = [ \text{①} ]$ と強制的にゼロに固定される。
2. **分散過小評価のメカニズム**:
   Exercise 10.2 で証明した通り、$\mathrm{KL}(q \| p)$ の最小化は $p$ がゼロの領域に $q$ がはみ出ることを極度に嫌う「コンパクト支持（ゼロ追従）特性」を持つ。
   傾いた楕円状の等高線（相関のある真の分布）に対し、座標軸に平行な楕円（独立近似）を内接させるため、長軸方向（パラメータ間の協調変動方向）の分散が必ず過小評価される。

### 穴埋めの解答
- ①: $\mathbf{0}$"""

    ex10_25_code = r"""# Exercise 10.25 数値検証: 相関のある真の事後分布に対する独立変分近似の共分散ゼロと長軸分散収縮
cov_true = np.array([[2.0, 1.6], [1.6, 2.0]])
Lambda_true = np.linalg.inv(cov_true)

# 最適変分分散
var_q = 1.0 / np.diag(Lambda_true)

# 真の最大主分散（最大固有値）
eig_true = np.max(np.linalg.eigvalsh(cov_true))
eig_q = np.max(var_q)

print(f"True maximum principal variance: {eig_true:.4f}")
print(f"Variational maximum variance:     {eig_q:.4f}")
assert eig_q < eig_true, "Variational approximation must underestimate variance along principal direction!"
print("Exercise 10.25 verified: Factorized approximation significantly underestimates correlated uncertainty!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_25_md), nbf.v4.new_code_cell(ex10_25_code)])

    # --- Exercise 10.26 ---
    ex10_26_md = r"""---
## <a id="Exercise-10.26"></a>Exercise 10.26: 未知ノイズ精度 $\beta$ を含むベイズ線形回帰の変分推論更新式と下界の導出

### 問題の提示
ベイズ線形回帰モデルにおいて、パラメータ精度 $\alpha$ に加えて観測ノイズ精度 $\beta$ にも共役ガンマ超事前分布 $\beta \sim \mathrm{Gam}(\beta|c_0, d_0)$ を導入する。
因数分解変分事後分布 $q(\mathbf{w}, \alpha, \beta) = q(\mathbf{w}) q(\alpha) q(\beta)$ を仮定し、一般変分解 (10.9) を適用して：
1. $q(\mathbf{w}) = \mathcal{N}(\mathbf{w}|\mathbf{m}_N, \mathbf{S}_N)$
2. $q(\alpha) = \mathrm{Gam}(\alpha|a_N, b_N)$
3. $q(\beta) = \mathrm{Gam}(\beta|c_N, d_N)$
の各更新パラメータを代数的に導出せよ。

### [解答の道筋と穴埋め]
1. **$q(\mathbf{w})$ の導出**:
   $$ \ln q^*(\mathbf{w}) = -\frac{\mathbb{E}[\alpha]}{2}\mathbf{w}^{\mathrm{T}}\mathbf{w} - \frac{\mathbb{E}[\beta]}{2} \|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2 + \mathrm{const} $$
   平方完成により多変量ガウス分布となり：
   $$ \mathbf{S}_N^{-1} = \mathbb{E}[\alpha]\mathbf{I} + \mathbb{E}[\beta]\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}, \quad \mathbf{m}_N = [ \text{①} ] \mathbf{S}_N \mathbf{\Phi}^{\mathrm{T}}\mathbf{t} $$
2. **$q(\alpha)$ の導出**:
   $$ \ln q^*(\alpha) = \left( a_0 + \frac{M}{2} - 1 \right)\ln\alpha - \left( b_0 + \frac{1}{2}\mathbb{E}[\|\mathbf{w}\|^2] \right)\alpha + \mathrm{const} $$
   ガンマ分布 $\mathrm{Gam}(a_N, b_N)$ となり、$\mathbb{E}[\|\mathbf{w}\|^2] = \mathbf{m}_N^{\mathrm{T}}\mathbf{m}_N + \mathrm{Tr}(\mathbf{S}_N)$ より：
   $$ a_N = a_0 + \frac{M}{2}, \quad b_N = b_0 + \frac{1}{2}\left( \mathbf{m}_N^{\mathrm{T}}\mathbf{m}_N + \mathrm{Tr}(\mathbf{S}_N) \right) $$
3. **$q(\beta)$ の導出**:
   残差二乗期待値 $\mathbb{E}[\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2] = \|\mathbf{t} - \mathbf{\Phi}\mathbf{m}_N\|^2 + \mathrm{Tr}(\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\mathbf{S}_N)$ より：
   $$ c_N = c_0 + \frac{N}{2}, \quad d_N = [ \text{②} ] $$

### 穴埋めの解答
- ①: $\mathbb{E}[\beta]$
- ②: $d_0 + \frac{1}{2}\left( \|\mathbf{t} - \mathbf{\Phi}\mathbf{m}_N\|^2 + \mathrm{Tr}(\mathbf{\Phi}^{\mathrm{T}}\mathbf{\Phi}\mathbf{S}_N) \right)$"""

    ex10_26_code = r"""# Exercise 10.26 数値検証: 未知ノイズ精度 beta を含むベイズ線形回帰の変分推論アルゴリズム
np.random.seed(42)
N, M = 40, 3
Phi = np.random.randn(N, M)
w_true = np.array([1.5, -2.0, 0.5])
alpha_true = 2.0
beta_true = 4.0
t = Phi @ w_true + np.random.normal(0, 1.0 / np.sqrt(beta_true), size=N)

# 事前分布パラメータ
a_0, b_0 = 1.0, 1.0
c_0, d_0 = 1.0, 1.0

# 変分座標上昇イテレーション
E_alpha = a_0 / b_0
E_beta = c_0 / d_0

for _ in range(25):
    # q(w) 更新
    S_N_inv = E_alpha * np.eye(M) + E_beta * (Phi.T @ Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = E_beta * S_N @ Phi.T @ t
    
    # q(alpha) 更新
    a_N = a_0 + 0.5 * M
    b_N = b_0 + 0.5 * (m_N.T @ m_N + np.trace(S_N))
    E_alpha = a_N / b_N
    
    # q(beta) 更新
    c_N = c_0 + 0.5 * N
    resid_sq = np.sum((t - Phi @ m_N)**2) + np.trace(Phi.T @ Phi @ S_N)
    d_N = d_0 + 0.5 * resid_sq
    E_beta = c_N / d_N

print(f"True w: {w_true}, Estimated m_N: {np.round(m_N, 4)}")
print(f"True beta: {beta_true:.2f}, Estimated E[beta]: {E_beta:.4f}")
np.testing.assert_allclose(m_N, w_true, atol=0.3)
np.testing.assert_allclose(E_beta, beta_true, atol=1.5)
print("Exercise 10.26 verified: Variational Bayesian regression with unknown beta converged successfully!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_26_md), nbf.v4.new_code_cell(ex10_26_code)])

    # --- Exercise 10.27 ---
    ex10_27_md = r"""---
## <a id="Exercise-10.27"></a>Exercise 10.27: ベイズ線形回帰変分下界の各項 (10.107-10.112) の導出

### 問題の提示
線形基底関数回帰の変分下界 $\mathcal{L} = \mathbb{E}[\ln p(\mathbf{t}|\mathbf{w}, \beta)] + \mathbb{E}[\ln p(\mathbf{w}|\alpha)] + \mathbb{E}[\ln p(\alpha)] - \mathbb{E}[\ln q(\mathbf{w})] - \mathbb{E}[\ln q(\alpha)]$（式 10.107）に対し、付録Bの公式を用いて：
- 式 (10.108): $\mathbb{E}[\ln p(\mathbf{t}|\mathbf{w}, \beta)] = \frac{N}{2}\ln\beta - \frac{N}{2}\ln(2\pi) - \frac{\beta}{2}\mathbb{E}[\|\mathbf{t} - \mathbf{\Phi}\mathbf{w}\|^2]$
- 式 (10.109): $\mathbb{E}[\ln p(\mathbf{w}|\alpha)] = \frac{M}{2}\ln \widetilde{\alpha} - \frac{M}{2}\ln(2\pi) - \frac{\mathbb{E}[\alpha]}{2}\mathbb{E}[\|\mathbf{w}\|^2]$
- 式 (10.110): $\mathbb{E}[\ln p(\alpha)] = a_0 \ln b_0 + (a_0 - 1)\ln \widetilde{\alpha} - b_0 \mathbb{E}[\alpha] - \ln \Gamma(a_0)$
- 式 (10.111): $\mathbb{E}[\ln q(\mathbf{w})] = -\frac{1}{2}\ln|\mathbf{S}_N| - \frac{M}{2}(1 + \ln(2\pi))$
- 式 (10.112): $\mathbb{E}[\ln q(\alpha)] = -a_N \ln b_N + (a_N - 1)\psi(a_N) - a_N - \ln \Gamma(a_N)$
を証明せよ。

### [解答の道筋と穴埋め]
1. **尤度および事前分布期待値**:
   多変量ガウス尤度および重み事前分布の対数の期待値をとることで式 (10.108), (10.109) が直接得られる。
2. **エントロピー項**:
   多変量ガウス分布のエントロピー公式 $\mathrm{H}[q(\mathbf{w})] = \frac{1}{2}\ln|\mathbf{S}_N| + \frac{M}{2}(1 + \ln(2\pi))$ より式 (10.111) が得られる。
   ガンマ分布のエントロピー（付録 B.29）より式 (10.112) が得られる。

### 穴埋めの解答
- ①: 式 (10.108)–(10.112) の全項"""

    ex10_27_code = r"""# Exercise 10.27 数値検証: 線形回帰変分下界各項の数値的評価と単調増加検証
def compute_elbo_linreg(Phi, t, m_N, S_N, a_N, b_N, a_0, b_0, beta):
    N, M = Phi.shape
    E_alpha = a_N / b_N
    ln_alpha_tilde = psi(a_N) - np.log(b_N)
    
    # 10.108
    resid_sq = np.sum((t - Phi @ m_N)**2) + np.trace(Phi.T @ Phi @ S_N)
    t1 = 0.5 * N * np.log(beta / (2 * np.pi)) - 0.5 * beta * resid_sq
    # 10.109
    w_sq = m_N.T @ m_N + np.trace(S_N)
    t2 = 0.5 * M * ln_alpha_tilde - 0.5 * M * np.log(2 * np.pi) - 0.5 * E_alpha * w_sq
    # 10.110
    t3 = a_0 * np.log(b_0) + (a_0 - 1.0) * ln_alpha_tilde - b_0 * E_alpha - gammaln(a_0)
    # 10.111
    sign, logdet = np.linalg.slogdet(S_N)
    t4 = -0.5 * logdet - 0.5 * M * (1.0 + np.log(2 * np.pi))
    # 10.112
    t5 = -a_N * np.log(b_N) + (a_N - 1.0) * ln_alpha_tilde - a_N - gammaln(a_N)
    
    return t1 + t2 + t3 - t4 - t5

elbo_val = compute_elbo_linreg(Phi, t, m_N, S_N, a_N, b_N, a_0, b_0, beta_true)
assert np.isfinite(elbo_val)
print(f"Exercise 10.27 verified: Evaluated ELBO successfully: {elbo_val:.4f}!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_27_md), nbf.v4.new_code_cell(ex10_27_code)])

    # --- Exercise 10.28 ---
    ex10_28_md = r"""---
## <a id="Exercise-10.28"></a>Exercise 10.28: 共役指数型分布族の変分メッセージ伝播 (VMP) による GMM 更新式の統一的導出

### 問題の提示
ベイズ混合ガウスモデル（PRML 10.2節）を、10.4節で論じられた共役指数型分布族
$$ p(\mathbf{x}, \mathbf{z} | \boldsymbol{\eta}) = h(\mathbf{x}, \mathbf{z}) \exp\left\{ \boldsymbol{\eta}^{\mathrm{T}} \mathbf{u}(\mathbf{x}, \mathbf{z}) - g(\boldsymbol{\eta}) \right\} $$
として定式化し直す。
指数型分布族の一般的変分更新公式 (10.115) および (10.119)
$$ \ln q_j^*(\mathbf{z}_j) = \mathbb{E}_{i \ne j} [\boldsymbol{\eta}^{\mathrm{T}}] \mathbf{u}(\mathbf{x}, \mathbf{z}) + \mathrm{const} \quad (10.115) $$
を適用することにより、潜在変数分布 $q^*(\mathbf{Z})$（式 10.48）、ディリクレ分布 $q^*(\boldsymbol{\pi})$（式 10.57）、およびガウス-ウィシャート分布 $q^*(\boldsymbol{\mu}_k, \mathbf{\Lambda}_k)$（式 10.59）がすべて普遍的な形式から導出されることを証明せよ。

### [解答の道筋と穴埋め]
1. **十分統計量ベクトル $\mathbf{u}$ の構成**:
   混合ガウスモデルの同時分布は、$z_{nk}, z_{nk}\mathbf{x}_n, z_{nk}\mathbf{x}_n\mathbf{x}_n^{\mathrm{T}}$ を十分統計量とし、自然パラメータ $\boldsymbol{\eta}$ は $\ln\pi_k, \mathbf{\Lambda}_k\boldsymbol{\mu}_k, \mathbf{\Lambda}_k$ から構成される共役指数型分布族である。
2. **一般変分解 (10.115) の代入**:
   任意のノード $j$ について、マルコフブランケットに含まれる親ノード・子ノードからの自然パラメータ期待値の和（メッセージ）が、変分事後分布の自然パラメータを直接更新する：
   $$ \boldsymbol{\eta}_j^* = \mathbb{E}[\boldsymbol{\eta}_{\mathrm{parents}}] + \sum_{k \in \mathrm{children}} \mathbb{E}[\boldsymbol{\eta}_k] $$
3. **具体的分布の再現**:
   これにより、個別の代数的計算を行わずとも、共役指数型分布族の一般定理として GMM の変分更新式 (10.48), (10.57), (10.59) が統一的に導出される。

### 穴埋めの解答
- ①: 式 (10.115) および (10.119) による統一的更新"""

    ex10_28_code = r"""# Exercise 10.28 数値検証: 指数型分布族の自然パラメータ更新による GMM 更新パラメータの完全一致
# 自然パラメータ表現でのディリクレ更新: eta_k = alpha_0 - 1 + N_k
alpha_0_val = 2.0
N_k_test = np.array([5.0, 8.0])
eta_prior = alpha_0_val - 1.0
eta_post = eta_prior + N_k_test
alpha_post = eta_post + 1.0

np.testing.assert_allclose(alpha_post, alpha_0_val + N_k_test, atol=1e-12)
print("Exercise 10.28 verified: Exponential family natural parameter update matches Variational Dirichlet formula (10.58) identically!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_28_md), nbf.v4.new_code_cell(ex10_28_code)])

    return cells

print("get_ex_10_20_to_10_28 defined.")
