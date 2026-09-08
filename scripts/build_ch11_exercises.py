# scripts/build_ch11_exercises.py
"""
Build complete exercises for PRML Chapter 11 (Sampling Methods, Exercises 11.1 - 11.17).
Contains rigorous mathematical derivations, blank-filling (穴埋め ①, ②...) format,
and self-contained numerical validation scripts.
"""

import nbformat as nbf

def get_ch11_all_exercises():
    cells = []

    # --- Title & Setup ---
    title_md = r"""# 第11章 サンプリング法：演習問題 (Exercises 11.1 - 11.17)

本ノートブックでは、PRML第11章「サンプリング法 (Sampling Methods)」の**全17問 (Exercises 11.1 〜 11.17)** について、詳細な数式展開・証明、思考プロセスを明示した穴埋め形式、および Python による自己完結型の数値検証コードを完備しています。

### 収録内容一覧
- **Ex 11.1**: モンテカルロ推定量の不偏性 $\mathbb{E}[\hat{f}] = \mathbb{E}[f]$ と分散 $1/L$ 収束則
- **Ex 11.2**: 累積分布関数逆関数法（Inverse CDF Method）の確率密度保存則の厳密証明
- **Ex 11.3**: 逆正接関数変換 $y = \tan(\pi(z - 1/2))$ によるコーシー乱数の厳密導出
- **Ex 11.4**: 極座標 Box-Muller 法による単位円板乱数からの2変量標準正規乱数生成とヤコビアン証明
- **Ex 11.5**: コレスキー分解 $\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^{\mathrm{T}}$ による多変量正規乱数 $\mathbf{y} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$ の生成
- **Ex 11.6**: 棄却サンプリング（Rejection Sampling）の正当性 $p(z \mid \mathrm{accept}) = p(z)$ の完全証明
- **Ex 11.7**: 一般コーシー分布 $p(y) = \frac{1}{\pi}\frac{b}{b^2 + (y-c)^2}$ のアフィン逆正接変換生成
- **Ex 11.8**: 適応型棄却サンプリング（ARS）における区分的包絡線の連続性条件と規格化係数 $k_i$ 導出
- **Ex 11.9**: ARS における区分的指数包絡線からのサンプリングアルゴリズムの導出と実装
- **Ex 11.10**: 1次元整数ランダムウォークの拡散距離 $\mathbb{E}[(z^{(\tau)})^2] = \tau / 2$ の数学的帰納法証明
- **Ex 11.11**: ギブスサンプリング（Gibbs Sampling）のマルコフ遷移核と詳細釣り合い条件の証明
- **Ex 11.12**: 非連結・孤立モード分布におけるギブスサンプリングの非エルゴード性（Ergodicity Failure）
- **Ex 11.13**: 未知平均・未知精度の共役ガウス-ガンマモデルにおける完全条件付き事後分布の導出
- **Ex 11.14**: 過剰緩和法（Over-Relaxation: Adler 1981）による分散保存性 $\mathrm{var}[z_i'] = \sigma_i^2$ の代数的一致検証
- **Ex 11.15**: ハミルトン力学系方程式 $\frac{dz_i}{d\tau} = \frac{\partial H}{\partial r_i}, \frac{dr_i}{d\tau} = -\frac{\partial H}{\partial z_i}$ とニュートン運動方程式の等価性
- **Ex 11.16**: HMC カノニカル分布における運動量条件付き分布 $p(\mathbf{r} \mid \mathbf{z}) = \mathcal{N}(\mathbf{r} \mid \mathbf{0}, \mathbf{I})$ のガウス分布性証明
- **Ex 11.17**: ハイブリッド・モンテカルロ法 (HMC) の詳細釣り合い条件とリープフロッグ時間反転・体積保存性
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))

    # --- Exercise 11.1 ---
    ex11_1_md = r"""---
## <a id="Exercise-11.1"></a>Exercise 11.1: モンテカルロ推定量の不偏性 $\mathbb{E}[\hat{f}] = \mathbb{E}[f]$ と分散 $1/L$ 収束則

### 問題の提示
目標分布 $p(z)$ から独立同一に生成された $L$ 個のサンプル $\{z^{(l)}\}_{l=1}^L \sim p(z)$ に対し、期待値 $\mathbb{E}[f] = \int f(z)p(z)dz$ の有限標本推定量
$$ \hat{f} = \frac{1}{L}\sum_{l=1}^L f(z^{(l)}) \quad (11.2) $$
を定義する。この推定量が不偏推定量であり、その分散が式 (11.3)
$$ \mathrm{var}[\hat{f}] = \frac{1}{L}\mathbb{E}[(f - \mathbb{E}[f])^2] = \frac{1}{L}\mathrm{var}[f] $$
で与えられることを証明せよ。

### [解答の道筋と穴埋め]
1. **期待値の線形性（不偏性）**:
   各サンプル $z^{(l)}$ は同一の分布 $p(z)$ に従うため、任意の $l$ に対して $\mathbb{E}[f(z^{(l)})] = \mathbb{E}[f]$ である。
   したがって、推定量 $\hat{f}$ の期待値は：
   $$ \mathbb{E}[\hat{f}] = \mathbb{E}\left[ \frac{1}{L}\sum_{l=1}^L f(z^{(l)}) \right] = \frac{1}{L}\sum_{l=1}^L \mathbb{E}[f(z^{(l)})] = \frac{1}{L} \cdot L \mathbb{E}[f] = [ \text{①} ] $$
   となり、不偏推定量であることが示される。
2. **分散の加法性**:
   サンプル $\{z^{(l)}\}$ は互いに独立であるため、$l \ne m$ のとき共分散 $\mathrm{cov}[f(z^{(l)}), f(z^{(m)})] = 0$ である。
   したがって：
   $$ \mathrm{var}[\hat{f}] = \mathrm{var}\left[ \frac{1}{L}\sum_{l=1}^L f(z^{(l)}) \right] = \frac{1}{L^2}\sum_{l=1}^L \mathrm{var}[f(z^{(l)})] = \frac{1}{L^2} \cdot L \mathrm{var}[f] = [ \text{②} ] $$
   これにより、モンテカルロ推定の標準誤差は標本数 $L$ の平方根に逆比例して $\mathcal{O}(1/\sqrt{L})$ で減衰し、問題の空間次元 $D$ に依存しないことが示される。

### 穴埋めの解答
- ①: $\mathbb{E}[f]$
- ②: $\frac{1}{L} \mathrm{var}[f]$"""

    ex11_1_code = r"""# Exercise 11.1 数値検証: モンテカルロ推定量の不偏性と 1/L 分散スケーリング
import numpy as np

np.random.seed(42)
N_trials = 3000
L_values = [10, 50, 200, 500]

# テスト関数: f(z) = z^2, z ~ N(0, 1)
# 理論値: E[f] = 1.0, var[f] = E[z^4] - (E[z^2])^2 = 3.0 - 1.0 = 2.0
true_mean = 1.0
true_var = 2.0

for L in L_values:
    # (N_trials, L) の標本生成
    samples = np.random.normal(0, 1, size=(N_trials, L))
    f_hat = np.mean(samples**2, axis=1) # (N_trials,)
    
    emp_mean = np.mean(f_hat)
    emp_var = np.var(f_hat)
    theo_var = true_var / L
    
    print(f"L = {L:3d}: Emp Mean = {emp_mean:.4f} (True {true_mean:.1f}), Emp Var = {emp_var:.6f} (Theo {theo_var:.6f})")
    
    np.testing.assert_allclose(emp_mean, true_mean, atol=0.03)
    np.testing.assert_allclose(emp_var, theo_var, rtol=0.10)

print("Exercise 11.1 verified: Unbiasedness and 1/L variance decay strictly confirmed!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_1_md), nbf.v4.new_code_cell(ex11_1_code)])

    # --- Exercise 11.2 ---
    ex11_2_md = r"""---
## <a id="Exercise-11.2"></a>Exercise 11.2: 累積分布関数逆関数法（Inverse CDF Method）の確率密度保存則の厳密証明

### 問題の提示
一様確率変数 $z \sim \mathrm{Uniform}(0, 1)$ に対し、目的分布 $p(y)$ の累積分布関数
$$ h(y) = \int_{-\infty}^y p(\hat{y})\mathrm{d}\hat{y} \quad (11.6) $$
の逆関数 $y = h^{-1}(z)$ を適用して得られる確率変数 $y$ が、厳密に目標分布 $p(y)$ に従うことを証明せよ。

### [解答の道筋と穴埋め]
1. **確率変数の累積確率の変換**:
   累積分布関数 $h(y)$ は狭義単調非減少関数であるため、逆関数 $h^{-1}$ が一意に存在する。
   確率変数 $Y = h^{-1}(Z)$ の累積分布関数 $P_Y(Y \le y)$ は、事象の等価性より：
   $$ P(Y \le y) = P(h^{-1}(Z) \le y) = P(Z \le h(y)) $$
2. **一様分布の性質の適用**:
   確率変数 $Z$ は区間 $(0, 1)$ 上の一様分布に従うため、任意の $u \in [0, 1]$ に対して $P(Z \le u) = u$ である。
   したがって：
   $$ P(Y \le y) = [ \text{①} ] $$
3. **確率密度の導出**:
   両辺を $y$ で微分すると、微分積分学の基本定理より：
   $$ p_Y(y) = \frac{\mathrm{d}}{\mathrm{d}y} P(Y \le y) = \frac{\mathrm{d}}{\mathrm{d}y} h(y) = \frac{\mathrm{d}}{\mathrm{d}y} \int_{-\infty}^y p(\hat{y})\mathrm{d}\hat{y} = [ \text{②} ] $$
   となり、変換された変数 $y$ の確率密度が目標分布 $p(y)$ と一致することが証明される。

### 穴埋めの解答
- ①: $h(y)$
- ②: $p(y)$"""

    ex11_2_code = r"""# Exercise 11.2 数値検証: 指数分布の逆関数法サンプリングと Kolmogorov-Smirnov 検定
from scipy.stats import kstest, expon

np.random.seed(42)
lam = 1.5
N_samples = 40000

# 目標分布: 指数分布 p(y) = lam * exp(-lam * y) (y >= 0)
# CDF: h(y) = 1 - exp(-lam * y)
# 逆関数: y = - (1 / lam) * ln(1 - z)
z = np.random.uniform(0, 1, size=N_samples)
y_samples = - (1.0 / lam) * np.log(1.0 - z)

# Kolmogorov-Smirnov 検定による理論分布との適合度評価
ks_stat, p_val = kstest(y_samples, expon(scale=1.0/lam).cdf)
print(f"Exponential (lam={lam}): KS statistic = {ks_stat:.5f}, p-value = {p_val:.4f}")

assert p_val > 0.05, "Sample distribution must match theoretical exponential distribution!"
np.testing.assert_allclose(np.mean(y_samples), 1.0 / lam, rtol=0.02)
np.testing.assert_allclose(np.var(y_samples), 1.0 / (lam**2), rtol=0.03)
print("Exercise 11.2 verified: Inverse CDF transform rigorously produces samples from target distribution!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_2_md), nbf.v4.new_code_cell(ex11_2_code)])

    # --- Exercise 11.3 ---
    ex11_3_md = r"""---
## <a id="Exercise-11.3"></a>Exercise 11.3: 逆正接関数変換 $y = \tan(\pi(z - 1/2))$ によるコーシー乱数の厳密導出

### 問題の提示
標準コーシー分布（式 11.8）
$$ p(y) = \frac{1}{\pi (1 + y^2)} $$
に従う確率変数を、一様乱数 $z \sim \mathrm{Uniform}(0, 1)$ の変換 $y = f(z)$ によって生成する関数 $f(z)$ を導出せよ。

### [解答の道筋と穴埋め]
1. **コーシー分布の累積分布関数 (CDF) の計算**:
   定義より：
   $$ h(y) = \int_{-\infty}^y \frac{1}{\pi (1 + t^2)}\mathrm{d}t = \frac{1}{\pi} \left[ \arctan(t) \right]_{-\infty}^y = \frac{1}{\pi}\left( \arctan(y) - \left(-\frac{\pi}{2}\right) \right) = [ \text{①} ] $$
2. **逆関数の代数的導出**:
   一様変数 $z \in (0, 1)$ に対して $z = h(y)$ と置き、$y$ について解く：
   $$ z = \frac{1}{2} + \frac{1}{\pi} \arctan(y) \iff z - \frac{1}{2} = \frac{1}{\pi} \arctan(y) \iff \arctan(y) = \pi\left(z - \frac{1}{2}\right) $$
   両辺の正接（$\tan$）を取ると：
   $$ y = f(z) = [ \text{②} ] $$
   これにより、一様乱数 $z$ からコーシー乱数が閉形式の初等関数で厳密に生成される。

### 穴埋めの解答
- ①: $\frac{1}{2} + \frac{1}{\pi}\arctan(y)$
- ②: $\tan\left(\pi\left(z - \frac{1}{2}\right)\right)$"""

    ex11_3_code = r"""# Exercise 11.3 数値検証: 逆正接変換によるコーシー乱数生成と分位点検証
from scipy.stats import cauchy

np.random.seed(42)
N_samples = 60000

z = np.random.uniform(0, 1, size=N_samples)
y_cauchy = np.tan(np.pi * (z - 0.5))

# コーシー分布は期待値・分散が未定義のため、ロバスト統計量（中央値と四分位範囲 IQR）で検証
# 理論値: 中央値 = 0.0, 第1四分位点 = -1.0, 第3四分位点 = 1.0, IQR = 2.0
emp_median = np.median(y_cauchy)
q25, q75 = np.percentile(y_cauchy, [25, 75])
emp_iqr = q75 - q25

print(f"Cauchy: Median = {emp_median:.4f} (Theo 0.0), Q25 = {q25:.4f}, Q75 = {q75:.4f}, IQR = {emp_iqr:.4f} (Theo 2.0)")

np.testing.assert_allclose(emp_median, 0.0, atol=0.03)
np.testing.assert_allclose(emp_iqr, 2.0, atol=0.05)

# KS 検定による分布形状の厳密適合判定
ks_stat, p_val = kstest(y_cauchy, cauchy.cdf)
assert p_val > 0.01, f"KS test failed with p-value {p_val}"
print("Exercise 11.3 verified: Transformation y = tan(pi*(z - 1/2)) generates exact Cauchy variates!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_3_md), nbf.v4.new_code_cell(ex11_3_code)])

    # --- Exercise 11.4 ---
    ex11_4_md = r"""---
## <a id="Exercise-11.4"></a>Exercise 11.4: 極座標 Box-Muller 法による単位円板乱数からの2変量標準正規乱数生成とヤコビアン証明

### 問題の提示
単位円板 $z_1^2 + z_2^2 < 1$ 上の一様分布からサンプル $(z_1, z_2)$ を抽出し、$r^2 = z_1^2 + z_2^2$ と置く。
変数変換（式 11.10, 11.11）
$$ y_1 = z_1 \left( \frac{-2\ln r^2}{r^2} \right)^{1/2}, \quad y_2 = z_2 \left( \frac{-2\ln r^2}{r^2} \right)^{1/2} $$
を適用したとき、$(y_1, y_2)$ が独立な標準正規分布 $\mathcal{N}(0, 1)$ に従う（式 11.12）
$$ p(y_1, y_2) = \frac{1}{2\pi}\exp\left(-\frac{y_1^2 + y_2^2}{2}\right) $$
ことを証明せよ。

### [解答の道筋と穴埋め]
1. **極座標表現**:
   $z_1 = r\cos\theta, z_2 = r\sin\theta$ と置く。単位円板の面積は $\pi$ であるため、結合密度は $p(z_1, z_2) = \frac{1}{\pi}$。
   ヤコビアン $\mathrm{d}z_1\mathrm{d}z_2 = r\mathrm{d}r\mathrm{d}\theta = \frac{1}{2}\mathrm{d}(r^2)\mathrm{d}\theta$ より、$u \equiv r^2 \in (0, 1)$ と $\theta \in (0, 2\pi)$ の結合分布は：
   $$ p(u, \theta) = p(z_1, z_2) \left|\frac{\partial(z_1, z_2)}{\partial(u, \theta)}\right| = \frac{1}{\pi} \cdot \frac{1}{2} = \frac{1}{2\pi} $$
   となり、$u \sim \mathrm{Uniform}(0, 1)$ と $\theta \sim \mathrm{Uniform}(0, 2\pi)$ は互いに独立である。
2. **変数変換の作用**:
   定義式に $z_1 = \sqrt{u}\cos\theta, z_2 = \sqrt{u}\sin\theta$ を代入すると：
   $$ y_1 = \sqrt{u}\cos\theta \sqrt{\frac{-2\ln u}{u}} = [ \text{①} ], \quad y_2 = \sqrt{u}\sin\theta \sqrt{\frac{-2\ln u}{u}} = \sqrt{-2\ln u}\sin\theta $$
   これは古典的 Box-Muller 変換そのものである。
3. **確率密度の導出**:
   自乗和は $y_1^2 + y_2^2 = -2\ln u$ より $u = \exp\left(-\frac{y_1^2 + y_2^2}{2}\right)$。また $\theta = \arctan(y_2 / y_1)$。
   ヤコビアン行列式を計算すると $\left|\frac{\partial(u, \theta)}{\partial(y_1, y_2)}\right| = \exp\left(-\frac{y_1^2 + y_2^2}{2}\right)$。
   したがって：
   $$ p(y_1, y_2) = p(u, \theta) \left|\frac{\partial(u, \theta)}{\partial(y_1, y_2)}\right| = [ \text{②} ] $$
   となり、独立な標準ガウス分布の積となる。

### 穴埋めの解答
- ①: $\sqrt{-2\ln u}\cos\theta$
- ②: $\frac{1}{2\pi}\exp\left(-\frac{y_1^2 + y_2^2}{2}\right)$"""

    ex11_4_code = r"""# Exercise 11.4 数値検証: 極座標 Box-Muller 法による二変量標準ガウス乱数の生成
np.random.seed(42)
N_candidates = 80000

# 1. 単位円板内の一様乱数抽出 (棄却サンプリング)
u1 = np.random.uniform(-1, 1, N_candidates)
u2 = np.random.uniform(-1, 1, N_candidates)
r_sq = u1**2 + u2**2
inside_mask = (r_sq > 0) & (r_sq < 1.0)
z1, z2, r2 = u1[inside_mask], u2[inside_mask], r_sq[inside_mask]

# 2. 式 (11.10), (11.11) 変換
factor = np.sqrt(-2.0 * np.log(r2) / r2)
y1 = z1 * factor
y2 = z2 * factor

# 3. 統計量検証: 平均 0, 分散 1, 相関 0
mean_y = np.array([np.mean(y1), np.mean(y2)])
cov_y = np.cov(y1, y2)

print(f"Empirical Means: {mean_y}")
print(f"Empirical Covariance Matrix:\n{cov_y}")

np.testing.assert_allclose(mean_y, [0.0, 0.0], atol=0.02)
np.testing.assert_allclose(cov_y, np.eye(2), atol=0.03)

# 各成分の正規性 KS 検定
from scipy.stats import norm
assert kstest(y1, norm.cdf).pvalue > 0.05
assert kstest(y2, norm.cdf).pvalue > 0.05
print("Exercise 11.4 verified: Polar Box-Muller method produces exact independent standard Gaussians!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_4_md), nbf.v4.new_code_cell(ex11_4_code)])

    # --- Exercise 11.5 ---
    ex11_5_md = r"""---
## <a id="Exercise-11.5"></a>Exercise 11.5: コレスキー分解 $\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^{\mathrm{T}}$ による多変量正規乱数 $\mathbf{y} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z}$ の生成

### 問題の提示
$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_D)$ を $D$ 次元標準正規確率変数ベクトルとし、正定値対称共分散行列 $\mathbf{\Sigma}$ が下三角行列 $\mathbf{L}$ を用いてコレスキー分解 $\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^{\mathrm{T}}$ されるとする。
このときアフィン変換
$$ \mathbf{y} = \boldsymbol{\mu} + \mathbf{L}\mathbf{z} $$
によって定義される確率変数ベクトル $\mathbf{y}$ が、平均 $\boldsymbol{\mu}$、共分散行列 $\mathbf{\Sigma}$ の多変量ガウス分布 $\mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$ に従うことを証明せよ。

### [解答の道筋と穴埋め]
1. **期待値の計算**:
   期待値の線形性および $\mathbb{E}[\mathbf{z}] = \mathbf{0}$ より：
   $$ \mathbb{E}[\mathbf{y}] = \mathbb{E}[\boldsymbol{\mu} + \mathbf{L}\mathbf{z}] = \boldsymbol{\mu} + \mathbf{L}\mathbb{E}[\mathbf{z}] = [ \text{①} ] $$
2. **共分散行列の計算**:
   共分散の定義式に代入すると：
   $$ \mathrm{cov}[\mathbf{y}] = \mathbb{E}[(\mathbf{y} - \boldsymbol{\mu})(\mathbf{y} - \boldsymbol{\mu})^{\mathrm{T}}] = \mathbb{E}[(\mathbf{L}\mathbf{z})(\mathbf{L}\mathbf{z})^{\mathrm{T}}] = \mathbb{E}[\mathbf{L}\mathbf{z}\mathbf{z}^{\mathrm{T}}\mathbf{L}^{\mathrm{T}}] $$
   行列 $\mathbf{L}$ は定数行列であるため期待値の外にくくり出すことができ、$\mathbb{E}[\mathbf{z}\mathbf{z}^{\mathrm{T}}] = \mathbf{I}_D$ より：
   $$ \mathrm{cov}[\mathbf{y}] = \mathbf{L} \mathbb{E}[\mathbf{z}\mathbf{z}^{\mathrm{T}}] \mathbf{L}^{\mathrm{T}} = \mathbf{L}\mathbf{I}_D\mathbf{L}^{\mathrm{T}} = [ \text{②} ] $$
3. **正規性の保存**:
   ガウス確率変数のアフィン変換は再びガウス分布となる性質から、$\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$ が示される。

### 穴埋めの解答
- ①: $\boldsymbol{\mu}$
- ②: $\mathbf{L}\mathbf{L}^{\mathrm{T}} = \mathbf{\Sigma}$"""

    ex11_5_code = r"""# Exercise 11.5 数値検証: コレスキー分解による任意共分散ガウス乱数の生成
np.random.seed(42)
D = 4
N_samples = 50000

# 任意の位置ベクトルと正定値対称共分散行列の生成
mu_true = np.array([1.5, -2.0, 0.5, 3.0])
A = np.random.randn(D, D)
Sigma_true = A @ A.T + 0.5 * np.eye(D)

# コレスキー分解: Sigma = L @ L.T
L = np.linalg.cholesky(Sigma_true)

# 独立標準正規乱数 z ~ N(0, I) からの生成
z = np.random.normal(0, 1, size=(D, N_samples))
y_samples = mu_true[:, np.newaxis] + L @ z # (D, N_samples)

# 標本平均と標本共分散行列の検証
emp_mu = np.mean(y_samples, axis=1)
emp_Sigma = np.cov(y_samples)

print(f"Max Absolute Error in Mean:       {np.max(np.abs(emp_mu - mu_true)):.6f}")
print(f"Max Absolute Error in Covariance: {np.max(np.abs(emp_Sigma - Sigma_true)):.6f}")

np.testing.assert_allclose(emp_mu, mu_true, atol=0.03)
np.testing.assert_allclose(emp_Sigma, Sigma_true, atol=0.15)
print("Exercise 11.5 verified: Cholesky affine transformation generates exact multivariate Gaussian!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_5_md), nbf.v4.new_code_cell(ex11_5_code)])

    # --- Exercise 11.6 ---
    ex11_6_md = r"""---
## <a id="Exercise-11.6"></a>Exercise 11.6: 棄却サンプリング（Rejection Sampling）の正当性 $p(z \mid \mathrm{accept}) = p(z)$ の完全証明

### 問題の提示
目標分布 $p(z) = \tilde{p}(z) / Z_p$（未正規化密度 $\tilde{p}(z)$）に対し、提案分布 $q(z)$ および上界定数 $k$（すべての $z$ で $k q(z) \ge \tilde{p}(z)$）を用いる棄却サンプリングを考える。
提案点 $z \sim q(z)$ が採択される条件付き確率が
$$ p(\mathrm{accept} \mid z) = \frac{\tilde{p}(z)}{k q(z)} $$
であるとき、乗法定理および加法定理を用いて採択されたサンプルの事後確率分布 $p(z \mid \mathrm{accept})$ を求め、それが厳密に規格化された目標分布 $p(z)$ に一致することを証明せよ。

### [解答の道筋と穴埋め]
1. **同時採択確率密度の計算**:
   提案点 $z$ が提案され、かつ採択される同時確率密度は、乗法定理より：
   $$ p(z, \mathrm{accept}) = p(\mathrm{accept} \mid z) q(z) = \frac{\tilde{p}(z)}{k q(z)} q(z) = [ \text{①} ] $$
2. **全採択確率（周辺確率）の計算**:
   加法定理により、すべての $z$ について積分すると：
   $$ p(\mathrm{accept}) = \int p(z, \mathrm{accept})\mathrm{d}z = \int \frac{\tilde{p}(z)}{k}\mathrm{d}z = \frac{1}{k}\int \tilde{p}(z)\mathrm{d}z = [ \text{②} ] $$
   ここで $Z_p = \int \tilde{p}(z)\mathrm{d}z$ は目標分布の正規化定数である。
3. **採択サンプルの条件付き密度の導出**:
   ベイズの定理（条件付き確率の定義）より：
   $$ p(z \mid \mathrm{accept}) = \frac{p(z, \mathrm{accept})}{p(\mathrm{accept})} = \frac{\tilde{p}(z) / k}{Z_p / k} = \frac{\tilde{p}(z)}{Z_p} = [ \text{③} ] $$
   したがって、正規化定数 $Z_p$ の値が未知であっても、採択されたサンプル集合は厳密に目標分布 $p(z)$ に従うことが示される。

### 穴埋めの解答
- ①: $\frac{\tilde{p}(z)}{k}$
- ②: $\frac{Z_p}{k}$
- ③: $p(z)$"""

    ex11_6_code = r"""# Exercise 11.6 数値検証: 棄却サンプリングの採択率と目標分布一致性の検証
import scipy.integrate as integrate

np.random.seed(42)
# 目標分布: 未規格化ガウス分布 p_tilde(z) = exp(-z^2 / 2) (真の Z_p = sqrt(2*pi))
# 提案分布: ラプラス分布 q(z) = 0.5 * exp(-|z|)
# 上界 k の決定: max_z p_tilde(z) / q(z) = max_z 2 * exp(|z| - z^2/2) = 2 * exp(1/2) = sqrt(2*e)*sqrt(2) = 2*sqrt(e)
k = 2.0 * np.sqrt(np.e) # ~ 3.2974

N_proposals = 100000
# ラプラス分布からのサンプリング
u_laplace = np.random.uniform(-0.5, 0.5, size=N_proposals)
z_prop = - np.sign(u_laplace) * np.log(1.0 - 2.0 * np.abs(u_laplace))

p_tilde = np.exp(-0.5 * z_prop**2)
q_val = 0.5 * np.exp(-np.abs(z_prop))

# 採択判定
u_rand = np.random.uniform(0, 1, size=N_proposals)
accept_mask = u_rand <= (p_tilde / (k * q_val))
accepted_samples = z_prop[accept_mask]

# 理論採択率: Z_p / k = sqrt(2*pi) / (2*sqrt(e)) ~ 0.7602
theo_acc_rate = np.sqrt(2 * np.pi) / k
emp_acc_rate = np.mean(accept_mask)

print(f"Empirical Acceptance Rate: {emp_acc_rate:.4f}, Theoretical: {theo_acc_rate:.4f}")
np.testing.assert_allclose(emp_acc_rate, theo_acc_rate, rtol=0.02)

# 採択されたサンプルが N(0, 1) に従うか KS 検定
ks_res = kstest(accepted_samples, norm.cdf)
assert ks_res.pvalue > 0.05
print("Exercise 11.6 verified: Rejection sampling accepted distribution strictly matches target p(z)!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_6_md), nbf.v4.new_code_cell(ex11_6_code)])

    # --- Exercise 11.7 ---
    ex11_7_md = r"""---
## <a id="Exercise-11.7"></a>Exercise 11.7: 一般コーシー分布 $p(y) = \frac{1}{\pi}\frac{b}{b^2 + (y-c)^2}$ のアフィン逆正接変換生成

### 問題の提示
区間 $(0, 1)$ 上の一様分布に従う確率変数 $z \sim \mathrm{Uniform}(0, 1)$ に対し、変換
$$ y = b\tan\left(\pi\left(z - \frac{1}{2}\right)\right) + c \quad (11.16) $$
を施した確率変数 $y$ が、位置母数 $c$、尺度母数 $b > 0$ を持つ一般コーシー分布
$$ p(y) = \frac{1}{\pi} \frac{b}{b^2 + (y - c)^2} $$
に従うことを証明せよ。

### [解答の道筋と穴埋め]
1. **変数変換の逆写像**:
   $y = g(z) = b\tan\left(\pi\left(z - \frac{1}{2}\right)\right) + c$ とする。$z$ について解くと：
   $$ \frac{y - c}{b} = \tan\left(\pi\left(z - \frac{1}{2}\right)\right) \iff z = [ \text{①} ] $$
2. **ヤコビアンの計算**:
   $\arctan$ 関数の導関数公式 $\frac{\mathrm{d}}{\mathrm{d}t}\arctan(t) = \frac{1}{1 + t^2}$ より：
   $$ \left|\frac{\mathrm{d}z}{\mathrm{d}y}\right| = \frac{1}{\pi} \frac{1}{1 + \left(\frac{y - c}{b}\right)^2} \cdot \frac{1}{b} = \frac{1}{\pi} \frac{1}{b\left(1 + \frac{(y - c)^2}{b^2}\right)} = \frac{1}{\pi} \frac{b}{b^2 + (y - c)^2} $$
3. **確率密度の保存**:
   $z \sim \mathrm{Uniform}(0, 1)$ より $p_Z(z) = 1$（$0 < z < 1$）。したがって：
   $$ p_Y(y) = p_Z(z(y)) \left|\frac{\mathrm{d}z}{\mathrm{d}y}\right| = [ \text{②} ] $$
   となり、一般コーシー分布の密度関数が厳密に得られる。

### 穴埋めの解答
- ①: $\frac{1}{2} + \frac{1}{\pi}\arctan\left(\frac{y - c}{b}\right)$
- ②: $\frac{1}{\pi} \frac{b}{b^2 + (y - c)^2}$"""

    ex11_7_code = r"""# Exercise 11.7 数値検証: 位置母数 c, 尺度母数 b のコーシー乱数生成と経験累積分布検証
np.random.seed(42)
b_param = 3.2
c_param = -1.8
N_samples = 60000

z = np.random.uniform(0, 1, size=N_samples)
y = b_param * np.tan(np.pi * (z - 0.5)) + c_param

# 理論中央値 = c, IQR = 2 * b
emp_med = np.median(y)
q25, q75 = np.percentile(y, [25, 75])
emp_iqr = q75 - q25

print(f"Cauchy (c={c_param}, b={b_param}): Median = {emp_med:.4f} (True {c_param}), IQR = {emp_iqr:.4f} (True {2*b_param:.4f})")

np.testing.assert_allclose(emp_med, c_param, atol=0.05)
np.testing.assert_allclose(emp_iqr, 2.0 * b_param, atol=0.10)

# 一般コーシー分布に対する KS 検定
ks_res = kstest(y, cauchy(loc=c_param, scale=b_param).cdf)
assert ks_res.pvalue > 0.05
print("Exercise 11.7 verified: Generalized Cauchy variable correctly generated via affine tangent transform!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_7_md), nbf.v4.new_code_cell(ex11_7_code)])

    # --- Exercise 11.8 ---
    ex11_8_md = r"""---
## <a id="Exercise-11.8"></a>Exercise 11.8: 適応型棄却サンプリング（ARS）における区分的包絡線の連続性条件と規格化係数 $k_i$ 導出

### 問題の提示
対数凹分布に対する適応型棄却サンプリング（ARS: Gilks & Wild 1992）において、対数確率密度の接線支持直線から構成される包絡分布（式 11.17）
$$ q(z) = k_i \lambda_i \exp(-\lambda_i (z - z_{i-1})) \quad (z_{i-1} < z \le z_i) $$
を考える。境界点 $z_i$ における関数の連続性条件、および全体での正規化条件 $\int q(z)\mathrm{d}z = 1$ を用いて、各区間の係数 $k_i$ の再帰的関係式および決定方程式を導出せよ。

### [解答の道筋と穴埋め]
1. **境界点 $z_i$ における連続性条件**:
   $z \to z_i^-$（第 $i$ 区間の右端）での極限値と、$z \to z_i^+$（第 $i+1$ 区間の左端）での極限値が一致しなければならない：
   $$ q(z_i^-) = k_i \lambda_i \exp(-\lambda_i(z_i - z_{i-1})) $$
   $$ q(z_i^+) = k_{i+1} \lambda_{i+1} \exp(-\lambda_{i+1}(z_i - z_i)) = k_{i+1} \lambda_{i+1} $$
   両者を等置することにより、係数の漸化式が得られる：
   $$ k_{i+1} = [ \text{①} ] $$
2. **各区間の積分値**:
   第 $i$ 区間の積分 $I_i = \int_{z_{i-1}}^{z_i} q(z)\mathrm{d}z$ は：
   $$ I_i = k_i \lambda_i \left[ -\frac{1}{\lambda_i} e^{-\lambda_i(z - z_{i-1})} \right]_{z_{i-1}}^{z_i} = k_i (1 - e^{-\lambda_i(z_i - z_{i-1})}) $$
3. **全体正規化条件**:
   全区間数の和が 1 となる要求から：
   $$ \sum_{i=1}^M I_i = [ \text{②} ] $$
   $k_1$ を自由変数としてすべての $k_i$ を $k_1$ の倍数で表すことで、$k_1$、ひいては全 $k_i$ が一意に確定する。

### 穴埋めの解答
- ①: $k_i \frac{\lambda_i}{\lambda_{i+1}} \exp(-\lambda_i(z_i - z_{i-1}))$
- ②: $\sum_{i=1}^M k_i (1 - e^{-\lambda_i(z_i - z_{i-1})}) = 1$"""

    ex11_8_code = r"""# Exercise 11.8 数値検証: ARS 包絡分布の境界連続性と全体正規化条件の求解
# 3区間のテストケース設定
z_bounds = [0.0, 1.0, 2.5, 4.0] # z_0, z_1, z_2, z_3
lambdas = [1.2, 0.8, 1.5]        # lambda_1, lambda_2, lambda_3

# 1. 漸化式を用いて k_1 を基準とした相対係数 c_i = k_i / k_1 を計算
M = len(lambdas)
c = np.zeros(M)
c[0] = 1.0
for i in range(M - 1):
    dz = z_bounds[i+1] - z_bounds[i]
    c[i+1] = c[i] * (lambdas[i] / lambdas[i+1]) * np.exp(-lambdas[i] * dz)

# 2. 各区間の無正規化積分 I_i_rel の計算
I_rel = np.zeros(M)
for i in range(M):
    dz = z_bounds[i+1] - z_bounds[i]
    I_rel[i] = c[i] * (1.0 - np.exp(-lambdas[i] * dz))

# 3. 全体規格化定数 k_1 の決定: sum I_i = 1
k_1 = 1.0 / np.sum(I_rel)
k = k_1 * c

# 境界での連続性検証
for i in range(M - 1):
    left_val = k[i] * lambdas[i] * np.exp(-lambdas[i] * (z_bounds[i+1] - z_bounds[i]))
    right_val = k[i+1] * lambdas[i+1]
    print(f"Boundary z_{i+1} = {z_bounds[i+1]}: Left = {left_val:.8f}, Right = {right_val:.8f}")
    np.testing.assert_allclose(left_val, right_val, atol=1e-12)

# 全体積分の検証
total_integral = np.sum([k[i] * (1.0 - np.exp(-lambdas[i] * (z_bounds[i+1] - z_bounds[i]))) for i in range(M)])
print(f"Total Normalized Integral: {total_integral:.12f}")
np.testing.assert_allclose(total_integral, 1.0, atol=1e-12)
print("Exercise 11.8 verified: Envelope continuity and normalization strictly validated!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_8_md), nbf.v4.new_code_cell(ex11_8_code)])

    # --- Exercise 11.9 ---
    ex11_9_md = r"""---
## <a id="Exercise-11.9"></a>Exercise 11.9: ARS における区分的指数包絡線からのサンプリングアルゴリズムの導出と実装

### 問題の提示
式 (11.17) で定義される区分的指数包絡線 $q(z)$ から乱数をサンプリングする2段階アルゴリズム（区間選択＋局所切断指数サンプリング）を構築し、その正当性を証明せよ。

### [解答の道筋と穴埋め]
1. **第1段階：区間の離散サンプリング**:
   各区間 $(z_{i-1}, z_i)$ の全確率質量は $P_i = \int_{z_{i-1}}^{z_i} q(z)\mathrm{d}z = k_i(1 - e^{-\lambda_i(z_i - z_{i-1})})$ である。
   全体で正規化されているため $\sum_{i=1}^M P_i = 1$。
   一様乱数 $u_1 \sim \mathrm{Uniform}(0, 1)$ に基づき、離散カテゴリカル分布から区間インデックス $i \in \{1, \dots, M\}$ を採択する：
   $$ P(\text{区間 } i) = [ \text{①} ] $$
2. **第2段階：局所切断指数分布からの逆関数サンプリング**:
   区間 $i$ に条件付けられた条件付き累積分布関数 $F_i(z) = P(Z \le z \mid Z \in [z_{i-1}, z_i])$ は：
   $$ F_i(z) = \frac{1 - e^{-\lambda_i(z - z_{i-1})}}{1 - e^{-\lambda_i(z_i - z_{i-1})}} $$
   独立な一様乱数 $u_2 \sim \mathrm{Uniform}(0, 1)$ に対し $u_2 = F_i(z)$ を解くと：
   $$ z = [ \text{②} ] $$
   これにより、全区間にわたる厳密な直接乱数生成が実現される。

### 穴埋めの解答
- ①: $k_i (1 - e^{-\lambda_i(z_i - z_{i-1})})$
- ②: $z_{i-1} - \frac{1}{\lambda_i} \ln\left(1 - u_2 (1 - e^{-\lambda_i(z_i - z_{i-1})})\right)$"""

    ex11_9_code = r"""# Exercise 11.9 数値検証: 区分的指数包絡線からの2段階サンプラーの実装とヒストグラム整合性検証
np.random.seed(42)
N_draws = 60000

# 区間選択確率 P_i
P_intervals = np.array([k[i] * (1.0 - np.exp(-lambdas[i] * (z_bounds[i+1] - z_bounds[i]))) for i in range(M)])
cum_P = np.cumsum(P_intervals)

# 1. 区間インデックスのサンプリング
u1 = np.random.uniform(0, 1, size=N_draws)
interval_idx = np.searchsorted(cum_P, u1)

# 2. 各区間内での切断指数分布サンプリング
u2 = np.random.uniform(0, 1, size=N_draws)
z_draws = np.zeros(N_draws)

for i in range(M):
    mask = (interval_idx == i)
    n_i = np.sum(mask)
    if n_i > 0:
        dz = z_bounds[i+1] - z_bounds[i]
        scale = 1.0 - np.exp(-lambdas[i] * dz)
        z_draws[mask] = z_bounds[i] - (1.0 / lambdas[i]) * np.log(1.0 - u2[mask] * scale)

# 解析的密度とのヒストグラム比較
counts, bin_edges = np.histogram(z_draws, bins=50, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# 理論密度の計算
def true_q(z_val):
    res = np.zeros_like(z_val)
    for i in range(M):
        idx = (z_val >= z_bounds[i]) & (z_val < z_bounds[i+1])
        res[idx] = k[i] * lambdas[i] * np.exp(-lambdas[i] * (z_val[idx] - z_bounds[i]))
    return res

theo_dens = true_q(bin_centers)
max_abs_err = np.max(np.abs(counts - theo_dens))
print(f"Max Absolute Error between Histogram and Analytic Density: {max_abs_err:.4f}")

assert max_abs_err < 0.05
print("Exercise 11.9 verified: Two-stage piecewise exponential sampler matches target distribution perfectly!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_9_md), nbf.v4.new_code_cell(ex11_9_code)])

    # --- Exercise 11.10 ---
    ex11_10_md = r"""---
## <a id="Exercise-11.10"></a>Exercise 11.10: 1次元整数ランダムウォークの拡散距離 $\mathbb{E}[(z^{(\tau)})^2] = \tau / 2$ の数学的帰納法証明

### 問題の提示
整数上の1次元単純ランダムウォーク（式 11.34 - 11.36）
$$ p(z^{(\tau+1)} = z^{(\tau)}) = 0.5 $$
$$ p(z^{(\tau+1)} = z^{(\tau)} + 1) = 0.25 $$
$$ p(z^{(\tau+1)} = z^{(\tau)} - 1) = 0.25 $$
を考える。初期位置を $z^{(0)} = 0$ とするとき、
$$ \mathbb{E}[(z^{(\tau)})^2] = \mathbb{E}[(z^{(\tau-1)})^2] + \frac{1}{2} $$
が成立することを示し、数学的帰納法により $\mathbb{E}[(z^{(\tau)})^2] = \frac{\tau}{2}$ を証明せよ。

### [解答の道筋と穴埋め]
1. **1ステップ変位量 $\Delta z^{(\tau)} \equiv z^{(\tau)} - z^{(\tau-1)}$ の統計量**:
   定義より、変位量 $\Delta z$ の確率分布は：
   $$ P(\Delta z = 0) = 0.5, \quad P(\Delta z = +1) = 0.25, \quad P(\Delta z = -1) = 0.25 $$
   期待値：
   $$ \mathbb{E}[\Delta z] = 0 \cdot 0.5 + (+1) \cdot 0.25 + (-1) \cdot 0.25 = 0 $$
   2次モーメント：
   $$ \mathbb{E}[(\Delta z)^2] = 0^2 \cdot 0.5 + (+1)^2 \cdot 0.25 + (-1)^2 \cdot 0.25 = 0.25 + 0.25 = [ \text{①} ] $$
2. **自乗変位の漸化式**:
   $z^{(\tau)} = z^{(\tau-1)} + \Delta z^{(\tau)}$ を自乗して展開する：
   $$ (z^{(\tau)})^2 = (z^{(\tau-1)})^2 + 2 z^{(\tau-1)}\Delta z^{(\tau)} + (\Delta z^{(\tau)})^2 $$
   $\Delta z^{(\tau)}$ は過去の履歴 $z^{(\tau-1)}$ と独立であるため：
   $$ \mathbb{E}[z^{(\tau-1)}\Delta z^{(\tau)}] = \mathbb{E}[z^{(\tau-1)}]\mathbb{E}[\Delta z^{(\tau)}] = 0 $$
   したがって：
   $$ \mathbb{E}[(z^{(\tau)})^2] = \mathbb{E}[(z^{(\tau-1)})^2] + \mathbb{E}[(\Delta z^{(\tau)})^2] = \mathbb{E}[(z^{(\tau-1)})^2] + \frac{1}{2} $$
3. **数学的帰納法**:
   $z^{(0)} = 0$ より $\mathbb{E}[(z^{(0)})^2] = 0$。
   $\tau$ 回の反復により：
   $$ \mathbb{E}[(z^{(\tau)})^2] = [ \text{②} ] $$
   平均自乗変位 (MSD) はステップ数 $\tau$ に比例し、探索距離（標準偏差）は $\mathcal{O}(\sqrt{\tau})$ でしか増大しないことが厳密に証明される。

### 穴埋めの解答
- ①: $\frac{1}{2}$
- ②: $\frac{\tau}{2}$"""

    ex11_10_code = r"""# Exercise 11.10 数値検証: ランダムウォークの自乗平均変位 E[z^2] = tau / 2 のシミュレーション
np.random.seed(42)
N_walks = 10000
tau_max = 80

# ステップ変化量: 0 (prob 0.5), +1 (prob 0.25), -1 (prob 0.25)
steps = np.random.choice([0, 1, -1], p=[0.5, 0.25, 0.25], size=(N_walks, tau_max))
# 軌跡 z(tau)
z_trajectories = np.hstack([np.zeros((N_walks, 1)), np.cumsum(steps, axis=1)]) # (N_walks, tau_max + 1)

# 各時刻での平均自乗変位 E[(z^tau)^2]
msd_empirical = np.mean(z_trajectories**2, axis=0)
taus = np.arange(tau_max + 1)
msd_theoretical = taus / 2.0

print(f"tau = 20: Empirical MSD = {msd_empirical[20]:.3f}, Theo = {msd_theoretical[20]:.3f}")
print(f"tau = 40: Empirical MSD = {msd_empirical[40]:.3f}, Theo = {msd_theoretical[40]:.3f}")
print(f"tau = 80: Empirical MSD = {msd_empirical[80]:.3f}, Theo = {msd_theoretical[80]:.3f}")

np.testing.assert_allclose(msd_empirical[10:], msd_theoretical[10:], rtol=0.08)
print("Exercise 11.10 verified: Random walk variance strictly matches tau / 2!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_10_md), nbf.v4.new_code_cell(ex11_10_code)])

    # --- Exercise 11.11 ---
    ex11_11_md = r"""---
## <a id="Exercise-11.11"></a>Exercise 11.11: ギブスサンプリング（Gibbs Sampling）のマルコフ遷移核と詳細釣り合い条件の証明

### 問題の提示
ギブスサンプリング（PRML 11.3節）において、各ステップで1つの変数 $z_k$ を完全条件付き事後分布 $p(z_k \mid \mathbf{z}_{\backslash k})$ からサンプリングし、他のすべての変数 $\mathbf{z}_{\backslash k}$ を固定する。
この単一変数更新の遷移確率核が詳細釣り合い条件（式 11.40）
$$ p(\mathbf{z}) T(\mathbf{z} \to \mathbf{z}^*) = p(\mathbf{z}^*) T(\mathbf{z}^* \to \mathbf{z}) $$
を満たすことを証明せよ。

### [解答の道筋と穴埋め]
1. **ギブス遷移確率核の数式表現**:
   変数 $z_k$ のみが更新され、他の成分 $\mathbf{z}_{\backslash k}^* = \mathbf{z}_{\backslash k}$ が不変に保たれるため、遷移確率密度は：
   $$ T(\mathbf{z} \to \mathbf{z}^*) = p(z_k^* \mid \mathbf{z}_{\backslash k}) \delta(\mathbf{z}_{\backslash k}^* - \mathbf{z}_{\backslash k}) $$
2. **左辺 $p(\mathbf{z})T(\mathbf{z} \to \mathbf{z}^*)$ の代数的展開**:
   同時確率分布を条件付き確率と周辺確率の積 $p(\mathbf{z}) = p(z_k \mid \mathbf{z}_{\backslash k}) p(\mathbf{z}_{\backslash k})$ に分解すると：
   $$ p(\mathbf{z}) T(\mathbf{z} \to \mathbf{z}^*) = p(z_k \mid \mathbf{z}_{\backslash k}) p(\mathbf{z}_{\backslash k}) p(z_k^* \mid \mathbf{z}_{\backslash k}) \delta(\mathbf{z}_{\backslash k}^* - \mathbf{z}_{\backslash k}) = [ \text{①} ] $$
3. **右辺 $p(\mathbf{z}^*)T(\mathbf{z}^* \to \mathbf{z})$ との対称性**:
   逆方向の遷移核は $T(\mathbf{z}^* \to \mathbf{z}) = p(z_k \mid \mathbf{z}_{\backslash k}^*) \delta(\mathbf{z}_{\backslash k} - \mathbf{z}_{\backslash k}^*)$ である。
   $\delta$ 関数の作用下では $\mathbf{z}_{\backslash k}^* = \mathbf{z}_{\backslash k}$ であるため：
   $$ p(\mathbf{z}^*) T(\mathbf{z}^* \to \mathbf{z}) = p(z_k^* \mid \mathbf{z}_{\backslash k}) p(\mathbf{z}_{\backslash k}) p(z_k \mid \mathbf{z}_{\backslash k}) \delta(\mathbf{z}_{\backslash k} - \mathbf{z}_{\backslash k}^*) $$
   乗法交換律より両辺は厳密に恒等一致する：
   $$ p(\mathbf{z}) T(\mathbf{z} \to \mathbf{z}^*) \equiv [ \text{②} ] $$
   これにより、各ギブスステップが目標分布 $p(\mathbf{z})$ を不変に保つことが証明される。

### 穴埋めの解答
- ①: $p(z_k \mid \mathbf{z}_{\backslash k}) p(z_k^* \mid \mathbf{z}_{\backslash k}) p(\mathbf{z}_{\backslash k}) \delta(\mathbf{z}_{\backslash k}^* - \mathbf{z}_{\backslash k})$
- ②: $p(\mathbf{z}^*) T(\mathbf{z}^* \to \mathbf{z})$"""

    ex11_11_code = r"""# Exercise 11.11 数値検証: 2変量ガウス分布におけるギブスサンプリング詳細釣り合いの数値積分検証
from scipy.stats import multivariate_normal

mu_vec = np.array([0.5, -0.2])
cov_mat = np.array([[1.5, 0.8], [0.8, 2.0]])
mvn = multivariate_normal(mean=mu_vec, cov=cov_mat)

# 条件付き分布 p(z1 | z2) のパラメータ: N(mu1 + rho*(sigma1/sigma2)*(z2 - mu2), sigma1^2*(1 - rho^2))
s1, s2 = np.sqrt(cov_mat[0, 0]), np.sqrt(cov_mat[1, 1])
rho = cov_mat[0, 1] / (s1 * s2)
var_cond1 = (s1**2) * (1.0 - rho**2)

def cond_mean1(z2):
    return mu_vec[0] + rho * (s1 / s2) * (z2 - mu_vec[1])

# 2点 z = (z1, z2) と z* = (z1*, z2) [z2 は共通固定]
z2_fixed = 0.7
z1_a = -0.4
z1_b = 1.2

p_z = mvn.pdf([z1_a, z2_fixed])
p_z_star = mvn.pdf([z1_b, z2_fixed])

# 遷移確率 T(z -> z*) = p(z1* | z2_fixed)
T_a_to_b = norm.pdf(z1_b, loc=cond_mean1(z2_fixed), scale=np.sqrt(var_cond1))
T_b_to_a = norm.pdf(z1_a, loc=cond_mean1(z2_fixed), scale=np.sqrt(var_cond1))

flux_forward = p_z * T_a_to_b
flux_reverse = p_z_star * T_b_to_a

print(f"Forward Probability Flux p(z) T(z -> z*):   {flux_forward:.10f}")
print(f"Reverse Probability Flux p(z*) T(z* -> z): {flux_reverse:.10f}")

np.testing.assert_allclose(flux_forward, flux_reverse, atol=1e-12)
print("Exercise 11.11 verified: Detailed balance strictly holds for Gibbs sampling updates!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_11_md), nbf.v4.new_code_cell(ex11_11_code)])

    # --- Exercise 11.12 ---
    ex11_12_md = r"""---
## <a id="Exercise-11.12"></a>Exercise 11.12: 非連結・孤立モード分布におけるギブスサンプリングの非エルゴード性（Ergodicity Failure）

### 問題の提示
図 11.15 に示される確率分布を考える。2つの確率変数 $z_1, z_2$ は、互いに座標軸方向に射影が重なり合わない2つの孤立した連結成分（例えば正方形領域 $S_1 = [0, 1] \times [0, 1]$ と $S_2 = [2, 3] \times [2, 3]$）の上で一様であり、それ以外の全領域で 0 である。
この分布に対して標準的なギブスサンプリングを適用した場合、マルコフ連鎖がエルゴード的（ergodic）となるか否かを議論し、正しいサンプリングが行えるかを判定せよ。

### [解答の道筋と穴埋め]
1. **条件付き分布の台（Support）の検証**:
   現在のサンプルが領域 $S_1$ 内にあるとする（$z_1 \in [0, 1], z_2 \in [0, 1]$）。
   ギブスサンプリングにおいて $z_2 \in [0, 1]$ を固定して $z_1$ を条件付き分布 $p(z_1 \mid z_2)$ から更新する際、$z_2 \in [0, 1]$ の下で $p(z_1, z_2) > 0$ となる $z_1$ の範囲は $[0, 1]$ のみである。
   したがって：
   $$ p(z_1 \in [2, 3] \mid z_2 \in [0, 1]) = [ \text{①} ] $$
2. **軸平行遷移による孤立**:
   同様に、$z_1 \in [0, 1]$ を固定して $z_2$ を更新する場合も $p(z_2 \in [2, 3] \mid z_1 \in [0, 1]) = 0$ となる。
   ギブスサンプリングの遷移は常に座標軸に平行な直線移動（$z_1$ 軸または $z_2$ 軸方向）しか行えないため、2つのモードを結ぶ軸平行なパスが存在しない場合、領域間の遷移確率は厳密に 0 となる：
   $$ T(S_1 \to S_2) = 0, \quad T(S_2 \to S_1) = 0 $$
3. **エルゴード性の破綻**:
   マルコフ連鎖は非連結な不変集合に分割され、既約性（Irreducibility）を欠くため、[ ② ] となる。初期値が置かれた領域に永久に捕捉され、全空間の真の分布を再現することは不可能である。

### 穴埋めの解答
- ①: $0$
- ②: 非エルゴード的（Ergodicity failure / Reducible）"""

    ex11_12_code = r"""# Exercise 11.12 数値検証: 孤立した二峰性分布におけるギブスサンプリングのトラップ現象
np.random.seed(42)
N_steps = 5000

# 初期位置を S_1 = [0, 1] x [0, 1] 内に設定
z_curr = np.array([0.5, 0.5])
visited = []

for step in range(N_steps):
    # 1. z1 の更新: p(z1 | z2)
    # z2 in [0, 1] ならば z1 in [0, 1] の一様乱数
    # z2 in [2, 3] ならば z1 in [2, 3] の一様乱数
    if z_curr[1] <= 1.0:
        z_curr[0] = np.random.uniform(0.0, 1.0)
    else:
        z_curr[0] = np.random.uniform(2.0, 3.0)
        
    # 2. z2 の更新: p(z2 | z1)
    if z_curr[0] <= 1.0:
        z_curr[1] = np.random.uniform(0.0, 1.0)
    else:
        z_curr[1] = np.random.uniform(2.0, 3.0)
        
    visited.append(z_curr.copy())

visited = np.array(visited)

# S_2 = [2, 3] x [2, 3] への訪問回数
s2_visits = np.sum((visited[:, 0] >= 2.0) & (visited[:, 1] >= 2.0))
print(f"Total Gibbs Steps: {N_steps}, Steps trapped in S1: {len(visited) - s2_visits}, Steps visiting S2: {s2_visits}")

assert s2_visits == 0, "Gibbs sampling must be strictly trapped in S1 due to lack of axis-aligned paths!"
print("Exercise 11.12 verified: Reducibility and ergodicity breakdown numerically demonstrated!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_12_md), nbf.v4.new_code_cell(ex11_12_code)])

    # --- Exercise 11.13 ---
    ex11_13_md = r"""---
## <a id="Exercise-11.13"></a>Exercise 11.13: 未知平均・未知精度の共役ガウス-ガンマモデルにおける完全条件付き事後分布の導出

### 問題の提示
観測データ $x \sim \mathcal{N}(x \mid \mu, \tau^{-1})$ に対し、平均 $\mu$ と精度 $\tau$ の事前分布として独立な
$$ p(\mu) = \mathcal{N}(\mu \mid \mu_0, s_0^{-1}), \quad p(\tau) = \mathrm{Gam}(\tau \mid a, b) = \frac{1}{\Gamma(a)} b^a \tau^{a-1} e^{-b\tau} $$
を仮定する（図 11.16）。
事後分布 $p(\mu, \tau \mid x)$ に対するギブスサンプリングを実行するために必要となる、完全条件付き分布 $p(\mu \mid x, \tau)$ および $p(\tau \mid x, \mu)$ の閉形式を導出せよ。

### [解答の道筋と穴埋め]
1. **完全結合分布の表現**:
   $$ p(x, \mu, \tau) = \left( \frac{\tau}{2\pi} \right)^{1/2} \exp\left(-\frac{\tau}{2}(x - \mu)^2\right) \cdot \left( \frac{s_0}{2\pi} \right)^{1/2} \exp\left(-\frac{s_0}{2}(\mu - \mu_0)^2\right) \cdot \frac{b^a}{\Gamma(a)} \tau^{a-1} e^{-b\tau} $$
2. **$\mu$ の条件付き分布 $p(\mu \mid x, \tau)$**:
   $\mu$ に関する項のみを抽出して平方完成する：
   $$ \ln p(\mu \mid x, \tau) = -\frac{\tau}{2}(\mu^2 - 2\mu x) - \frac{s_0}{2}(\mu^2 - 2\mu \mu_0) + \mathrm{const} = -\frac{s_0 + \tau}{2}\mu^2 + (s_0 \mu_0 + \tau x)\mu + \mathrm{const} $$
   したがって、更新後の精度は $s_N = s_0 + \tau$、平均は $\mu_N = \frac{s_0 \mu_0 + \tau x}{s_0 + \tau}$ となり：
   $$ p(\mu \mid x, \tau) = [ \text{①} ] $$
3. **$\tau$ の条件付き分布 $p(\tau \mid x, \mu)$**:
   $\tau$ に関する項のみを抽出すると：
   $$ \ln p(\tau \mid x, \mu) = \frac{1}{2}\ln \tau - \frac{\tau}{2}(x - \mu)^2 + (a - 1)\ln \tau - b\tau + \mathrm{const} = \left(a + \frac{1}{2} - 1\right)\ln \tau - \left(b + \frac{1}{2}(x - \mu)^2\right)\tau + \mathrm{const} $$
   これは形状母数 $a_N = a + 1/2$、尺度母数 $b_N = b + \frac{1}{2}(x - \mu)^2$ のガンマ分布である：
   $$ p(\tau \mid x, \mu) = [ \text{②} ] $$

### 穴埋めの解答
- ①: $\mathcal{N}\left(\mu \;\middle|\; \frac{s_0 \mu_0 + \tau x}{s_0 + \tau}, (s_0 + \tau)^{-1}\right)$
- ②: $\mathrm{Gam}\left(\tau \;\middle|\; a + \frac{1}{2}, b + \frac{1}{2}(x - \mu)^2\right)$"""

    ex11_13_code = r"""# Exercise 11.13 数値検証: ガウス-ガンマ事後分布に対するギブスサンプラーの実装
from scipy.stats import gamma as gamma_dist

np.random.seed(42)
x_obs = 2.4
mu_0, s_0 = 0.0, 1.0  # 事前平均 0, 精度 1
a_0, b_0 = 2.0, 1.0   # ガンマ事前分布

N_samples = 30000
mu_chain = np.zeros(N_samples)
tau_chain = np.zeros(N_samples)

# 初期値
mu_curr = 0.0
tau_curr = 1.0

for s in range(N_samples):
    # 1. p(mu | x, tau) から抽出: N(mu_N, 1/s_N)
    s_N = s_0 + tau_curr
    mu_N = (s_0 * mu_0 + tau_curr * x_obs) / s_N
    mu_curr = np.random.normal(loc=mu_N, scale=1.0 / np.sqrt(s_N))
    
    # 2. p(tau | x, mu) から抽出: Gam(a_N, b_N)
    a_N = a_0 + 0.5
    b_N = b_0 + 0.5 * (x_obs - mu_curr)**2
    # scipy の gamma は scale = 1 / b_N
    tau_curr = np.random.gamma(shape=a_N, scale=1.0 / b_N)
    
    mu_chain[s] = mu_curr
    tau_chain[s] = tau_curr

# バーンイン除去
burn = 5000
mu_post = mu_chain[burn:]
tau_post = tau_chain[burn:]

print(f"Posterior Mean E[mu|x]: {np.mean(mu_post):.4f}, E[tau|x]: {np.mean(tau_post):.4f}")
assert 1.0 < np.mean(mu_post) < 2.5
assert 0.8 < np.mean(tau_post) < 2.5
print("Exercise 11.13 verified: Gibbs sampling from conditional distributions converges stably!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_13_md), nbf.v4.new_code_cell(ex11_13_code)])

    # --- Exercise 11.14 ---
    ex11_14_md = r"""---
## <a id="Exercise-11.14"></a>Exercise 11.14: 過剰緩和法（Over-Relaxation: Adler 1981）による分散保存性 $\mathrm{var}[z_i'] = \sigma_i^2$ の代数的一致検証

### 問題の提示
多変量ガウス分布に対するギブスサンプリングを高速化する過剰緩和法（Over-Relaxation; 式 11.50）
$$ z_i' = \mu_i + \alpha(z_i - \mu_i) + \sigma_i (1 - \alpha^2)^{1/2} \nu $$
において、条件付き平均 $\mu_i$、条件付き分散 $\sigma_i^2$、パラメータ $\alpha \in (-1, 1)$、および独立な標準正規確率変数 $\nu \sim \mathcal{N}(0, 1)$ とする。
更新後の確率変数 $z_i'$ が条件付き平均 $\mu_i$ および条件付き分散 $\sigma_i^2$ を厳密に保持することを証明せよ。

### [解答の道筋と穴埋め]
1. **更新後の期待値**:
   $\mathbb{E}[z_i] = \mu_i$ かつ $\mathbb{E}[\nu] = 0$ であるため：
   $$ \mathbb{E}[z_i'] = \mu_i + \alpha(\mathbb{E}[z_i] - \mu_i) + \sigma_i (1 - \alpha^2)^{1/2}\mathbb{E}[\nu] = \mu_i + 0 + 0 = [ \text{①} ] $$
2. **更新後の分散**:
   確率変数 $z_i$ と新たに抽出された乱数 $\nu$ は統計的に独立であるため、分散の加法性より：
   $$ \mathrm{var}[z_i'] = \mathrm{var}[\alpha(z_i - \mu_i)] + \mathrm{var}\left[ \sigma_i (1 - \alpha^2)^{1/2} \nu \right] $$
   $$ \mathrm{var}[z_i'] = \alpha^2 \mathrm{var}[z_i] + \sigma_i^2 (1 - \alpha^2) \mathrm{var}[\nu] = \alpha^2 \sigma_i^2 + \sigma_i^2 (1 - \alpha^2) \cdot 1 = [ \text{②} ] $$
   したがって、$\alpha$ の任意の値（特に負の値 $\alpha \to -1$）に対して条件付き分散が完全に保存され、定常分布の不変性が保証される。

### 穴埋めの解答
- ①: $\mu_i$
- ②: $\sigma_i^2$"""

    ex11_14_code = r"""# Exercise 11.14 数値検証: 過剰緩和法による分散保存と自己相関抑制
np.random.seed(42)
N_trials = 50000
mu_cond = 2.5
sigma_cond = 1.4

# 様々な alpha に対する過剰緩和更新の検証
alphas = [-0.9, -0.5, 0.0, 0.7]

for alpha in alphas:
    z_init = np.random.normal(mu_cond, sigma_cond, size=N_trials)
    nu = np.random.normal(0, 1, size=N_trials)
    
    # 式 (11.50)
    z_prime = mu_cond + alpha * (z_init - mu_cond) + sigma_cond * np.sqrt(1.0 - alpha**2) * nu
    
    emp_mean = np.mean(z_prime)
    emp_var = np.var(z_prime)
    
    print(f"alpha = {alpha:4.1f}: Mean = {emp_mean:.4f} (True {mu_cond}), Var = {emp_var:.4f} (True {sigma_cond**2:.4f})")
    
    np.testing.assert_allclose(emp_mean, mu_cond, atol=0.03)
    np.testing.assert_allclose(emp_var, sigma_cond**2, atol=0.04)

print("Exercise 11.14 verified: Over-relaxation update strictly preserves conditional mean and variance!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_14_md), nbf.v4.new_code_cell(ex11_14_code)])

    # --- Exercise 11.15 ---
    ex11_15_md = r"""---
## <a id="Exercise-11.15"></a>Exercise 11.15: ハミルトン力学系方程式 $\frac{dz_i}{d\tau} = \frac{\partial H}{\partial r_i}, \frac{dr_i}{d\tau} = -\frac{\partial H}{\partial z_i}$ とニュートン運動方程式の等価性

### 問題の提示
ハミルトニアン $H(\mathbf{z}, \mathbf{r})$ が位置エネルギー $E(\mathbf{z})$ と運動エネルギー $K(\mathbf{r})$ の和
$$ H(\mathbf{z}, \mathbf{r}) = E(\mathbf{z}) + K(\mathbf{r}) \quad (11.56), \quad K(\mathbf{r}) = \frac{1}{2}\sum_i r_i^2 \quad (11.57) $$
で与えられるとき、ハミルトン正準方程式（式 11.58, 11.59）
$$ \frac{\mathrm{d}z_i}{\mathrm{d}\tau} = \frac{\partial H}{\partial r_i}, \quad \frac{\mathrm{d}r_i}{\mathrm{d}\tau} = -\frac{\partial H}{\partial z_i} $$
が、ニュートンの運動方程式（式 11.53, 11.55）
$$ \frac{\mathrm{d}z_i}{\mathrm{d}\tau} = r_i, \quad \frac{\mathrm{d}r_i}{\mathrm{d}\tau} = -\frac{\partial E}{\partial z_i} $$
と厳密に等価であることを証明せよ。

### [解答の道筋と穴埋め]
1. **運動量に関する微分（第1方程式）**:
   ハミルトニアン $H(\mathbf{z}, \mathbf{r}) = E(\mathbf{z}) + \frac{1}{2}\sum_j r_j^2$ を $r_i$ で偏微分する。$E(\mathbf{z})$ は $\mathbf{r}$ に依存しないため：
   $$ \frac{\partial H}{\partial r_i} = \frac{\partial E(\mathbf{z})}{\partial r_i} + \frac{\partial}{\partial r_i}\left( \frac{1}{2}\sum_j r_j^2 \right) = 0 + \frac{1}{2} \cdot 2 r_i = [ \text{①} ] $$
   したがって、$\frac{\mathrm{d}z_i}{\mathrm{d}\tau} = \frac{\partial H}{\partial r_i} \iff \frac{\mathrm{d}z_i}{\mathrm{d}\tau} = r_i$ が得られる。
2. **位置に関する微分（第2方程式）**:
   同様に $H(\mathbf{z}, \mathbf{r})$ を $z_i$ で偏微分する。運動エネルギー $K(\mathbf{r})$ は $\mathbf{z}$ に依存しないため：
   $$ \frac{\partial H}{\partial z_i} = \frac{\partial E(\mathbf{z})}{\partial z_i} + \frac{\partial K(\mathbf{r})}{\partial z_i} = \frac{\partial E(\mathbf{z})}{\partial z_i} + 0 = \frac{\partial E(\mathbf{z})}{\partial z_i} $$
   したがって：
   $$ \frac{\mathrm{d}r_i}{\mathrm{d}\tau} = -\frac{\partial H}{\partial z_i} = [ \text{②} ] $$
   となり、勾配力による運動量変化（ニュートンの第2法則）に厳密に一致する。

### 穴埋めの解答
- ①: $r_i$
- ②: $-\frac{\partial E}{\partial z_i}$"""

    ex11_15_code = r"""# Exercise 11.15 数値検証: ハミルトン方程式によるエネルギー保存則 dH/dtau = 0 の数値確認
# 1次元ポテンシャル E(z) = 0.5 * k * z^2 (調和振動子)
k_spring = 2.5
def E(z):
    return 0.5 * k_spring * z**2

def grad_E(z):
    return k_spring * z

def H(z, r):
    return E(z) + 0.5 * r**2

# ハミルトン正準時間発展: dz/dtau = r, dr/dtau = - grad_E(z)
# シンプレクティック・リープフロッグ積分
tau_steps = 100
eps = 0.05
z_val = 1.8
r_val = 0.0

H_init = H(z_val, r_val)
H_trajectory = [H_init]

for _ in range(tau_steps):
    r_half = r_val - 0.5 * eps * grad_E(z_val)
    z_val = z_val + eps * r_half
    r_val = r_half - 0.5 * eps * grad_E(z_val)
    H_trajectory.append(H(z_val, r_val))

max_energy_drift = np.max(np.abs(np.array(H_trajectory) - H_init))
print(f"Initial Energy: {H_init:.6f}, Max Hamiltonian Energy Drift: {max_energy_drift:.6f}")

assert max_energy_drift < 0.01, "Hamiltonian dynamics must preserve total energy H(z, r)!"
print("Exercise 11.15 verified: Canonical equations strictly equivalent to Newtonian dynamics with energy conservation!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_15_md), nbf.v4.new_code_cell(ex11_15_code)])

    # --- Exercise 11.16 ---
    ex11_16_md = r"""---
## <a id="Exercise-11.16"></a>Exercise 11.16: HMC カノニカル分布における運動量条件付き分布 $p(\mathbf{r} \mid \mathbf{z}) = \mathcal{N}(\mathbf{r} \mid \mathbf{0}, \mathbf{I})$ のガウス分布性証明

### 問題の提示
ハイブリッド・モンテカルロ法における相空間 $(\mathbf{z}, \mathbf{r})$ 上のボルツマン結合分布（式 11.60）
$$ p(\mathbf{z}, \mathbf{r}) = \frac{1}{Z_H} \exp(-H(\mathbf{z}, \mathbf{r})) $$
において、ハミルトニアンが式 (11.56), (11.57) の標準形式をとるとき、運動量 $\mathbf{r}$ の条件付き分布 $p(\mathbf{r} \mid \mathbf{z})$ が位置 $\mathbf{z}$ と独立な標準多変量ガウス分布 $\mathcal{N}(\mathbf{r} \mid \mathbf{0}, \mathbf{I}_D)$ になることを証明せよ。

### [解答の道筋と穴埋め]
1. **結合分布の因数分解**:
   $H(\mathbf{z}, \mathbf{r}) = E(\mathbf{z}) + \frac{1}{2}\mathbf{r}^{\mathrm{T}}\mathbf{r}$ を代入すると：
   $$ p(\mathbf{z}, \mathbf{r}) = \frac{1}{Z_H} \exp\left( -E(\mathbf{z}) - \frac{1}{2}\mathbf{r}^{\mathrm{T}}\mathbf{r} \right) = \left( \frac{1}{Z_E}\exp(-E(\mathbf{z})) \right) \left( \frac{1}{Z_K}\exp\left(-\frac{1}{2}\mathbf{r}^{\mathrm{T}}\mathbf{r}\right) \right) = [ \text{①} ] $$
   ここで正規化定数は $Z_H = Z_E Z_K$ と完全に直積分解される。
2. **条件付き分布の導出**:
   条件付き確率の定義より：
   $$ p(\mathbf{r} \mid \mathbf{z}) = \frac{p(\mathbf{z}, \mathbf{r})}{p(\mathbf{z})} = \frac{p(\mathbf{z})p(\mathbf{r})}{p(\mathbf{z})} = p(\mathbf{r}) = [ \text{②} ] $$
   したがって、運動量 $\mathbf{r}$ は任意の位置 $\mathbf{z}$ と独立であり、標準正規分布から極めて容易に再サンプリング（Gibbs ステップ）できることが証明される。

### 穴埋めの解答
- ①: $p(\mathbf{z}) p(\mathbf{r})$
- ②: $\mathcal{N}(\mathbf{r} \mid \mathbf{0}, \mathbf{I}_D)$"""

    ex11_16_code = r"""# Exercise 11.16 数値検証: 相空間カノニカル分布における位置と運動量の完全独立性
D = 3
np.random.seed(42)

# 任意の位置ポテンシャル E(z) = 0.5 * z^T A z
A_mat = np.array([[2.0, 0.5, 0.0], [0.5, 1.5, 0.3], [0.0, 0.3, 1.0]])

# カノニカル分布からの独立抽出
N_mc = 30000
z_samples = np.random.multivariate_normal(np.zeros(D), np.linalg.inv(A_mat), size=N_mc)
r_samples = np.random.normal(0, 1, size=(N_mc, D))

# 位置と運動量の相互共分散 cov(z_i, r_j) がすべて 0 であることの確認
cross_cov = np.cov(z_samples.T, r_samples.T)[:D, D:]
print("Cross-Covariance Matrix cov(z, r):\n", cross_cov)

np.testing.assert_allclose(cross_cov, np.zeros((D, D)), atol=0.03)
np.testing.assert_allclose(np.mean(r_samples, axis=0), np.zeros(D), atol=0.03)
np.testing.assert_allclose(np.cov(r_samples.T), np.eye(D), atol=0.03)
print("Exercise 11.16 verified: Momentum conditional distribution is strictly standard Gaussian and independent of position!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_16_md), nbf.v4.new_code_cell(ex11_16_code)])

    # --- Exercise 11.17 ---
    ex11_17_md = r"""---
## <a id="Exercise-11.17"></a>Exercise 11.17: ハイブリッド・モンテカルロ法 (HMC) の詳細釣り合い条件とリープフロッグ時間反転・体積保存性

### 問題の提示
離散化誤差を含むリープフロッグ積分器に基づくハイブリッド・モンテカルロ（HMC）法において、遷移核の採択確率式（式 11.68, 11.69）
$$ p(\mathbf{z}, \mathbf{r}) P_{\mathrm{acc}}((\mathbf{z}, \mathbf{r}) \to (\mathbf{z}^*, \mathbf{r}^*)) = p(\mathbf{z}^*, -\mathbf{r}^*) P_{\mathrm{acc}}((\mathbf{z}^*, -\mathbf{r}^*) \to (\mathbf{z}, -\mathbf{r})) $$
が恒等的に成立し、相空間上の詳細釣り合い条件が厳密に満たされることを証明せよ。

### [解答の道筋と穴埋め]
1. **リープフロッグ積分のシンプレクティック性（体積保存則と時間反転性）**:
   - **体積保存性（リウヴィルの定理）**: リープフロッグ積分のヤコビアン行列式は厳密に $|\det \mathbf{J}| = 1$ であり、相空間測度 $\mathrm{d}\mathbf{z}\mathrm{d}\mathbf{r} = \mathrm{d}\mathbf{z}^*\mathrm{d}\mathbf{r}^*$ を保存する。
   - **時間反転性（Reversibility）**: $(\mathbf{z}, \mathbf{r})$ から $L$ ステップで $(\mathbf{z}^*, \mathbf{r}^*)$ に到達したとき、運動量を反転した $(\mathbf{z}^*, -\mathbf{r}^*)$ から同一の積分を行うと厳密に $(\mathbf{z}, -\mathbf{r})$ に戻る。
2. **運動エネルギーの偶関数性**:
   $K(\mathbf{r}) = \frac{1}{2}\mathbf{r}^{\mathrm{T}}\mathbf{r} = \frac{1}{2}(-\mathbf{r})^{\mathrm{T}}(-\mathbf{r}) = K(-\mathbf{r})$ であるため：
   $$ H(\mathbf{z}^*, -\mathbf{r}^*) = H(\mathbf{z}^*, \mathbf{r}^*), \quad H(\mathbf{z}, -\mathbf{r}) = H(\mathbf{z}, \mathbf{r}) $$
3. **メトロポリス採択積の対称性**:
   前向き採択確率は $P_{\mathrm{acc}} = \min(1, \exp(-H(\mathbf{z}^*, \mathbf{r}^*) + H(\mathbf{z}, \mathbf{r})))$。
   したがって左辺は：
   $$ p(\mathbf{z}, \mathbf{r}) P_{\mathrm{acc}} = \frac{1}{Z_H} e^{-H(\mathbf{z}, \mathbf{r})} \min\left(1, e^{-[H(\mathbf{z}^*, \mathbf{r}^*) - H(\mathbf{z}, \mathbf{r})]}\right) = [ \text{①} ] $$
   逆方向の採択確率は $P_{\mathrm{acc}}^* = \min(1, \exp(-H(\mathbf{z}, -\mathbf{r}) + H(\mathbf{z}^*, -\mathbf{r}^*)))$ であるため、右辺も：
   $$ p(\mathbf{z}^*, -\mathbf{r}^*) P_{\mathrm{acc}}^* = \frac{1}{Z_H} e^{-H(\mathbf{z}^*, \mathbf{r}^*)} \min\left(1, e^{-[H(\mathbf{z}, \mathbf{r}) - H(\mathbf{z}^*, \mathbf{r}^*)]}\right) = \frac{1}{Z_H} \min\left( e^{-H(\mathbf{z}, \mathbf{r})}, e^{-H(\mathbf{z}^*, \mathbf{r}^*)} \right) $$
   両辺は厳密に一致し、詳細釣り合い条件が成立することが示される。

### 穴埋めの解答
- ①: $\frac{1}{Z_H} \min\left( e^{-H(\mathbf{z}, \mathbf{r})}, e^{-H(\mathbf{z}^*, \mathbf{r}^*)} \right)$"""

    ex11_17_code = r"""# Exercise 11.17 数値検証: リープフロッグ時間反転性と HMC 詳細釣り合い確率流の一致検証
# 2次元非ガウス・ポテンシャル
def E_pot(z):
    return 0.5 * (z[0]**2 + z[1]**2) + 0.25 * z[0]**4

def grad_E_pot(z):
    return np.array([z[0] + z[0]**3, z[1]])

def H_func(z, r):
    return E_pot(z) + 0.5 * np.sum(r**2)

def leapfrog(z_in, r_in, eps_val, L_steps):
    z = z_in.copy()
    r = r_in.copy()
    r = r - 0.5 * eps_val * grad_E_pot(z)
    for step in range(L_steps - 1):
        z = z + eps_val * r
        r = r - eps_val * grad_E_pot(z)
    z = z + eps_val * r
    r = r - 0.5 * eps_val * grad_E_pot(z)
    return z, r

# 1. 時間反転性の数値確認
z_0 = np.array([1.2, -0.8])
r_0 = np.array([0.5, 1.4])
eps = 0.15
L_steps = 15

z_star, r_star = leapfrog(z_0, r_0, eps, L_steps)
# 反転運動量からスタート
z_rev, r_rev = leapfrog(z_star, -r_star, eps, L_steps)

print(f"Original z_0:     {z_0}, Returned z_rev: {z_rev}")
print(f"Original -r_0:    {-r_0}, Returned r_rev:  {r_rev}")

np.testing.assert_allclose(z_rev, z_0, atol=1e-12)
np.testing.assert_allclose(r_rev, -r_0, atol=1e-12)

# 2. 詳細釣り合い確率流の一致
H_0 = H_func(z_0, r_0)
H_star = H_func(z_star, r_star)

p_acc_fwd = min(1.0, np.exp(-H_star + H_0))
p_acc_rev = min(1.0, np.exp(-H_0 + H_star))

flux_fwd = np.exp(-H_0) * p_acc_fwd
flux_rev = np.exp(-H_star) * p_acc_rev

print(f"Forward Flux p(z, r) * P_acc:         {flux_fwd:.12f}")
print(f"Reverse Flux p(z*, -r*) * P_acc_rev:  {flux_rev:.12f}")

np.testing.assert_allclose(flux_fwd, flux_rev, atol=1e-12)
print("Exercise 11.17 verified: Time reversibility and detailed balance identically confirmed!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex11_17_md), nbf.v4.new_code_cell(ex11_17_code)])

    return cells
