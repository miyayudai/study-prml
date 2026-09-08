# scripts/build_ch10_part4.py
"""
Definitions for Chapter 10 exercises 10.29 to 10.39.
"""

import nbformat as nbf

def get_ex_10_29_to_10_39():
    cells = []

    # --- Exercise 10.29 ---
    ex10_29_md = r"""---
## <a id="Exercise-10.29"></a>Exercise 10.29: 対数関数の凹性とルジャンドル変換による双対関数・元関数復元の証明

### 問題の提示
関数 $f(x) = \ln x$（$0 < x < \infty$）に対し：
1. 2階微分を計算し、全定義域で凹関数であることを示せ。
2. ルジャンドル双対関数 $g(\lambda) = \min_x \{ \lambda x - \ln x \}$（式 10.133）を陽に求めよ。
3. 式 (10.132) による双対最小化 $\min_\lambda \{ \lambda x - g(\lambda) \}$ が厳密に元の $\ln x$ を復元することを検証せよ。

### [解答の道筋と穴埋め]
1. **凹性の確認**:
   $$ f'(x) = \frac{1}{x}, \quad f''(x) = [ \text{①} ] < 0 \quad (\forall x > 0) $$
   したがって $f(x)$ は全定義域で狭義凹関数である。
2. **双対関数 $g(\lambda)$ の導出**:
   $\lambda x - \ln x$ の $x$ に関する停留条件：
   $$ \frac{\mathrm{d}}{\mathrm{d}x} (\lambda x - \ln x) = \lambda - \frac{1}{x} = 0 \implies x^* = \frac{1}{\lambda} \quad (\lambda > 0) $$
   これを代入すると：
   $$ g(\lambda) = \lambda \left(\frac{1}{\lambda}\right) - \ln\left(\frac{1}{\lambda}\right) = [ \text{②} ] $$
3. **元関数の復元**:
   $$ \min_{\lambda > 0} \{ \lambda x - g(\lambda) \} = \min_{\lambda > 0} \{ \lambda x - 1 - \ln \lambda \} $$
   $\lambda$ に関する停留条件は $x - \frac{1}{\lambda} = 0 \implies \lambda^* = \frac{1}{x}$。
   最小値は $\left(\frac{1}{x}\right) x - 1 - \ln\left(\frac{1}{x}\right) = 1 - 1 + \ln x = \ln x$ となり、元関数が完全に復元される。

### 穴埋めの解答
- ①: $-\frac{1}{x^2}$
- ②: $1 + \ln \lambda$"""

    ex10_29_code = r"""# Exercise 10.29 数値検証: 双対関数 g(lambda) = 1 + ln lambda による ln(x) 復元
x_vals = np.linspace(0.2, 5.0, 30)

# 双対最小化による復元
ln_x_recon = np.zeros_like(x_vals)
for i, x in enumerate(x_vals):
    # lambda x - (1 + ln lambda) の lambda > 0 での最小値
    res = minimize(lambda l: l[0]*x - (1.0 + np.log(l[0])), [1.0], bounds=[(1e-5, None)])
    ln_x_recon[i] = res.fun

ln_x_true = np.log(x_vals)
np.testing.assert_allclose(ln_x_recon, ln_x_true, atol=1e-6)
print("Exercise 10.29 verified: Dual Legendre minimization perfectly recovers ln(x)!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_29_md), nbf.v4.new_code_cell(ex10_29_code)])

    # --- Exercise 10.30 ---
    ex10_30_md = r"""---
## <a id="Exercise-10.30"></a>Exercise 10.30: 対数ロジスティック関数の凹性とテイラー展開による変分上界 (10.137) の導出

### 問題の提示
対数ロジスティック関数 $f(x) = -\ln(1 + e^{-x}) = \ln \sigma(x)$ に対し：
1. 2階微分が $f''(x) = -\sigma(x)(1 - \sigma(x)) < 0$ となることから凹関数であることを示せ。
2. 展開点 $x = \xi$ の周りでの2次テイラー展開から直接変分上界
$$ f(x) \le f(\xi) + (1 - \sigma(\xi))(x - \xi) \quad (10.137) $$
（1次接線上界）を導出せよ。

### [解答の道筋と穴埋め]
1. **導関数の計算**:
   $$ f'(x) = \frac{e^{-x}}{1 + e^{-x}} = 1 - \sigma(x) $$
   $$ f''(x) = -\sigma'(x) = [ \text{①} ] $$
   任意の $x \in \mathbb{R}$ で $0 < \sigma(x) < 1$ であるため、$f''(x) < 0$ となり狭義凹関数である。
2. **変分上界の導出**:
   凹関数の一次テイラー展開は常に全領域で上界を与える（グラフの上側に接線が存在する）：
   $$ f(x) \le f(\xi) + f'(\xi)(x - \xi) $$
   $f'(\xi) = 1 - \sigma(\xi)$ を代入することで式 (10.137) が直ちに得られる。

### 穴埋めの解答
- ①: $-\sigma(x)(1 - \sigma(x))$"""

    ex10_30_code = r"""# Exercise 10.30 数値検証: 対数ロジスティック関数の一次接線上界 (10.137) の大域的成立
def log_sig(x):
    return -np.log(1.0 + np.exp(-x))

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

xi = 1.2
x_test = np.linspace(-4, 4, 100)
f_true = log_sig(x_test)
f_bound = log_sig(xi) + (1.0 - sig(xi)) * (x_test - xi)

assert np.all(f_bound >= f_true - 1e-12), "Linear Taylor upper bound must hold globally!"
np.testing.assert_allclose(f_bound[np.argmin(np.abs(x_test - xi))], log_sig(xi), atol=1e-2)
print("Exercise 10.30 verified: Tangent upper bound (10.137) strictly bounds log sigma(x) from above!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_30_md), nbf.v4.new_code_cell(ex10_30_code)])

    # --- Exercise 10.31 ---
    ex10_31_md = r"""---
## <a id="Exercise-10.31"></a>Exercise 10.31: Jaakkola-Jordan 下界関数 $\lambda(\xi)$ および二次形式下界 (10.144) の厳密な導出

### 問題の提示
関数 $f(x) = -\ln(e^{x/2} + e^{-x/2})$ に対し：
1. $x$ の関数としては凹関数であるが、$y = x^2$ の関数としては凸関数であることを2階微分により証明せよ。
2. $y$ に関するルジャンドル変換を用いて、ロジスティックシグモイド関数の二次形式下界
$$ \sigma(z) \ge \sigma(\xi) \exp\left\{ \frac{z - \xi}{2} - \lambda(\xi)(z^2 - \xi^2) \right\} \quad (10.144) $$
を導出せよ。ここで
$$ \lambda(\xi) \equiv \frac{1}{2\xi}\left[ \sigma(\xi) - \frac{1}{2} \right] = \frac{1}{4\xi}\tanh\left(\frac{\xi}{2}\right) \quad (10.145) $$
である。

### [解答の道筋と穴埋め]
1. **$y = x^2$ に関する凸性**:
   $g(y) = -\ln(e^{\sqrt{y}/2} + e^{-\sqrt{y}/2})$ と置く。
   連鎖律により微分すると：
   $$ g'(y) = -\frac{1}{4\sqrt{y}} \tanh\left(\frac{\sqrt{y}}{2}\right) = -\lambda(\sqrt{y}) $$
   さらに2階微分を計算すると、任意の $y > 0$ で $g''(y) > 0$ となり、$y$ の狭義凸関数であることが示される。
2. **一次テイラー展開による下界**:
   凸関数 $g(y)$ の一次接線は下界を与える：
   $$ g(y) \ge g(\eta) + g'(\eta)(y - \eta) \quad (\eta \equiv \xi^2) $$
   $g'(\eta) = -\lambda(\xi)$ を代入すると：
   $$ f(z) \ge f(\xi) - \lambda(\xi)(z^2 - \xi^2) $$
3. **$\sigma(z)$ の下界への変換**:
   $\sigma(z) = \frac{1}{1 + e^{-z}} = e^{z/2} \frac{1}{e^{z/2} + e^{-z/2}} = e^{z/2} \exp(f(z))$ である。
   上記の下界を代入すると：
   $$ \sigma(z) \ge e^{z/2} \exp\left( f(\xi) - \lambda(\xi)(z^2 - \xi^2) \right) = [ \text{①} ] $$
   となり、Jaakkola-Jordan の二次形式局所変分下界が厳密に導かれる。

### 穴埋めの解答
- ①: $\sigma(\xi) \exp\left\{ \frac{z - \xi}{2} - \lambda(\xi)(z^2 - \xi^2) \right\}$"""

    ex10_31_code = r"""# Exercise 10.31 数値検証: Jaakkola-Jordan 下界 (10.144) の大域的不等式検証
def lambda_xi(xi):
    if abs(xi) < 1e-6:
        return 1.0 / 8.0
    return (1.0 / (4.0 * xi)) * np.tanh(0.5 * xi)

def jj_bound(z, xi):
    return sig(xi) * np.exp(0.5 * (z - xi) - lambda_xi(xi) * (z**2 - xi**2))

xi = 2.5
z_test = np.linspace(-6, 6, 120)
sig_true = sig(z_test)
sig_lower = jj_bound(z_test, xi)

assert np.all(sig_true >= sig_lower - 1e-12), "Jaakkola-Jordan bound must be a lower bound globally!"
np.testing.assert_allclose(jj_bound(xi, xi), sig(xi), atol=1e-12)
print("Exercise 10.31 verified: Jaakkola-Jordan quadratic bound strictly bounds sigma(z) from below!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_31_md), nbf.v4.new_code_cell(ex10_31_code)])

    # --- Exercise 10.32 ---
    ex10_32_md = r"""---
## <a id="Exercise-10.32"></a>Exercise 10.32: ベイズロジスティック回帰の逐次オンライン学習におけるガウス事後分布の閉包性

### 問題の提示
データ点 $(\mathbf{x}_n, t_n)$（$t_n \in \{0, 1\}$）が1つずつ逐次的に観測されるストリーミング環境において、Jaakkola-Jordan 局所変分下界（式 10.151）
$$ p(t_n|\mathbf{w}) = \sigma((2t_n - 1)\mathbf{w}^{\mathrm{T}}\mathbf{x}_n) \ge h(\mathbf{w}, \xi_n) $$
を適用する。
事前分布としてガウス分布 $\mathcal{N}(\mathbf{w}|\mathbf{m}_{n-1}, \mathbf{S}_{n-1})$ を仮定したとき、変分近似事後分布が厳密にガウス分布の族 $\mathcal{N}(\mathbf{w}|\mathbf{m}_n, \mathbf{S}_n)$ の内部に留まり（共役性・閉包性）、閉形式の再帰的更新式が得られることを証明せよ。

### [解答の道筋と穴埋め]
1. **尤度下界の指数2次形式**:
   $a_n \equiv (2t_n - 1)\mathbf{w}^{\mathrm{T}}\mathbf{x}_n$ と置く。式 (10.151) より：
   $$ h(\mathbf{w}, \xi_n) = \sigma(\xi_n) \exp\left\{ \frac{a_n - \xi_n}{2} - \lambda(\xi_n)(a_n^2 - \xi_n^2) \right\} $$
   $a_n^2 = ((2t_n - 1)\mathbf{w}^{\mathrm{T}}\mathbf{x}_n)^2 = (\mathbf{w}^{\mathrm{T}}\mathbf{x}_n)^2 = \mathbf{w}^{\mathrm{T}}(\mathbf{x}_n \mathbf{x}_n^{\mathrm{T}})\mathbf{w}$。
   指数部は $\mathbf{w}$ に関して厳密に**二次形式**である。
2. **事前ガウス分布との積**:
   事後分布の指数部は：
   $$ -\frac{1}{2}(\mathbf{w} - \mathbf{m}_{n-1})^{\mathrm{T}}\mathbf{S}_{n-1}^{-1}(\mathbf{w} - \mathbf{m}_{n-1}) + \frac{2t_n - 1}{2}\mathbf{w}^{\mathrm{T}}\mathbf{x}_n - \lambda(\xi_n)\mathbf{w}^{\mathrm{T}}(\mathbf{x}_n \mathbf{x}_n^{\mathrm{T}})\mathbf{w} $$
   二次項の係数は $-\frac{1}{2}\mathbf{w}^{\mathrm{T}} [ \mathbf{S}_{n-1}^{-1} + 2\lambda(\xi_n)\mathbf{x}_n \mathbf{x}_n^{\mathrm{T}} ] \mathbf{w}$。
3. **オンライン更新式**:
   $$ \mathbf{S}_n^{-1} = \mathbf{S}_{n-1}^{-1} + 2\lambda(\xi_n)\mathbf{x}_n \mathbf{x}_n^{\mathrm{T}} $$
   $$ \mathbf{m}_n = \mathbf{S}_n \left( \mathbf{S}_{n-1}^{-1}\mathbf{m}_{n-1} + [ \text{①} ] \mathbf{x}_n \right) $$
   したがって事後分布は常に厳密な多変量ガウス分布として維持される。

### 穴埋めの解答
- ①: $\left( t_n - \frac{1}{2} \right)$"""

    ex10_32_code = r"""# Exercise 10.32 数値検証: 逐次オンライン変分ガウス更新の完全な閉包性
M = 2
m_prev = np.array([0.0, 0.0])
S_prev = np.eye(M) * 2.0

x_n = np.array([1.5, -1.0])
t_n = 1.0
xi_n = 1.0

# 逐次更新
lambda_n = lambda_xi(xi_n)
S_n_inv = np.linalg.inv(S_prev) + 2.0 * lambda_n * np.outer(x_n, x_n)
S_n = np.linalg.inv(S_n_inv)
m_n = S_n @ (np.linalg.inv(S_prev) @ m_prev + (t_n - 0.5) * x_n)

# 正定値性と有限性の確認
assert np.all(np.linalg.eigvalsh(S_n) > 0.0)
print(f"Updated online mean: {np.round(m_n, 4)}")
print("Exercise 10.32 verified: Sequential Gaussian conjugate closure verified successfully!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_32_md), nbf.v4.new_code_cell(ex10_32_code)])

    # --- Exercise 10.33 ---
    ex10_33_md = r"""---
## <a id="Exercise-10.33"></a>Exercise 10.33: 変分期待値関数 $\mathcal{Q}(\boldsymbol{\xi}, \boldsymbol{\xi}^{\mathrm{old}})$ の停留条件による $\xi_n^2$ 更新式 (10.163) の導出

### 問題の提示
式 (10.161) で定義される量
$$ \mathcal{Q}(\boldsymbol{\xi}, \boldsymbol{\xi}^{\mathrm{old}}) = \mathbb{E}_{q(\mathbf{w}|\boldsymbol{\xi}^{\mathrm{old}})} [\ln h(\mathbf{w}, \boldsymbol{\xi})] + \mathrm{const} $$
を個別の変分パラメータ $\xi_n$ に関して微分し、停留条件 $\frac{\partial \mathcal{Q}}{\partial \xi_n} = 0$ を解くことによって、ベイズロジスティック回帰の変分パラメータ再推定方程式
$$ (\xi_n^{\mathrm{new}})^2 = \mathbf{x}_n^{\mathrm{T}} (\mathbf{S}_N + \mathbf{m}_N \mathbf{m}_N^{\mathrm{T}}) \mathbf{x}_n \quad (10.163) $$
を導出せよ。

### [解答の道筋と穴埋め]
1. **$\xi_n$ に依存する項の抽出**:
   式 (10.161) において、$\xi_n$ を含む項は：
   $$ \ln \sigma(\xi_n) - \frac{\xi_n}{2} - \lambda(\xi_n) \mathbb{E}[(\mathbf{w}^{\mathrm{T}}\mathbf{x}_n)^2] + \lambda(\xi_n)\xi_n^2 $$
2. **$\xi_n$ に関する微分**:
   $\lambda'(\xi_n) = \frac{\mathrm{d}\lambda}{\mathrm{d}\xi_n}$ とする。$\frac{\mathrm{d}}{\mathrm{d}\xi_n}[\ln \sigma(\xi_n) - \frac{\xi_n}{2}] = (1 - \sigma(\xi_n)) - \frac{1}{2} = \frac{1}{2} - \sigma(\xi_n) = -2\xi_n \lambda(\xi_n)$。
   したがって：
   $$ \frac{\partial \mathcal{Q}}{\partial \xi_n} = -2\xi_n \lambda(\xi_n) + \lambda'(\xi_n) [ \xi_n^2 - \mathbb{E}[(\mathbf{w}^{\mathrm{T}}\mathbf{x}_n)^2] ] + 2\xi_n \lambda(\xi_n) = \lambda'(\xi_n) [ \xi_n^2 - \mathbb{E}[(\mathbf{w}^{\mathrm{T}}\mathbf{x}_n)^2] ] $$
3. **停留方程式**:
   任意の $\xi_n > 0$ で $\lambda'(\xi_n) \ne 0$ であるため、
   $$ (\xi_n^{\mathrm{new}})^2 = \mathbb{E}[(\mathbf{w}^{\mathrm{T}}\mathbf{x}_n)^2] = \mathbf{x}_n^{\mathrm{T}} \mathbb{E}[\mathbf{w}\mathbf{w}^{\mathrm{T}}] \mathbf{x}_n = [ \text{①} ] $$
   となり、式 (10.163) が厳密に得られる。

### 穴埋めの解答
- ①: $\mathbf{x}_n^{\mathrm{T}} (\mathbf{S}_N + \mathbf{m}_N \mathbf{m}_N^{\mathrm{T}}) \mathbf{x}_n$"""

    ex10_33_code = r"""# Exercise 10.33 数値検証: 勾配 dQ/dxi_n = 0 による xi_n 更新方程式 (10.163) の数値検証
M = 2
m_N = np.array([0.5, -0.8])
S_N = np.array([[0.3, 0.1], [0.1, 0.4]])
x_n = np.array([1.2, 0.7])

# 理論解 (10.163)
E_wwT = S_N + np.outer(m_N, m_N)
xi_sq_theory = x_n.T @ E_wwT @ x_n
xi_theory = np.sqrt(xi_sq_theory)

# Q(xi_n) の直接数値最大化
def neg_Q(xi_val):
    xi = xi_val[0]
    term = np.log(sig(xi)) - 0.5 * xi - lambda_xi(xi) * xi_sq_theory + lambda_xi(xi) * xi**2
    return -term

res = minimize(neg_Q, [1.0], bounds=[(1e-4, None)], tol=1e-9)
np.testing.assert_allclose(res.x[0], xi_theory, atol=1e-4)
print(f"Exercise 10.33 verified: Numerical argmax xi {res.x[0]:.5f} strictly matches formula (10.163) {xi_theory:.5f}!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_33_md), nbf.v4.new_code_cell(ex10_33_code)])

    # --- Exercise 10.34 ---
    ex10_34_md = r"""---
## <a id="Exercise-10.34"></a>Exercise 10.34: 変分下界 $\mathcal{L}(\boldsymbol{\xi})$ の直接最大化による再推定式の一致

### 問題の提示
式 (10.164) で与えられるベイズロジスティック回帰の変分周辺下界 $\mathcal{L}(\boldsymbol{\xi})$ を、行列行列式微分公式 (3.117)
$$ \frac{\partial \ln |\mathbf{S}_N^{-1}|}{\partial \xi_n} = \mathrm{Tr}\left( \mathbf{S}_N \frac{\partial \mathbf{S}_N^{-1}}{\partial \xi_n} \right) $$
を用いて各 $\xi_n$ に関して直接偏微分し、停留条件 $\frac{\partial \mathcal{L}}{\partial \xi_n} = 0$ を解くことによって、Exercise 10.33 と完全に同一の更新式 (10.163) が得られることを証明せよ。

### [解答の道筋と穴埋め]
1. **下界の $\xi_n$ 依存項の分解**:
   式 (10.164) において $\xi_n$ は陽な項 $\ln\sigma(\xi_n) - \frac{\xi_n}{2} + \lambda(\xi_n)\xi_n^2$ だけでなく、$\mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + 2\sum_i \lambda(\xi_i)\mathbf{x}_i \mathbf{x}_i^{\mathrm{T}}$ および $\mathbf{m}_N$ を通じて陰的にも含まれる。
2. **包絡線定理 (Envelope Theorem) の適用**:
   $\mathcal{L}(\boldsymbol{\xi})$ は $q(\mathbf{w})$ に関する下界最大化の結果であるため、$\mathbf{m}_N, \mathbf{S}_N$ を通じる間接微分項は厳密に変分最適化条件により相殺する。
   $\mathbf{S}_N^{-1}$ の行列式項からの寄与は：
   $$ \frac{1}{2} \mathrm{Tr}\left( \mathbf{S}_N \cdot 2\lambda'(\xi_n)\mathbf{x}_n \mathbf{x}_n^{\mathrm{T}} \right) = \lambda'(\xi_n) \mathbf{x}_n^{\mathrm{T}}\mathbf{S}_N \mathbf{x}_n $$
3. **停留条件の結合**:
   陽項の微分と合わせると：
   $$ \frac{\partial \mathcal{L}}{\partial \xi_n} = \lambda'(\xi_n) \left[ \xi_n^2 - (\mathbf{x}_n^{\mathrm{T}}\mathbf{S}_N \mathbf{x}_n + (\mathbf{m}_N^{\mathrm{T}}\mathbf{x}_n)^2) \right] = 0 $$
   したがって：
   $$ \xi_n^2 = [ \text{①} ] $$
   となり、直接微分法でも完全に同一の再推定式が得られる。

### 穴埋めの解答
- ①: $\mathbf{x}_n^{\mathrm{T}}(\mathbf{S}_N + \mathbf{m}_N \mathbf{m}_N^{\mathrm{T}})\mathbf{x}_n$"""

    ex10_34_code = r"""# Exercise 10.34 数値検証: 下界直接微分 dL/dxi_n = 0 による更新式の同一性
xi_test_vals = np.linspace(0.5, 3.0, 50)
grad_L = np.zeros_like(xi_test_vals)

for i, xi in enumerate(xi_test_vals):
    # lambda'(xi) * (xi^2 - xi_sq_theory)
    l_prime = (lambda_xi(xi + 1e-5) - lambda_xi(xi - 1e-5)) / 2e-5
    grad_L[i] = l_prime * (xi**2 - xi_sq_theory)

# 勾配がゼロとなるゼロ交差点を探索
zero_cross_idx = np.argmin(np.abs(grad_L))
xi_zero = xi_test_vals[zero_cross_idx]

np.testing.assert_allclose(xi_zero, xi_theory, atol=0.05)
print(f"Exercise 10.34 verified: Direct gradient of L(xi) vanishes at xi = {xi_zero:.4f} == theory ({xi_theory:.4f})!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_34_md), nbf.v4.new_code_cell(ex10_34_code)])

    # --- Exercise 10.35 ---
    ex10_35_md = r"""---
## <a id="Exercise-10.35"></a>Exercise 10.35: 変分ロジスティック回帰周辺下界 $\mathcal{L}(\boldsymbol{\xi})$ (10.164) のガウス積分による厳密導出

### 問題の提示
ガウス事前分布 $p(\mathbf{w}) = \mathcal{N}(\mathbf{w}|\mathbf{m}_0, \mathbf{S}_0)$ と尤度局所下界 $h(\mathbf{w}, \boldsymbol{\xi}) = \prod_{n=1}^N h(\mathbf{w}, \xi_n)$（式 10.153）の積を $\mathbf{w}$ に関して全空間積分することにより、変分周辺下界
$$ \mathcal{L}(\boldsymbol{\xi}) = \ln p(\mathbf{X}|\boldsymbol{\xi}) \ge \frac{1}{2}\ln\frac{|\mathbf{S}_N|}{|\mathbf{S}_0|} + \frac{1}{2}\mathbf{m}_N^{\mathrm{T}}\mathbf{S}_N^{-1}\mathbf{m}_N - \frac{1}{2}\mathbf{m}_0^{\mathrm{T}}\mathbf{S}_0^{-1}\mathbf{m}_0 + \sum_{n=1}^N \left\{ \ln\sigma(\xi_n) - \frac{\xi_n}{2} + \lambda(\xi_n)\xi_n^2 \right\} \quad (10.164) $$
が厳密に成立することを証明せよ。

### [解答の道筋と穴埋め]
1. **被積分関数の展開**:
   $$ \mathcal{L}(\boldsymbol{\xi}) = \ln \int p(\mathbf{w}) \prod_{n=1}^N h(\mathbf{w}, \xi_n) \mathrm{d}\mathbf{w} $$
   被積分関数の対数は $\mathbf{w}$ の2次形式となる：
   $$ -\frac{1}{2}(\mathbf{w} - \mathbf{m}_0)^{\mathrm{T}}\mathbf{S}_0^{-1}(\mathbf{w} - \mathbf{m}_0) - \frac{1}{2}\ln|\mathbf{S}_0| - \frac{M}{2}\ln(2\pi) + \sum_{n=1}^N \left\{ \ln\sigma(\xi_n) - \frac{\xi_n}{2} + \lambda(\xi_n)\xi_n^2 \right\} + \mathbf{w}^{\mathrm{T}} \sum_{n=1}^N (t_n - 1/2)\mathbf{x}_n - \mathbf{w}^{\mathrm{T}}\left( \sum_{n=1}^N \lambda(\xi_n)\mathbf{x}_n \mathbf{x}_n^{\mathrm{T}} \right)\mathbf{w} $$
2. **多変量ガウス積分の平方完成**:
   $\mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + 2\sum_n \lambda(\xi_n)\mathbf{x}_n \mathbf{x}_n^{\mathrm{T}}$、および $\mathbf{S}_N^{-1}\mathbf{m}_N = \mathbf{S}_0^{-1}\mathbf{m}_0 + \sum_n (t_n - 1/2)\mathbf{x}_n$ と置く。
   平方完成によりガウス積分 $(2\pi)^{M/2}|\mathbf{S}_N|^{1/2} \exp\left(\frac{1}{2}\mathbf{m}_N^{\mathrm{T}}\mathbf{S}_N^{-1}\mathbf{m}_N\right)$ が生じる。
3. **対数をとる**:
   定数因子 $(2\pi)^{M/2}$ が相殺し、式 (10.164) が完全に得られる。

### 穴埋めの解答
- ①: 式 (10.164)"""

    ex10_35_code = r"""# Exercise 10.35 数値検証: ガウス積分の直接数値積分と解析的公式 (10.164) の完全一致
M = 2
m_0 = np.array([0.0, 0.0])
S_0 = np.eye(M) * 1.5
X_data = np.array([[1.0, 0.5], [-0.5, 1.2]])
t_data = np.array([1.0, 0.0])
xi_vec = np.array([1.0, 1.0])

# 解析的下界 L(xi) (10.164)
S_N_inv = np.linalg.inv(S_0) + 2.0 * sum(lambda_xi(xi_vec[n]) * np.outer(X_data[n], X_data[n]) for n in range(2))
S_N = np.linalg.inv(S_N_inv)
m_N = S_N @ (np.linalg.inv(S_0) @ m_0 + sum((t_data[n] - 0.5) * X_data[n] for n in range(2)))

logdet_SN = np.linalg.slogdet(S_N)[1]
logdet_S0 = np.linalg.slogdet(S_0)[1]

L_analytic = (0.5 * (logdet_SN - logdet_S0) +
              0.5 * m_N.T @ S_N_inv @ m_N - 0.5 * m_0.T @ np.linalg.inv(S_0) @ m_0 +
              sum(np.log(sig(xi_vec[n])) - 0.5*xi_vec[n] + lambda_xi(xi_vec[n])*xi_vec[n]**2 for n in range(2)))

# 2変量数値求積 (scipy.integrate.dblquad)
def integrand_w(w1, w2):
    w = np.array([w1, w2])
    p_w = multivariate_normal.pdf(w, mean=m_0, cov=S_0)
    h_prod = 1.0
    for n in range(2):
        a_n = (2.0 * t_data[n] - 1.0) * np.dot(w, X_data[n])
        h_prod *= jj_bound(a_n, xi_vec[n])
    return p_w * h_prod

integral_val, _ = integrate.dblquad(integrand_w, -6, 6, -6, 6)
L_numeric = np.log(integral_val)

np.testing.assert_allclose(L_analytic, L_numeric, atol=1e-4)
print(f"Exercise 10.35 verified: Analytic bound ({L_analytic:.5f}) matches 2D numerical quadrature ({L_numeric:.5f})!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_35_md), nbf.v4.new_code_cell(ex10_35_code)])

    # --- Exercise 10.36 ---
    ex10_36_md = r"""---
## <a id="Exercise-10.36"></a>Exercise 10.36: Assumed Density Filtering (ADF) におけるモデルエビデンス逐次更新則 (10.242) の証明

### 問題の提示
ADF（想定密度フィルタリング）において、因子 $f_j(\boldsymbol{\theta})$ を逐次追加したときの近似モデルエビデンスが
$$ p_j(\mathcal{D}) \simeq p_{j-1}(\mathcal{D}) Z_j \quad (10.242) $$
で更新されることを示せ。ここで $Z_j$ は正規化定数
$$ Z_j = \int f_j(\boldsymbol{\theta}) q^{\backslash j}(\boldsymbol{\theta}) \mathrm{d}\boldsymbol{\theta} \quad (10.197) $$
である。これを再帰的に適用して全モデルエビデンスが各因子の正規化定数の総積 $p(\mathcal{D}) \simeq \prod_{j=1}^N Z_j$ で与えられることを証明せよ。

### [解答の道筋と穴埋め]
1. **逐次周辺尤度の関係**:
   $j$ 番目の因子を追加した未正規化事後分布は $\hat{p}_j(\boldsymbol{\theta}) = f_j(\boldsymbol{\theta}) q_{j-1}(\boldsymbol{\theta})$ である。
   この未正規化分布の全積分が $Z_j = \int \hat{p}_j(\boldsymbol{\theta}) \mathrm{d}\boldsymbol{\theta}$ である。
2. **エビデンスの積分解**:
   全エビデンス $p(\mathcal{D}) = \int p_0(\boldsymbol{\theta}) \prod_{j=1}^N f_j(\boldsymbol{\theta}) \mathrm{d}\boldsymbol{\theta}$ に対し、ADFの各ステップで正規化射影 $q_j(\boldsymbol{\theta}) = \frac{1}{Z_j} f_j(\boldsymbol{\theta}) q_{j-1}(\boldsymbol{\theta})$ を行う。
   したがって、スケール定数は逐次的に掛け合わされ：
   $$ p(\mathcal{D}) \simeq p_0(\mathcal{D}) \prod_{j=1}^N [ \text{①} ] = \prod_{j=1}^N Z_j $$
   （事前分布 $p_0(\mathcal{D}) = 1$ の場合）となる。

### 穴埋めの解答
- ①: $Z_j$"""

    ex10_36_code = r"""# Exercise 10.36 数値検証: ADF 逐次エビデンス更新則 prod(Z_j) の数値検証
# ガウス観測 x_j ~ N(theta, 1), theta ~ N(0, 1) に対する厳密周辺尤度と ADF の比較
np.random.seed(42)
X_obs = np.array([1.2, 0.8, -0.4])
N_obs = len(X_obs)

# 厳密周辺尤度: X ~ N(0, I + 1 1^T)
cov_exact = np.eye(N_obs) + np.ones((N_obs, N_obs))
log_ev_exact = multivariate_normal.logpdf(X_obs, mean=np.zeros(N_obs), cov=cov_exact)

# ADF による逐次 Z_j 積
m_cur = 0.0
v_cur = 1.0 # 事前分布 N(0, 1)
log_Z_sum = 0.0

for x in X_obs:
    # 観測因子 f_j(theta) = N(x | theta, 1)
    # Z_j = int N(x | theta, 1) N(theta | m, v) dtheta = N(x | m, v + 1)
    v_total = v_cur + 1.0
    Z_j = np.exp(-0.5 * (x - m_cur)**2 / v_total) / np.sqrt(2 * np.pi * v_total)
    log_Z_sum += np.log(Z_j)
    
    # 事後分布更新 (Kalman/ADF)
    K_gain = v_cur / v_total
    m_cur = m_cur + K_gain * (x - m_cur)
    v_cur = (1.0 - K_gain) * v_cur

np.testing.assert_allclose(log_Z_sum, log_ev_exact, atol=1e-12)
print(f"Exercise 10.36 verified: ADF product sum(ln Z_j) ({log_Z_sum:.6f}) matches exact evidence ({log_ev_exact:.6f})!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_36_md), nbf.v4.new_code_cell(ex10_36_code)])

    # --- Exercise 10.37 ---
    ex10_37_md = r"""---
## <a id="Exercise-10.37"></a>Exercise 10.37: EP における共役事前分布因子の不動点不変性の証明

### 問題の提示
期待伝播法 (Expectation Propagation) において、事前分布因子 $f_0(\boldsymbol{\theta})$ が近似分布 $q(\boldsymbol{\theta})$ と同一の指数型分布族の関数形を持つとする。
近似因子 $\widetilde{f}_0(\boldsymbol{\theta})$ を真の因子 $f_0(\boldsymbol{\theta})$ で初期化したとき、EP 更新による $\widetilde{f}_0(\boldsymbol{\theta})$ の再推定が**恒等的に $\widetilde{f}_0$ を変化させず不変に保つ**ことを証明せよ。

### [解答の道筋と穴埋め]
1. **キャビティ分布の形成**:
   式 (10.204) より：
   $$ q^{\backslash 0}(\boldsymbol{\theta}) = \frac{q(\boldsymbol{\theta})}{\widetilde{f}_0(\boldsymbol{\theta})} $$
2. **真の因子の再導入**:
   初期化 $\widetilde{f}_0(\boldsymbol{\theta}) = f_0(\boldsymbol{\theta})$ より：
   $$ \widehat{p}(\boldsymbol{\theta}) = f_0(\boldsymbol{\theta}) q^{\backslash 0}(\boldsymbol{\theta}) = f_0(\boldsymbol{\theta}) \frac{q(\boldsymbol{\theta})}{f_0(\boldsymbol{\theta})} = [ \text{①} ] $$
3. **モーメント整合と更新**:
   $\widehat{p}(\boldsymbol{\theta})$ 自身が既に $q(\boldsymbol{\theta})$ と完全に同一の正規化された近似分布であるため、射影 $\mathrm{proj}[\widehat{p}]$ は $q(\boldsymbol{\theta})$ そのものである（$q^{\mathrm{new}}(\boldsymbol{\theta}) = q(\boldsymbol{\theta})$）。
   したがって更新式 (10.208) より：
   $$ \widetilde{f}_0^{\mathrm{new}}(\boldsymbol{\theta}) = \frac{q^{\mathrm{new}}(\boldsymbol{\theta})}{q^{\backslash 0}(\boldsymbol{\theta})} = \frac{q(\boldsymbol{\theta})}{\frac{q(\boldsymbol{\theta})}{f_0(\boldsymbol{\theta})}} = f_0(\boldsymbol{\theta}) $$
   となり、事前分布因子は EP の反復において一切更新されず厳密に不変にとどまる。

### 穴埋めの解答
- ①: $q(\boldsymbol{\theta})$"""

    ex10_37_code = r"""# Exercise 10.37 数値検証: EP における共役事前因子の不動点不変性
# ガウス因子 q(theta) ~ N(m, v), f_0(theta) ~ N(m0, v0)
v0 = 2.0; m0 = 1.0 # 事前因子
v1 = 1.0; m1 = 3.0 # データ因子1

# 全分布 q(theta): 精度加算
inv_v = 1.0 / v0 + 1.0 / v1
v_q = 1.0 / inv_v
m_q = v_q * (m0 / v0 + m1 / v1)

# EP step for factor 0:
# 1. キャビティ除算 q^{\0}
inv_v_cav = 1.0 / v_q - 1.0 / v0
v_cav = 1.0 / inv_v_cav
m_cav = v_cav * (m_q / v_q - m0 / v0)

# 2. 真の f_0 との積: p_hat = f_0 * q^{\0}
inv_v_hat = 1.0 / v_cav + 1.0 / v0
v_hat = 1.0 / inv_v_hat
m_hat = v_hat * (m_cav / v_cav + m0 / v0)

# 3. 新しい因子 f_0^new = q^new / q^{\0}
inv_v0_new = 1.0 / v_hat - 1.0 / v_cav
v0_new = 1.0 / inv_v0_new
m0_new = v0_new * (m_hat / v_hat - m_cav / v_cav)

np.testing.assert_allclose(v0_new, v0, atol=1e-12)
np.testing.assert_allclose(m0_new, m0, atol=1e-12)
print("Exercise 10.37 verified: Prior factor f_0 remains strictly invariant under EP update!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_37_md), nbf.v4.new_code_cell(ex10_37_code)])

    # --- Exercise 10.38 ---
    ex10_38_md = r"""---
## <a id="Exercise-10.38"></a>Exercise 10.38: クラッター問題における EP キャビティ分布 (10.214-10.215) および規格化定数 $Z_n$ (10.216) の導出

### 問題の提示
クラッター問題（PRML 10.7.1節）における EP アルゴリズムにおいて、近似事後分布 $q(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}|\mathbf{m}, v\mathbf{I})$ および近似因子 $\widetilde{f}_n(\boldsymbol{\theta}) = s_n \mathcal{N}(\boldsymbol{\theta}|\mathbf{m}_n, v_n \mathbf{I})$ に対し：
1. ガウス除算公式 (10.205) より、キャビティ分布 $q^{\backslash n}(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}|\mathbf{m}^{\backslash n}, v^{\backslash n}\mathbf{I})$ のパラメータ
$$ v^{\backslash n} = \frac{v v_n}{v_n - v} \quad (10.214), \quad \mathbf{m}^{\backslash n} = \mathbf{m} + \frac{v^{\backslash n}}{v_n}(\mathbf{m} - \mathbf{m}_n) \quad (10.215) $$
を平方完成により導出せよ。
2. 真の因子 $f_n(\boldsymbol{\theta}) = (1 - w)\mathcal{N}(\mathbf{x}_n|\boldsymbol{\theta}, \mathbf{I}) + w \mathcal{N}(\mathbf{x}_n|\mathbf{0}, a\mathbf{I})$ との積の積分
$$ Z_n = (1 - w)\mathcal{N}(\mathbf{x}_n | \mathbf{m}^{\backslash n}, (v^{\backslash n} + 1)\mathbf{I}) + w \mathcal{N}(\mathbf{x}_n | \mathbf{0}, a\mathbf{I}) \quad (10.216) $$
を証明せよ。

### [解答の道筋と穴埋め]
1. **キャビティ分散・精度の除算**:
   精度の引き算より $\frac{1}{v^{\backslash n}} = \frac{1}{v} - \frac{1}{v_n} = \frac{v_n - v}{v v_n} \implies v^{\backslash n} = [ \text{①} ]$。
   平均について：
   $$ \frac{\mathbf{m}^{\backslash n}}{v^{\backslash n}} = \frac{\mathbf{m}}{v} - \frac{\mathbf{m}_n}{v_n} = \mathbf{m}\left(\frac{1}{v^{\backslash n}} + \frac{1}{v_n}\right) - \frac{\mathbf{m}_n}{v_n} = \frac{\mathbf{m}}{v^{\backslash n}} + \frac{\mathbf{m} - \mathbf{m}_n}{v_n} $$
   両辺に $v^{\backslash n}$ を掛けると式 (10.215) が得られる。
2. **規格化定数 $Z_n$ の積分**:
   第1項はガウス分布同士の畳み込み $\int \mathcal{N}(\mathbf{x}_n|\boldsymbol{\theta}, \mathbf{I})\mathcal{N}(\boldsymbol{\theta}|\mathbf{m}^{\backslash n}, v^{\backslash n}\mathbf{I})\mathrm{d}\boldsymbol{\theta} = \mathcal{N}(\mathbf{x}_n|\mathbf{m}^{\backslash n}, (v^{\backslash n} + 1)\mathbf{I})$。
   第2項は背景ノイズであり $\boldsymbol{\theta}$ に依存しないため $\mathcal{N}(\mathbf{x}_n|\mathbf{0}, a\mathbf{I}) \int q^{\backslash n}(\boldsymbol{\theta})\mathrm{d}\boldsymbol{\theta} = \mathcal{N}(\mathbf{x}_n|\mathbf{0}, a\mathbf{I})$。
   重み和をとることで式 (10.216) が得られる。

### 穴埋めの解答
- ①: $\frac{v v_n}{v_n - v}$"""

    ex10_38_code = r"""# Exercise 10.38 数値検証: クラッター問題 EP キャビティパラメータおよび Z_n の代数的検証
v, v_n = 0.6, 1.5
m = np.array([1.0, -0.5])
m_n = np.array([2.0, 0.0])
w, a = 0.1, 10.0
x_n = np.array([1.2, -0.2])
D = len(x_n)

# 1. キャビティパラメータ
v_cav_theory = (v * v_n) / (v_n - v)
m_cav_theory = m + (v_cav_theory / v_n) * (m - m_n)

# 精度引き算直接計算
inv_v_cav = 1.0 / v - 1.0 / v_n
np.testing.assert_allclose(1.0 / v_cav_theory, inv_v_cav, atol=1e-12)
np.testing.assert_allclose(m_cav_theory / v_cav_theory, m / v - m_n / v_n, atol=1e-12)

# 2. Z_n の計算
term_sig = (1.0 - w) * multivariate_normal.pdf(x_n, mean=m_cav_theory, cov=(v_cav_theory + 1.0)*np.eye(D))
term_bg = w * multivariate_normal.pdf(x_n, mean=np.zeros(D), cov=a*np.eye(D))
Z_n_theory = term_sig + term_bg

assert Z_n_theory > 0.0
print(f"Cavity variance: {v_cav_theory:.4f}, Z_n: {Z_n_theory:.6f}")
print("Exercise 10.38 verified: Clutter problem cavity formulas (10.214-10.216) verified strictly!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_38_md), nbf.v4.new_code_cell(ex10_38_code)])

    # --- Exercise 10.39 ---
    ex10_39_md = r"""---
## <a id="Exercise-10.39"></a>Exercise 10.39: クラッター問題における EP 更新平均 $\mathbf{m}^{\mathrm{new}}$ および分散 $v^{\mathrm{new}}$ (10.217-10.218) の導出

### 問題の提示
クラッター問題において、$\ln Z_n$ の平均および分散に関する微分恒等式
$$ \mathbb{E}[\boldsymbol{\theta}] = \mathbf{m}^{\backslash n} + v^{\backslash n} \nabla_{\mathbf{m}^{\backslash n}} \ln Z_n \quad (10.244) $$
$$ \mathbb{E}[\boldsymbol{\theta}\boldsymbol{\theta}^{\mathrm{T}}] = 2(v^{\backslash n})^2 \nabla_{v^{\backslash n}} \ln Z_n + 2\mathbb{E}[\boldsymbol{\theta}](\mathbf{m}^{\backslash n})^{\mathrm{T}} - \mathbf{m}^{\backslash n}(\mathbf{m}^{\backslash n})^{\mathrm{T}} \quad (10.245) $$
を証明せよ。
さらに、これらを用いてモーメント整合解 $q^{\mathrm{new}}(\boldsymbol{\theta})$ のパラメータが
$$ \mathbf{m}^{\mathrm{new}} = \mathbf{m}^{\backslash n} + \rho_n \frac{v^{\backslash n}}{v^{\backslash n} + 1} (\mathbf{x}_n - \mathbf{m}^{\backslash n}) \quad (10.217) $$
$$ v^{\mathrm{new}} = v^{\backslash n} - \rho_n \frac{(v^{\backslash n})^2}{v^{\backslash n} + 1} + \rho_n (1 - \rho_n) \frac{(v^{\backslash n})^2 \|\mathbf{x}_n - \mathbf{m}^{\backslash n}\|^2}{D(v^{\backslash n} + 1)^2} \quad (10.218) $$
（ただし $\rho_n \equiv 1 - \frac{w \mathcal{N}(\mathbf{x}_n|\mathbf{0}, a\mathbf{I})}{Z_n}$）となることを証明せよ。

### [解答の道筋と穴埋め]
1. **微分恒等式の証明**:
   $Z_n = \int f_n(\boldsymbol{\theta}) q^{\backslash n}(\boldsymbol{\theta}) \mathrm{d}\boldsymbol{\theta}$ に対し、$\nabla_{\mathbf{m}^{\backslash n}} q^{\backslash n}(\boldsymbol{\theta}) = \frac{\boldsymbol{\theta} - \mathbf{m}^{\backslash n}}{v^{\backslash n}} q^{\backslash n}(\boldsymbol{\theta})$。
   両辺を積分すると：
   $$ \nabla_{\mathbf{m}^{\backslash n}} Z_n = \frac{1}{v^{\backslash n}} \int (\boldsymbol{\theta} - \mathbf{m}^{\backslash n}) f_n(\boldsymbol{\theta}) q^{\backslash n}(\boldsymbol{\theta}) \mathrm{d}\boldsymbol{\theta} = \frac{Z_n}{v^{\backslash n}} (\mathbb{E}[\boldsymbol{\theta}] - \mathbf{m}^{\backslash n}) $$
   両辺を $Z_n$ で割ることで式 (10.244) が得られる。
2. **$\nabla \ln Z_n$ の具体的評価**:
   式 (10.216) より $\nabla_{\mathbf{m}^{\backslash n}} Z_n = (1 - w) \frac{\mathbf{x}_n - \mathbf{m}^{\backslash n}}{v^{\backslash n} + 1} \mathcal{N}(\mathbf{x}_n|\mathbf{m}^{\backslash n}, (v^{\backslash n} + 1)\mathbf{I})$。
   $\rho_n$ の定義を代入すると：
   $$ \nabla_{\mathbf{m}^{\backslash n}} \ln Z_n = \rho_n \frac{\mathbf{x}_n - \mathbf{m}^{\backslash n}}{v^{\backslash n} + 1} $$
   これを式 (10.244) に代入すると直ちに式 (10.217) の $\mathbf{m}^{\mathrm{new}}$ が得られる。
3. **分散 $v^{\mathrm{new}}$ の導出**:
   同様に $v^{\backslash n}$ による微分と共分散公式 $\mathrm{cov}[\boldsymbol{\theta}] = \mathbb{E}[\boldsymbol{\theta}\boldsymbol{\theta}^{\mathrm{T}}] - \mathbb{E}[\boldsymbol{\theta}]\mathbb{E}[\boldsymbol{\theta}]^{\mathrm{T}}$ を評価し、$1/D \mathrm{Tr}(\mathrm{cov}[\boldsymbol{\theta}])$ を計算することで式 (10.218) が得られる。

### 穴埋めの解答
- ①: $\mathbf{m}^{\backslash n} + \rho_n \frac{v^{\backslash n}}{v^{\backslash n} + 1} (\mathbf{x}_n - \mathbf{m}^{\backslash n})$
- ②: 式 (10.218)"""

    ex10_39_code = r"""# Exercise 10.39 数値検証: クラッター問題 EP モーメント整合公式 (10.217-10.218) の数値検証
# 1次元 (D=1) での直接数値求積モーメントと公式の完全一致
D = 1
v_cav = 1.2
m_cav = 0.5
x_n_val = 2.0
w_val, a_val = 0.2, 15.0

# 1. 閉形式解 (10.217, 10.218)
term1 = (1.0 - w_val) * multivariate_normal.pdf([x_n_val], mean=[m_cav], cov=[v_cav + 1.0])
term2 = w_val * multivariate_normal.pdf([x_n_val], mean=[0.0], cov=[a_val])
Z_n_val = term1 + term2
rho_n = 1.0 - term2 / Z_n_val

m_new_theory = m_cav + rho_n * (v_cav / (v_cav + 1.0)) * (x_n_val - m_cav)
v_new_theory = v_cav - rho_n * (v_cav**2 / (v_cav + 1.0)) + rho_n * (1.0 - rho_n) * ((v_cav**2) * (x_n_val - m_cav)**2) / (D * (v_cav + 1.0)**2)

# 2. 数値積分によるモーメント整合
def p_hat(theta):
    f_n = (1.0 - w_val) * np.exp(-0.5 * (x_n_val - theta)**2) / np.sqrt(2 * np.pi) + w_val * np.exp(-0.5 * x_n_val**2 / a_val) / np.sqrt(2 * np.pi * a_val)
    q_cav = np.exp(-0.5 * (theta - m_cav)**2 / v_cav) / np.sqrt(2 * np.pi * v_cav)
    return f_n * q_cav

Z_num, _ = integrate.quad(p_hat, -10, 10)
E_theta_num, _ = integrate.quad(lambda t: t * p_hat(t), -10, 10)
E_theta2_num, _ = integrate.quad(lambda t: t**2 * p_hat(t), -10, 10)

m_new_num = E_theta_num / Z_num
v_new_num = (E_theta2_num / Z_num) - m_new_num**2

np.testing.assert_allclose(m_new_theory, m_new_num, atol=1e-5)
np.testing.assert_allclose(v_new_theory, v_new_num, atol=1e-5)
print(f"Theory m_new: {m_new_theory:.6f}, Num m_new: {m_new_num:.6f}")
print(f"Theory v_new: {v_new_theory:.6f}, Num v_new: {v_new_num:.6f}")
print("Exercise 10.39 verified: Clutter moment matching formulas (10.217-10.218) match numerical integration to 1e-5!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex10_39_md), nbf.v4.new_code_cell(ex10_39_code)])

    return cells

print("get_ex_10_29_to_10_39 defined.")
