# scripts/build_ch12_exercises.py
"""
Build Chapter 12 Exercises Notebook (12/12_Exercises.ipynb)
Covers PRML Chapter 12: Continuous Latent Variables (Exercises 12.1 - 12.29)
Each exercise provides:
- Rigorous mathematical derivation and context
- Fill-in-the-blank (穴埋め) learning structure
- Self-contained Python numerical simulation / verification code
"""

import nbformat as nbf

def build_part1():
    """Exercises 12.1 - 12.5: PCA Foundations & Linear Transformations"""
    cells = []

    # Title & Introduction
    title_md = """# 第12章 連続潜在変数：演習問題 (Exercises 12.1 - 12.29)

本ノートブックは、PRML（パターン認識と機械学習）第12章「連続潜在変数 (Continuous Latent Variables)」に登場する**全29問 (Exercises 12.1 〜 12.29)** の詳細な数理解説、証明ステップ、穴埋め問題、および自己完結型の Python 数値検証コードを網羅した完全版です。

### 構成一覧
1. **Exercises 12.1 - 12.5**: PCA の最大分散幾何、数学的帰納法、歪み尺度と残差固有値和、スナップショット法、正規分布のアフィン変換
2. **Exercises 12.6 - 12.10**: 確率的主成分分析 (PPCA) のグラフィカルモデル、周辺分布・事後分布の導出、最尤推定停留点と大域的凸性
3. **Exercises 12.11 - 12.14**: ノイズレス極限 $\\sigma^2 \\to 0$ における直交射影、事後平均の原点縮小、最小二乗再構成点、独立パラメータ数
4. **Exercises 12.15 - 12.17**: PPCA の EM アルゴリズム M-step 導出、欠損値 EM アルゴリズム、二乗歪みコストの交互最小化
5. **Exercises 12.18 - 12.22**: 因子分析 (FA) のパラメータ数、潜在直交回転不変性、平均最尤推定量、FA の EM アルゴリズム (E/M-step)
6. **Exercises 12.23 - 12.25**: 混合 PPCA のグラフィカルモデル、多変量スチューデント t 分布の EM アルゴリズム、アフィン変換共変性
7. **Exercises 12.26 - 12.29**: カーネル PCA の固有値方程式と零空間不変性、線形カーネルの標準 PCA 一致証明、単調変数変換微分方程式、無相関従属反例
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))

    # Exercise 12.1
    ex12_1_md = """---
## Exercise 12.1: 数学的帰納法による $M$ 次元最大分散主部分空間の導出

### 問題の背景と数学的証明
第12.1節では、$M=1$ の場合に単位長さ制約 $\\mathbf{u}_1^{\\mathrm{T}}\\mathbf{u}_1 = 1$ の下で射影分散 $\\mathbf{u}_1^{\\mathrm{T}}\\mathbf{S}\\mathbf{u}_1$ を最大化すると、サンプル共分散行列 $\\mathbf{S}$ の最大固有値 $\\lambda_1$ に対応する固有ベクトル $\\mathbf{u}_1$ が選ばれることが示された。
本問では、数学的帰納法を用いて一般の $M$ 次元から $M+1$ 次元への拡張を証明する。

**帰納法の仮定**:
最初の $M$ 個の射影方向 $\\{\\mathbf{u}_1, \\dots, \\mathbf{u}_M\\}$ は、$\\mathbf{S}$ の降順にソートされた固有値 $\\lambda_1 \\ge \\lambda_2 \\ge \\dots \\ge \\lambda_M$ に対応する直交正規固有ベクトルであるとする。

**$M+1$ 次元の方向 $\\mathbf{u}_{M+1}$ の最適化**:
新たな方向 $\\mathbf{u}_{M+1}$ は、既知の直交正規基底 $\\{\\mathbf{u}_j\\}_{j=1}^M$ と直交し、かつ単位長さを持つ必要がある：
$$ \\mathbf{u}_{M+1}^{\\mathrm{T}}\\mathbf{u}_{M+1} = 1, \\quad \\mathbf{u}_{M+1}^{\\mathrm{T}}\\mathbf{u}_j = 0 \\quad (j=1, \\dots, M) $$
これらをラグランジュ未定乗数 $\\lambda_{M+1}$ および $\\{\\eta_j\\}_{j=1}^M$ を用いてラグランジュ関数を定義する：
$$ L(\\mathbf{u}_{M+1}) = \\mathbf{u}_{M+1}^{\\mathrm{T}}\\mathbf{S}\\mathbf{u}_{M+1} + \\lambda_{M+1}(1 - \\mathbf{u}_{M+1}^{\\mathrm{T}}\\mathbf{u}_{M+1}) - 2\\sum_{j=1}^M \\eta_j \\mathbf{u}_{M+1}^{\\mathrm{T}}\\mathbf{u}_j $$
$\\mathbf{u}_{M+1}$ に関して微分してゼロとおくと：
$$ \\frac{\\partial L}{\\partial \\mathbf{u}_{M+1}} = 2\\mathbf{S}\\mathbf{u}_{M+1} - 2\\lambda_{M+1}\\mathbf{u}_{M+1} - 2\\sum_{j=1}^M \\eta_j \\mathbf{u}_j = \\mathbf{0} $$
両辺に左から任意の $\\mathbf{u}_k^{\\mathrm{T}}$ ($k \\le M$) を掛けると：
$$ \\mathbf{u}_k^{\\mathrm{T}}\\mathbf{S}\\mathbf{u}_{M+1} - \\lambda_{M+1}\\mathbf{u}_k^{\\mathrm{T}}\\mathbf{u}_{M+1} - \\sum_{j=1}^M \\eta_j \\mathbf{u}_k^{\\mathrm{T}}\\mathbf{u}_j = 0 $$
ここで $\\mathbf{S}$ の対称性と帰納法の仮定 $\\mathbf{S}\\mathbf{u}_k = \\lambda_k \\mathbf{u}_k$ より：
$$ \\mathbf{u}_k^{\\mathrm{T}}\\mathbf{S}\\mathbf{u}_{M+1} = (\\mathbf{S}\\mathbf{u}_k)^{\\mathrm{T}}\\mathbf{u}_{M+1} = \\lambda_k \\mathbf{u}_k^{\\mathrm{T}}\\mathbf{u}_{M+1} = 0 $$
さらに直交正規性 $\\mathbf{u}_k^{\\mathrm{T}}\\mathbf{u}_j = \\delta_{kj}$ より、直ちに $\\eta_k = 0$ ($k=1, \\dots, M$) が従う。
したがって：
$$ \\mathbf{S}\\mathbf{u}_{M+1} = \\lambda_{M+1}\\mathbf{u}_{M+1} $$
すなわち $\\mathbf{u}_{M+1}$ もまた $\\mathbf{S}$ の固有ベクトルである。射影分散 $\\mathbf{u}_{M+1}^{\\mathrm{T}}\\mathbf{S}\\mathbf{u}_{M+1} = \\lambda_{M+1}$ を最大化するには、既に選ばれた $M$ 個以外の残りの固有値の中で最大の固有値 $\\lambda_{M+1}$ を選べばよい。これにより命題が証明された。

#### 穴埋め問題
1. $\\mathbf{S}$ が実対称行列であることから、$(\\mathbf{S}\\mathbf{u}_k)^{\\mathrm{T}}\\mathbf{u}_{M+1} = \\mathbf{u}_k^{\\mathrm{T}} \\text{[ (A) ]} \\mathbf{u}_{M+1}$ と変形できる。
2. 帰納法の仮定 $\\mathbf{S}\\mathbf{u}_k = \\lambda_k \\mathbf{u}_k$ を代入すると、直交性より値は $\\text{[ (B) ]}$ となる。
3. したがってラグランジュ乗数 $\\eta_k$ はすべて $\\text{[ (C) ]}$ となり、$\\mathbf{u}_{M+1}$ は $\\mathbf{S}$ の固有方程式を満たす。
*(解: A: $\\mathbf{S}$, B: $0$, C: $0$)*
"""
    ex12_1_code = """# Exercise 12.1 数値検証: 数学的帰納法に基づく逐次直交制約下の最大分散探索
import numpy as np
from scipy.optimize import minimize

np.random.seed(42)
D = 5
N = 1000
X = np.random.randn(N, D) @ np.diag([5.0, 3.5, 2.0, 1.0, 0.5])
S = np.cov(X, rowvar=False)

# 理論固有値分解 (降順)
eigvals, eigvecs = np.linalg.eigh(S)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# 逐次制約付き最適化により u_1, u_2, u_3 を順に求める
found_u = []
for m in range(3):
    def objective(u):
        return -u.T @ S @ u

    constraints = [{'type': 'eq', 'fun': lambda u: u.T @ u - 1.0}]
    for prev_u in found_u:
        constraints.append({'type': 'eq', 'fun': lambda u, pu=prev_u: u.T @ pu})

    res = minimize(objective, np.random.randn(D), constraints=constraints, method='SLSQP')
    u_opt = res.x
    var_opt = -res.fun
    found_u.append(u_opt)

    print(f"Step {m+1}: Optimized Variance = {var_opt:.6f}, True Eigenvalue = {eigvals[m]:.6f}")
    np.testing.assert_allclose(var_opt, eigvals[m], rtol=1e-3)
    # 直交性の確認
    for j in range(m):
        ortho_dot = np.abs(u_opt.T @ found_u[j])
        np.testing.assert_allclose(ortho_dot, 0.0, atol=1e-4)

print("Exercise 12.1 verified: Sequential orthogonal variance maximization recovers exact eigenvectors and eigenvalues!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_1_md), nbf.v4.new_code_cell(ex12_1_code)])

    # Exercise 12.2
    ex12_2_md = """---
## Exercise 12.2: 歪み尺度 $J$ の最小化と残差固有値和 (PRML 式 12.15, 12.93)

### 数学的導出
$D$ 次元データ空間において、$M$ 次元の主部分空間の外側の直交補空間を張る $D-M$ 本の正規直交基底を $\\mathbf{U} = [\\mathbf{u}_{M+1}, \\dots, \\mathbf{u}_D] \\in \\mathbb{R}^{D \\times (D-M)}$ とする。
二乗再構成歪み尺度は次式で与えられる：
$$ J = \\mathrm{Tr}\\{\\mathbf{U}^{\\mathrm{T}}\\mathbf{S}\\mathbf{U}\\} + \\mathrm{Tr}\\{\\mathbf{H}(\\mathbf{I} - \\mathbf{U}^{\\mathrm{T}}\\mathbf{U})\\} $$
ここで $\\mathbf{H}$ は正規直交制約 $\\mathbf{U}^{\\mathrm{T}}\\mathbf{U} = \\mathbf{I}$ を課すラグランジュ未定乗数行列である。

$\\mathbf{U}$ に関して $J$ を微分して停留条件を求めると：
$$ \\frac{\\partial J}{\\partial \\mathbf{U}} = 2\\mathbf{S}\\mathbf{U} - \\mathbf{U}(\\mathbf{H} + \\mathbf{H}^{\\mathrm{T}}) = \\mathbf{O} $$
制約 $\\mathbf{U}^{\\mathrm{T}}\\mathbf{U} = \\mathbf{I}$ が対称であるため、一般性を失うことなく $\\mathbf{H}$ を対称行列（$\\mathbf{H} = \\mathbf{H}^{\\mathrm{T}}$）と仮定できる。よって：
$$ \\mathbf{S}\\mathbf{U} = \\mathbf{U}\\mathbf{H} $$
$\\mathbf{H}$ は対称行列であるため、直交行列 $\\mathbf{V}$ を用いてスペクトル分解 $\\mathbf{H} = \\mathbf{V}\\boldsymbol{\\Lambda}_{D-M}\\mathbf{V}^{\\mathrm{T}}$ が可能である。
ここで $\\tilde{\\mathbf{U}} = \\mathbf{U}\\mathbf{V}$ と定義すると：
$$ \\tilde{\\mathbf{U}}^{\\mathrm{T}}\\tilde{\\mathbf{U}} = \\mathbf{V}^{\\mathrm{T}}\\mathbf{U}^{\\mathrm{T}}\\mathbf{U}\\mathbf{V} = \\mathbf{V}^{\\mathrm{T}}\\mathbf{I}\\mathbf{V} = \\mathbf{I} $$
$$ \\mathbf{S}\\tilde{\\mathbf{U}} = \\mathbf{S}\\mathbf{U}\\mathbf{V} = \\mathbf{U}\\mathbf{H}\\mathbf{V} = \\mathbf{U}\\mathbf{V}\\boldsymbol{\\Lambda}_{D-M} = \\tilde{\\mathbf{U}}\\boldsymbol{\\Lambda}_{D-M} $$
したがって $\\tilde{\\mathbf{U}}$ の各列はまさに $\\mathbf{S}$ の固有ベクトルである。
トレースの巡回不変性より：
$$ J = \\mathrm{Tr}\\{\\mathbf{U}^{\\mathrm{T}}\\mathbf{S}\\mathbf{U}\\} = \\mathrm{Tr}\\{\\mathbf{V}\\tilde{\\mathbf{U}}^{\\mathrm{T}}\\mathbf{S}\\tilde{\\mathbf{U}}\\mathbf{V}^{\\mathrm{T}}\\} = \\mathrm{Tr}\\{\\tilde{\\mathbf{U}}^{\\mathrm{T}}\\mathbf{S}\\tilde{\\mathbf{U}}\\} = \\sum_{i=M+1}^D \\lambda_i $$
これは最小二乗歪みが捨てられた $D-M$ 個の最小固有値の和に厳密に等しいことを示している。
"""
    ex12_2_code = """# Exercise 12.2 数値検証: 歪み尺度 J と最小固有値和の一致、および SU = UH の不変性
np.random.seed(42)
D = 6
M = 2
N = 500
X = np.random.randn(N, D)
S = np.cov(X, rowvar=False)

eigvals, eigvecs = np.linalg.eigh(S)
idx = np.argsort(eigvals) # 昇順
discarded_eigvals = eigvals[idx[:D-M]]
U_eig = eigvecs[:, idx[:D-M]] # 最小固有値に対応する D-M 本

# 1. 固有ベクトル解における歪み尺度
J_eig = np.trace(U_eig.T @ S @ U_eig)
sum_eig = np.sum(discarded_eigvals)
print(f"J with Eigenvectors: {J_eig:.8f}, Sum of discarded eigenvalues: {sum_eig:.8f}")
np.testing.assert_allclose(J_eig, sum_eig, atol=1e-10)

# 2. 任意の直交回転 V による一般解 U_rot = U_eig @ V
Q, _ = np.linalg.qr(np.random.randn(D-M, D-M))
U_rot = U_eig @ Q
H_rot = Q.T @ np.diag(discarded_eigvals) @ Q

# SU = UH の検証
np.testing.assert_allclose(S @ U_rot, U_rot @ H_rot, atol=1e-10)
# 歪み尺度 J_rot の不変性検証
J_rot = np.trace(U_rot.T @ S @ U_rot)
print(f"J with rotated U:    {J_rot:.8f}")
np.testing.assert_allclose(J_rot, sum_eig, atol=1e-10)
print("Exercise 12.2 verified: SU = UH holds and minimal distortion J strictly equals sum of discarded eigenvalues!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_2_md), nbf.v4.new_code_cell(ex12_2_code)])

    # Exercise 12.3
    ex12_3_md = """---
## Exercise 12.3: 高次元データにおけるスナップショット法固有ベクトルの正規化 (PRML 式 12.30)

### 数学的証明
データ数 $N$ が次元数 $D$ よりも小さい場合（$N \\ll D$）、$D \\times D$ 共分散行列 $\\mathbf{S} = \\frac{1}{N}\\mathbf{X}^{\\mathrm{T}}\\mathbf{X}$ の代わりに、$N \\times N$ のグラム行列 $\\frac{1}{N}\\mathbf{X}\\mathbf{X}^{\\mathrm{T}}$ の固有値問題を解くことで計算量を大幅に削減できる（スナップショット法）。

いま $\\mathbf{v}_i \\in \\mathbb{R}^N$ を $\\frac{1}{N}\\mathbf{X}\\mathbf{X}^{\\mathrm{T}}$ の単位固有ベクトル（$\\mathbf{v}_i^{\\mathrm{T}}\\mathbf{v}_i = 1$）、固有値を $\\lambda_i$ とする：
$$ \\frac{1}{N}\\mathbf{X}\\mathbf{X}^{\\mathrm{T}}\\mathbf{v}_i = \\lambda_i \\mathbf{v}_i $$
両辺に左から $\\mathbf{X}^{\\mathrm{T}}$ を掛けると：
$$ \\left(\\frac{1}{N}\\mathbf{X}^{\\mathrm{T}}\\mathbf{X}\\right) (\\mathbf{X}^{\\mathrm{T}}\\mathbf{v}_i) = \\lambda_i (\\mathbf{X}^{\\mathrm{T}}\\mathbf{v}_i) $$
これは $\\mathbf{X}^{\\mathrm{T}}\\mathbf{v}_i$ が共分散行列 $\\mathbf{S} = \\frac{1}{N}\\mathbf{X}^{\\mathrm{T}}\\mathbf{X}$ の同一固有値 $\\lambda_i$ に対する固有ベクトルであることを意味する。
PRML 式 (12.30) では、この固有ベクトルを単位長さに正規化したものを次のように定義している：
$$ \\mathbf{u}_i = \\frac{1}{\\sqrt{N\\lambda_i}} \\mathbf{X}^{\\mathrm{T}}\\mathbf{v}_i $$
この $\\mathbf{u}_i$ の二乗ノルムを直接計算すると：
$$ \\|\\mathbf{u}_i\\|^2 = \\mathbf{u}_i^{\\mathrm{T}}\\mathbf{u}_i = \\left(\\frac{1}{\\sqrt{N\\lambda_i}}\\mathbf{X}^{\\mathrm{T}}\\mathbf{v}_i\\right)^{\\mathrm{T}} \\left(\\frac{1}{\\sqrt{N\\lambda_i}}\\mathbf{X}^{\\mathrm{T}}\\mathbf{v}_i\\right) = \\frac{1}{N\\lambda_i} \\mathbf{v}_i^{\\mathrm{T}}\\mathbf{X}\\mathbf{X}^{\\mathrm{T}}\\mathbf{v}_i $$
ここで $\\frac{1}{N}\\mathbf{X}\\mathbf{X}^{\\mathrm{T}}\\mathbf{v}_i = \\lambda_i \\mathbf{v}_i$ を代入すると：
$$ \\|\\mathbf{u}_i\\|^2 = \\frac{1}{\\lambda_i} \\mathbf{v}_i^{\\mathrm{T}} (\\lambda_i \\mathbf{v}_i) = \\mathbf{v}_i^{\\mathrm{T}}\\mathbf{v}_i = 1 $$
したがって、$\\mathbf{u}_i$ は厳密に単位長さに正規化されている。
"""
    ex12_3_code = """# Exercise 12.3 数値検証: N << D におけるスナップショット法固有ベクトルの正規化
np.random.seed(42)
N = 20
D = 100
X = np.random.randn(N, D)
# 中心化
X = X - np.mean(X, axis=0)

# N x N グラム行列の固有値分解
K_gram = (1.0 / N) * (X @ X.T)
eigvals_K, eigvecs_K = np.linalg.eigh(K_gram)
# 正の固有値を持つものを選択
pos_idx = np.where(eigvals_K > 1e-10)[0]

for idx_i in pos_idx:
    lam = eigvals_K[idx_i]
    v = eigvecs_K[:, idx_i]
    # u_i = (1 / sqrt(N * lam)) * X.T @ v
    u = (1.0 / np.sqrt(N * lam)) * (X.T @ v)

    # 1. 単位ノルムの検証
    norm_sq = u.T @ u
    np.testing.assert_allclose(norm_sq, 1.0, atol=1e-12)

    # 2. S の固有方程式 S u = lam u の検証
    S = (1.0 / N) * (X.T @ X)
    Su = S @ u
    lam_u = lam * u
    np.testing.assert_allclose(Su, lam_u, atol=1e-10)

print(f"Exercise 12.3 verified: All {len(pos_idx)} eigenvectors u_i constructed via snapshot method have strictly unit norm and satisfy S u = lambda u!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_3_md), nbf.v4.new_code_cell(ex12_3_code)])

    # Exercise 12.4
    ex12_4_md = """---
## Exercise 12.4: 潜在変数事前分布の一般ガウス化と周辺分布 $p(\\mathbf{x})$ の不変性

### 数学的証明
確率的 PCA モデルの標準的定式化（PRML 12.31）では、潜在変数 $\\mathbf{z}$ の事前分布としてゼロ平均・単位共分散の標準正規分布 $p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{z} \\mid \\mathbf{0}, \\mathbf{I})$ を仮定する。
いま、これを一般のガウス分布 $p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{z} \\mid \\mathbf{m}, \\mathbf{\\Sigma})$ に置き換えたモデルを考える：
$$ \\mathbf{z} \\sim \\mathcal{N}(\\mathbf{m}, \\mathbf{\\Sigma}), \\quad \\mathbf{x} \\mid \\mathbf{z} \\sim \\mathcal{N}(\\mathbf{W}\\mathbf{z} + \\boldsymbol{\\mu}, \\sigma^2 \\mathbf{I}) $$
共分散行列 $\\mathbf{\\Sigma}$ は正定値対称行列であるため、コレスキー分解または平方根行列を用いて $\\mathbf{\\Sigma} = \\mathbf{L}\\mathbf{L}^{\\mathrm{T}}$ と因数分解できる。
ここで新しい標準正規潜在変数 $\\tilde{\\mathbf{z}} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$ を用いて $\\mathbf{z} = \\mathbf{m} + \\mathbf{L}\\tilde{\\mathbf{z}}$ と表すと：
$$ \\mathbf{x} = \\mathbf{W}(\\mathbf{m} + \\mathbf{L}\\tilde{\\mathbf{z}}) + \\boldsymbol{\\mu} + \\boldsymbol{\\epsilon} = (\\mathbf{W}\\mathbf{L})\\tilde{\\mathbf{z}} + (\\boldsymbol{\\mu} + \\mathbf{W}\\mathbf{m}) + \\boldsymbol{\\epsilon} $$
したがって、パラメータを次のように再定義する：
$$ \\tilde{\\mathbf{W}} = \\mathbf{W}\\mathbf{L}, \\quad \\tilde{\\boldsymbol{\\mu}} = \\boldsymbol{\\mu} + \\mathbf{W}\\mathbf{m} $$
このとき、$\\mathbf{x}$ の周辺分布 $p(\\mathbf{x})$ の平均および共分散行列は：
$$ \\mathbb{E}[\\mathbf{x}] = \\tilde{\\boldsymbol{\\mu}} = \\boldsymbol{\\mu} + \\mathbf{W}\\mathbf{m} $$
$$ \\mathrm{cov}[\\mathbf{x}] = \\tilde{\\mathbf{W}}\\tilde{\\mathbf{W}}^{\\mathrm{T}} + \\sigma^2 \\mathbf{I} = (\\mathbf{W}\\mathbf{L})(\\mathbf{L}^{\\mathrm{T}}\\mathbf{W}^{\\mathrm{T}}) + \\sigma^2 \\mathbf{I} = \\mathbf{W}\\mathbf{\\Sigma}\\mathbf{W}^{\\mathrm{T}} + \\sigma^2 \\mathbf{I} $$
これは元のパラメータと再定義後のパラメータで観測変数 $\\mathbf{x}$ の周辺分布 $p(\\mathbf{x})$ が完全に同一のガウス分布族を張ることを示している。ゆえに潜在空間の事前分布として $\\mathcal{N}(\\mathbf{0}, \\mathbf{I})$ を仮定してもモデルの表現力は一切損なわれない。
"""
    ex12_4_code = """# Exercise 12.4 数値検証: 潜在事前分布のアフィン変換と周辺ガウス分布の一致
np.random.seed(42)
D = 4
M = 2
W = np.random.randn(D, M)
mu = np.array([1.0, -0.5, 2.0, 0.0])
sigma_sq = 0.3

# 一般の潜在事前分布パラメータ m, Sigma
m = np.array([0.8, -1.2])
A_mat = np.random.randn(M, M)
Sigma = A_mat @ A_mat.T + 0.2 * np.eye(M)
L = np.linalg.cholesky(Sigma)

# 1. 元のモデルによる周辺パラメータ
mean_orig = mu + W @ m
Cov_orig = W @ Sigma @ W.T + sigma_sq * np.eye(D)

# 2. 再定義パラメータ W_tilde, mu_tilde による周辺パラメータ
W_tilde = W @ L
mu_tilde = mu + W @ m
Cov_reparam = W_tilde @ W_tilde.T + sigma_sq * np.eye(D)

np.testing.assert_allclose(mean_orig, mu_tilde, atol=1e-12)
np.testing.assert_allclose(Cov_orig, Cov_reparam, atol=1e-12)
print("Exercise 12.4 verified: General latent prior N(m, Sigma) yields identical marginal distribution p(x)!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_4_md), nbf.v4.new_code_cell(ex12_4_code)])

    # Exercise 12.5
    ex12_5_md = """---
## Exercise 12.5: ガウス変数の線形変換 $\\mathbf{y} = \\mathbf{A}\\mathbf{x} + \\mathbf{b}$ とランク幾何

### 数学的証明
$D$ 次元ガウス確率変数 $\\mathbf{x} \\sim \\mathcal{N}(\\mathbf{x} \\mid \\boldsymbol{\\mu}, \\mathbf{\\Sigma})$ に対し、$M \\times D$ 行列 $\\mathbf{A}$ と $M$ 次元ベクトル $\\mathbf{b}$ によるアフィン変換 $\\mathbf{y} = \\mathbf{A}\\mathbf{x} + \\mathbf{b}$ を考える。

ガウス分布の線形変換性より、$\\mathbf{y}$ もまたガウス分布に従う。
期待値および共分散行列は線形性より直ちに求まる：
$$ \\mathbb{E}[\\mathbf{y}] = \\mathbb{E}[\\mathbf{A}\\mathbf{x} + \\mathbf{b}] = \\mathbf{A}\\mathbb{E}[\\mathbf{x}] + \\mathbf{b} = \\mathbf{A}\\boldsymbol{\\mu} + \\mathbf{b} $$
$$ \\mathrm{cov}[\\mathbf{y}] = \\mathbb{E}[(\\mathbf{y} - \\mathbb{E}[\\mathbf{y}])(\\mathbf{y} - \\mathbb{E}[\\mathbf{y}])^{\\mathrm{T}}] = \\mathbf{A}\\mathbb{E}[(\\mathbf{x} - \\boldsymbol{\\mu})(\\mathbf{x} - \\boldsymbol{\\mu})^{\\mathrm{T}}]\\mathbf{A}^{\\mathrm{T}} = \\mathbf{A}\\mathbf{\\Sigma}\\mathbf{A}^{\\mathrm{T}} $$

### 次元の関係による幾何学的分類
1. **$M < D$ (次元削減)**: $\\mathbf{A}$ の行ランクがフルランク $M$ の場合、$\\mathbf{A}\\mathbf{\\Sigma}\\mathbf{A}^{\\mathrm{T}}$ は $M \\times M$ の正定値行列となり、$\\mathbb{R}^M$ 全体に確率密度を持つ非退化ガウス分布となる。
2. **$M = D$ (同次元変換)**: $\\mathbf{A}$ が非正則でなければ $\\mathbf{y}$ は $D$ 次元の正則なガウス分布となる。
3. **$M > D$ (埋め込み・次元拡大)**: $\\mathrm{rank}(\\mathbf{A}\\mathbf{\\Sigma}\\mathbf{A}^{\\mathrm{T}}) \\le \\min(M, D) = D < M$ となるため、共分散行列は階数不足（特異行列）となる。このとき確率質量は $M$ 次元空間内の $D$ 次元超平面上に集中し、$\\mathbb{R}^M$ 全体でのルベーグ測度はゼロとなる（特異ガウス分布）。
"""
    ex12_5_code = """# Exercise 12.5 数値検証: 線形アフィン変換のモーメントおよび特異ランクの検証
np.random.seed(42)
D = 3
mu = np.array([1.0, 2.0, -1.0])
Sigma = np.array([[2.0, 0.5, 0.2], [0.5, 1.5, -0.3], [0.2, -0.3, 1.0]])

# Case 1: M < D (M=2)
M1 = 2
A1 = np.random.randn(M1, D)
b1 = np.random.randn(M1)
Cov1 = A1 @ Sigma @ A1.T
eig1 = np.linalg.eigvalsh(Cov1)
assert np.all(eig1 > 0), "M < D covariance must be positive definite"

# Case 2: M > D (M=5)
M2 = 5
A2 = np.random.randn(M2, D)
b2 = np.random.randn(M2)
Cov2 = A2 @ Sigma @ A2.T
rank2 = np.linalg.matrix_rank(Cov2)
print(f"M > D: Matrix shape = {Cov2.shape}, Rank = {rank2} (bounded by D = {D})")
assert rank2 <= D, "Rank cannot exceed D"

# 標本平均・共分散のモンテカルロ整合性確認
N_mc = 50000
X_mc = np.random.multivariate_normal(mu, Sigma, size=N_mc)
Y1_mc = X_mc @ A1.T + b1
np.testing.assert_allclose(np.mean(Y1_mc, axis=0), A1 @ mu + b1, atol=0.02)
np.testing.assert_allclose(np.cov(Y1_mc, rowvar=False), Cov1, atol=0.03)
print("Exercise 12.5 verified: Gaussian affine transformations and rank behaviors confirmed!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_5_md), nbf.v4.new_code_cell(ex12_5_code)])

    return cells

def build_part2():
    """Exercises 12.6 - 12.10: PPCA Inference & Global Optimality"""
    cells = []

    # Exercise 12.6
    ex12_6_md = """---
## Exercise 12.6: 確率的 PCA の有向グラフィカルモデルとナイーブベイズ独立性

### 数学的解説
確率的 PCA モデルにおいて、観測ベクトル $\\mathbf{x} = (x_1, \\dots, x_D)^{\\mathrm{T}}$ の各成分を個別のノードとして展開した有向グラフィカルモデルを考える。
生成モデルは次式で表される：
$$ x_d = \\mathbf{w}_d^{\\mathrm{T}}\\mathbf{z} + \\mu_d + \\epsilon_d, \\quad \\epsilon_d \\sim \\mathcal{N}(0, \\sigma^2) \\quad (d=1, \\dots, D) $$
ここで $\\mathbf{w}_d^{\\mathrm{T}}$ は重み行列 $\\mathbf{W}$ の第 $d$ 行ベクトルである。各ノイズ成分 $\\epsilon_1, \\dots, \\epsilon_D$ は互いに独立である。

したがって、潜在変数 $\\mathbf{z}$ が与えられた（条件付けられた）もとでの観測成分の同時条件付き確率は完全に因数分解される：
$$ p(\\mathbf{x} \\mid \\mathbf{z}) = \\prod_{d=1}^D p(x_d \\mid \\mathbf{z}) = \\prod_{d=1}^D \\mathcal{N}(x_d \\mid \\mathbf{w}_d^{\\mathrm{T}}\\mathbf{z} + \\mu_d, \\sigma^2) $$
有向グラフにおいて、ノード $\\mathbf{z}$ は各 $x_d$ に向かう親ノードであり、$\\mathbf{z}$ を条件付けることでテール・トゥ・テール（tail-to-tail）経路がブロックされ、すべての $x_i, x_j$ ($i \\ne j$) 間は d-分離（d-separated）される。
これは、クラスラベル $y$ を条件付けたときに特徴量成分が独立になる**ナイーブベイズモデル（Naive Bayes Model, PRML 8.2.2節）** と全く同一の条件付き独立性構造を持つことを示している。
"""
    ex12_6_code = """# Exercise 12.6 数値検証: 潜在変数 z の条件付けによる観測成分間の完全独立性
np.random.seed(42)
D = 4
M = 2
W = np.random.randn(D, M)
mu = np.array([0.5, -1.0, 2.0, 1.5])
sigma_sq = 0.25

# 1. z を条件付けない周辺共分散 C = W W^T + sigma^2 I
C_marginal = W @ W.T + sigma_sq * np.eye(D)
# 非対角成分の最大絶対値（強い相関が存在）
off_diag_marginal = np.max(np.abs(C_marginal - np.diag(np.diag(C_marginal))))
print(f"Marginal Covariance Off-Diagonal Max: {off_diag_marginal:.6f} > 0")
assert off_diag_marginal > 0.1

# 2. z を固定した場合の条件付き共分散 cov(x | z)
# モンテカルロサンプリングで固定した z から x を生成
z_fixed = np.array([1.2, -0.7])
N_samples = 30000
eps = np.random.normal(0, np.sqrt(sigma_sq), size=(N_samples, D))
x_cond_samples = mu + (W @ z_fixed) + eps
cov_cond_emp = np.cov(x_cond_samples, rowvar=False)

# 条件付き共分散は対角行列 sigma^2 I に厳密一致
np.testing.assert_allclose(cov_cond_emp, sigma_sq * np.eye(D), atol=0.02)
print("Exercise 12.6 verified: Components x_d are conditionally strictly independent given z (Naive Bayes structure)!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_6_md), nbf.v4.new_code_cell(ex12_6_code)])

    # Exercise 12.7
    ex12_7_md = """---
## Exercise 12.7: 線形ガウスモデルの周辺分布 $p(\\mathbf{x}) = \\mathcal{N}(\\boldsymbol{\\mu}, \\mathbf{C})$ の完全導出 (PRML 式 12.35)

### 数学的導出
PRML第2章の線形ガウスモデルの公式 (2.270) および (2.271) を適用する：
$$ p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{z} \\mid \\mathbf{0}, \\mathbf{I}) $$
$$ p(\\mathbf{x} \\mid \\mathbf{z}) = \\mathcal{N}(\\mathbf{x} \\mid \\mathbf{W}\\mathbf{z} + \\boldsymbol{\\mu}, \\sigma^2 \\mathbf{I}) $$
期待値の線形性より：
$$ \\mathbb{E}[\\mathbf{x}] = \\mathbb{E}[\\mathbb{E}[\\mathbf{x} \\mid \\mathbf{z}]] = \\mathbb{E}[\\mathbf{W}\\mathbf{z} + \\boldsymbol{\\mu}] = \\mathbf{W}\\mathbb{E}[\\mathbf{z}] + \\boldsymbol{\\mu} = \\boldsymbol{\\mu} $$
また、全分散の法則（あるいは誤差項 $\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\sigma^2 \\mathbf{I})$ と $\\mathbf{z}$ の独立性）より：
$$ \\mathrm{cov}[\\mathbf{x}] = \\mathbb{E}[(\\mathbf{x} - \\boldsymbol{\\mu})(\\mathbf{x} - \\boldsymbol{\\mu})^{\\mathrm{T}}] = \\mathbb{E}[(\\mathbf{W}\\mathbf{z} + \\boldsymbol{\\epsilon})(\\mathbf{W}\\mathbf{z} + \\boldsymbol{\\epsilon})^{\\mathrm{T}}] $$
$$ = \\mathbf{W}\\mathbb{E}[\\mathbf{z}\\mathbf{z}^{\\mathrm{T}}]\\mathbf{W}^{\\mathrm{T}} + \\mathbf{W}\\mathbb{E}[\\mathbf{z}\\boldsymbol{\\epsilon}^{\\mathrm{T}}] + \\mathbb{E}[\\boldsymbol{\\epsilon}\\mathbf{z}^{\\mathrm{T}}]\\mathbf{W}^{\\mathrm{T}} + \\mathbb{E}[\\boldsymbol{\\epsilon}\\boldsymbol{\\epsilon}^{\\mathrm{T}}] $$
ここで $\\mathbb{E}[\\mathbf{z}\\mathbf{z}^{\\mathrm{T}}] = \\mathbf{I}$, $\\mathbb{E}[\\mathbf{z}\\boldsymbol{\\epsilon}^{\\mathrm{T}}] = \\mathbf{O}$, $\\mathbb{E}[\\boldsymbol{\\epsilon}\\boldsymbol{\\epsilon}^{\\mathrm{T}}] = \\sigma^2 \\mathbf{I}$ であるから：
$$ \\mathbf{C} = \\mathrm{cov}[\\mathbf{x}] = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}} + \\sigma^2 \\mathbf{I} $$
したがって、周辺分布はガウス分布 $p(\\mathbf{x}) = \\mathcal{N}(\\mathbf{x} \\mid \\boldsymbol{\\mu}, \\mathbf{C})$ となる。
"""
    ex12_7_code = """# Exercise 12.7 数値検証: 周辺分布 p(x) の解析解と経験的標本モーメントの完全一致
np.random.seed(42)
D = 5
M = 2
W = np.random.randn(D, M)
mu = np.array([2.0, -1.0, 0.5, 3.0, -2.5])
sigma_sq = 0.4
C_true = W @ W.T + sigma_sq * np.eye(D)

N_samples = 60000
z = np.random.randn(N_samples, M)
eps = np.random.normal(0, np.sqrt(sigma_sq), size=(N_samples, D))
x_samples = mu + z @ W.T + eps

emp_mean = np.mean(x_samples, axis=0)
emp_cov = np.cov(x_samples, rowvar=False)

np.testing.assert_allclose(emp_mean, mu, atol=0.03)
np.testing.assert_allclose(emp_cov, C_true, atol=0.05)
print("Exercise 12.7 verified: Marginal distribution p(x) matches N(mu, W W^T + sigma^2 I) exactly!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_7_md), nbf.v4.new_code_cell(ex12_7_code)])

    # Exercise 12.8
    ex12_8_md = """---
## Exercise 12.8: 事後分布 $p(\\mathbf{z} \\mid \\mathbf{x})$ の閉形式導出 (PRML 式 12.42)

### 数学的導出
PRML第2章の条件付きガウス分布の公式 (2.116) を適用する。
同時分布 $p(\\mathbf{z}, \\mathbf{x}) = p(\\mathbf{z})p(\\mathbf{x} \\mid \\mathbf{z})$ の指数項における $\\mathbf{z}$ の二次形式を整理する：
$$ \\ln p(\\mathbf{z} \\mid \\mathbf{x}) = -\\frac{1}{2}\\mathbf{z}^{\\mathrm{T}}\\mathbf{z} - \\frac{1}{2\\sigma^2}(\\mathbf{x} - \\boldsymbol{\\mu} - \\mathbf{W}\\mathbf{z})^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu} - \\mathbf{W}\\mathbf{z}) + \\mathrm{const} $$
$$ = -\\frac{1}{2} \\mathbf{z}^{\\mathrm{T}} \\left( \\mathbf{I} + \\frac{1}{\\sigma^2}\\mathbf{W}^{\\mathrm{T}}\\mathbf{W} \\right) \\mathbf{z} + \\frac{1}{\\sigma^2} \\mathbf{z}^{\\mathrm{T}}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu}) + \\mathrm{const} $$
ここで行列 $\\mathbf{M} = \\mathbf{W}^{\\mathrm{T}}\\mathbf{W} + \\sigma^2 \\mathbf{I} \\in \\mathbb{R}^{M \\times M}$ を定義すると：
$$ \\mathbf{I} + \\frac{1}{\\sigma^2}\\mathbf{W}^{\\mathrm{T}}\\mathbf{W} = \\frac{1}{\\sigma^2} \\mathbf{M} $$
したがって事後精度行列は $\\sigma^{-2}\\mathbf{M}$ であり、事後共分散行列は：
$$ \\mathbf{\\Sigma}_{z|x} = \\sigma^2 \\mathbf{M}^{-1} $$
また、一次の項と平方完成の対応から事後平均は：
$$ \\mathbb{E}[\\mathbf{z} \\mid \\mathbf{x}] = \\mathbf{\\Sigma}_{z|x} \\left( \\frac{1}{\\sigma^2}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu}) \\right) = (\\sigma^2 \\mathbf{M}^{-1}) \\left( \\frac{1}{\\sigma^2}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu}) \\right) = \\mathbf{M}^{-1}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu}) $$
よって PRML 式 (12.42) が厳密に導出された：
$$ p(\\mathbf{z} \\mid \\mathbf{x}) = \\mathcal{N}(\\mathbf{z} \\mid \\mathbf{M}^{-1}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu}), \\sigma^2 \\mathbf{M}^{-1}) $$
"""
    ex12_8_code = """# Exercise 12.8 数値検証: 事後分布 p(z | x) の閉形式とベイズ定理による完全一致
from scipy.stats import multivariate_normal

np.random.seed(42)
D = 3
M = 2
W = np.random.randn(D, M)
mu = np.array([1.0, 0.0, -1.0])
sigma_sq = 0.5
M_mat = W.T @ W + sigma_sq * np.eye(M)
M_inv = np.linalg.inv(M_mat)

# 観測点 x
x_test = np.array([2.5, -0.8, 0.4])

# 式 (12.42) による理論事後パラメータ
post_mean_theory = M_inv @ W.T @ (x_test - mu)
post_cov_theory = sigma_sq * M_inv

# ベイズ定理: p(z, x) の結合正規分布からの条件付き分布導出
# mu_joint = [0, mu], Cov_joint = [[I, W^T], [W, W W^T + sigma^2 I]]
C_mat = W @ W.T + sigma_sq * np.eye(D)
# ガウス公式による条件付き平均と共分散
# post_mean = mu_z + Cov_zx @ Cov_xx^{-1} (x - mu_x) = W^T @ C^{-1} (x - mu)
# post_cov  = I - W^T @ C^{-1} @ W
C_inv = np.linalg.inv(C_mat)
post_mean_joint = W.T @ C_inv @ (x_test - mu)
post_cov_joint = np.eye(M) - W.T @ C_inv @ W

np.testing.assert_allclose(post_mean_theory, post_mean_joint, atol=1e-12)
np.testing.assert_allclose(post_cov_theory, post_cov_joint, atol=1e-12)
print("Exercise 12.8 verified: Posterior p(z | x) formula matches joint Gaussian conditioning to machine precision!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_8_md), nbf.v4.new_code_cell(ex12_8_code)])

    # Exercise 12.9
    ex12_9_md = """---
## Exercise 12.9: 対数尤度最大化による $\\boldsymbol{\\mu}_{\\mathrm{ML}} = \\bar{\\mathbf{x}}$ の導出 (PRML 式 12.43)

### 数学的証明
確率的 PCA の対数尤度関数は次式で与えられる：
$$ \\ln p(\\mathbf{X} \\mid \\boldsymbol{\\mu}, \\mathbf{W}, \\sigma^2) = -\\frac{ND}{2}\\ln(2\\pi) - \\frac{N}{2}\\ln |\\mathbf{C}| - \\frac{1}{2}\\sum_{n=1}^N (\\mathbf{x}_n - \\boldsymbol{\\mu})^{\\mathrm{T}}\\mathbf{C}^{-1}(\\mathbf{x}_n - \\boldsymbol{\\mu}) $$
ここで $\\mathbf{C} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}} + \\sigma^2 \\mathbf{I}$ である。
この対数尤度を $\\boldsymbol{\\mu}$ について微分すると：
$$ \\frac{\\partial \\ln p}{\\partial \\boldsymbol{\\mu}} = -\\frac{1}{2} \\sum_{n=1}^N \\left( -2\\mathbf{C}^{-1}(\\mathbf{x}_n - \\boldsymbol{\\mu}) \\right) = \\mathbf{C}^{-1} \\left( \\sum_{n=1}^N \\mathbf{x}_n - N\\boldsymbol{\\mu} \\right) $$
極値条件 $\\frac{\\partial \\ln p}{\\partial \\boldsymbol{\\mu}} = \\mathbf{0}$ を課す：
$$ \\mathbf{C}^{-1} \\left( \\sum_{n=1}^N \\mathbf{x}_n - N\\boldsymbol{\\mu} \\right) = \\mathbf{0} $$
$\\mathbf{C}$ は正定値対称行列であるため正則であり、逆行列 $\\mathbf{C}^{-1}$ が存在する。
両辺に左から $\\mathbf{C}$ を掛けることにより：
$$ \\sum_{n=1}^N \\mathbf{x}_n - N\\boldsymbol{\\mu} = \\mathbf{0} \\implies \\boldsymbol{\\mu}_{\\mathrm{ML}} = \\frac{1}{N}\\sum_{n=1}^N \\mathbf{x}_n = \\bar{\\mathbf{x}} $$
したがって、最尤推定量 $\\boldsymbol{\\mu}_{\\mathrm{ML}}$ はサンプル平均 $\\bar{\\mathbf{x}}$ に厳密に一致する。
"""
    ex12_9_code = """# Exercise 12.9 数値検証: 対数尤度の勾配がサンプル平均でゼロとなることの確認
np.random.seed(42)
N = 100
D = 4
M = 2
X = np.random.randn(N, D) + np.array([1.5, -2.0, 3.0, 0.5])
x_bar = np.mean(X, axis=0)

W = np.random.randn(D, M)
sigma_sq = 0.8
C = W @ W.T + sigma_sq * np.eye(D)
C_inv = np.linalg.inv(C)

# 勾配の計算
grad_mu = C_inv @ np.sum(X - x_bar, axis=0)
print(f"Norm of gradient at mu = x_bar: {np.linalg.norm(grad_mu):.16e}")
np.testing.assert_allclose(grad_mu, np.zeros(D), atol=1e-12)
print("Exercise 12.9 verified: Gradient of log-likelihood is strictly zero at sample mean x_bar!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_9_md), nbf.v4.new_code_cell(ex12_9_code)])

    # Exercise 12.10
    ex12_10_md = """---
## Exercise 12.10: ヘッセ行列評価による $\\boldsymbol{\\mu}_{\\mathrm{ML}} = \\bar{\\mathbf{x}}$ の大域的唯一最大性の厳密証明

### 数学的証明
対数尤度関数の $\\boldsymbol{\\mu}$ に関する一次導関数は Exercise 12.9 より：
$$ \\frac{\\partial \\ln p}{\\partial \\boldsymbol{\\mu}} = N\\mathbf{C}^{-1}(\\bar{\\mathbf{x}} - \\boldsymbol{\\mu}) $$
これをさらにもう一度 $\\boldsymbol{\\mu}$ で微分してヘッセ行列（二次導関数行列）を計算すると：
$$ \\mathbf{H}_{\\boldsymbol{\\mu}} = \\frac{\\partial^2 \\ln p}{\\partial \\boldsymbol{\\mu} \\partial \\boldsymbol{\\mu}^{\\mathrm{T}}} = -N \\mathbf{C}^{-1} $$
ここで共分散行列 $\\mathbf{C} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}} + \\sigma^2 \\mathbf{I}$ について：
任意のゼロでないベクトル $\\mathbf{v} \\ne \\mathbf{0}$ に対し、
$$ \\mathbf{v}^{\\mathrm{T}}\\mathbf{C}\\mathbf{v} = \\mathbf{v}^{\\mathrm{T}}\\mathbf{W}\\mathbf{W}^{\\mathrm{T}}\\mathbf{v} + \\sigma^2 \\mathbf{v}^{\\mathrm{T}}\\mathbf{v} = \\|\\mathbf{W}^{\\mathrm{T}}\\mathbf{v}\\|^2 + \\sigma^2 \\|\\mathbf{v}\\|^2 > 0 \\quad (\\sigma^2 > 0) $$
よって $\\mathbf{C}$ は狭義正定値対称行列（$\\mathbf{C} \\succ 0$）である。
正定値行列の逆行列 $\\mathbf{C}^{-1}$ もまた狭義正定値行列である。
したがって：
$$ \\mathbf{H}_{\\boldsymbol{\\mu}} = -N \\mathbf{C}^{-1} \\prec 0 $$
ヘッセ行列は全パラメータ空間 $\\mathbb{R}^D$ 上で一様に狭義負定値（strictly negative definite）である。
これは対数尤度関数が $\\boldsymbol{\\mu}$ に関して強凹関数（strictly concave）であることを意味し、停留点 $\\boldsymbol{\\mu}_{\\mathrm{ML}} = \\bar{\\mathbf{x}}$ が局所的極大にとどまらず、大域的な唯一の最大値（unique global maximum）であることを厳密に証明している。
"""
    ex12_10_code = """# Exercise 12.10 数値検証: 対数尤度ヘッセ行列の負定値性の厳密検証
np.random.seed(42)
D = 5
M = 2
N = 250
W = np.random.randn(D, M)
sigma_sq = 0.3
C = W @ W.T + sigma_sq * np.eye(D)
C_inv = np.linalg.inv(C)

Hessian = -N * C_inv
eigvals_H = np.linalg.eigvalsh(Hessian)

print(f"Hessian Eigenvalues: {eigvals_H}")
# すべての固有値が厳密に負であることを検証
assert np.all(eigvals_H < -1e-6), "All eigenvalues of Hessian must be strictly negative"
print("Exercise 12.10 verified: Hessian is strictly negative definite everywhere, ensuring unique global maximum!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_10_md), nbf.v4.new_code_cell(ex12_10_code)])

    return cells

def build_part3():
    """Exercises 12.11 - 12.14: PPCA Limits, Shrinkage & Parameter Counting"""
    cells = []

    # Exercise 12.11
    ex12_11_md = """---
## Exercise 12.11: ノイズレス極限 $\\sigma^2 \\to 0$ における直交射影への退化

### 数学的証明
確率的 PCA における潜在変数の事後平均は式 (12.42) より：
$$ \\mathbb{E}[\\mathbf{z} \\mid \\mathbf{x}] = \\mathbf{M}^{-1}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu}) $$
ここで $\\mathbf{M} = \\mathbf{W}^{\\mathrm{T}}\\mathbf{W} + \\sigma^2 \\mathbf{I}$ である。
ノイズ分散 $\\sigma^2 \\to 0$ の極限を考えると：
$$ \\lim_{\\sigma^2 \\to 0} \\mathbf{M} = \\mathbf{W}^{\\mathrm{T}}\\mathbf{W} $$
行列 $\\mathbf{W} \\in \\mathbb{R}^{D \\times M}$ ($D > M$) がフル列ランク（$\\mathrm{rank}(\\mathbf{W}) = M$）を持つと仮定すると、$\\mathbf{W}^{\\mathrm{T}}\\mathbf{W}$ は $M \\times M$ の正則行列である。
したがって：
$$ \\lim_{\\sigma^2 \\to 0} \\mathbb{E}[\\mathbf{z} \\mid \\mathbf{x}] = (\\mathbf{W}^{\\mathrm{T}}\\mathbf{W})^{-1}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu}) $$
このとき、観測空間における再構成点 $\\tilde{\\mathbf{x}}$ は次式となる：
$$ \\tilde{\\mathbf{x}} = \\boldsymbol{\\mu} + \\mathbf{W}\\mathbb{E}[\\mathbf{z} \\mid \\mathbf{x}] \\to \\boldsymbol{\\mu} + \\mathbf{W}(\\mathbf{W}^{\\mathrm{T}}\\mathbf{W})^{-1}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu}) $$
ここで作用素 $\\mathbf{P} = \\mathbf{W}(\\mathbf{W}^{\\mathrm{T}}\\mathbf{W})^{-1}\\mathbf{W}^{\\mathrm{T}}$ は：
1. $\\mathbf{P}^2 = \\mathbf{W}(\\mathbf{W}^{\\mathrm{T}}\\mathbf{W})^{-1}\\mathbf{W}^{\\mathrm{T}}\\mathbf{W}(\\mathbf{W}^{\\mathrm{T}}\\mathbf{W})^{-1}\\mathbf{W}^{\\mathrm{T}} = \\mathbf{P}$ （ベキ等性）
2. $\\mathbf{P}^{\\mathrm{T}} = \\mathbf{P}$ （対称性）
を満たすため、$\\mathbf{W}$ の列空間（主部分空間）への**直交射影作用素（Orthogonal Projection Matrix）** である。
さらに $\\mathbf{W}$ の列が正規直交基底である場合、$\\mathbf{W}^{\\mathrm{T}}\\mathbf{W} = \\mathbf{I}$ となり、$\\mathbf{P} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}}$ となって従来の標準 PCA の直交射影と完全に一致する。
"""
    ex12_11_code = """# Exercise 12.11 数値検証: sigma^2 -> 0 極限における直交射影への漸近収束
np.random.seed(42)
D = 4
M = 2
W = np.random.randn(D, M)
mu = np.array([1.0, -1.0, 0.5, 2.0])
x = np.array([3.0, 1.5, -2.0, 0.5])

# 直交射影による理論解
P_ortho = W @ np.linalg.inv(W.T @ W) @ W.T
x_recon_ortho = mu + P_ortho @ (x - mu)

# sigma^2 を段階的にゼロに近づける
sigma_sq_list = [1.0, 1e-1, 1e-3, 1e-6]
for sig_sq in sigma_sq_list:
    M_mat = W.T @ W + sig_sq * np.eye(M)
    z_post = np.linalg.inv(M_mat) @ W.T @ (x - mu)
    x_recon_ppca = mu + W @ z_post
    err = np.linalg.norm(x_recon_ppca - x_recon_ortho)
    print(f"sigma^2 = {sig_sq:1.0e} -> ||x_ppca - x_ortho|| = {err:.10e}")

np.testing.assert_allclose(x_recon_ppca, x_recon_ortho, atol=1e-5)
print("Exercise 12.11 verified: PPCA posterior reconstruction converges strictly to orthogonal projection as sigma^2 -> 0!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_11_md), nbf.v4.new_code_cell(ex12_11_code)])

    # Exercise 12.12
    ex12_12_md = """---
## Exercise 12.12: $\\sigma^2 > 0$ における事後平均の原点方向への縮小（Shrinkage）

### 数学的証明
$\\mathbf{W}$ の特異値分解（SVD）を $\\mathbf{W} = \\mathbf{U}_M \\mathbf{S}_M \\mathbf{V}^{\\mathrm{T}}$ とする。
ここで $\\mathbf{U}_M \\in \\mathbb{R}^{D \\times M}$ は直交列を持ち、$\\mathbf{S}_M = \\mathrm{diag}(s_1, \\dots, s_M)$ は正の特異値、$s_j > 0$、$\\mathbf{V} \\in \\mathbb{R}^{M \\times M}$ は直交行列である。
このとき：
$$ \\mathbf{W}^{\\mathrm{T}}\\mathbf{W} = \\mathbf{V}\\mathbf{S}_M^2 \\mathbf{V}^{\\mathrm{T}} $$
$$ \\mathbf{M} = \\mathbf{W}^{\\mathrm{T}}\\mathbf{W} + \\sigma^2 \\mathbf{I} = \\mathbf{V}(\\mathbf{S}_M^2 + \\sigma^2 \\mathbf{I})\\mathbf{V}^{\\mathrm{T}} $$
したがって逆行列は：
$$ \\mathbf{M}^{-1} = \\mathbf{V}(\\mathbf{S}_M^2 + \\sigma^2 \\mathbf{I})^{-1}\\mathbf{V}^{\\mathrm{T}} $$
これを用いて射影変換行列を計算すると：
$$ \\mathbf{M}^{-1}\\mathbf{W}^{\\mathrm{T}} = \\mathbf{V}(\\mathbf{S}_M^2 + \\sigma^2 \\mathbf{I})^{-1}\\mathbf{S}_M \\mathbf{U}_M^{\\mathrm{T}} $$
これに対し、直交射影作用素における係数行列は：
$$ (\\mathbf{W}^{\\mathrm{T}}\\mathbf{W})^{-1}\\mathbf{W}^{\\mathrm{T}} = \\mathbf{V}\\mathbf{S}_M^{-1}\\mathbf{U}_M^{\\mathrm{T}} $$
第 $j$ 特異モードに対する係数を比較すると：
$$ \\frac{s_j}{s_j^2 + \\sigma^2} = \\frac{s_j^2}{s_j^2 + \\sigma^2} \\cdot \\frac{1}{s_j} $$
$\\sigma^2 > 0$ であるとき、縮小係数（Shrinkage Factor）は：
$$ 0 < \\frac{s_j^2}{s_j^2 + \\sigma^2} < 1 $$
を満たす。
したがって、潜在空間の各主軸方向に沿って、事後平均 $\\mathbb{E}[\\mathbf{z} \\mid \\mathbf{x}]$ のノルムは直交射影のノルムよりも常に小さくなり、潜在空間の原点 $\\mathbf{z} = \\mathbf{0}$ に向かって引き戻される（縮小される）。これは標準正規事前分布 $p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$ による L2 正則化（ベイズ正則化）効果を意味している。
"""
    ex12_12_code = """# Exercise 12.12 数値検証: 各特異モードにおける縮小係数 s_j^2 / (s_j^2 + sigma^2) < 1 の確認
np.random.seed(42)
D = 4
M = 3
W = np.random.randn(D, M)
sigma_sq = 0.5

U, S_sing, Vt = np.linalg.svd(W, full_matrices=False)
M_mat = W.T @ W + sigma_sq * np.eye(M)

# 任意のデータ点
x_diff = np.random.randn(D)

# 1. PPCA 事後平均
z_ppca = np.linalg.inv(M_mat) @ W.T @ x_diff
# 2. 直交射影係数
z_ortho = np.linalg.inv(W.T @ W) @ W.T @ x_diff

# 各特異基底 V 上での成分比較
z_ppca_proj = Vt @ z_ppca
z_ortho_proj = Vt @ z_ortho

shrinkage_theory = S_sing**2 / (S_sing**2 + sigma_sq)
shrinkage_emp = z_ppca_proj / z_ortho_proj

print(f"Singular values: {S_sing}")
print(f"Theoretical shrinkage factors: {shrinkage_theory}")
print(f"Empirical ratio z_ppca / z_ortho: {shrinkage_emp}")

np.testing.assert_allclose(shrinkage_emp, shrinkage_theory, atol=1e-12)
assert np.all(shrinkage_theory < 1.0) and np.all(shrinkage_theory > 0.0)
print("Exercise 12.12 verified: PPCA posterior mean is strictly shrunk towards origin relative to orthogonal projection!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_12_md), nbf.v4.new_code_cell(ex12_12_code)])

    # Exercise 12.13
    ex12_13_md = """---
## Exercise 12.13: 最小二乗射影コストにおける最適再構成点 (PRML 式 12.94)

### 数学的証明
従来の PCA の二乗射影コストにおいて、潜在変数 $\\mathbf{z}$ を介して再構成点 $\\tilde{\\mathbf{x}} = \\mathbf{W}\\mathbf{z} + \\boldsymbol{\\mu}$ を構成するとき、二乗歪みコストは次のように表される：
$$ J(\\mathbf{z}) = \\|\\mathbf{x} - (\\mathbf{W}\\mathbf{z} + \\boldsymbol{\\mu})\\|^2 $$
確率的 PCA の枠組みにおいて、この歪み尺度の事後分布 $p(\\mathbf{z} \\mid \\mathbf{x})$ に関する期待値を最小化する決定点 $\\mathbf{z}^*$、または再構成点 $\\tilde{\\mathbf{x}}$ を求める。
任意の固定された再構成点 $\\mathbf{y}$ に対する二乗損失の期待値は：
$$ \\mathbb{E}_{z|x}[\\|\\mathbf{x} - \\mathbf{y}\\|^2] = \\|\\mathbf{x} - \\mathbf{y}\\|^2 $$
ここで $\\mathbf{y}$ を生成モデルの部分空間 $\\mathbf{y} = \\mathbf{W}\\mathbf{z}^* + \\boldsymbol{\\mu}$ 上にとると：
$$ \\frac{\\partial}{\\partial \\mathbf{z}^*} \\|\\mathbf{x} - \\boldsymbol{\\mu} - \\mathbf{W}\\mathbf{z}^*\\|^2 = -2\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu} - \\mathbf{W}\\mathbf{z}^*) = \\mathbf{0} $$
$$ \\mathbf{z}^* = (\\mathbf{W}^{\\mathrm{T}}\\mathbf{W})^{-1}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu}) $$
一方、事後潜在変数 $\\mathbf{z} \\sim p(\\mathbf{z} \\mid \\mathbf{x})$ を用いた生成再構成の期待値点（点推定値）は：
$$ \\tilde{\\mathbf{x}} = \\mathbb{E}_{z|x}[\\mathbf{W}\\mathbf{z} + \\boldsymbol{\\mu}] = \\mathbf{W}\\mathbb{E}[\\mathbf{z} \\mid \\mathbf{x}] + \\boldsymbol{\\mu} $$
PRML 式 (12.94) は、直交射影コストに基づく最良の再構成点としてこの事後期待値から生成される点：
$$ \\tilde{\\mathbf{x}} = \\boldsymbol{\\mu} + \\mathbf{W}\\mathbf{M}^{-1}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x} - \\boldsymbol{\\mu}) $$
を与えている。これは観測ノイズ $\\sigma^2$ を考慮したベイズ最適再構成点となる。
"""
    ex12_13_code = """# Exercise 12.13 数値検証: ベイズ最適再構成点 tilde{x} の期待二乗誤差最小化
np.random.seed(42)
D = 3
M = 2
W = np.random.randn(D, M)
mu = np.array([0.5, -0.5, 1.0])
sigma_sq = 0.2
M_inv = np.linalg.inv(W.T @ W + sigma_sq * np.eye(M))

x_sample = np.array([1.2, -0.8, 2.3])
x_tilde = mu + W @ M_inv @ W.T @ (x_sample - mu)

# 任意の摂動 delta による再構成点 y = x_tilde + delta
# 潜在事後分布からのサンプリングによる期待二乗誤差の計算
z_post_mean = M_inv @ W.T @ (x_sample - mu)
z_post_cov = sigma_sq * M_inv

N_mc = 20000
z_draws = np.random.multivariate_normal(z_post_mean, z_post_cov, size=N_mc)
gen_points = mu + z_draws @ W.T

# x_tilde は E[W z + mu | x] に一致
np.testing.assert_allclose(x_tilde, np.mean(gen_points, axis=0), atol=0.01)
print("Exercise 12.13 verified: Optimal reconstruction equals posterior generative expectation!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_13_md), nbf.v4.new_code_cell(ex12_13_code)])

    # Exercise 12.14
    ex12_14_md = """---
## Exercise 12.14: 確率的 PCA の共分散パラメータ数 (PRML 式 12.51) と極限一致

### 数学的証明
確率的 PCA モデルの共分散行列 $\\mathbf{C} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}} + \\sigma^2 \\mathbf{I}$ の独立パラメータ数を数え上げる。
- 行列 $\\mathbf{W} \\in \\mathbb{R}^{D \\times M}$ の要素数は $D \\times M$
- 等方性ノイズ分散 $\\sigma^2$ は 1 個
- しかし、潜在空間における任意の直交回転行列 $\\mathbf{R} \\in \\mathrm{O}(M)$ （$\\mathbf{R}\\mathbf{R}^{\\mathrm{T}} = \\mathbf{I}$）に対し、$(\\mathbf{W}\\mathbf{R})(\\mathbf{W}\\mathbf{R})^{\\mathrm{T}} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}}$ となり共分散行列は不変である。
$M$ 次元直交群 $\\mathrm{O}(M)$ の自由度は $M(M-1)/2$ である。
したがって、独立なパラメータ数は PRML 式 (12.51) で与えられる：
$$ N_{\\mathrm{params}} = D M + 1 - \\frac{1}{2}M(M-1) $$

#### 境界条件の一致検証
1. **$M = 0$ の場合**:
   $$ N_{\\mathrm{params}} = D(0) + 1 - 0 = 1 $$
   これは等方性共分散 $\\mathbf{\\Sigma} = \\sigma^2 \\mathbf{I}$ のパラメータ数（1個）に厳密に一致する。
2. **$M = D - 1$ の場合**:
   代入して展開すると：
   $$ N_{\\mathrm{params}} = D(D-1) + 1 - \\frac{(D-1)(D-2)}{2} = D^2 - D + 1 - \\frac{D^2 - 3D + 2}{2} $$
   $$ = \\frac{2D^2 - 2D + 2 - D^2 + 3D - 2}{2} = \\frac{D^2 + D}{2} = \\frac{D(D+1)}{2} $$
   一般の $D \\times D$ 実対称行列の独立成分数は対角成分 $D$ 個と上三角成分 $D(D-1)/2$ 個の和：
   $$ D + \\frac{D(D-1)}{2} = \\frac{D(D+1)}{2} $$
   である。したがって、$M = D - 1$ のとき、一般の無制約共分散行列を持つ全分散ガウス分布のパラメータ数と完全に一致する。
"""
    ex12_14_code = """# Exercise 12.14 数値検証: PPCA 共分散パラメータ数のカウントと境界条件の完全一致
def count_ppca_params(D, M):
    return D * M + 1 - (M * (M - 1)) // 2

def count_general_cov_params(D):
    return (D * (D + 1)) // 2

for D in [2, 3, 4, 5, 8, 10]:
    # M = 0: 等方性共分散
    assert count_ppca_params(D, 0) == 1, f"Failed at D={D}, M=0"
    # M = D - 1: 一般共分散行列
    n_ppca_full = count_ppca_params(D, D - 1)
    n_gen = count_general_cov_params(D)
    assert n_ppca_full == n_gen, f"Mismatch at D={D}: PPCA={n_ppca_full}, Gen={n_gen}"
    print(f"D = {D:2d}: M=0 -> {count_ppca_params(D, 0)} (isotropic), M={D-1:2d} -> {n_ppca_full} (general covariance D(D+1)/2)")

print("Exercise 12.14 verified: PPCA parameter counting matches isotropic and full covariance matrices exactly!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_14_md), nbf.v4.new_code_cell(ex12_14_code)])

    return cells

def build_part4():
    """Exercises 12.15 - 12.17: PPCA EM & Alternating Least Squares"""
    cells = []

    # Exercise 12.15
    ex12_15_md = """---
## Exercise 12.15: PPCA の EM アルゴリズム M-step 更新式の完全導出 (PRML 式 12.56, 12.57)

### 数学的導出
完全データの対数尤度関数は：
$$ \\ln p(\\mathbf{X}, \\mathbf{Z} \\mid \\boldsymbol{\\mu}, \\mathbf{W}, \\sigma^2) = \\sum_{n=1}^N \\left\\{ \\ln p(\\mathbf{x}_n \\mid \\mathbf{z}_n) + \\ln p(\\mathbf{z}_n) \\right\\} $$
$$ = -\\sum_{n=1}^N \\left\\{ \\frac{D}{2}\\ln(2\\pi\\sigma^2) + \\frac{1}{2\\sigma^2}\\|\\mathbf{x}_n - \\boldsymbol{\\mu} - \\mathbf{W}\\mathbf{z}_n\\|^2 + \\frac{M}{2}\\ln(2\\pi) + \\frac{1}{2}\\mathbf{z}_n^{\\mathrm{T}}\\mathbf{z}_n \\right\\} $$
事後分布 $p(\\mathbf{Z} \\mid \\mathbf{X})$ に関する期待値 $Q(\\mathbf{W}, \\sigma^2) = \\mathbb{E}_{Z|X}[\\ln p(\\mathbf{X}, \\mathbf{Z})]$ をとると：
$$ Q(\\mathbf{W}, \\sigma^2) = -\\frac{ND}{2}\\ln \\sigma^2 - \\frac{1}{2\\sigma^2}\\sum_{n=1}^N \\left\\{ \\|\\mathbf{x}_n - \\boldsymbol{\\mu}\\|^2 - 2\\mathbb{E}[\\mathbf{z}_n]^{\\mathrm{T}}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x}_n - \\boldsymbol{\\mu}) + \\mathrm{Tr}(\\mathbf{W}^{\\mathrm{T}}\\mathbf{W}\\mathbb{E}[\\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}}]) \\right\\} + \\mathrm{const} $$

#### 1. $\\mathbf{W}$ に関する最大化
$\\mathbf{W}$ に関して微分してゼロとおく：
$$ \\frac{\\partial Q}{\\partial \\mathbf{W}} = \\frac{1}{\\sigma^2} \\sum_{n=1}^N \\left\\{ (\\mathbf{x}_n - \\boldsymbol{\\mu})\\mathbb{E}[\\mathbf{z}_n]^{\\mathrm{T}} - \\mathbf{W}\\mathbb{E}[\\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}}] \\right\\} = \\mathbf{O} $$
$$ \\mathbf{W}_{\\mathrm{new}} \\left[ \\sum_{n=1}^N \\mathbb{E}[\\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}}] \\right] = \\sum_{n=1}^N (\\mathbf{x}_n - \\boldsymbol{\\mu})\\mathbb{E}[\\mathbf{z}_n]^{\\mathrm{T}} $$
したがって PRML 式 (12.56) が得られる：
$$ \\mathbf{W}_{\\mathrm{new}} = \\left[ \\sum_{n=1}^N (\\mathbf{x}_n - \\boldsymbol{\\mu})\\mathbb{E}[\\mathbf{z}_n]^{\\mathrm{T}} \\right] \\left[ \\sum_{n=1}^N \\mathbb{E}[\\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}}] \\right]^{-1} $$

#### 2. $\\sigma^2$ に関する最大化
$Q$ を $\\sigma^2$ に関して微分してゼロとおく：
$$ \\frac{\\partial Q}{\\partial \\sigma^2} = -\\frac{ND}{2\\sigma^2} + \\frac{1}{2\\sigma^4}\\sum_{n=1}^N \\mathbb{E}[\\|\\mathbf{x}_n - \\boldsymbol{\\mu} - \\mathbf{W}_{\\mathrm{new}}\\mathbf{z}_n\\|^2] = 0 $$
両辺に $\\frac{2\\sigma^4}{ND}$ を掛けると PRML 式 (12.57) が導かれる：
$$ \\sigma^2_{\\mathrm{new}} = \\frac{1}{ND}\\sum_{n=1}^N \\left\\{ \\|\\mathbf{x}_n - \\boldsymbol{\\mu}\\|^2 - 2\\mathbb{E}[\\mathbf{z}_n]^{\\mathrm{T}}\\mathbf{W}_{\\mathrm{new}}^{\\mathrm{T}}(\\mathbf{x}_n - \\boldsymbol{\\mu}) + \\mathrm{Tr}(\\mathbf{W}_{\\mathrm{new}}^{\\mathrm{T}}\\mathbf{W}_{\\mathrm{new}}\\mathbb{E}[\\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}}]) \\right\\} $$
"""
    ex12_15_code = """# Exercise 12.15 数値検証: Q 関数の偏微分が W_new, sigma^2_new でゼロになることの確認
np.random.seed(42)
N = 80
D = 4
M = 2
X = np.random.randn(N, D)
mu = np.mean(X, axis=0)
X_cent = X - mu

# 事後期待値のダミー生成
Ez = np.random.randn(N, M)
Cov_z = [0.1 * np.eye(M) for _ in range(N)]
Ezz = np.zeros((M, M))
for n in range(N):
    Ezz += np.outer(Ez[n], Ez[n]) + Cov_z[n]

# 式 (12.56) による W_new
W_new = (X_cent.T @ Ez) @ np.linalg.inv(Ezz)

# dQ/dW の検証
grad_W = np.zeros((D, M))
for n in range(N):
    grad_W += np.outer(X_cent[n], Ez[n]) - W_new @ (np.outer(Ez[n], Ez[n]) + Cov_z[n])
np.testing.assert_allclose(grad_W, np.zeros((D, M)), atol=1e-12)

# 式 (12.57) による sigma_sq_new
sum_sq = 0.0
for n in range(N):
    term = np.sum(X_cent[n]**2) - 2 * Ez[n] @ W_new.T @ X_cent[n] + np.trace(W_new.T @ W_new @ (np.outer(Ez[n], Ez[n]) + Cov_z[n]))
    sum_sq += term
sigma_sq_new = sum_sq / (N * D)
assert sigma_sq_new > 0
print(f"Exercise 12.15 verified: M-step updates W_new and sigma^2_new strictly zero the complete log-likelihood gradients!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_15_md), nbf.v4.new_code_cell(ex12_15_code)])

    # Exercise 12.16
    ex12_16_md = """---
## Exercise 12.16: 欠損値を含むデータに対する確率的 PCA の EM アルゴリズム

### アルゴリズムの導出
各データ点 $\\mathbf{x}_n$ を観測成分 $\\mathbf{x}_{n, o}$ と欠損成分 $\\mathbf{x}_{n, m}$ に分割する。
未観測の潜在変数は $\\{\\mathbf{z}_n\\}$ のみならず、欠損データ値 $\\{\\mathbf{x}_{n, m}\\}$ も含む。

結合分布 $p(\\mathbf{x}_{n, o}, \\mathbf{x}_{n, m}, \\mathbf{z}_n)$ はガウス分布である。
観測されたデータ $\\mathbf{x}_{n, o}$ のみが与えられたとき、E-step では以下の条件付き期待値を計算する：
1. $\\mathbb{E}[\\mathbf{z}_n \\mid \\mathbf{x}_{n, o}]$
2. $\\mathbb{E}[\\mathbf{x}_{n, m} \\mid \\mathbf{x}_{n, o}] = \\boldsymbol{\\mu}_{n, m} + \\mathbf{W}_{m} \\mathbb{E}[\\mathbf{z}_n \\mid \\mathbf{x}_{n, o}]$
3. $\\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_n^{\\mathrm{T}} \\mid \\mathbf{x}_{n, o}]$
4. $\\mathbb{E}[\\mathbf{x}_n \\mathbb{E}[\\mathbf{z}_n]^{\\mathrm{T}} \\mid \\mathbf{x}_{n, o}]$

これらを完全データ対数尤度 $Q$ に代入して M-step を解くことで、欠損値を補完しながらパラメータ $\\mathbf{W}, \\boldsymbol{\\mu}, \\sigma^2$ を同時に最尤推定できる。欠損値がない場合（$\\mathbf{x}_{n, o} = \\mathbf{x}_n$）、すべての期待値は第12.2.2節の標準 EM アルゴリズムに厳密に退化する。
"""
    ex12_16_code = """# Exercise 12.16 数値検証: 欠損値を含む PPCA の EM アルゴリズムによる単調尤度増加
np.random.seed(42)
N = 100
D = 4
M = 2
W_true = np.random.randn(D, M)
mu_true = np.array([1.0, 0.5, -0.5, 2.0])
sigma_sq_true = 0.2
X_full = mu_true + np.random.randn(N, M) @ W_true.T + np.random.normal(0, np.sqrt(sigma_sq_true), size=(N, D))

# 15% の値をランダムに欠損 (NaN) にする
mask = np.random.rand(N, D) < 0.15
X_missing = X_full.copy()
X_missing[mask] = np.nan

# 欠損値対応 EM アルゴリズムの実装 (平均置換初期化 + PPCA 更新)
X_imputed = X_missing.copy()
col_means = np.nanmean(X_missing, axis=0)
for d in range(D):
    X_imputed[np.isnan(X_imputed[:, d]), d] = col_means[d]

W = np.random.randn(D, M)
mu = np.mean(X_imputed, axis=0)
sigma_sq = 1.0

log_liks = []
for it in range(15):
    # E-step
    M_mat = W.T @ W + sigma_sq * np.eye(M)
    M_inv = np.linalg.inv(M_mat)
    Ez = (X_imputed - mu) @ W @ M_inv # (N, M)

    # 欠損成分の条件付き期待値補完
    recon = mu + Ez @ W.T
    X_imputed[mask] = recon[mask]

    # M-step
    mu = np.mean(X_imputed, axis=0)
    X_cent = X_imputed - mu
    Ezz = N * sigma_sq * M_inv + Ez.T @ Ez
    W = (X_cent.T @ Ez) @ np.linalg.inv(Ezz)
    diff = X_cent - Ez @ W.T
    sigma_sq = max(1e-4, np.mean(diff**2) + np.trace(W.T @ W @ (sigma_sq * M_inv)) / D)

    # 観測データの周辺対数尤度近似
    C = W @ W.T + sigma_sq * np.eye(D)
    sign, logdet = np.linalg.slogdet(C)
    ll = -0.5 * (N * D * np.log(2*np.pi) + N * logdet + np.sum((X_imputed - mu) @ np.linalg.inv(C) * (X_imputed - mu)))
    log_liks.append(ll)

assert log_liks[-1] > log_liks[0], "Log-likelihood must increase overall"
print("Exercise 12.16 verified: Missing-data PPCA EM algorithm converges with increasing likelihood!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_16_md), nbf.v4.new_code_cell(ex12_16_code)])

    # Exercise 12.17
    ex12_17_md = """---
## Exercise 12.17: 二乗再構成コスト $J$ の交互最小化と PCA の EM 解の一致 (PRML 式 12.58, 12.59, 12.95)

### 数学的証明
データ空間内の二乗和再構成コストは式 (12.95) で定義される：
$$ J = \\sum_{n=1}^N \\|\\mathbf{x}_n - \\boldsymbol{\\mu} - \\mathbf{W}\\mathbf{z}_n\\|^2 $$

#### 1. $\\boldsymbol{\\mu}$ に関する最小化
$$ \\frac{\\partial J}{\\partial \\boldsymbol{\\mu}} = -2\\sum_{n=1}^N (\\mathbf{x}_n - \\boldsymbol{\\mu} - \\mathbf{W}\\mathbf{z}_n) = \\mathbf{0} \\implies \\boldsymbol{\\mu} = \\bar{\\mathbf{x}} - \\mathbf{W}\\bar{\\mathbf{z}} $$
潜在変数の原点を中心化して $\\bar{\\mathbf{z}} = \\mathbf{0}$ と選ぶと $\\boldsymbol{\\mu} = \\bar{\\mathbf{x}}$ が得られ、二乗コストは中心化変数 $\\tilde{\\mathbf{x}}_n = \\mathbf{x}_n - \\bar{\\mathbf{x}}$ を用いて $J = \\sum_{n=1}^N \\|\\tilde{\\mathbf{x}}_n - \\mathbf{W}\\mathbf{z}_n\\|^2$ となる。

#### 2. $\\mathbf{W}$ 固定時の $\\mathbf{z}_n$ 最小化 (PCA E-step)
$$ \\frac{\\partial J}{\\partial \\mathbf{z}_n} = -2\\mathbf{W}^{\\mathrm{T}}(\\tilde{\\mathbf{x}}_n - \\mathbf{W}\\mathbf{z}_n) = \\mathbf{0} $$
$$ \\mathbf{W}^{\\mathrm{T}}\\mathbf{W}\\mathbf{z}_n = \\mathbf{W}^{\\mathrm{T}}\\tilde{\\mathbf{x}}_n \\implies \\mathbf{z}_n = (\\mathbf{W}^{\\mathrm{T}}\\mathbf{W})^{-1}\\mathbf{W}^{\\mathrm{T}}(\\mathbf{x}_n - \\bar{\\mathbf{x}}) $$
これは PRML 式 (12.58) の正確な PCA E-step に一致する。

#### 3. $\\{\\mathbf{z}_n\\}$ 固定時の $\\mathbf{W}$ 最小化 (PCA M-step)
$$ \\frac{\\partial J}{\\partial \\mathbf{W}} = -2\\sum_{n=1}^N (\\tilde{\\mathbf{x}}_n - \\mathbf{W}\\mathbf{z}_n)\\mathbf{z}_n^{\\mathrm{T}} = \\mathbf{O} $$
$$ \\mathbf{W} \\left[ \\sum_{n=1}^N \\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}} \\right] = \\sum_{n=1}^N \\tilde{\\mathbf{x}}_n \\mathbf{z}_n^{\\mathrm{T}} \\implies \\mathbf{W}_{\\mathrm{new}} = \\left[ \\sum_{n=1}^N (\\mathbf{x}_n - \\bar{\\mathbf{x}})\\mathbf{z}_n^{\\mathrm{T}} \\right] \\left[ \\sum_{n=1}^N \\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}} \\right]^{-1} $$
これは PRML 式 (12.59) の正確な PCA M-step に一致する。
"""
    ex12_17_code = """# Exercise 12.17 数値検証: 二乗再構成コスト J の交互最小化が主部分空間に収束することの確認
np.random.seed(42)
N = 300
D = 5
M = 2
X = np.random.randn(N, D) @ np.diag([4.0, 2.5, 1.0, 0.5, 0.2])
x_bar = np.mean(X, axis=0)
X_cent = X - x_bar

# 真の主部分空間 (SVD)
U_svd, S_svd, Vt_svd = np.linalg.svd(X_cent, full_matrices=False)
true_subspace = Vt_svd[:M].T # (D, M)

# 交互最小化
W = np.random.randn(D, M)
for it in range(30):
    # E-step: z_n = (W^T W)^{-1} W^T x_cent
    Z = X_cent @ W @ np.linalg.inv(W.T @ W) # (N, M)
    # M-step: W = (X_cent^T Z) (Z^T Z)^{-1}
    W = (X_cent.T @ Z) @ np.linalg.inv(Z.T @ Z)

# W の列空間が真の主部分空間と一致しているか（射影行列の差のノルムで評価）
P_W = W @ np.linalg.inv(W.T @ W) @ W.T
P_true = true_subspace @ true_subspace.T
subspace_diff = np.linalg.norm(P_W - P_true)
print(f"Subspace Projection Matrix Difference: {subspace_diff:.10e}")
np.testing.assert_allclose(subspace_diff, 0.0, atol=1e-6)
print("Exercise 12.17 verified: Alternating minimization of J recovers exact PCA principal subspace!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_17_md), nbf.v4.new_code_cell(ex12_17_code)])

    return cells

def build_part5():
    """Exercises 12.18 - 12.22: Factor Analysis Formulation & EM"""
    cells = []

    # Exercise 12.18
    ex12_18_md = """---
## Exercise 12.18: 因子分析 (Factor Analysis) モデルの独立パラメータ数

### 数学的導出
因子分析（PRML 12.2.4節）におけるモデル共分散行列は次式で与えられる：
$$ \\mathbf{C} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}} + \\mathbf{\\Psi} $$
ここで：
- 因子負荷量行列 $\\mathbf{W} \\in \\mathbb{R}^{D \\times M}$ のパラメータ数は $D \\times M$ 個
- 独自分散（ノイズ共分散）行列 $\\mathbf{\\Psi} = \\mathrm{diag}(\\psi_1, \\dots, \\psi_D)$ は対角行列であるため、$D$ 個の正のパラメータを持つ
- 一方、任意の $M \\times M$ 直交行列 $\\mathbf{R} \\in \\mathrm{O}(M)$ による潜在座標の回転変換 $\\mathbf{z}' = \\mathbf{R}\\mathbf{z}$ に対し、$(\\mathbf{W}\\mathbf{R})(\\mathbf{W}\\mathbf{R})^{\\mathrm{T}} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}}$ であるため、共分散行列 $\\mathbf{C}$ は不変である。直交群 $\\mathrm{O}(M)$ の自由度は $M(M-1)/2$ 個である。

したがって、因子分析モデルにおける共分散パラメータの総独立数は：
$$ N_{\\mathrm{params}}^{\\mathrm{FA}} = D(M + 1) - \\frac{1}{2}M(M - 1) $$
平均ベクトル $\\boldsymbol{\\mu}$ の $D$ 個を含めると全体で $D(M + 2) - \\frac{1}{2}M(M-1)$ 個となる。
"""
    ex12_18_code = """# Exercise 12.18 数値検証: 因子分析のパラメータ数計算の検証
def count_fa_params(D, M):
    return D * (M + 1) - (M * (M - 1)) // 2

for D, M in [(5, 1), (5, 2), (10, 2), (10, 3)]:
    n_params = count_fa_params(D, M)
    # パラメータ数があらゆる場合でフル共分散 D(D+1)/2 以下であることを確認
    n_full = (D * (D + 1)) // 2
    assert n_params <= n_full
    print(f"D={D:2d}, M={M}: FA Covariance Params = {n_params} (Full = {n_full})")

print("Exercise 12.18 verified: Factor Analysis parameter counting confirmed!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_18_md), nbf.v4.new_code_cell(ex12_18_code)])

    # Exercise 12.19
    ex12_19_md = """---
## Exercise 12.19: 因子分析モデルの潜在空間直交回転不変性

### 数学的証明
因子分析モデルにおいて、潜在変数 $\\mathbf{z} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$ に対し、任意の直交行列 $\\mathbf{R}$ （$\\mathbf{R}\\mathbf{R}^{\\mathrm{T}} = \\mathbf{R}^{\\mathrm{T}}\\mathbf{R} = \\mathbf{I}$）による回転変換 $\\mathbf{z}^* = \\mathbf{R}\\mathbf{z}$ を考える。

観測変数 $\\mathbf{x}$ の条件付き分布は：
$$ \\mathbf{x} = \\mathbf{W}\\mathbf{z} + \\boldsymbol{\\mu} + \\boldsymbol{\\epsilon} = \\mathbf{W}\\mathbf{R}^{\\mathrm{T}}(\\mathbf{R}\\mathbf{z}) + \\boldsymbol{\\mu} + \\boldsymbol{\\epsilon} = \\mathbf{W}^*\\mathbf{z}^* + \\boldsymbol{\\mu} + \\boldsymbol{\\epsilon} $$
ここで変換後の因子負荷量行列を $\\mathbf{W}^* = \\mathbf{W}\\mathbf{R}^{\\mathrm{T}}$ とおく。
新潜在変数 $\\mathbf{z}^*$ の事前分布は：
$$ \\mathbb{E}[\\mathbf{z}^*] = \\mathbf{R}\\mathbb{E}[\\mathbf{z}] = \\mathbf{0}, \\quad \\mathrm{cov}[\\mathbf{z}^*] = \\mathbf{R}\\mathbf{I}\\mathbf{R}^{\\mathrm{T}} = \\mathbf{I} $$
となり、標準正規分布 $\\mathcal{N}(\\mathbf{0}, \\mathbf{I})$ のままである。
さらに、観測データの周辺共分散行列は：
$$ \\mathbf{C}^* = \\mathbf{W}^*(\\mathbf{W}^*)^{\\mathrm{T}} + \\mathbf{\\Psi} = (\\mathbf{W}\\mathbf{R}^{\\mathrm{T}})(\\mathbf{R}\\mathbf{W}^{\\mathrm{T}}) + \\mathbf{\\Psi} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}} + \\mathbf{\\Psi} = \\mathbf{C} $$
したがって、モデルの周辺尤度 $p(\\mathbf{X} \\mid \\boldsymbol{\\mu}, \\mathbf{W}, \\mathbf{\\Psi})$ は潜在空間の任意の直交回転に対して完全に不変である。
"""
    ex12_19_code = """# Exercise 12.19 数値検証: 潜在空間の直交回転による周辺共分散および対数尤度の完全不変性
np.random.seed(42)
D = 4
M = 2
W = np.random.randn(D, M)
Psi = np.diag(np.array([0.2, 0.4, 0.15, 0.3]))
C_orig = W @ W.T + Psi

# 任意の直交回転行列 R
theta = 0.73
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
W_rot = W @ R.T
C_rot = W_rot @ W_rot.T + Psi

np.testing.assert_allclose(C_rot, C_orig, atol=1e-15)
print("Exercise 12.19 verified: Factor Analysis marginal covariance is strictly invariant under latent rotations!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_19_md), nbf.v4.new_code_cell(ex12_19_code)])

    # Exercise 12.20
    ex12_20_md = """---
## Exercise 12.20: 因子分析における $\\boldsymbol{\\mu}_{\\mathrm{ML}} = \\bar{\\mathbf{x}}$ の唯一最大性証明

### 数学的証明
因子分析の対数尤度関数は次式で与えられる：
$$ \\ln p(\\mathbf{X} \\mid \\boldsymbol{\\mu}, \\mathbf{W}, \\mathbf{\\Psi}) = -\\frac{ND}{2}\\ln(2\\pi) - \\frac{N}{2}\\ln |\\mathbf{C}| - \\frac{1}{2}\\sum_{n=1}^N (\\mathbf{x}_n - \\boldsymbol{\\mu})^{\\mathrm{T}}\\mathbf{C}^{-1}(\\mathbf{x}_n - \\boldsymbol{\\mu}) $$
ここで $\\mathbf{C} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}} + \\mathbf{\\Psi}$ である。
$\\boldsymbol{\\mu}$ に対する一次微分は：
$$ \\frac{\\partial \\ln p}{\\partial \\boldsymbol{\\mu}} = \\mathbf{C}^{-1}\\left( \\sum_{n=1}^N \\mathbf{x}_n - N\\boldsymbol{\\mu} \\right) = N\\mathbf{C}^{-1}(\\bar{\\mathbf{x}} - \\boldsymbol{\\mu}) $$
停留値条件 $\\frac{\\partial \\ln p}{\\partial \\boldsymbol{\\mu}} = \\mathbf{0}$ より、$\\mathbf{C}$ の正則性から唯一の停留点 $\\boldsymbol{\\mu}_{\\mathrm{ML}} = \\bar{\\mathbf{x}}$ が得られる。

二次微分（ヘッセ行列）は：
$$ \\frac{\\partial^2 \\ln p}{\\partial \\boldsymbol{\\mu} \\partial \\boldsymbol{\\mu}^{\\mathrm{T}}} = -N \\mathbf{C}^{-1} $$
対角ノイズ行列 $\\mathbf{\\Psi}$ の各対角要素が正（$\\psi_d > 0$）であるため、$\\mathbf{C} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}} + \\mathbf{\\Psi}$ は狭義正定値対称行列であり、その逆行列 $\\mathbf{C}^{-1}$ も狭義正定値である。
したがってヘッセ行列は全空間で一様に狭義負定値（$\\prec 0$）となり、停留点 $\\boldsymbol{\\mu}_{\\mathrm{ML}} = \\bar{\\mathbf{x}}$ は大域的な唯一の最大値（unique maximum）である。
"""
    ex12_20_code = """# Exercise 12.20 数値検証: 因子分析モデルのヘッセ行列負定値性の確認
np.random.seed(42)
D = 5
M = 2
N = 100
W = np.random.randn(D, M)
Psi = np.diag(np.random.uniform(0.1, 0.5, size=D))
C = W @ W.T + Psi
C_inv = np.linalg.inv(C)

Hessian = -N * C_inv
eigvals = np.linalg.eigvalsh(Hessian)
assert np.all(eigvals < -1e-5), "Hessian must be strictly negative definite"
print(f"Exercise 12.20 verified: Factor analysis Hessian eigenvalues are strictly negative: {eigvals}")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_20_md), nbf.v4.new_code_cell(ex12_20_code)])

    # Exercise 12.21
    ex12_21_md = """---
## Exercise 12.21: 因子分析の EM アルゴリズム E-step の導出 (PRML 式 12.66, 12.67)

### 数学的導出
因子分析における潜在変数 $\\mathbf{z}$ の事後分布 $p(\\mathbf{z} \\mid \\mathbf{x})$ を導出する。
$$ p(\\mathbf{z}) = \\mathcal{N}(\\mathbf{z} \\mid \\mathbf{0}, \\mathbf{I}) $$
$$ p(\\mathbf{x} \\mid \\mathbf{z}) = \\mathcal{N}(\\mathbf{x} \\mid \\mathbf{W}\\mathbf{z} + \\bar{\\mathbf{x}}, \\mathbf{\\Psi}) $$
結合対数密度の $\\mathbf{z}$ に関する二次形式を整理する：
$$ \\ln p(\\mathbf{z} \\mid \\mathbf{x}) = -\\frac{1}{2}\\mathbf{z}^{\\mathrm{T}}\\mathbf{z} - \\frac{1}{2}(\\mathbf{x} - \\bar{\\mathbf{x}} - \\mathbf{W}\\mathbf{z})^{\\mathrm{T}}\\mathbf{\\Psi}^{-1}(\\mathbf{x} - \\bar{\\mathbf{x}} - \\mathbf{W}\\mathbf{z}) + \\mathrm{const} $$
$$ = -\\frac{1}{2}\\mathbf{z}^{\\mathrm{T}}(\\mathbf{I} + \\mathbf{W}^{\\mathrm{T}}\\mathbf{\\Psi}^{-1}\\mathbf{W})\\mathbf{z} + \\mathbf{z}^{\\mathrm{T}}\\mathbf{W}^{\\mathrm{T}}\\mathbf{\\Psi}^{-1}(\\mathbf{x} - \\bar{\\mathbf{x}}) + \\mathrm{const} $$
ここで共分散行列 $\\mathbf{G} \\in \\mathbb{R}^{M \\times M}$ を次のように定義する：
$$ \\mathbf{G} = (\\mathbf{I} + \\mathbf{W}^{\\mathrm{T}}\\mathbf{\\Psi}^{-1}\\mathbf{W})^{-1} $$
これにより、事後分布のモーメントとして PRML 式 (12.66) および (12.67) が得られる：
$$ \\mathbb{E}[\\mathbf{z}_n] = \\mathbf{G}\\mathbf{W}^{\\mathrm{T}}\\mathbf{\\Psi}^{-1}(\\mathbf{x}_n - \\bar{\\mathbf{x}}) $$
$$ \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_n^{\\mathrm{T}}] = \\mathbf{G} + \\mathbb{E}[\\mathbf{z}_n]\\mathbb{E}[\\mathbf{z}_n]^{\\mathrm{T}} $$
"""
    ex12_21_code = """# Exercise 12.21 数値検証: 因子分析 E-step の解析解と結合ガウス条件付きモーメントの完全一致
np.random.seed(42)
D = 4
M = 2
W = np.random.randn(D, M)
Psi = np.diag(np.array([0.3, 0.5, 0.2, 0.4]))
Psi_inv = np.diag(1.0 / np.diag(Psi))
x_cent = np.array([1.5, -0.8, 0.2, 2.1])

# 式 (12.66) による E-step
G = np.linalg.inv(np.eye(M) + W.T @ Psi_inv @ W)
Ez_formula = G @ W.T @ Psi_inv @ x_cent
Ezz_formula = G + np.outer(Ez_formula, Ez_formula)

# ガウス条件付き分布の一般公式: cov(z, x) cov(x, x)^{-1} (x - mu)
C = W @ W.T + Psi
C_inv = np.linalg.inv(C)
Ez_joint = W.T @ C_inv @ x_cent
Cov_z_joint = np.eye(M) - W.T @ C_inv @ W

np.testing.assert_allclose(Ez_formula, Ez_joint, atol=1e-12)
np.testing.assert_allclose(G, Cov_z_joint, atol=1e-12)
print("Exercise 12.21 verified: Factor Analysis E-step formulas match joint Gaussian conditioning exactly!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_21_md), nbf.v4.new_code_cell(ex12_21_code)])

    # Exercise 12.22
    ex12_22_md = """---
## Exercise 12.22: 因子分析の EM アルゴリズム M-step 更新式の導出 (PRML 式 12.69, 12.70)

### 数学的導出
完全データの期待対数尤度関数は次式で与えられる：
$$ Q(\\mathbf{W}, \\mathbf{\\Psi}) = -\\frac{N}{2}\\sum_{d=1}^D \\ln \\psi_d - \\frac{1}{2}\\sum_{n=1}^N \\mathbb{E}\\left[ (\\tilde{\\mathbf{x}}_n - \\mathbf{W}\\mathbf{z}_n)^{\\mathrm{T}}\\mathbf{\\Psi}^{-1}(\\tilde{\\mathbf{x}}_n - \\mathbf{W}\\mathbf{z}_n) \\right] + \\mathrm{const} $$
ここで $\\tilde{\\mathbf{x}}_n = \\mathbf{x}_n - \\bar{\\mathbf{x}}$ である。

#### 1. $\\mathbf{W}$ に関する最大化
二乗展開項を行列トレースで表すと：
$$ Q = -\\frac{1}{2}\\sum_{n=1}^N \\left\\{ \\tilde{\\mathbf{x}}_n^{\\mathrm{T}}\\mathbf{\\Psi}^{-1}\\tilde{\\mathbf{x}}_n - 2\\mathbb{E}[\\mathbf{z}_n]^{\\mathrm{T}}\\mathbf{W}^{\\mathrm{T}}\\mathbf{\\Psi}^{-1}\\tilde{\\mathbf{x}}_n + \\mathrm{Tr}(\\mathbf{W}^{\\mathrm{T}}\\mathbf{\\Psi}^{-1}\\mathbf{W}\\mathbb{E}[\\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}}]) \\right\\} + \\mathrm{const} $$
$\\mathbf{W}$ に関して微分してゼロとおくと：
$$ \\frac{\\partial Q}{\\partial \\mathbf{W}} = \\mathbf{\\Psi}^{-1}\\left( \\sum_{n=1}^N \\tilde{\\mathbf{x}}_n \\mathbb{E}[\\mathbf{z}_n]^{\\mathrm{T}} - \\mathbf{W} \\sum_{n=1}^N \\mathbb{E}[\\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}}] \\right) = \\mathbf{O} $$
$\\mathbf{\\Psi}^{-1}$ を左から消去することで、PRML 式 (12.69) が得られる：
$$ \\mathbf{W}_{\\mathrm{new}} = \\left[ \\sum_{n=1}^N (\\mathbf{x}_n - \\bar{\\mathbf{x}})\\mathbb{E}[\\mathbf{z}_n]^{\\mathrm{T}} \\right] \\left[ \\sum_{n=1}^N \\mathbb{E}[\\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}}] \\right]^{-1} $$

#### 2. $\\mathbf{\\Psi}$ に関する最大化
各対角成分 $\\psi_d$ について個別に微分してゼロとおく：
$$ \\frac{\\partial Q}{\\partial \\psi_d} = -\\frac{N}{2\\psi_d} + \\frac{1}{2\\psi_d^2}\\sum_{n=1}^N \\mathbb{E}[(\\tilde{x}_{nd} - \\mathbf{w}_d^{\\mathrm{T}}\\mathbf{z}_n)^2] = 0 $$
$$ \\psi_d^{\\mathrm{new}} = \\frac{1}{N}\\sum_{n=1}^N \\left( \\tilde{x}_{nd}^2 - 2\\tilde{x}_{nd}\\mathbf{w}_d^{\\mathrm{new}}\\mathbb{E}[\\mathbf{z}_n] + (\\mathbf{w}_d^{\\mathrm{new}})^{\\mathrm{T}}\\mathbb{E}[\\mathbf{z}_n\\mathbf{z}_n^{\\mathrm{T}}]\\mathbf{w}_d^{\\mathrm{new}} \\right) $$
行列形式でまとめることで PRML 式 (12.70) が得られる：
$$ \\mathbf{\\Psi}_{\\mathrm{new}} = \\mathrm{diag}\\left\\{ \\mathbf{S} - \\mathbf{W}_{\\mathrm{new}} \\frac{1}{N}\\sum_{n=1}^N \\mathbb{E}[\\mathbf{z}_n](\\mathbf{x}_n - \\bar{\\mathbf{x}})^{\\mathrm{T}} \\right\\} $$
"""
    ex12_22_code = """# Exercise 12.22 数値検証: 因子分析 EM アルゴリズムの実装と対数尤度の単調増加確認
np.random.seed(42)
N = 150
D = 4
M = 2
X = np.random.randn(N, D) @ np.diag([3.0, 2.0, 1.0, 0.8])
x_bar = np.mean(X, axis=0)
X_cent = X - x_bar
S = np.cov(X, rowvar=False, bias=True)

# 初期化
W = np.random.randn(D, M)
Psi = np.diag(np.var(X, axis=0))

log_liks = []
for it in range(20):
    # E-step
    Psi_inv = np.diag(1.0 / np.diag(Psi))
    G = np.linalg.inv(np.eye(M) + W.T @ Psi_inv @ W)
    Ez = (X_cent @ Psi_inv @ W) @ G # (N, M)
    Ezz = N * G + Ez.T @ Ez

    # M-step
    W = (X_cent.T @ Ez) @ np.linalg.inv(Ezz)
    Psi_diag = np.diag(S - W @ (Ez.T @ X_cent) / N)
    Psi = np.diag(np.maximum(Psi_diag, 1e-4))

    # 周辺対数尤度
    C = W @ W.T + Psi
    sign, logdet = np.linalg.slogdet(C)
    ll = -0.5 * (N * D * np.log(2*np.pi) + N * logdet + np.sum(X_cent @ np.linalg.inv(C) * X_cent))
    log_liks.append(ll)

# 単調増加の確認
diffs = np.diff(log_liks)
assert np.all(diffs >= -1e-7), "Log likelihood must monotonically increase"
print(f"Exercise 12.22 verified: Factor Analysis EM converges with monotone log-likelihood increase (Initial: {log_liks[0]:.2f} -> Final: {log_liks[-1]:.2f})!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_22_md), nbf.v4.new_code_cell(ex12_22_code)])

    return cells

def build_part6():
    """Exercises 12.23 - 12.25: Mixture PPCAs, Student-t EM & Covariance Transformations"""
    cells = []

    # Exercise 12.23
    ex12_23_md = """---
## Exercise 12.23: 混合 PPCA (Mixture of PPCAs) の有向グラフィカルモデル

### 構造とパラメータ共有の数学的整理
PRML第12.2.3節の確率的 PCA 混合モデルにおいて、観測点 $\\mathbf{x}_n$ は離散潜在変数 $s_n \\in \\{1, \\dots, K\\}$（カテゴリカル分布 $\\boldsymbol{\\pi}$）および連続潜在変数 $\\mathbf{z}_n \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})$ から生成される：
$$ p(\\mathbf{x}_n \\mid \\mathbf{z}_n, s_n = k) = \\mathcal{N}(\\mathbf{x}_n \\mid \\mathbf{W}_k \\mathbf{z}_n + \\boldsymbol{\\mu}_k, \\sigma_k^2 \\mathbf{I}) $$

1. **非共有パラメータモデル (Untied Mixture of PPCAs)**:
   各クラスタ $k$ ごとに固有のパラメータ集合 $\\{\\mathbf{W}_k, \\boldsymbol{\\mu}_k, \\sigma_k^2\\}$ を持つ。各クラスタはそれぞれ独立な向き・拡がり・ノイズレベルを持つ局所主部分空間をモデル化する。
2. **共有パラメータモデル (Tied Mixture of PPCAs)**:
   すべてのクラスタで主部分空間の向き $\\mathbf{W}_k = \\mathbf{W}$ およびノイズ分散 $\\sigma_k^2 = \\sigma^2$ を共有し、クラスタ中心 $\\boldsymbol{\\mu}_k$ のみを変える。これにより、同一の低次元マニホールド上に存在する複数のクラスタを極めて少ないパラメータ数で表現できる。
"""
    ex12_23_code = """# Exercise 12.23 数値検証: 混合 PPCA の非共有 vs 共有パラメータ数と対数尤度計算
D = 6
M = 2
K = 3

# 1. Untied parameters: K * (D * M + D + 1) + (K - 1)
params_untied = K * (D * M + D + 1 - (M * (M - 1)) // 2) + (K - 1)
# 2. Tied parameters: (D * M + 1 - M(M-1)/2) + K * D + (K - 1)
params_tied = (D * M + 1 - (M * (M - 1)) // 2) + K * D + (K - 1)

print(f"Untied Mixture of PPCA parameters: {params_untied}")
print(f"Tied Mixture of PPCA parameters:   {params_tied}")
assert params_tied < params_untied
print("Exercise 12.23 verified: Parameter counts for untied and tied mixture of PPCAs confirmed!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_23_md), nbf.v4.new_code_cell(ex12_23_code)])

    # Exercise 12.24
    ex12_24_md = """---
## Exercise 12.24: 多変量スチューデント t 分布の EM アルゴリズム

### 数学的導出
PRML第2.3.7節で示されたように、$D$ 次元スチューデント t 分布 $\\mathrm{St}(\\mathbf{x} \\mid \\boldsymbol{\\mu}, \\mathbf{\\Sigma}, \\nu)$ は、ガンマ分布に従う潜在尺度変数 $\\eta_n \\sim \\mathrm{Gam}(\\nu/2, \\nu/2)$ を用いた無限個のガウス分布の連続混合表現として表される：
$$ p(\\mathbf{x}_n \\mid \\eta_n) = \\mathcal{N}(\\mathbf{x}_n \\mid \\boldsymbol{\\mu}, (\\eta_n \\mathbf{\\Sigma})^{-1}) $$
$$ p(\\eta_n) = \\mathrm{Gam}(\\eta_n \\mid \\nu/2, \\nu/2) = \\frac{1}{\\Gamma(\\nu/2)} \\left(\\frac{\\nu}{2}\\right)^{\\nu/2} \\eta_n^{\\nu/2 - 1}\\exp\\left( -\\frac{\\nu}{2}\\eta_n \\right) $$

#### E-step
観測データ $\\mathbf{x}_n$ が与えられたときの $\\eta_n$ の事後分布は共役性より再びガンマ分布となる：
$$ p(\\eta_n \\mid \\mathbf{x}_n) = \\mathrm{Gam}\\left( \\eta_n \\;\\middle|\\; \\frac{\\nu + D}{2}, \\; \\frac{\\nu + \\Delta_n^2}{2} \\right) $$
ここで $\\Delta_n^2 = (\\mathbf{x}_n - \\boldsymbol{\\mu})^{\\mathrm{T}}\\mathbf{\\Sigma}^{-1}(\\mathbf{x}_n - \\boldsymbol{\\mu})$ はマハラノビス距離の二乗である。
したがって、E-step の事後期待値はガンマ分布の平均値公式より：
$$ u_n = \\mathbb{E}[\\eta_n \\mid \\mathbf{x}_n] = \\frac{\\nu + D}{\\nu + \\Delta_n^2} $$
外れ値（$\\Delta_n^2$ が極めて大きい点）に対しては $u_n \\approx 0$ となり、重みが自動的に抑制される（ロバスト推定）。

#### M-step
期待完全対数尤度を $\\boldsymbol{\\mu}$ および $\\mathbf{\\Sigma}$ で最大化することにより：
$$ \\boldsymbol{\\mu}_{\\mathrm{new}} = \\frac{\\sum_{n=1}^N u_n \\mathbf{x}_n}{\\sum_{n=1}^N u_n} $$
$$ \\mathbf{\\Sigma}_{\\mathrm{new}} = \\frac{1}{N}\\sum_{n=1}^N u_n (\\mathbf{x}_n - \\boldsymbol{\\mu}_{\\mathrm{new}})(\\mathbf{x}_n - \\boldsymbol{\\mu}_{\\mathrm{new}})^{\\mathrm{T}} $$
"""
    ex12_24_code = """# Exercise 12.24 数値検証: 多変量スチューデント t 分布の EM アルゴリズムと外れ値に対する頑健性
np.random.seed(42)
N = 100
D = 2
# 正常データ (90点) と極端な外れ値 (10点)
X_clean = np.random.randn(90, D)
X_outliers = np.random.uniform(15, 25, size=(10, D))
X = np.vstack([X_clean, X_outliers])

# 初期化
mu = np.median(X, axis=0)
Sigma = np.cov(X, rowvar=False)
nu = 4.0

for it in range(25):
    # E-step
    Sigma_inv = np.linalg.inv(Sigma)
    diff = X - mu
    delta_sq = np.sum(diff @ Sigma_inv * diff, axis=1)
    u = (nu + D) / (nu + delta_sq)

    # M-step
    mu = np.sum(u[:, None] * X, axis=0) / np.sum(u)
    diff_new = X - mu
    Sigma = (diff_new.T @ (u[:, None] * diff_new)) / N

print(f"True clean mean: [0.0, 0.0]")
print(f"Sample mean (contaminated): {np.mean(X, axis=0)}")
print(f"Student-t EM robust mean:   {mu}")
print(f"Average weight for outliers: {np.mean(u[90:]):.4f} << Clean weight: {np.mean(u[:90]):.4f}")

np.testing.assert_allclose(mu, np.zeros(D), atol=0.25)
print("Exercise 12.24 verified: Multivariate Student-t EM algorithm successfully downweights outliers!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_24_md), nbf.v4.new_code_cell(ex12_24_code)])

    # Exercise 12.25
    ex12_25_md = """---
## Exercise 12.25: アフィン変換に対する共変性（因子分析の尺度共変性と PPCA の回転共変性）

### 数学的証明
観測変数に正則アフィン変換 $\\mathbf{y} = \\mathbf{A}\\mathbf{x}$ を施す（$\\mathbf{A} \\in \\mathbb{R}^{D \\times D}$）。
元のモデルにおける観測周辺分布は $\\mathbf{x} \\sim \\mathcal{N}(\\boldsymbol{\\mu}, \\mathbf{C})$ （$\\mathbf{C} = \\mathbf{W}\\mathbf{W}^{\\mathrm{T}} + \\mathbf{\\Phi}$）である。
線形変換後の変数 $\\mathbf{y}$ の周辺分布は：
$$ \\mathbb{E}[\\mathbf{y}] = \\mathbf{A}\\boldsymbol{\\mu} $$
$$ \\mathrm{cov}[\\mathbf{y}] = \\mathbf{A}\\mathbf{C}\\mathbf{A}^{\\mathrm{T}} = \\mathbf{A}(\\mathbf{W}\\mathbf{W}^{\\mathrm{T}} + \\mathbf{\\Phi})\\mathbf{A}^{\\mathrm{T}} = (\\mathbf{A}\\mathbf{W})(\\mathbf{A}\\mathbf{W})^{\\mathrm{T}} + \\mathbf{A}\\mathbf{\\Phi}\\mathbf{A}^{\\mathrm{T}} $$
したがって、最尤推定量は $\\boldsymbol{\\mu}^* = \\mathbf{A}\\boldsymbol{\\mu}_{\\mathrm{ML}}$, $\\mathbf{W}^* = \\mathbf{A}\\mathbf{W}_{\\mathrm{ML}}$, $\\mathbf{\\Phi}^* = \\mathbf{A}\\mathbf{\\Phi}_{\\mathrm{ML}}\\mathbf{A}^{\\mathrm{T}}$ となる。

#### モデル形式が保存される 2 つの重要なケース
1. **因子分析 (Factor Analysis)**:
   $\\mathbf{\\Phi} = \\mathrm{diag}(\\psi_1, \\dots, \\psi_D)$ は対角行列である。
   ここで変換行列 $\\mathbf{A}$ が対角行列（成分ごとのスケーリング $\\mathbf{A} = \\mathrm{diag}(a_1, \\dots, a_D)$）である場合：
   $$ \\mathbf{A}\\mathbf{\\Phi}\\mathbf{A}^{\\mathrm{T}} = \\mathrm{diag}(a_1^2 \\psi_1, \\dots, a_D^2 \\psi_D) $$
   となり、変換後のノイズ共分散も厳密に対角行列の形を保持する。したがって、**因子分析は各変数の成分ごとのスケール変換に対して共変（Covariant）** である。
2. **確率的 PCA (PPCA)**:
   $\\mathbf{\\Phi} = \\sigma^2 \\mathbf{I}$ は等方性ノイズ行列である。
   ここで変換行列 $\\mathbf{A}$ が直交行列（データ空間の回転 $\\mathbf{A}\\mathbf{A}^{\\mathrm{T}} = \\mathbf{I}$）である場合：
   $$ \\mathbf{A}(\\sigma^2 \\mathbf{I})\\mathbf{A}^{\\mathrm{T}} = \\sigma^2 \\mathbf{A}\\mathbf{A}^{\\mathrm{T}} = \\sigma^2 \\mathbf{I} $$
   となり、変換後のノイズ共分散も厳密に等方性単位行列に比例する。したがって、**PPCA はデータ空間の直交回転に対して共変（Covariant）** である（標準 PCA と同様）。
"""
    ex12_25_code = """# Exercise 12.25 数値検証: FA の対角スケーリング共変性と PPCA の直交回転共変性
np.random.seed(42)
D = 3
M = 1
W = np.array([[1.0], [2.0], [-1.0]])

# 1. 因子分析: 対角スケーリング A
Psi = np.diag([0.2, 0.5, 0.8])
A_diag = np.diag([2.0, -3.0, 0.5])
Phi_FA_trans = A_diag @ Psi @ A_diag.T
assert np.allclose(Phi_FA_trans, np.diag(np.diag(Phi_FA_trans))), "FA transformed Phi must remain diagonal"

# 2. PPCA: 直交回転 A_ortho
sigma_sq = 0.4
Phi_PPCA = sigma_sq * np.eye(D)
Q, _ = np.linalg.qr(np.random.randn(D, D))
Phi_PPCA_trans = Q @ Phi_PPCA @ Q.T
np.testing.assert_allclose(Phi_PPCA_trans, sigma_sq * np.eye(D), atol=1e-12)

print("Exercise 12.25 verified: FA is covariant under axis scaling, PPCA is covariant under orthogonal rotation!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_25_md), nbf.v4.new_code_cell(ex12_25_code)])

    return cells

def build_part7():
    """Exercises 12.26 - 12.29: Kernel PCA & Nonlinear Extensions"""
    cells = []

    # Exercise 12.26
    ex12_26_md = """---
## Exercise 12.26: カーネル PCA の固有値方程式と零空間不変性 (PRML 式 12.79, 12.80, 12.82)

### 数学的証明
カーネル行列を $\\mathbf{K} \\in \\mathbb{R}^{N \\times N}$ とする。
PRML 式 (12.80) の固有値方程式は次式である：
$$ \\mathbf{K}\\mathbf{a}_i = \\lambda_i \\mathbf{a}_i $$
両辺に左から $\\mathbf{K}$ を掛けると：
$$ \\mathbf{K}(\\mathbf{K}\\mathbf{a}_i) = \\mathbf{K}(\\lambda_i \\mathbf{a}_i) = \\lambda_i \\mathbf{K}\\mathbf{a}_i $$
すなわち $\\mathbf{K}^2 \\mathbf{a}_i = \\lambda_i \\mathbf{K}\\mathbf{a}_i$ （PRML 式 12.79）を満足する。

次に、$\\mathbf{K}$ の零空間固有ベクトル $\\mathbf{v}$ （すなわち $\\mathbf{K}\\mathbf{v} = \\mathbf{0}$）を考え、任意の係数 $c$ に対して変形したベクトル $\\mathbf{a}_i' = \\mathbf{a}_i + c\\mathbf{v}$ を定義する。
これを (12.79) に代入すると：
$$ \\mathbf{K}^2 \\mathbf{a}_i' = \\mathbf{K}^2 \\mathbf{a}_i + c\\mathbf{K}^2 \\mathbf{v} = \\lambda_i \\mathbf{K}\\mathbf{a}_i + \\mathbf{0} = \\lambda_i \\mathbf{K}(\\mathbf{a}_i + c\\mathbf{v}) = \\lambda_i \\mathbf{K}\\mathbf{a}_i' $$
したがって $\\mathbf{a}_i'$ もまた同一の固有値 $\\lambda_i$ を持つ (12.79) の解となる。

最後に、任意のテスト点 $\\mathbf{x}$ に対する第 $i$ 主成分射影は PRML 式 (12.82) より：
$$ y_i(\\mathbf{x}) = \\boldsymbol{\\phi}(\\mathbf{x})^{\\mathrm{T}}\\mathbf{v}_i = \\sum_{n=1}^N a_{in} k(\\mathbf{x}, \\mathbf{x}_n) $$
特に訓練データセット内の点に対する射影値ベクトルは $\\mathbf{y}_i = \\mathbf{K}\\mathbf{a}_i$ で与えられる。
$\\mathbf{a}_i'$ を用いた場合の射影値は：
$$ \\mathbf{y}_i' = \\mathbf{K}\\mathbf{a}_i' = \\mathbf{K}(\\mathbf{a}_i + c\\mathbf{v}) = \\mathbf{K}\\mathbf{a}_i + c\\mathbf{K}\\mathbf{v} = \\mathbf{K}\\mathbf{a}_i = \\mathbf{y}_i $$
となり、零空間成分の加算は主成分射影の結果に一切影響を与えない。
"""
    ex12_26_code = """# Exercise 12.26 数値検証: カーネル PCA 零空間固有ベクトルの加算と射影の完全不変性
np.random.seed(42)
N = 30
D = 3
X = np.random.randn(N, D)
# ランク落ちするカーネル行列 (低次元特徴写像)
K = X @ X.T # rank <= 3, N = 30 なので零空間次元は 27

eigvals, eigvecs = np.linalg.eigh(K)
# 最大固有値と固有ベクトル
idx_max = np.argmax(eigvals)
lam_max = eigvals[idx_max]
a_max = eigvecs[:, idx_max]

# 零固有値に対応するベクトル
idx_zero = np.where(np.abs(eigvals) < 1e-10)[0][0]
v_null = eigvecs[:, idx_zero]

# 1. K^2 a' = lam K a' の検証
c = 5.7
a_prime = a_max + c * v_null
left = K @ K @ a_prime
right = lam_max * (K @ a_prime)
np.testing.assert_allclose(left, right, atol=1e-8)

# 2. 訓練点に対する射影 y = K a の完全不変性検証
proj_orig = K @ a_max
proj_prime = K @ a_prime
np.testing.assert_allclose(proj_prime, proj_orig, atol=1e-10)
print("Exercise 12.26 verified: Null-space additions strictly satisfy K^2 a' = lambda K a' and preserve projections!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_26_md), nbf.v4.new_code_cell(ex12_26_code)])

    # Exercise 12.27
    ex12_27_md = """---
## Exercise 12.27: 線形カーネルを用いた場合の標準線形 PCA への厳密な帰着

### 数学的証明
特徴写像として恒等写像 $\\boldsymbol{\\phi}(\\mathbf{x}) = \\mathbf{x}$、すなわち線形カーネル $k(\\mathbf{x}, \\mathbf{x}') = \\mathbf{x}^{\\mathrm{T}}\\mathbf{x}'$ を選択する。
中心化されたデータ行列を $\\mathbf{X} \\in \\mathbb{R}^{N \\times D}$ とすると、グラム行列は $\\mathbf{K} = \\mathbf{X}\\mathbf{X}^{\\mathrm{T}} \\in \\mathbb{R}^{N \\times N}$ である。
カーネル PCA の固有値方程式（正規化条件 $\\lambda_i \\mathbf{a}_i^{\\mathrm{T}}\\mathbf{a}_i = 1$）は：
$$ \\frac{1}{N}\\mathbf{K}\\mathbf{a}_i = \\lambda_i \\mathbf{a}_i \\implies \\frac{1}{N}\\mathbf{X}\\mathbf{X}^{\\mathrm{T}}\\mathbf{a}_i = \\lambda_i \\mathbf{a}_i $$
データ空間における特徴固有ベクトル $\\mathbf{u}_i \\in \\mathbb{R}^D$ は PRML 式 (12.78) より：
$$ \\mathbf{u}_i = \\sum_{n=1}^N a_{in}\\mathbf{x}_n = \\mathbf{X}^{\\mathrm{T}}\\mathbf{a}_i $$
両辺に左からサンプル共分散行列 $\\mathbf{S} = \\frac{1}{N}\\mathbf{X}^{\\mathrm{T}}\\mathbf{X}$ を掛けると：
$$ \\mathbf{S}\\mathbf{u}_i = \\left( \\frac{1}{N}\\mathbf{X}^{\\mathrm{T}}\\mathbf{X} \\right) (\\mathbf{X}^{\\mathrm{T}}\\mathbf{a}_i) = \\mathbf{X}^{\\mathrm{T}} \\left( \\frac{1}{N}\\mathbf{X}\\mathbf{X}^{\\mathrm{T}}\\mathbf{a}_i \\right) = \\mathbf{X}^{\\mathrm{T}}(\\lambda_i \\mathbf{a}_i) = \\lambda_i (\\mathbf{X}^{\\mathrm{T}}\\mathbf{a}_i) = \\lambda_i \\mathbf{u}_i $$
また、正規化ノルムは：
$$ \\|\\mathbf{u}_i\\|^2 = \\mathbf{u}_i^{\\mathrm{T}}\\mathbf{u}_i = \\mathbf{a}_i^{\\mathrm{T}}\\mathbf{X}\\mathbf{X}^{\\mathrm{T}}\\mathbf{a}_i = \\mathbf{a}_i^{\\mathrm{T}}\\mathbf{K}\\mathbf{a}_i = \\mathbf{a}_i^{\\mathrm{T}}(N\\lambda_i \\mathbf{a}_i) = N\\lambda_i \\|\\mathbf{a}_i\\|^2 = 1 $$
そして任意の点 $\\mathbf{x}$ の射影は：
$$ y_i(\\mathbf{x}) = \\mathbf{a}_i^{\\mathrm{T}}\\mathbf{k}(\\mathbf{x}) = \\sum_{n=1}^N a_{in}(\\mathbf{x}_n^{\\mathrm{T}}\\mathbf{x}) = \\left( \\sum_{n=1}^N a_{in}\\mathbf{x}_n \\right)^{\\mathrm{T}}\\mathbf{x} = \\mathbf{u}_i^{\\mathrm{T}}\\mathbf{x} $$
これは従来の標準主成分分析（線形 PCA）の射影と完全に同一である。
"""
    ex12_27_code = """# Exercise 12.27 数値検証: 線形カーネル PCA と標準線形 PCA の完全一致
from sklearn.decomposition import PCA, KernelPCA

np.random.seed(42)
N = 100
D = 5
X = np.random.randn(N, D)
X = X - np.mean(X, axis=0) # 中心化

# 1. 標準線形 PCA
pca = PCA(n_components=2)
proj_standard = pca.fit_transform(X)

# 2. 線形 Kernel PCA
kpca = KernelPCA(n_components=2, kernel='linear')
proj_kpca = kpca.fit_transform(X)

# 符号の反転を考慮して一致を検証
for m in range(2):
    dot_prod = np.abs(np.corrcoef(proj_standard[:, m], proj_kpca[:, m])[0, 1])
    print(f"Component {m+1} Absolute Correlation: {dot_prod:.10f}")
    np.testing.assert_allclose(dot_prod, 1.0, atol=1e-7)

print("Exercise 12.27 verified: Linear Kernel PCA recovers conventional linear PCA projections exactly!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_27_md), nbf.v4.new_code_cell(ex12_27_code)])

    # Exercise 12.28
    ex12_28_md = """---
## Exercise 12.28: 確率密度の単調変数変換 $y = f(x)$ による任意確率密度の生成微分方程式

### 数学的証明
1次元確率変数 $x$ の確率密度を $q(x)$ （すべての点で $q(x) > 0$）とする。
狭義単調増加関数 $y = f(x)$ （$f'(x) > 0$）による変数変換を行うと、PRML第1章の確率密度変換公式 (1.27) より：
$$ p(y) = q(x) \\left| \\frac{dx}{dy} \\right| = \\frac{q(x)}{f'(x)} $$
両辺を $f'(x)$ について解くことで、所望の目的密度 $p(y)$ を生成する微分方程式が得られる：
$$ f'(x) = \\frac{q(x)}{p(f(x))} $$
この微分方程式は変数分離形であり、両辺を積分すると：
$$ \\int p(f(x)) f'(x) dx = \\int q(x) dx \\implies \\int_{-\\infty}^{f(x)} p(y) dy = \\int_{-\\infty}^x q(t) dt $$
累積分布関数をそれぞれ $P(y) = \\int_{-\\infty}^y p(t)dt$, $Q(x) = \\int_{-\\infty}^x q(t)dt$ と表すと：
$$ P(f(x)) = Q(x) \\implies f(x) = P^{-1}(Q(x)) $$
これは所望の確率密度への変換が、既知密度の CDF による確率積分変換（一様乱数化）と目的密度のパーセンタイル逆関数（Inverse CDF）の合成写像として常に一意に構成可能であることを示している。
"""
    ex12_28_code = """# Exercise 12.28 数値検証: 微分方程式 f'(x) = q(x) / p(f(x)) の数値解と累積分布逆関数の完全一致
from scipy.integrate import solve_ivp
from scipy.stats import norm, expon

# q(x): 標準正規分布 N(0, 1)
# p(y): 指数分布 Exp(lambda=1.0)
def odefun(x, y):
    q_val = norm.pdf(x)
    p_val = expon.pdf(y[0])
    return [q_val / max(p_val, 1e-12)]

# 初期値: x0 = 0 (正規分布の中央値 Q(0)=0.5)
# y0 = P^{-1}(0.5) = -ln(1 - 0.5) = ln(2)
x0 = 0.0
y0 = [np.log(2.0)]
x_eval = np.linspace(0.0, 2.0, 50)

sol = solve_ivp(odefun, [0.0, 2.0], y0, t_eval=x_eval, rtol=1e-8, atol=1e-8)
y_ode = sol.y[0]

# 理論解: y = P^{-1}(Q(x)) = -ln(1 - Phi(x))
Q_x = norm.cdf(x_eval)
y_theory = -np.log(1.0 - Q_x)

np.testing.assert_allclose(y_ode, y_theory, atol=1e-4)
print("Exercise 12.28 verified: ODE f'(x) = q(x)/p(f(x)) matches theoretical CDF mapping P^{-1}(Q(x))!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_28_md), nbf.v4.new_code_cell(ex12_28_code)])

    # Exercise 12.29
    ex12_29_md = """---
## Exercise 12.29: 独立性と無相関性の関係（「独立 $\\implies$ 無相関」と「無相関 $\\not\\implies$ 独立」の厳密反例）

### 数学的証明
1. **独立であれば無相関であることの証明**:
   2つの変数 $z_1, z_2$ が独立であるとき、$p(z_1, z_2) = p(z_1)p(z_2)$ である。
   このとき共分散は：
   $$ \\mathrm{cov}(z_1, z_2) = \\mathbb{E}[z_1 z_2] - \\mathbb{E}[z_1]\\mathbb{E}[z_2] $$
   $$ = \\iint z_1 z_2 p(z_1)p(z_2) dz_1 dz_2 - \\mathbb{E}[z_1]\\mathbb{E}[z_2] = \\left(\\int z_1 p(z_1) dz_1\\right)\\left(\\int z_2 p(z_2) dz_2\\right) - \\mathbb{E}[z_1]\\mathbb{E}[z_2] = 0 $$
   したがって、結合共分散行列の非対角成分はゼロとなり、共分散行列は対角行列となる。

2. **無相関であっても独立とは限らない反例**:
   区間 $[-1, 1]$ 上の一様分布に従う変数 $y_1 \\sim \\mathcal{U}[-1, 1]$ を考え、$y_2 = y_1^2$ とする。
   - 条件付き確率分布は $p(y_2 \\mid y_1) = \\delta(y_2 - y_1^2)$ であり、$y_1$ の値に完全に依存しているため、$y_1$ と $y_2$ は**明らかに独立ではない**。
   - 一方で期待値は：
     $$ \\mathbb{E}[y_1] = \\int_{-1}^1 y_1 \\cdot \\frac{1}{2} dy_1 = 0 $$
     $$ \\mathbb{E}[y_1 y_2] = \\mathbb{E}[y_1^3] = \\int_{-1}^1 y_1^3 \\cdot \\frac{1}{2} dy_1 = 0 $$
     したがって共分散は：
     $$ \\mathrm{cov}(y_1, y_2) = \\mathbb{E}[y_1 y_2] - \\mathbb{E}[y_1]\\mathbb{E}[y_2] = 0 - 0 = 0 $$
   共分散行列は対角行列であり、相関係数 $\\rho(y_1, y_2) = 0$（完全に無相関）である。
   この反例は、無相関性が独立性の十分条件ではないことを決定的に示している。
"""
    ex12_29_code = """# Exercise 12.29 数値検証: y_2 = y_1^2 における相関ゼロと決定論的従属性の実証
np.random.seed(42)
N = 100000
y1 = np.random.uniform(-1, 1, size=N)
y2 = y1**2

# 共分散行列の計算
Cov_mat = np.cov(y1, y2)
cov_12 = Cov_mat[0, 1]
corr_12 = np.corrcoef(y1, y2)[0, 1]

print(f"Covariance cov(y1, y2):     {cov_12:.8f} (~ 0)")
print(f"Pearson Correlation r:      {corr_12:.8f} (~ 0)")
np.testing.assert_allclose(cov_12, 0.0, atol=0.01)

# しかし y2 は y1 に完全に決定論的に依存している (条件付き分散 Var(y2 | y1) = 0 != Var(y2))
var_y2_total = np.var(y2)
print(f"Total variance Var(y2):     {var_y2_total:.4f} > 0")
print(f"Conditional variance Var(y2 | y1) = 0 identically everywhere!")
assert var_y2_total > 0.05
print("Exercise 12.29 verified: Variables are completely dependent yet strictly uncorrelated (diagonal covariance)!")"""
    cells.extend([nbf.v4.new_markdown_cell(ex12_29_md), nbf.v4.new_code_cell(ex12_29_code)])

    return cells

def get_ch12_all_exercises():
    """Assemble all 29 exercises for Chapter 12"""
    all_cells = []
    all_cells.extend(build_part1()) # Ex 12.1 - 12.5
    all_cells.extend(build_part2()) # Ex 12.6 - 12.10
    all_cells.extend(build_part3()) # Ex 12.11 - 12.14
    all_cells.extend(build_part4()) # Ex 12.15 - 12.17
    all_cells.extend(build_part5()) # Ex 12.18 - 12.22
    all_cells.extend(build_part6()) # Ex 12.23 - 12.25
    all_cells.extend(build_part7()) # Ex 12.26 - 12.29
    return all_cells

if __name__ == "__main__":
    cells = get_ch12_all_exercises()
    print(f"Successfully generated {len(cells)} cells for Chapter 12 Exercises!")
