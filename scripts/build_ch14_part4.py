# scripts/build_ch14_part4.py
"""
Build Chapter 14 Exercises: Part 4 (Exercises 14.13 - 14.17)
Mixture of Experts EM Algorithm, IRLS, Multimodality Failure of Conditional Mean, and HME
"""

import nbformat as nbf

def build_part4():
    """Exercises 14.13 - 14.17: EM Optimization and Hierarchical Extensions"""
    cells = []

    # Exercise 14.13
    ex14_13_md = """---
## Exercise 14.13: 条件付き混合モデルの事後負担率 $\\gamma_{nk}$ と完全データ期待対数尤度 $Q$ 関数の導出

### 問題の背景と数学的証明
各データ点 $(\\mathbf{x}_n, t_n)$ に対し、どのエキスパートが選択されたかを表す 1-of-$K$ 表現の潜在変数 $\\mathbf{z}_n \\in \\{0, 1\\}^K$（$\\sum_{k=1}^K z_{nk} = 1$）を導入する。
完全データ対数尤度は次のように因数分解される（(14.44)式）：
$$ \\ln p(\\mathbf{t}, \\mathbf{Z} | \\mathbf{X}, \\boldsymbol{\\theta}) = \\sum_{n=1}^N \\sum_{k=1}^K z_{nk} \\left\\{ \\ln \\pi_k(\\mathbf{x}_n) + \\ln p_k(t_n | \\mathbf{x}_n) \\right\\} $$
本問では、事後潜在変数期待値（負担率）：
$$ \\gamma_{nk} \\equiv \\mathbb{E}[z_{nk} | \\mathbf{x}_n, t_n, \\boldsymbol{\\theta}^{\\text{old}}] = \\frac{\\pi_k(\\mathbf{x}_n) p_k(t_n | \\mathbf{x}_n, \\boldsymbol{\\theta}_k^{\\text{old}})}{\\sum_{j=1}^K \\pi_j(\\mathbf{x}_n) p_j(t_n | \\mathbf{x}_n, \\boldsymbol{\\theta}_j^{\\text{old}})} $$
および期待完全データ対数尤度 $Q$ 関数を厳密に導出する。

**代数的導出**:
1. ベイズの定理より、潜在変数 $z_{nk}=1$ の事後確率は：
$$ p(z_{nk} = 1 | \\mathbf{x}_n, t_n) = \\frac{p(z_{nk} = 1 | \\mathbf{x}_n) p(t_n | \\mathbf{x}_n, z_{nk} = 1)}{\\sum_{j=1}^K p(z_{nj} = 1 | \\mathbf{x}_n) p(t_n | \\mathbf{x}_n, z_{nj} = 1)} $$
2. $p(z_{nk} = 1 | \\mathbf{x}_n) = \\pi_k(\\mathbf{x}_n)$、および $p(t_n | \\mathbf{x}_n, z_{nk} = 1) = p_k(t_n | \\mathbf{x}_n)$ を代入すると、求める事後負担率が得られる：
$$ \\gamma_{nk} = \\frac{\\pi_k(\\mathbf{x}_n) p_k(t_n | \\mathbf{x}_n)}{\\sum_{j=1}^K \\pi_j(\\mathbf{x}_n) p_j(t_n | \\mathbf{x}_n)} $$
3. 完全データ対数尤度の期待値をとると、線形性により $z_{nk}$ のみが $\\mathbb{E}[z_{nk}] = \\gamma_{nk}$ に置き換わる：
$$ Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{\\text{old}}) = \\sum_{n=1}^N \\sum_{k=1}^K \\gamma_{nk} \\ln \\pi_k(\\mathbf{x}_n) + \\sum_{n=1}^N \\sum_{k=1}^K \\gamma_{nk} \\ln p_k(t_n | \\mathbf{x}_n, \\boldsymbol{\\theta}_k) $$
4. これにより、$Q$ 関数はゲーティングネットワークのパラメータ $\\{\\boldsymbol{\\eta}_k\\}$ に依存する項と、各エキスパートのパラメータ $\\{\\mathbf{w}_k, \\beta_k\\}$ に依存する項に**完全に分離**され、個別に独立して最大化できる。

#### 穴埋め問題
1. 事後負担率 $\\gamma_{nk}$ は、観測値 $(\\mathbf{x}_n, t_n)$ が与えられた下でエキスパート $k$ が生成した $\\text{[ (A) ]}$ である。
2. 期待対数尤度 $Q$ 関数は、ゲーティング項とエキスパート項の $\\text{[ (B) ]}$ に分離される。
3. これにより、M ステップにおいて各エキスパートのパラメータは互いに $\\text{[ (C) ]}$ に更新可能となる。
*(解: A: 条件付き事後確率, B: 和, C: 独立)*
"""
    ex14_13_code = """# Exercise 14.13 数値検証: 混合エキスパートの E ステップ（事後負担率の計算）
# 事後負担率 gamma の計算
gamma_mat = np.zeros((N, K))
for n in range(N):
    lik_row = np.zeros(K)
    for k in range(K):
        norm_pdf = np.sqrt(beta[k] / (2.0 * np.pi)) * np.exp(-0.5 * beta[k] * (t[n] - W[k] @ X[n])**2)
        lik_row[k] = pi_all[n, k] * norm_pdf
    gamma_mat[n, :] = lik_row / np.sum(lik_row)

# 各行の和が 1 であることを検証
np.testing.assert_allclose(np.sum(gamma_mat, axis=1), np.ones(N), atol=1e-12)

# Q 関数の計算
Q_val = 0.0
for n in range(N):
    for k in range(K):
        norm_pdf = np.sqrt(beta[k] / (2.0 * np.pi)) * np.exp(-0.5 * beta[k] * (t[n] - W[k] @ X[n])**2)
        Q_val += gamma_mat[n, k] * (np.log(pi_all[n, k] + 1e-15) + np.log(norm_pdf + 1e-15))

print(f"Exercise 14.13 verified: Posterior responsibilities sum to 1 strictly; Q value = {Q_val:.4f}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_13_md), nbf.v4.new_code_cell(ex14_13_code)])

    # Exercise 14.14
    ex14_14_md = """---
## Exercise 14.14: エキスパートパラメータ $\\mathbf{w}_k, \\beta_k$ の重み付き最小二乗（WLS）閉形式更新式の導出

### 問題の背景と数学的証明
M ステップにおいて、$Q$ 関数のエキスパート項：
$$ Q_k(\\mathbf{w}_k, \\beta_k) = \\sum_{n=1}^N \\gamma_{nk} \\left\\{ \\frac{1}{2}\\ln\\beta_k - \\frac{1}{2}\\ln(2\\pi) - \\frac{\\beta_k}{2} (t_n - \\mathbf{w}_k^{\\mathrm{T}} \\mathbf{x}_n)^2 \\right\\} $$
を最大化するパラメータ $\\mathbf{w}_k$ および精度 $\\beta_k$ の閉形式更新式（(14.47)-(14.48)式）を導出する。

**代数的導出**:
1. **回帰重み $\\mathbf{w}_k$ の導出**:
   $Q_k$ を $\\mathbf{w}_k$ で偏微分してゼロとおく：
   $$ \\frac{\\partial Q_k}{\\partial \\mathbf{w}_k} = \\beta_k \\sum_{n=1}^N \\gamma_{nk} (t_n - \\mathbf{w}_k^{\\mathrm{T}} \\mathbf{x}_n) \\mathbf{x}_n = \\mathbf{0} $$
   $\\beta_k > 0$ であるため：
   $$ \\sum_{n=1}^N \\gamma_{nk} \\mathbf{x}_n \\mathbf{x}_n^{\\mathrm{T}} \\mathbf{w}_k = \\sum_{n=1}^N \\gamma_{nk} t_n \\mathbf{x}_n $$
   対角重み行列 $\\mathbf{R}_k = \\operatorname{diag}(\\gamma_{1k}, \\dots, \\gamma_{Nk})$ を定義すると、重み付き最小二乗推定量：
   $$ \\mathbf{w}_k^{\\text{new}} = (\\mathbf{X}^{\\mathrm{T}} \\mathbf{R}_k \\mathbf{X})^{-1} \\mathbf{X}^{\\mathrm{T}} \\mathbf{R}_k \\mathbf{t} $$
   が得られる。
2. **精度 $\\beta_k$ の導出**:
   $Q_k$ を $\\beta_k$ で偏微分してゼロとおく：
   $$ \\frac{\\partial Q_k}{\\partial \\beta_k} = \\sum_{n=1}^N \\gamma_{nk} \\left[ \\frac{1}{2\\beta_k} - \\frac{1}{2}(t_n - \\mathbf{w}_k^{\\mathrm{T}} \\mathbf{x}_n)^2 \\right] = 0 $$
   両辺を 2 倍して整理すると：
   $$ \\frac{1}{\\beta_k} \\sum_{n=1}^N \\gamma_{nk} = \\sum_{n=1}^N \\gamma_{nk} (t_n - \\mathbf{w}_k^{\\mathrm{T}} \\mathbf{x}_n)^2 $$
   したがって：
   $$ \\frac{1}{\\beta_k^{\\text{new}}} = \\frac{\\sum_{n=1}^N \\gamma_{nk} (t_n - \\mathbf{w}_k^{\\mathrm{T}} \\mathbf{x}_n)^2}{\\sum_{n=1}^N \\gamma_{nk}} $$
   が得られる。これは負担率で重み付けされた残差分散である。

#### 穴埋め問題
1. 回帰ベクトル $\\mathbf{w}_k$ の最適解は、負担率 $\\gamma_{nk}$ を対角重みとする $\\text{[ (A) ]}$ 解である。
2. エキスパートの分散 $1/\\beta_k$ は、負担率による $\\text{[ (B) ]}$ に一致する。
*(解: A: 重み付き最小二乗 (WLS), B: 重み付き平均二乗残差)*
"""
    ex14_14_code = """# Exercise 14.14 数値検証: エキスパート WLS 解の導出と勾配ゼロの確認
W_new = np.zeros_like(W)
beta_new = np.zeros_like(beta)

for k in range(K):
    R_k = np.diag(gamma_mat[:, k])
    # w_k = (X^T R_k X)^-1 X^T R_k t
    W_new[k] = np.linalg.solve(X.T @ R_k @ X, X.T @ R_k @ t)
    
    # 残差二乗
    residuals = t - X @ W_new[k]
    var_k = np.sum(gamma_mat[:, k] * (residuals**2)) / np.sum(gamma_mat[:, k])
    beta_new[k] = 1.0 / var_k
    
    # 勾配ゼロの検証
    grad_w = X.T @ R_k @ (t - X @ W_new[k])
    np.testing.assert_allclose(grad_w, np.zeros(D), atol=1e-12)

print("Exercise 14.14 verified: Expert parameters solve WLS equations strictly:")
print("Updated W:\\n", W_new)
print("Updated beta:", beta_new)
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_14_md), nbf.v4.new_code_cell(ex14_14_code)])

    # Exercise 14.15
    ex14_15_md = """---
## Exercise 14.15: ゲーティングネットワーク（softmax）の勾配ベクトルとヘッセ行列の導出 (IRLS)

### 問題の背景と数学的証明
M ステップにおいて、ゲーティングパラメータ $\\boldsymbol{\\eta}_k$ に依存する目的関数は：
$$ Q_{\\text{gate}}(\\{\\boldsymbol{\\eta}_k\\}) = \\sum_{n=1}^N \\sum_{k=1}^K \\gamma_{nk} \\ln \\pi_k(\\mathbf{x}_n) $$
である（(14.49)式）。これは多クラスロジスティック回帰（第4章）のクロスエントロピー誤差と同一の形式を持つ。
本問では、$\\boldsymbol{\\eta}_j$ に関する勾配ベクトルおよびヘッセ行列を導出し、ニュートン・ラフソン法（IRLS）による更新公式を証明する。

**代数的導出**:
1. softmax 関数の導関数：
$$ \\frac{\\partial \\pi_k(\\mathbf{x}_n)}{\\partial \\boldsymbol{\\eta}_j} = \\pi_k(\\mathbf{x}_n)(\\delta_{jk} - \\pi_j(\\mathbf{x}_n)) \\mathbf{x}_n $$
2. $Q_{\\text{gate}}$ の $\\boldsymbol{\\eta}_j$ に関する勾配：
$$ \\nabla_{\\boldsymbol{\\eta}_j} Q_{\\text{gate}} = \\sum_{n=1}^N \\sum_{k=1}^K \\frac{\\gamma_{nk}}{\\pi_k(\\mathbf{x}_n)} \\frac{\\partial \\pi_k(\\mathbf{x}_n)}{\\partial \\boldsymbol{\\eta}_j} = \\sum_{n=1}^N \\sum_{k=1}^K \\gamma_{nk} (\\delta_{jk} - \\pi_j(\\mathbf{x}_n)) \\mathbf{x}_n $$
$\\sum_{k=1}^K \\gamma_{nk} = 1$ であるため：
$$ \\nabla_{\\boldsymbol{\\eta}_j} Q_{\\text{gate}} = \\sum_{n=1}^N (\\gamma_{nj} - \\pi_j(\\mathbf{x}_n)) \\mathbf{x}_n $$
目標値（事後負担率）$\\gamma_{nj}$ と予測確率 $\\pi_j(\\mathbf{x}_n)$ の差の線形和という極めて美しい形となる。
3. ヘッセ行列（2階偏微分）：
$$ \\mathbf{H}_{jk} = \\frac{\\partial^2 Q_{\\text{gate}}}{\\partial \\boldsymbol{\\eta}_j \\partial \\boldsymbol{\\eta}_k^{\\mathrm{T}}} = -\\sum_{n=1}^N \\frac{\\partial \\pi_j(\\mathbf{x}_n)}{\\partial \\boldsymbol{\\eta}_k} \\mathbf{x}_n^{\\mathrm{T}} = -\\sum_{n=1}^N \\pi_j(\\mathbf{x}_n)(\\delta_{jk} - \\pi_k(\\mathbf{x}_n)) \\mathbf{x}_n \\mathbf{x}_n^{\\mathrm{T}} $$
これは全ブロックにおいて負の半定値行列であり、$Q_{\\text{gate}}$ が凹関数（単峰性）であることを保証する。

#### 穴埋め問題
1. ゲーティングネットワークの学習は、目標値を事後負担率 $\\gamma_{nk}$ とする $\\text{[ (A) ]}$ 回帰に等価である。
2. 勾配は残差 $\\gamma_{nj} - \\pi_j(\\mathbf{x}_n)$ に入力 $\\mathbf{x}_n$ を乗じたものである。
3. ヘッセ行列が負半定値であるため、目的関数は $\\text{[ (B) ]}$ 関数であり局所解に捕らわれず最適解に収束する。
*(解: A: 多クラスロジスティック (softmax), B: 凹 (上に凸))*
"""
    ex14_15_code = """# Exercise 14.15 数値検証: ゲーティングの解析的勾配と数値微分の完全一致
# パラメータ eta を 1次元ベクトルに平坦化してテスト
eta_flat = eta.flatten()

def gate_objective(eta_params):
    eta_m = eta_params.reshape(K, D)
    pi_m = softmax(eta_m, X)
    return np.sum(gamma_mat * np.log(pi_m + 1e-15))

def gate_gradient_analytical(eta_params):
    eta_m = eta_params.reshape(K, D)
    pi_m = softmax(eta_m, X)
    grad = np.zeros((K, D))
    for k in range(K):
        grad[k] = np.sum((gamma_mat[:, k] - pi_m[:, k])[:, None] * X, axis=0)
    return grad.flatten()

# 数値勾配 (有限差分法)
eps_fd = 1e-6
grad_numerical = np.zeros_like(eta_flat)
base_val = gate_objective(eta_flat)

for i in range(len(eta_flat)):
    eta_step = eta_flat.copy()
    eta_step[i] += eps_fd
    val_step = gate_objective(eta_step)
    grad_numerical[i] = (val_step - base_val) / eps_fd

grad_ana = gate_gradient_analytical(eta_flat)
np.testing.assert_allclose(grad_ana, grad_numerical, atol=1e-5)

print("Exercise 14.15 verified: Analytical gradient strictly matches finite-difference numerical gradient:")
print("Analytical Gradient:", grad_ana)
print("Numerical  Gradient:", grad_numerical)
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_15_md), nbf.v4.new_code_cell(ex14_15_code)])

    # Exercise 14.16
    ex14_16_md = """---
## Exercise 14.16: 多峰性 (Multimodal) 分布における条件付き期待値 $\\mathbb{E}[t|\\mathbf{x}]$ の破綻と混合エキスパートの優位性

### 問題の背景と数学的証明
標準的な二乗誤差最小化による回帰モデル（線形回帰、ニューラルネットワーク、ガウス過程等）は、条件付き期待値：
$$ y(\\mathbf{x}) = \\mathbb{E}[t | \\mathbf{x}] = \\int t \\, p(t | \\mathbf{x}) dt $$
を出力する。
しかし、ロボットの障害物回避や逆問題（Inverse kinematics）のように、$p(t | \\mathbf{x})$ が複数の峰を持つ（多峰性である）場合、条件付き期待値は**確率がほぼゼロである「山と山の中間値」**を出力してしまい、致命的な大事故や予測の破綻を引き起こす。
本問では、この現象を数値的に構成し、混合エキスパートモデルが個々の峰（モード）を正確に分離してモデル化できる優位性を検証する。

**反例の数理**:
1. 目標変数 $t$ が入力 $x$ に対して2つの枝 $t = x$ または $t = -x$ を確率 0.5 ずつでとる二峰性分布：
$$ p(t | x) = 0.5 \\, \\mathcal{N}(t \\mid x, \\sigma^2) + 0.5 \\, \\mathcal{N}(t \\mid -x, \\sigma^2) $$
2. 条件付き期待値は相加平均である：
$$ \\mathbb{E}[t | x] = 0.5 (x) + 0.5 (-x) = 0 $$
3. $x \\ne 0$ において、真のデータは $t = +x$ または $t = -x$ の周辺に集中しており、$t = 0$ 付近にはデータが**1点も存在しない（確率密度が極小）**。
4. 単一の回帰モデルは常に $y(x) = 0$ と予測するため、すべての予測が外れる。
5. 一方、混合エキスパートモデルは2つのエキスパート $y_1(x) = x, y_2(x) = -x$ を割り当てることで、個々のモードを完璧に表現できる。

#### 穴埋め問題
1. 二乗誤差を最小化する決定論的予測モデルは、条件付き分布の $\\text{[ (A) ]}$ を出力する。
2. 分布が二峰性を持つ場合、条件付き期待値は谷間（確率密度が $\\text{[ (B) ]}$ な領域）を出力する。
3. 混合エキスパートモデルは、潜在変数を導入することで個々の $\\text{[ (C) ]}$ を同時に捕捉できる。
*(解: A: 平均 (期待値), B: 極小 (ゼロに近い), C: モード (峰))*
"""
    ex14_16_code = """# Exercise 14.16 数値検証: 多峰性における条件付き期待値の破綻と混合モデルの優位性
# x = 4.0 における二峰性分布: モード +4 と -4
x_val = 4.0
# 真のデータ生成
t_mode1 = np.random.normal(loc=x_val, scale=0.3, size=500)
t_mode2 = np.random.normal(loc=-x_val, scale=0.3, size=500)
t_all = np.concatenate([t_mode1, t_mode2])

# 1. 単一の二乗誤差最小化モデルの予測 (平均値)
cond_mean = np.mean(t_all)
# 平均値 t=0 における真のデータ密度は極めて小さい
num_near_mean = np.sum(np.abs(t_all - cond_mean) < 1.0)

# 2. 混合モデル (2つのモード)
mode1_est = np.mean(t_mode1)
mode2_est = np.mean(t_mode2)

print("Exercise 14.16 verified:")
print(f"  Conditional Mean Prediction E[t|x]: {cond_mean:.4f}")
print(f"  Number of actual data points near mean (within 1.0): {num_near_mean} / 1000 (Catastrophic failure!)")
print(f"  Mixture Expert 1 captures Mode 1: {mode1_est:.4f}")
print(f"  Mixture Expert 2 captures Mode 2: {mode2_est:.4f}")

assert np.abs(cond_mean) < 0.2
assert num_near_mean == 0  # 平均値付近には全くサンプルが存在しない！
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_16_md), nbf.v4.new_code_cell(ex14_16_code)])

    # Exercise 14.17
    ex14_17_md = """---
## Exercise 14.17: 階層的混合エキスパート (Hierarchical Mixture of Experts: HME) のマルチレベル確率モデルと EM 更新

### 問題の背景と数学的証明
階層的混合エキスパート (HME) は、混合エキスパートモデルを木構造状に入れ子にした高度なアンサンブルアーキテクチャである（(14.53)式）。
深さ2の木を考えると、上位ノードのゲーティング確率 $\\pi_j(\\mathbf{x})$ と下位ノードのゲーティング確率 $\\pi_{k|j}(\\mathbf{x})$ の積によって各葉エキスパートの結合選択確率が定義される：
$$ p(t | \\mathbf{x}) = \\sum_{j=1}^{J} \\pi_j(\\mathbf{x}) \\sum_{k=1}^K \\pi_{k|j}(\\mathbf{x}) p_{jk}(t | \\mathbf{x}) $$
本問では、結合事後負担率：
$$ \\gamma_{j, k|j} = \\frac{\\pi_j(\\mathbf{x}) \\pi_{k|j}(\\mathbf{x}) p_{jk}(t | \\mathbf{x})}{\\sum_{j'} \\pi_{j'}(\\mathbf{x}) \\sum_{k'} \\pi_{k'|j'}(\\mathbf{x}) p_{j'k'}(t | \\mathbf{x})} $$
および上位ノードの周辺負担率 $\\gamma_j = \\sum_{k=1}^K \\gamma_{j, k|j}$ を導出し、マルチレベル EM アルゴリズムの構造的整合性を証明する。

#### 穴埋め問題
1. HME は、木構造の上位から下位へ向けた $\\text{[ (A) ]}$ の積として各葉ノードの選択確率を表現する。
2. 上位ノードの負担率は、配下の葉ノードの負担率の $\\text{[ (B) ]}$ として得られる。
3. これにより、複雑な決定境界を $\\text{[ (C) ]}$ 的に空間分割しながら局所モデルを適合できる。
*(解: A: 条件付きゲーティング確率, B: 周辺和, C: 階層)*
"""
    ex14_17_code = """# Exercise 14.17 数値検証: 階層的混合エキスパート (HME) の結合負担率と周辺化整合性
J_top = 2
K_sub = 3

# トップレベルのゲーティング確率 (和が1)
pi_top = np.array([0.4, 0.6])
# サブレベルの条件付きゲーティング確率 (各行の和が1)
pi_sub = np.array([[0.2, 0.5, 0.3],
                   [0.7, 0.1, 0.2]])

# 各葉エキスパートの尤度 p_jk(t|x)
p_leaf = np.array([[1.2, 0.8, 0.1],
                   [0.3, 2.5, 0.4]])

# 1. 結合事前重み pi_jk = pi_j * pi_{k|j}
pi_joint = pi_top[:, None] * pi_sub
np.testing.assert_allclose(np.sum(pi_joint), 1.0, atol=1e-12)

# 2. 全周辺尤度
joint_unnorm = pi_joint * p_leaf
p_total = np.sum(joint_unnorm)

# 3. 結合事後負担率 gamma_{j, k|j}
gamma_joint = joint_unnorm / p_total
np.testing.assert_allclose(np.sum(gamma_joint), 1.0, atol=1e-12)

# 4. 上位ノードの周辺負担率 gamma_j = sum_k gamma_{jk}
gamma_top = np.sum(gamma_joint, axis=1)

print("Exercise 14.17 verified: Hierarchical Mixture of Experts marginal consistency holds strictly:")
print("Top-level responsibilities gamma_j:", gamma_top)
print("Sum of joint responsibilities:", np.sum(gamma_joint))
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_17_md), nbf.v4.new_code_cell(ex14_17_code)])

    return cells
