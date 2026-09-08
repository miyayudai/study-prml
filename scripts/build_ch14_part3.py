# scripts/build_ch14_part3.py
"""
Build Chapter 14 Exercises: Part 3 (Exercises 14.9 - 14.12)
Decision Trees, Gini Impurity, Entropy, Cost-Complexity Pruning, and Mixture of Experts
"""

import nbformat as nbf

def build_part3():
    """Exercises 14.9 - 14.12: Trees and Mixture of Experts Formulation"""
    cells = []

    # Exercise 14.9
    ex14_9_md = """---
## Exercise 14.9: 決定木 (CART) における分割基準：誤分類率、交差エントロピー、ジニ不純度の定義と凹性の数学的比較

### 問題の背景と数学的証明
決定木（CART）のノード $m$ において、クラス $k \\in \\{1, \\dots, K\\}$ に属するサンプルの割合を $p_{mk}$ とする（$\\sum_{k=1}^K p_{mk} = 1$）。
ノードの不純度（不規則性）を測定する3大基準（(14.28)-(14.30)式）：
1. **誤分類率**: $Q_m(T) = 1 - \\max_k p_{mk}$
2. **交差エントロピー**: $Q_m(T) = -\\sum_{k=1}^K p_{mk} \\ln p_{mk}$
3. **ジニ不純度**: $Q_m(T) = \\sum_{k=1}^K p_{mk} (1 - p_{mk}) = 1 - \\sum_{k=1}^K p_{mk}^2$
について、それぞれの定義と数学的凹性（上に凸であること）を比較・証明する。

**数理的凹性の証明**:
1. **ジニ不純度の凹性**:
   $g(\\mathbf{p}) = 1 - \\sum_{k=1}^K p_k^2$ とする。2階偏導関数は：
   $$ \\frac{\\partial^2 g}{\\partial p_i \\partial p_j} = -2 \\delta_{ij} $$
   したがってヘッセ行列は $\\mathbf{H} = -2\\mathbf{I}$ であり、負定値である。ゆえにジニ不純度は全域で厳密に凹（上に凸）である。
2. **交差エントロピーの凹性**:
   $h(\\mathbf{p}) = -\\sum_{k=1}^K p_k \\ln p_k$ とする。2階偏導関数は：
   $$ \\frac{\\partial^2 h}{\\partial p_i \\partial p_j} = -\\frac{1}{p_i} \\delta_{ij} $$
   ヘッセ行列は対角成分が $-1/p_i < 0$ の負定値対角行列であるため、エントロピーも厳密に凹（上に凸）である。
3. **凹性と分割利得の正値性（イェンセンの不等式）**:
   親ノードの確率分布 $\\mathbf{p}$ が、左右の子ノードに比率 $\\lambda, 1-\\lambda$ で分割されて $\\mathbf{p}_L, \\mathbf{p}_R$ となったとする（$\\mathbf{p} = \\lambda \\mathbf{p}_L + (1-\\lambda) \\mathbf{p}_R$）。
   凹関数のイェンセンの不等式より：
   $$ Q(\\lambda \\mathbf{p}_L + (1-\\lambda) \\mathbf{p}_R) \\ge \\lambda Q(\\mathbf{p}_L) + (1-\\lambda) Q(\\mathbf{p}_R) $$
   したがって、分割利得 $\\Delta Q = Q(\\mathbf{p}) - [\\lambda Q(\\mathbf{p}_L) + (1-\\lambda) Q(\\mathbf{p}_R)] \\ge 0$ は常に非負であり、分割によって不純度が決して悪化しないことが数学的に保証される。

#### 穴埋め問題
1. ジニ不純度のヘッセ行列は $\\text{[ (A) ]}$ であり、全域で上に凸（凹関数）である。
2. 誤分類率に比べ、交差エントロピーやジニ不純度はノードの純度向上に対して $\\text{[ (B) ]}$ に反応する。
3. 凹関数のイェンセンの不等式により、分割による不純度利得は常に $\\text{[ (C) ]}$ である。
*(解: A: 負定値 ($-2\\mathbf{I}$), B: 滑らかかつ敏感, C: 非負 ($\\ge 0$))*
"""
    ex14_9_code = """# Exercise 14.9 数値検証: 3大不純度基準の計算と凹性・分割利得非負性の確認
import numpy as np

def misclassification_rate(p):
    return 1.0 - np.max(p)

def cross_entropy(p):
    p = np.clip(p, 1e-15, 1.0)
    return -np.sum(p * np.log(p))

def gini_impurity(p):
    return 1.0 - np.sum(p**2)

# 親ノードと子ノードの確率ベクトル (K=3)
p_L = np.array([0.8, 0.1, 0.1])
p_R = np.array([0.1, 0.7, 0.2])
lam = 0.6
p_parent = lam * p_L + (1.0 - lam) * p_R

for name, func in [("Misclass", misclassification_rate), ("Entropy", cross_entropy), ("Gini", gini_impurity)]:
    q_parent = func(p_parent)
    q_children = lam * func(p_L) + (1.0 - lam) * func(p_R)
    gain = q_parent - q_children
    assert gain >= -1e-12, f"Gain for {name} must be non-negative!"
    print(f"  {name:10s}: Parent={q_parent:.4f}, Children={q_children:.4f}, Gain={gain:.4f} >= 0")

print("Exercise 14.9 verified: Splitting gain is strictly non-negative for all criteria!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_9_md), nbf.v4.new_code_cell(ex14_9_code)])

    # Exercise 14.10
    ex14_10_md = """---
## Exercise 14.10: 2クラス分類におけるジニ不純度とエントロピーの対称性と最大値 $p=0.5$ の証明

### 問題の背景と数学的証明
$K=2$ クラス分類において、クラス1の所属確率を $p \\in [0, 1]$、クラス2の確率を $1-p$ とする。
このとき：
- ジニ不純度: $G(p) = 2p(1 - p)$
- エントロピー: $H(p) = -p \\ln p - (1 - p) \\ln(1 - p)$
- 誤分類率: $M(p) = 1 - \\max(p, 1-p) = \\min(p, 1-p)$
について、以下の対称性と極値の性質を数学的に証明する：
1. **対称性**: すべての $p \\in [0, 1]$ に対して $G(p) = G(1-p)$、$H(p) = H(1-p)$。
2. **境界条件**: $p=0$ および $p=1$ において $G(0)=G(1)=0$、$H(0)=H(1)=0$（完全純粋ノード）。
3. **最大値**: $p=0.5$（最も不純なノード）において最大値 $G(0.5) = 0.5$、$H(0.5) = \\ln 2 \\approx 0.693$ をとる。

**代数的証明**:
1. **対称性**:
   $G(1-p) = 2(1-p)(1-(1-p)) = 2(1-p)p = G(p)$。
   $H(1-p) = -(1-p)\\ln(1-p) - (1-(1-p))\\ln(1-(1-p)) = -(1-p)\\ln(1-p) - p\\ln p = H(p)$。
2. **1階微分と極値**:
   - $G'(p) = 2(1 - 2p) = 0 \\implies p = 0.5$。
   - $H'(p) = -\\ln p - 1 + \\ln(1-p) + 1 = \\ln\\left(\\frac{1-p}{p}\\right) = 0 \\implies \\frac{1-p}{p} = 1 \\implies p = 0.5$。
3. **2階微分**:
   - $G''(p) = -4 < 0$。
   - $H''(p) = -\\frac{1}{p} - \\frac{1}{1-p} = -\\frac{1}{p(1-p)} < 0$（$p \\in (0, 1)$）。
   したがって、$p=0.5$ は唯一の大域的最大値である。

#### 穴埋め問題
1. 2クラス問題における不純度関数は、直線 $p = \\text{[ (A) ]}$ に関して線対称である。
2. 純粋なノード（$p=0$ または $p=1$）では不純度は $\\text{[ (B) ]}$ となる。
3. 2階微分が常に負であるため、$p=0.5$ で不純度は $\\text{[ (C) ]}$ となる。
*(解: A: $0.5$, B: $0$, C: 最大)*
"""
    ex14_10_code = """# Exercise 14.10 数値検証: 2クラス不純度関数の対称性と極値
p_grid = np.linspace(0.01, 0.99, 100)

gini_curve = 2.0 * p_grid * (1.0 - p_grid)
entropy_curve = -p_grid * np.log(p_grid) - (1.0 - p_grid) * np.log(1.0 - p_grid)

# 1. 対称性の検証: f(p) == f(1-p)
np.testing.assert_allclose(gini_curve, gini_curve[::-1], atol=1e-12)
np.testing.assert_allclose(entropy_curve, entropy_curve[::-1], atol=1e-12)

# 2. p=0.5 での最大値検証
p_mid = 0.5
max_gini = 2.0 * p_mid * (1.0 - p_mid)
max_entropy = -2.0 * 0.5 * np.log(0.5)

np.testing.assert_allclose(max_gini, 0.5, atol=1e-12)
np.testing.assert_allclose(max_entropy, np.log(2.0), atol=1e-12)

assert np.all(gini_curve <= max_gini + 1e-12)
assert np.all(entropy_curve <= max_entropy + 1e-12)

print(f"Exercise 14.10 verified: Symmetry holds and maximum is strictly at p=0.5 (Gini={max_gini}, Entropy={max_entropy:.4f})!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_10_md), nbf.v4.new_code_cell(ex14_10_code)])

    # Exercise 14.11
    ex14_11_md = """---
## Exercise 14.11: 決定木のコスト複雑度剪定基準 (Cost-Complexity Pruning) と正則化パラメータ $\\alpha$

### 問題の背景と数学的証明
決定木が過学習するのを防ぐため、葉ノードの数 $|T|$ に対する正則化ペナルティを加えた「コスト複雑度評価関数」（(14.34)式）：
$$ C_\\alpha(T) = \\sum_{m=1}^{|T|} N_m Q_m(T) + \\alpha |T| = R(T) + \\alpha |T| $$
を最小化する最適枝刈り（Minimal Cost-Complexity Pruning）を考える。
ここで $R(T) = \\sum_m N_m Q_m(T)$ は訓練誤差（全不純度和）、$|T|$ は部分木の葉の数、$\\alpha \\ge 0$ は木の複雑さに対するペナルティ係数である。
本問では、$\\alpha$ の増加に伴って最適な部分木のサイズ $|T_\\alpha|$ が単調非増加であることを証明する。

**数理的証明**:
1. 2つのペナルティ係数 $\\alpha_1 < \\alpha_2$ を考える。
2. $\\alpha_1$ における最適木を $T_1$、$\\alpha_2$ における最適木を $T_2$ とする。最適性の定義より：
$$ C_{\\alpha_1}(T_1) \\le C_{\\alpha_1}(T_2) \\implies R(T_1) + \\alpha_1 |T_1| \\le R(T_2) + \\alpha_1 |T_2| $$
$$ C_{\\alpha_2}(T_2) \\le C_{\\alpha_2}(T_1) \\implies R(T_2) + \\alpha_2 |T_2| \\le R(T_1) + \\alpha_2 |T_1| $$
3. この2つの不等式を加える：
$$ [R(T_1) + R(T_2)] + \\alpha_1 |T_1| + \\alpha_2 |T_2| \\le [R(T_1) + R(T_2)] + \\alpha_1 |T_2| + \\alpha_2 |T_1| $$
4. 両辺から共通項 $R(T_1) + R(T_2)$ を引く：
$$ \\alpha_1 |T_1| + \\alpha_2 |T_2| \\le \\alpha_1 |T_2| + \\alpha_2 |T_1| $$
5. 同類項を整理する：
$$ (\\alpha_2 - \\alpha_1) |T_2| \\le (\\alpha_2 - \\alpha_1) |T_1| $$
6. 仮定より $\\alpha_2 - \\alpha_1 > 0$ であるため、両辺を割ると：
$$ |T_2| \\le |T_1| $$
が得られる。
7. したがって、ペナルティ係数 $\\alpha$ を大きくすると、最適木の葉の数（サイズ）$|T_\\alpha|$ は**単調非増加**（縮小または維持）することが厳密に証明された。

#### 穴埋め問題
1. コスト複雑度基準 $C_\\alpha(T)$ は、訓練不純度 $R(T)$ と葉の数 $|T|$ に対する $\\text{[ (A) ]}$ 項からなる。
2. $\\alpha = 0$ のとき、評価関数を最小化するのは最大の $\\text{[ (B) ]}$ である。
3. $\\alpha$ を増大させると、最適な部分木のサイズ $|T|$ は $\\text{[ (C) ]}$ する。
*(解: A: 正則化 (ペナルティ), B: 未剪定木 (完全成長木), C: 単調非増加 (縮小))*
"""
    ex14_11_code = """# Exercise 14.11 数値検証: コスト複雑度剪定における木サイズの単調非増加性
# 候補木群: (R(T), |T|)
# 木が大きいほど訓練誤差 R(T) は小さいが、葉の数 |T| は大きい
trees = [
    {"name": "Tree_1 (Full)", "R": 10.0, "size": 10},
    {"name": "Tree_2 (Medium)", "R": 15.0, "size": 6},
    {"name": "Tree_3 (Small)", "R": 22.0, "size": 3},
    {"name": "Tree_4 (Root)", "R": 35.0, "size": 1}
]

alpha_range = np.linspace(0.0, 10.0, 50)
optimal_sizes = []

for alpha in alpha_range:
    costs = [t["R"] + alpha * t["size"] for t in trees]
    best_idx = np.argmin(costs)
    optimal_sizes.append(trees[best_idx]["size"])

optimal_sizes = np.array(optimal_sizes)
# 単調非増加性の検証: 差分が常に <= 0
diffs = np.diff(optimal_sizes)
assert np.all(diffs <= 0), "Optimal tree size must be monotonically non-increasing in alpha!"

print("Exercise 14.11 verified: Optimal tree size decreases monotonically as alpha increases:")
print(f"  alpha=0.0 -> Best Size = {optimal_sizes[0]}")
print(f"  alpha=3.0 -> Best Size = {optimal_sizes[15]}")
print(f"  alpha=10.0 -> Best Size = {optimal_sizes[-1]}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_11_md), nbf.v4.new_code_cell(ex14_11_code)])

    # Exercise 14.12
    ex14_12_md = """---
## Exercise 14.12: 条件付き混合モデル（Mixture of Experts: 回帰）の定式化と対数尤度関数

### 問題の背景と数学的証明
混合エキスパートモデル (Mixture of Experts) は、空間分割と局所モデルを同時に学習する確率的モデルである（(14.35)式）：
$$ p(t | \\mathbf{x}, \\boldsymbol{\\theta}) = \\sum_{k=1}^K \\pi_k(\\mathbf{x}) p_k(t | \\mathbf{x}) $$
ここで $\\pi_k(\\mathbf{x})$ はゲーティング関数（$\\pi_k(\\mathbf{x}) \\ge 0, \\sum_k \\pi_k(\\mathbf{x}) = 1$）であり、入力依存の softmax 関数でモデル化される：
$$ \\pi_k(\\mathbf{x}) = \\frac{\\exp(\\boldsymbol{\\eta}_k^{\\mathrm{T}} \\mathbf{x})}{\\sum_{j=1}^K \\exp(\\boldsymbol{\\eta}_j^{\\mathrm{T}} \\mathbf{x})} $$
各エキスパート $p_k(t | \\mathbf{x})$ は局所的な線形ガウス回帰モデルである：
$$ p_k(t | \\mathbf{x}) = \\mathcal{N}(t \\mid \\mathbf{w}_k^{\\mathrm{T}} \\mathbf{x}, \\beta_k^{-1}) $$
本問では、独立同分布な $N$ 点のデータセットに対する全対数尤度関数：
$$ \\ln p(\\mathbf{t} | \\mathbf{X}, \\boldsymbol{\\theta}) = \\sum_{n=1}^N \\ln \\left\\{ \\sum_{k=1}^K \\pi_k(\\mathbf{x}_n) \\mathcal{N}(t_n \\mid \\mathbf{w}_k^{\\mathrm{T}} \\mathbf{x}_n, \\beta_k^{-1}) \\right\\} $$
を定式化し、その構造的性質を解析する。

#### 穴埋め問題
1. ゲーティングネットワーク $\\pi_k(\\mathbf{x})$ は入力に依存してエキスパートを選択する $\\text{[ (A) ]}$ 関数である。
2. 全体の対数尤度関数は、対数の中に $\\text{[ (B) ]}$ が入るため、パラメータに関する解析的な直接最大化解は存在しない。
3. このような潜在変数を持つモデルの最適化には $\\text{[ (C) ]}$ アルゴリズムが標準的に適用される。
*(解: A: softmax, B: 和 (Summation), C: EM)*
"""
    ex14_12_code = """# Exercise 14.12 数値検証: Mixture of Experts の対数尤度関数の評価
def softmax(eta_matrix, x):
    # eta_matrix: (K, D), x: (N, D)
    logits = x @ eta_matrix.T  # (N, K)
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)

# パラメータ設定 (K=2, D=2)
K = 2
D = 2
eta = np.array([[1.0, -0.5],
                [-1.0, 0.5]])
W = np.array([[2.0, 1.0],
              [-1.5, 0.5]])
beta = np.array([10.0, 10.0])

# 合成データ
N = 50
X = np.random.randn(N, D)
# 真の混合生成
pi_val = softmax(eta, X)
t = np.zeros(N)
for n in range(N):
    k_choice = np.random.choice(K, p=pi_val[n])
    t[n] = W[k_choice] @ X[n] + np.random.randn() / np.sqrt(beta[k_choice])

# 対数尤度計算
pi_all = softmax(eta, X)
log_lik = 0.0
for n in range(N):
    lik_n = 0.0
    for k in range(K):
        norm_pdf = np.sqrt(beta[k] / (2.0 * np.pi)) * np.exp(-0.5 * beta[k] * (t[n] - W[k] @ X[n])**2)
        lik_n += pi_all[n, k] * norm_pdf
    log_lik += np.log(lik_n + 1e-15)

assert np.isfinite(log_lik)
print(f"Exercise 14.12 verified: Mixture of Experts log-likelihood evaluated successfully: {log_lik:.4f}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex14_12_md), nbf.v4.new_code_cell(ex14_12_code)])

    return cells
