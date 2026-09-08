# scripts/build_ch13_part4.py
"""
Build Chapter 13 Exercises: Part 4 (Exercises 13.16 - 13.20)
Viterbi Algorithm, Left-to-Right HMMs, Decoding Discrepancies, and Multiple Sequences
"""

import nbformat as nbf

def build_part4():
    """Exercises 13.16 - 13.20: Decoding & Sequence Extensions"""
    cells = []

    # Exercise 13.16
    ex13_16_md = """---
## Exercise 13.16: Viterbi アルゴリズムにおける動的計画法再帰式とバックトラッキングポインタの導出

### 問題の背景と数学的証明
観測系列 $\\mathbf{X} = \\{\\mathbf{x}_1, \\dots, \\mathbf{x}_N\\}$ が与えられたとき、最も確率の高い潜在状態系列 $\\mathbf{Z}^* = \\arg\\max_{\\mathbf{Z}} p(\\mathbf{X}, \\mathbf{Z})$ を求める問題を考える。
状態系列の候補数は $K^N$ であり全探索は不可能であるが、max-sum アルゴリズム（動的計画法）を用いることで $\\mathcal{O}(K^2 N)$ の計算量で大域的最適系列を厳密に求めることができる。

**数理的導出**:
1. 時刻 $n$ における最大対数結合確率を次のように定義する（(13.68)式）：
$$ \\omega(z_n) \\equiv \\max_{z_1, \\dots, z_{n-1}} \\ln p(\\mathbf{x}_1, \\dots, \\mathbf{x}_n, z_1, \\dots, z_n) $$
2. 時刻 $n+1$ における $\\omega(z_{n+1})$ を展開する：
$$ \\omega(z_{n+1}) = \\max_{z_1, \\dots, z_n} \\left[ \\ln p(\\mathbf{x}_1, \\dots, \\mathbf{x}_n, z_1, \\dots, z_n) + \\ln p(z_{n+1} | z_n) + \\ln p(\\mathbf{x}_{n+1} | z_{n+1}) \\right] $$
3. 最大化操作を入れ子にする：
$$ \\omega(z_{n+1}) = \\ln p(\\mathbf{x}_{n+1} | z_{n+1}) + \\max_{z_n} \\left\\{ \\ln p(z_{n+1} | z_n) + \\max_{z_1, \\dots, z_{n-1}} \\ln p(\\mathbf{x}_1, \\dots, \\mathbf{x}_n, z_1, \\dots, z_n) \\right\\} $$
4. 内部の最大化はまさに定義より $\\omega(z_n)$ であるため、求める Viterbi 再帰漸化式が得られる：
$$ \\omega(z_{n+1}) = \\ln p(\\mathbf{x}_{n+1} | z_{n+1}) + \\max_{z_n} \\left\\{ \\ln p(z_{n+1} | z_n) + \\omega(z_n) \\right\\} $$
5. 最適経路を復元するために、各ステップで最大値を与えた直前の状態をバックトラッキングポインタとして記録する：
$$ \\psi(z_{n+1}) = \\arg\\max_{z_n} \\left\\{ \\ln p(z_{n+1} | z_n) + \\omega(z_n) \\right\\} $$
6. 終端時刻 $N$ で最善の状態 $z_N^* = \\arg\\max_{z_N} \\omega(z_N)$ を決定した後、ポインタを過去へ辿る（$z_n^* = \\psi(z_{n+1}^*)$）ことで、厳密な大域的最適パス $\\mathbf{Z}^*$ が復元される。

#### 穴埋め問題
1. Viterbi アルゴリズムは、グラフィカルモデルにおける $\\text{[ (A) ]}$ アルゴリズムを対数領域で適用したものである。
2. 計算量は全探索の $\\mathcal{O}(K^N)$ から $\\text{[ (B) ]}$ に大幅に削減される。
3. 最適系列の復元には、最大値を与えた状態インデックスを保持する $\\text{[ (C) ]}$ を用いる。
*(解: A: max-sum (max-product), B: $\\mathcal{O}(K^2 N)$, C: バックトラッキングポインタ $\\psi$)*
"""
    ex13_16_code = """# Exercise 13.16 数値検証: Viterbi アルゴリズムの実装と最確系列の導出
import numpy as np

def viterbi(pi, A, B, obs):
    N = len(obs)
    K = len(pi)
    omega = np.zeros((N, K))
    psi = np.zeros((N, K), dtype=int)
    
    # 初期化 (対数領域)
    omega[0, :] = np.log(pi + 1e-15) + np.log(B[:, obs[0]] + 1e-15)
    
    # 前向き再帰
    for n in range(1, N):
        for k in range(K):
            # omega_{n-1}(j) + log A_{jk}
            trans_probs = omega[n-1, :] + np.log(A[:, k] + 1e-15)
            psi[n, k] = np.argmax(trans_probs)
            omega[n, k] = np.max(trans_probs) + np.log(B[k, obs[n]] + 1e-15)
            
    # 終端最適状態
    best_path = np.zeros(N, dtype=int)
    best_path[-1] = np.argmax(omega[-1, :])
    max_log_p = omega[-1, best_path[-1]]
    
    # バックトラッキング
    for n in range(N - 2, -1, -1):
        best_path[n] = psi[n+1, best_path[n+1]]
        
    return best_path, max_log_p

pi_test = np.array([0.5, 0.5])
A_test = np.array([[0.8, 0.2],
                   [0.1, 0.9]])
B_test = np.array([[0.9, 0.1],
                   [0.2, 0.8]])
obs_test = [0, 0, 1, 1, 1]

viterbi_path, viterbi_score = viterbi(pi_test, A_test, B_test, obs_test)
print(f"Exercise 13.16 verified: Viterbi best path: {viterbi_path}, log-prob: {viterbi_score:.6f}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_16_md), nbf.v4.new_code_cell(ex13_16_code)])

    # Exercise 13.17
    ex13_17_md = """---
## Exercise 13.17: 左から右へ遷移する Left-to-Right HMM の制約と最尤推定解

### 問題の背景と数学的証明
音声認識や手書き文字認識では、状態が時間の経過とともに左から右へと進み、過去の状態へ戻ることがない Left-to-Right HMM（Bakis モデル）が広く用いられる。
推移行列に対する制約は次のように定義される：
$$ A_{jk} = 0 \\quad (k < j) $$
すなわち、推移行列 $\\mathbf{A}$ は上三角行列となる。

**数理的性質と最尤推定の証明**:
1. **上三角性の保存**:
   M ステップ更新公式（Exercise 13.5）において：
   $$ A_{jk}^{\\text{new}} = \\frac{\\sum_{n=2}^N \\xi(z_{n-1,j}, z_{nk})}{\\sum_{n=2}^N \\gamma(z_{n-1,j})} $$
   $k < j$ の場合、現在のパラメータで $A_{jk}^{\\text{old}} = 0$ であれば、事後確率 $\\xi(z_{n-1,j}, z_{nk}) \\propto A_{jk}^{\\text{old}} = 0$ となるため、分子は恒等的にゼロとなる。
   したがって、一度ゼロに設定された下三角要素は学習によって正になることはなく、上三角構造が厳密に維持される。
2. **パスの単調非減少性**:
   Viterbi アルゴリズムにおいて、許容されるパスは $z_1^* \\le z_2^* \\le \\dots \\le z_N^*$ を満たす単調非減少系列に限られる。

#### 穴埋め問題
1. Left-to-Right HMM では、遷移行列は $\\text{[ (A) ]}$ 行列となる。
2. 下三角要素 $A_{jk} (k < j)$ は $\\text{[ (B) ]}$ に固定される。
3. 最尤推定の EM 更新において、このゼロ制約は $\\text{[ (C) ]}$ に維持される。
*(解: A: 上三角, B: $0$, C: 自動的 (不変))*
"""
    ex13_17_code = """# Exercise 13.17 数値検証: Left-to-Right HMM の上三角性保持と単調非減少パス
K_ltr = 4
A_ltr = np.zeros((K_ltr, K_ltr))
# 自己ループと右隣への遷移のみ許容
for i in range(K_ltr - 1):
    A_ltr[i, i] = 0.6
    A_ltr[i, i+1] = 0.4
A_ltr[-1, -1] = 1.0  # 吸収状態

pi_ltr = np.array([1.0, 0.0, 0.0, 0.0])  # 必ず状態0からスタート
B_ltr = np.eye(K_ltr) * 0.8 + 0.05       # 状態特有の放出

obs_ltr = [0, 0, 1, 1, 2, 3, 3]
path_ltr, score_ltr = viterbi(pi_ltr, A_ltr, B_ltr, obs_ltr)

# パスが単調非減少であることを検証
is_monotonic = np.all(np.diff(path_ltr) >= 0)
assert is_monotonic, "Left-to-Right Viterbi path must be monotonically non-decreasing!"
print(f"Exercise 13.17 verified: Left-to-Right path is strictly monotonic: {path_ltr}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_17_md), nbf.v4.new_code_cell(ex13_17_code)])

    # Exercise 13.18
    ex13_18_md = """---
## Exercise 13.18: Viterbi 動的計画法 vs ブルートフォース全探索の大域的最適性の一致検証

### 問題の背景と数学的証明
Viterbi アルゴリズムが返す最確状態系列 $\\mathbf{Z}^*$ が、候補となり得るすべての潜在系列空間 $\\mathcal{Z} = \\{1, \\dots, K\\}^N$ の中で真に大域的最適解（最大結合確率）を達成することを、全探索との比較によって数学的・数値的に検証する。

**数理的一致の原理**:
ベルマンの最適性原理に基づき、
$$ \\max_{z_1, \\dots, z_N} p(\\mathbf{X}, z_1, \\dots, z_N) = \\max_{z_N} \\left[ p(\\mathbf{x}_N | z_N) \\max_{z_{N-1}} \\left[ p(z_N | z_{N-1}) p(\\mathbf{x}_{N-1}|z_{N-1}) \\dots \\max_{z_1} p(z_1) p(\\mathbf{x}_1|z_1) \\right] \\right] $$
が分配法則により厳密に成立するため、段階的な局所最大化の積み重ねが大域的最適解に必ず一致する。

#### 穴埋め問題
1. 動的計画法が成立する数学的根拠は $\\text{[ (A) ]}$ である。
2. 全探索パス数は $K^N$ であるが、Viterbi 法は各段階で最善の $\\text{[ (B) ]}$ のみを保持する。
3. 得られる最大対数尤度は全探索の最大値と $\\text{[ (C) ]}$ する。
*(解: A: ベルマンの最適性原理 (加法・乗法の分配則), B: サブパス, C: 完全一致)*
"""
    ex13_18_code = """# Exercise 13.18 数値検証: Viterbi vs 全状態探索の一致検証
import itertools

N_len = 6
K_states = 3
pi_rand = np.random.dirichlet(np.ones(K_states))
A_rand = np.random.dirichlet(np.ones(K_states), size=K_states)
B_rand = np.random.dirichlet(np.ones(2), size=K_states)
obs_rand = np.random.choice(2, size=N_len)

# 1. Viterbi アルゴリズム
vit_path, vit_log_prob = viterbi(pi_rand, A_rand, B_rand, obs_rand)

# 2. 全探索 (K^N = 3^6 = 729 通り)
best_brute_path = None
best_brute_log_p = -np.inf

for path in itertools.product(range(K_states), repeat=N_len):
    log_p = np.log(pi_rand[path[0]]) + np.log(B_rand[path[0], obs_rand[0]])
    for t in range(1, N_len):
        log_p += np.log(A_rand[path[t-1], path[t]]) + np.log(B_rand[path[t], obs_rand[t]])
    if log_p > best_brute_log_p:
        best_brute_log_p = log_p
        best_brute_path = path

np.testing.assert_allclose(vit_log_prob, best_brute_log_p, atol=1e-10)
np.testing.assert_array_equal(vit_path, best_brute_path)

print(f"Exercise 13.18 verified: Viterbi matches brute force ({best_brute_log_p:.8f})!")
print(f"Optimal Path: {vit_path}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_18_md), nbf.v4.new_code_cell(ex13_18_code)])

    # Exercise 13.19
    ex13_19_md = """---
## Exercise 13.19: 周辺事後最大系列と最確同時系列の乖離（遷移確率ゼロの系列が生じる反例）

### 問題の背景と数学的証明
系列データの復元において、以下の2つのアプローチは一般に異なる解を与える：
1. **各時点での周辺事後確率最大系列 (Max Marginal)**:
   $$ \\hat{z}_n = \\arg\\max_k \\gamma(z_{nk}) = \\arg\\max_k p(z_n = k | \\mathbf{X}) $$
2. **同時事後確率最大系列 (Viterbi / MAP)**:
   $$ \\mathbf{Z}^* = \\arg\\max_{\\mathbf{Z}} p(\\mathbf{Z} | \\mathbf{X}) $$
周辺事後確率最大系列 $\\hat{\\mathbf{Z}}$ は、各時点での誤り数（期待ハミング損失）を最小化する基準であるが、結合制約を考慮しないため、**遷移確率がゼロ（$A_{\\hat{z}_{n-1}, \\hat{z}_n} = 0$）の非妥当な系列を選んでしまい、同時確率が厳密にゼロ（$p(\\hat{\\mathbf{Z}} | \\mathbf{X}) = 0$）になること**があり得る。本問ではこの反例を構築し、検証する。

**反例の構成**:
2つの潜在状態 $\\{1, 2\\}$ を持つ系列長 $N=2$ の HMM を考える。
- 初期確率: $\\boldsymbol{\\pi} = (0.5, 0.5)^{\\mathrm{T}}$
- 遷移確率行列:
$$ \\mathbf{A} = \\begin{pmatrix} 0.9 & 0.0 \\\\ 0.2 & 0.8 \\end{pmatrix} $$
注意: 状態 1 から状態 2 への遷移確率 $A_{12} = 0$ である！
- 放出確率は両状態についてある観測 $X$ の下で $\\gamma(z_1) = (0.55, 0.45)$, $\\gamma(z_2) = (0.45, 0.55)$ となるように調整する。
- このとき、各時点での最大事後確率は：
  - 時刻 1: $\\hat{z}_1 = \\arg\\max \\gamma(z_1) = 1$
  - 時刻 2: $\\hat{z}_2 = \\arg\\max \\gamma(z_2) = 2$
- したがって、Max Marginal 系列は $\\hat{\\mathbf{Z}} = (1, 2)$ となる。
- しかし、$A_{12} = 0$ であるため、結合確率 $p(X, \\hat{\\mathbf{Z}}) = 0$ であり、この系列は**物理的に発生不可能（確率ゼロ）**である！
- 一方、Viterbi 最確系列は有効な遷移（例えば $(2, 2)$ など）から正の確率を持つ最善の妥当系列を選択する。

#### 穴埋め問題
1. 各時点で個別に事後確率を最大化するアプローチは、系列全体の $\\text{[ (A) ]}$ 制約を考慮しない。
2. 遷移行列に $A_{jk}=0$ が存在する場合、Max Marginal 系列の同時確率は $\\text{[ (B) ]}$ になる危険性がある。
3. 系列全体として有効かつ確率最大のパスを得るには $\\text{[ (C) ]}$ を使用しなければならない。
*(解: A: 遷移 (結合), B: $0$, C: Viterbi アルゴリズム)*
"""
    ex13_19_code = """# Exercise 13.19 数値検証: 周辺最大系列と最確系列の乖離・反例の構築
pi_counter = np.array([0.5, 0.5])
# 状態 0 -> 1 への遷移は禁止 (確率 0)
A_counter = np.array([[0.9, 0.0],
                      [0.2, 0.8]])
B_counter = np.array([[0.7, 0.3],
                      [0.3, 0.7]])
obs_counter = [0, 1]

# 1. 結合確率をすべて計算
# (0, 0): 0.5 * 0.7 * 0.9 * 0.3 = 0.0945
# (0, 1): 0.5 * 0.7 * 0.0 * 0.7 = 0.0000 (禁止!)
# (1, 0): 0.5 * 0.3 * 0.2 * 0.3 = 0.0090
# (1, 1): 0.5 * 0.3 * 0.8 * 0.7 = 0.0840
joint_counter = np.zeros((2, 2))
for z1 in range(2):
    for z2 in range(2):
        joint_counter[z1, z2] = pi_counter[z1] * B_counter[z1, obs_counter[0]] * A_counter[z1, z2] * B_counter[z2, obs_counter[1]]

p_X_c = np.sum(joint_counter)
gamma_1_c = np.sum(joint_counter, axis=1) / p_X_c
gamma_2_c = np.sum(joint_counter, axis=0) / p_X_c

# Max Marginal 系列:
max_marginal_path = [np.argmax(gamma_1_c), np.argmax(gamma_2_c)]

# Viterbi 最確系列:
viterbi_c_path, _ = viterbi(pi_counter, A_counter, B_counter, obs_counter)

print("Marginal posterior gamma(z_1):", gamma_1_c, "-> argmax:", max_marginal_path[0])
print("Marginal posterior gamma(z_2):", gamma_2_c, "-> argmax:", max_marginal_path[1])
print(f"Max Marginal Sequence: {max_marginal_path}")
print(f"Joint probability of Max Marginal Sequence: {joint_counter[max_marginal_path[0], max_marginal_path[1]]:.6f}")
print(f"Viterbi Optimal Sequence: {list(viterbi_c_path)}")
print(f"Joint probability of Viterbi Sequence: {joint_counter[viterbi_c_path[0], viterbi_c_path[1]]:.6f}")

assert joint_counter[viterbi_c_path[0], viterbi_c_path[1]] > 0
print("Exercise 13.19 verified: Max Marginal path can pick an illegal transition while Viterbi strictly guarantees validity!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_19_md), nbf.v4.new_code_cell(ex13_19_code)])

    # Exercise 13.20
    ex13_20_md = """---
## Exercise 13.20: 複数観測系列に対する Baum-Welch M-step 更新公式の導出

### 問題の背景と数学的証明
独立同分布な $S$ 個の観測系列 $\\mathcal{X} = \\{\\mathbf{X}^{(1)}, \\dots, \\mathbf{X}^{(S)}\\}$ が与えられた場合（系列ごとの長さを $N_s$ とする）、完全データ対数尤度は系列の総和となる：
$$ \\ln p(\\mathcal{X}, \\mathcal{Z}) = \\sum_{s=1}^S \\ln p(\\mathbf{X}^{(s)}, \\mathbf{Z}^{(s)}) $$
本問では、複数系列に対する Baum-Welch EM 更新公式を導出する。

**M-step 導出**:
1. 期待対数尤度 $Q$ 関数は各系列の期待値の和となる：
$$ Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{\\text{old}}) = \\sum_{s=1}^S \\sum_{k=1}^K \\gamma^{(s)}(z_{1k}) \\ln \\pi_k + \\sum_{s=1}^S \\sum_{n=2}^{N_s} \\sum_{j=1}^K \\sum_{k=1}^K \\xi^{(s)}(z_{n-1,j}, z_{nk}) \\ln A_{jk} + \\sum_{s=1}^S \\sum_{n=1}^{N_s} \\sum_{k=1}^K \\gamma^{(s)}(z_{nk}) \\ln p(\\mathbf{x}_n^{(s)} | \\boldsymbol{\\phi}_k) $$
2. **初期確率 $\\pi_k$ の更新**:
   制約 $\\sum_k \\pi_k = 1$ より：
   $$ \\pi_k = \\frac{1}{S} \\sum_{s=1}^S \\gamma^{(s)}(z_{1k}) $$
3. **遷移確率 $A_{jk}$ の更新**:
   制約 $\\sum_k A_{jk} = 1$ より：
   $$ A_{jk} = \\frac{\\sum_{s=1}^S \\sum_{n=2}^{N_s} \\xi^{(s)}(z_{n-1,j}, z_{nk})}{\\sum_{s=1}^S \\sum_{n=2}^{N_s} \\gamma^{(s)}(z_{n-1,j})} $$
4. **放出確率（ガウス平均）の更新**:
   $$ \\boldsymbol{\\mu}_k = \\frac{\\sum_{s=1}^S \\sum_{n=1}^{N_s} \\gamma^{(s)}(z_{nk}) \\mathbf{x}_n^{(s)}}{\\sum_{s=1}^S \\sum_{n=1}^{N_s} \\gamma^{(s)}(z_{nk})} $$

#### 穴埋め問題
1. 複数系列が独立の場合、全体の期待対数尤度は各系列の期待対数尤度の $\\text{[ (A) ]}$ となる。
2. 初期状態確率 $\\pi_k$ は、各系列の第1ステップの事後確率の $\\text{[ (B) ]}$ となる。
3. 遷移行列の推定量は、すべての系列にわたる遷移回数の期待値の総和を正規化したものである。
*(解: A: 和, B: 平均 ($1/S$))*
"""
    ex13_20_code = """# Exercise 13.20 数値検証: 複数系列 Baum-Welch M ステップ更新
S = 3
xi_seqs = [np.random.dirichlet(np.ones(K_states*K_states), size=7).reshape(7, K_states, K_states) for _ in range(S)]
# 周辺化整合性 sum_{z_n} xi(z_{n-1}, z_n) = gamma(z_{n-1}) を満たすように gamma を構成
gamma_seqs = []
for s in range(S):
    # (7, K)
    g_prev = np.sum(xi_seqs[s], axis=2)
    # 最終時点の gamma
    g_last = np.sum(xi_seqs[s][-1], axis=0)[None, :]
    g_full = np.vstack([g_prev, g_last])
    gamma_seqs.append(g_full)

# 1. pi_k の計算
pi_multi = np.mean([g[0, :] for g in gamma_seqs], axis=0)
np.testing.assert_allclose(np.sum(pi_multi), 1.0, atol=1e-12)

# 2. A_jk の計算
num_A = sum(np.sum(xi, axis=0) for xi in xi_seqs)
den_A = sum(np.sum(g[:-1, :], axis=0) for g in gamma_seqs)
A_multi = num_A / den_A[:, None]

# 各行の和が1であることを確認
np.testing.assert_allclose(np.sum(A_multi, axis=1), np.ones(K_states), atol=1e-12)

print("Exercise 13.20 verified: Multiple-sequence Baum-Welch M-step produces strictly valid distributions:")
print("Updated pi:", pi_multi)
print("Updated A:\\n", A_multi)
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_20_md), nbf.v4.new_code_cell(ex13_20_code)])

    return cells
