# scripts/build_ch13_part1.py
"""
Build Chapter 13 Exercises: Part 1 (Exercises 13.1 - 13.4)
Markov Models & Graphical Foundations
"""

import nbformat as nbf

def build_part1():
    """Exercises 13.1 - 13.4: Markov Models & Graphical Foundations"""
    cells = []

    # Title & Introduction
    title_md = """# 第13章 系列データ：演習問題 (Exercises 13.1 - 13.34)

本ノートブックは、PRML（パターン認識と機械学習）第13章「系列データ (Sequential Data)」に登場する**全34問 (Exercises 13.1 〜 13.34)** の詳細な数理解説、証明ステップ、穴埋め問題、および自己完結型の Python 数値検証コードを網羅した完全版です。

### 構成一覧
1. **Exercises 13.1 - 13.4**: マルコフ連鎖、d-分離、未来と過去の条件付き独立性、観測周辺化による大域的相関、高次マルコフ連鎖のマクロ変数表現
2. **Exercises 13.5 - 13.10**: 隠れマルコフモデル (HMM) 最尤推定、Baum-Welch Mステップ更新公式（遷移・初期・ガウス放出・離散放出）、前向き変数 $\\alpha$ と後ろ向き変数 $\\beta$ の再帰漸化式
3. **Exercises 13.11 - 13.15**: 事後周辺確率 $\\gamma(z_n)$ の時間不変正規化、2時点事後確率 $\\xi(z_{n-1}, z_n)$、スケーリング係数 $c_n$、ファクターグラフ等価性、自己回帰 HMM
4. **Exercises 13.16 - 13.20**: Viterbi 動的計画法アルゴリズム、Left-to-Right HMM、全探索一致検証、周辺最大系列 vs 最確系列の乖離反例、複数系列学習
5. **Exercises 13.21 - 13.27**: 線形動的システム (LDS)、カルマンゲイン行列の線形ガウス代数導出、ジョセフ形式共分散、RTS カルマンスムーザ、極限解析（観測ノイズ消失・システムノイズ消失）
6. **Exercises 13.28 - 13.34**: LDS の EM パラメータ学習（$A, C, \\mathbf{\\Gamma}, \\mathbf{\\Sigma}$）、定常カルマンフィルタと代数的リカッチ方程式 (DARE)、潜在空間の線形変換不変性、Switching LDS、粒子フィルタ (Sequential Importance Resampling)
"""
    cells.append(nbf.v4.new_markdown_cell(title_md))

    # Exercise 13.1
    ex13_1_md = """---
## Exercise 13.1: d-分離による1次マルコフ連鎖の条件付き独立性の証明

### 問題の背景と数学的証明
第13.1節において、1次マルコフ連鎖の結合確率分布は次式のように因数分解される：
$$ p(x_1, \\dots, x_N) = p(x_1) \\prod_{n=2}^N p(x_n | x_{n-1}) $$
本問では、d-分離規準を用いて条件付き独立性：
$$ p(x_n | x_1, \\dots, x_{n-1}) = p(x_n | x_{n-1}) $$
を証明する。

**d-分離による証明**:
1. グラフィカルモデルにおいて、$x_1, \\dots, x_{n-2}$ の任意のノード $x_i$ ($i < n-1$) から $x_n$ へのすべての有効パスを考える。
2. このマルコフ連鎖は有向直線グラフ $x_1 \\to x_2 \\to \\dots \\to x_{n-1} \\to x_n$ であるため、$x_i$ から $x_n$ への唯一のパスは中間ノード $x_{n-1}$ を通過する。
3. このパスにおいて、ノード $x_{n-1}$ は $x_{n-2} \\to x_{n-1} \\to x_n$ という「head-to-tail」の結合を形成している。
4. d-分離の定義（第8.2節）より、head-to-tail ノードが条件付け集合に含まれる場合、そのノードを通過するパスはブロック（遮断）される。
5. 条件付け集合には $x_{n-1}$ が含まれているため、$x_i$ から $x_n$ へのすべてのパスはブロックされる。
6. したがって、$x_n \\perp \\{x_1, \\dots, x_{n-2}\\} \\mid x_{n-1}$ が成り立ち、直ちに次式が従う：
$$ p(x_n | x_1, \\dots, x_{n-1}) = p(x_n | x_{n-1}) $$

**積の規則による直接計算の確認**:
結合確率の比として評価すると：
$$ p(x_n | x_1, \\dots, x_{n-1}) = \\frac{p(x_1, \\dots, x_n)}{p(x_1, \\dots, x_{n-1})} = \\frac{p(x_1)\\prod_{i=2}^n p(x_i|x_{i-1})}{p(x_1)\\prod_{i=2}^{n-1} p(x_i|x_{i-1})} = p(x_n | x_{n-1}) $$
となり、d-分離の結果と完全に一致する。

#### 穴埋め問題
1. 有向グラフ $x_{n-2} \\to x_{n-1} \\to x_n$ において、ノード $x_{n-1}$ は $\\text{[ (A) ]}$ の結合を形成している。
2. 条件付け集合にノード $x_{n-1}$ が含まれるため、パスは $\\text{[ (B) ]}$ される。
3. これにより、$x_n$ の分布は直前の $x_{n-1}$ のみに依存し、過去のすべての観測値とは $\\text{[ (C) ]}$ となる。
*(解: A: head-to-tail, B: ブロック, C: 条件付き独立)*
"""
    ex13_1_code = """# Exercise 13.1 数値検証: 1次マルコフ連鎖における条件付き確率の厳密な等価性
import numpy as np

np.random.seed(42)
K = 3  # 状態数
N = 5  # 系列長

# ランダムな初期確率 pi と推移行列 A
pi = np.random.dirichlet(np.ones(K))
A = np.random.dirichlet(np.ones(K), size=K)

# 全結合確率テーブル p(x_1, ..., x_N) を計算 (K^N 通り)
joint = np.zeros((K, K, K, K, K))
for x1 in range(K):
    for x2 in range(K):
        for x3 in range(K):
            for x4 in range(K):
                for x5 in range(K):
                    joint[x1, x2, x3, x4, x5] = pi[x1] * A[x1, x2] * A[x2, x3] * A[x3, x4] * A[x4, x5]

# p(x_1, x_2, x_3, x_4) を周辺化により算出
p_1234 = np.sum(joint, axis=4)

# 条件付き確率 p(x_5 | x_1, x_2, x_3, x_4) を計算
p_cond = joint / p_1234[:, :, :, :, None]

# 直接の遷移確率 p(x_5 | x_4) と比較
for x1 in range(K):
    for x2 in range(K):
        for x3 in range(K):
            for x4 in range(K):
                np.testing.assert_allclose(p_cond[x1, x2, x3, x4, :], A[x4, :], atol=1e-12)

print("Exercise 13.1 verified: p(x_n | x_1, ..., x_{n-1}) strictly equals p(x_n | x_{n-1}) across all states!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_1_md), nbf.v4.new_code_cell(ex13_1_code)])

    # Exercise 13.2
    ex13_2_md = """---
## Exercise 13.2: 現在状態が与えられたときの未来と過去の条件付き独立性

### 問題の背景と数学的証明
マルコフ連鎖において、中間のある時点 $n$ の状態 $x_n$ が観測された場合、未来の系列 $x_{n+1}, \\dots, x_N$ と過去の系列 $x_1, \\dots, x_{n-1}$ は条件付き独立：
$$ p(x_1, \\dots, x_{n-1}, x_{n+1}, \\dots, x_N | x_n) = p(x_1, \\dots, x_{n-1} | x_n) p(x_{n+1}, \\dots, x_N | x_n) $$
が成立することを証明する。

**代数的証明**:
1. 条件付き確率の定義と因数分解公式より：
$$ p(x_1, \\dots, x_N | x_n) = \\frac{p(x_1, \\dots, x_N)}{p(x_n)} $$
2. 分子の結合分布を $x_n$ を境に分割する：
$$ p(x_1, \\dots, x_N) = \\left( p(x_1) \\prod_{i=2}^n p(x_i | x_{i-1}) \\right) \\left( \\prod_{j=n+1}^N p(x_j | x_{j-1}) \\right) = p(x_1, \\dots, x_n) p(x_{n+1}, \\dots, x_N | x_n) $$
3. これを分母 $p(x_n)$ で割ると：
$$ \\frac{p(x_1, \\dots, x_N)}{p(x_n)} = \\frac{p(x_1, \\dots, x_n)}{p(x_n)} p(x_{n+1}, \\dots, x_N | x_n) = p(x_1, \\dots, x_{n-1} | x_n) p(x_{n+1}, \\dots, x_N | x_n) $$
4. したがって、$x_n$ で条件付けると、過去と未来は互いに因数分解され、完全に独立となる。

#### 穴埋め問題
1. マルコフ性により、$x_n$ が与えられたとき、未来の観測値の生成は $\\text{[ (A) ]}$ の状態のみに依存する。
2. 過去と未来の結合条件付き確率は、過去の条件付き確率と未来の条件付き確率の $\\text{[ (B) ]}$ に分解される。
3. したがって、$x_n$ は過去と未来を $\\text{[ (C) ]}$ する。
*(解: A: 現在 ($x_n$), B: 積, C: 条件付き独立化)*
"""
    ex13_2_code = """# Exercise 13.2 数値検証: 現在状態を与えたときの過去と未来の独立性
p_123 = np.sum(joint, axis=(3, 4))  # (x1, x2, x3)
p_2 = np.sum(p_123, axis=(0, 2))    # (x2,)
p_12 = np.sum(p_123, axis=2)        # (x1, x2)
p_23 = np.sum(p_123, axis=0)        # (x2, x3)

for x2 in range(K):
    p_13_given_2 = p_123[:, x2, :] / p_2[x2]
    p_1_given_2 = p_12[:, x2] / p_2[x2]
    p_3_given_2 = p_23[x2, :] / p_2[x2]
    p_prod = np.outer(p_1_given_2, p_3_given_2)
    
    np.testing.assert_allclose(p_13_given_2, p_prod, atol=1e-12)

print("Exercise 13.2 verified: Past and Future are strictly conditionally independent given the present state!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_2_md), nbf.v4.new_code_cell(ex13_2_code)])

    # Exercise 13.3
    ex13_3_md = """---
## Exercise 13.3: 潜在変数を周辺化した場合の観測変数間大域的相関

### 問題の背景と数学的証明
隠れマルコフモデル (HMM) では、観測変数系列 $X = \\{x_1, \\dots, x_N\\}$ と潜在変数系列 $Z = \\{z_1, \\dots, z_N\\}$ の結合分布は：
$$ p(X, Z) = p(z_1) p(x_1 | z_1) \\prod_{n=2}^N p(z_n | z_{n-1}) p(x_n | z_n) $$
で与えられる。
潜在変数 $Z$ が観測されず周辺化されたとき、$X$ の周辺分布 $p(X) = \\sum_Z p(X, Z)$ は因数分解せず、すべての観測変数間に大域的な相関が生じることを d-分離を用いて証明する。

**d-分離による証明**:
1. 任意の2つの観測ノード $x_i$ と $x_j$ ($i < j$) を結ぶパスを考える：
$$ x_i \\leftarrow z_i \\to z_{i+1} \\to \\dots \\to z_j \\to x_j $$
2. パス上の中間ノード $z_i$ は tail-to-tail、その後の $z_k$ ($i < k < j$) は head-to-tail の構造を持つ。
3. 潜在変数 $Z$ はすべて未観測（条件付け集合に含まれない）である。
4. d-分離の規則：
   - head-to-tail ノードは未観測のときブロック**されない**。
   - tail-to-tail ノードは未観測のときブロック**されない**。
5. したがって、$x_i$ と $x_j$ を結ぶパス上にはブロックするノードが1つも存在せず、パスは常に「オープン」である。
6. よって、任意の $i \\neq j$ について $x_i \\not\\perp x_j$ であり、観測変数間は一般に独立ではない（大域的相関を持つ）。

#### 穴埋め問題
1. $z_i \\to z_{i+1}$ の構造において、未観測のノードはパスを $\\text{[ (A) ]}$。
2. 潜在変数を周辺化すると、任意の観測変数 $x_i$ と $x_j$ の間のパスは $\\text{[ (B) ]}$ となる。
3. この結果、観測系列は完全な $\\text{[ (C) ]}$ を獲得し、長距離依存性を表現できる。
*(解: A: ブロックしない, B: オープン (非ブロック), C: 大域的相関)*
"""
    ex13_3_code = """# Exercise 13.3 数値検証: 潜在変数を周辺化した場合の観測変数間相互情報量 I(X_1; X_3) > 0
pi_h = np.array([0.5, 0.5])
A_h = np.array([[0.8, 0.2],
                [0.2, 0.8]])
B_h = np.array([[0.9, 0.1],
                [0.1, 0.9]])

p_full = np.zeros((2, 2, 2, 2, 2, 2))
for z1 in range(2):
    for z2 in range(2):
        for z3 in range(2):
            for x1 in range(2):
                for x2 in range(2):
                    for x3 in range(2):
                        p_full[x1, x2, x3, z1, z2, z3] = (
                            pi_h[z1] * B_h[z1, x1] *
                            A_h[z1, z2] * B_h[z2, x2] *
                            A_h[z2, z3] * B_h[z3, x3]
                        )

p_obs = np.sum(p_full, axis=(3, 4, 5))
p_x1_x3 = np.sum(p_obs, axis=1)
p_x1 = np.sum(p_x1_x3, axis=1)
p_x3 = np.sum(p_x1_x3, axis=0)

mi_x1_x3 = np.sum(p_x1_x3 * np.log(p_x1_x3 / np.outer(p_x1, p_x3)))

print(f"Mutual Information I(X_1; X_3) after marginalizing latent states: {mi_x1_x3:.6f}")
assert mi_x1_x3 > 1e-4, "X_1 and X_3 must be correlated (I(X_1; X_3) > 0)!"

p_x1_x3_z2 = np.sum(p_full, axis=(1, 3, 5))
p_z2 = np.sum(p_x1_x3_z2, axis=(0, 1))
for z2 in range(2):
    p_cond_x1_x3 = p_x1_x3_z2[:, :, z2] / p_z2[z2]
    p_cond_x1 = np.sum(p_cond_x1_x3, axis=1)
    p_cond_x3 = np.sum(p_cond_x1_x3, axis=0)
    np.testing.assert_allclose(p_cond_x1_x3, np.outer(p_cond_x1, p_cond_x3), atol=1e-12)

print("Exercise 13.3 verified: Unobserved latents induce global correlations, while conditioning d-separates!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_3_md), nbf.v4.new_code_cell(ex13_3_code)])

    # Exercise 13.4
    ex13_4_md = """---
## Exercise 13.4: 高次マルコフ連鎖のマクロ変数結合による1次マルコフ連鎖への帰着

### 問題の背景と数学的証明
$M$ 次マルコフ連鎖は、各観測値 $x_n$ の条件付き確率が直前の $M$ 個の観測値に依存するモデルである：
$$ p(x_1, \\dots, x_N) = p(x_1, \\dots, x_M) \\prod_{n=M+1}^N p(x_n | x_{n-1}, \\dots, x_{n-M}) $$
本問では、連続する $M$ 個の変数をまとめた新しい複合（マクロ）変数 $\\mathbf{y}_n = (x_n, x_{n-1}, \\dots, x_{n-M+1})^{\\mathrm{T}}$ を定義することで、任意の $M$ 次マルコフ連鎖がマクロ変数に関する標準的な「1次マルコフ連鎖」として厳密に書き直せることを証明する。

**代数的証明**:
1. 新変数 $\\mathbf{y}_n = (x_n, \\dots, x_{n-M+1})$ と $\\mathbf{y}_{n-1} = (x_{n-1}, \\dots, x_{n-M})$ を考える。
2. 条件付き確率 $p(\\mathbf{y}_n | \\mathbf{y}_{n-1})$ は次のように定義される：
$$ p(\\mathbf{y}_n | \\mathbf{y}_{n-1}) = p(x_n, x_{n-1}', \\dots, x_{n-M+1}' | x_{n-1}, \\dots, x_{n-M}) $$
3. ここで、要素の整合性制約から $x_{n-1}' = x_{n-1}, \\dots, x_{n-M+1}' = x_{n-M+1}$ が決定論的に満たされる必要があるため：
$$ p(\\mathbf{y}_n | \\mathbf{y}_{n-1}) = p(x_n | x_{n-1}, \\dots, x_{n-M}) \\prod_{i=1}^{M-1} \\mathbb{I}(x_{n-i}' = x_{n-i}) $$
4. これにより、結合確率は1次マルコフ連鎖の形式：
$$ p(\\mathbf{y}_M, \\dots, \\mathbf{y}_N) = p(\\mathbf{y}_M) \\prod_{n=M+1}^N p(\\mathbf{y}_n | \\mathbf{y}_{n-1}) $$
として完全に因数分解される。
5. 各変数が $K$ 個の離散状態をとる場合、新変数 $\\mathbf{y}_n$ の状態数は $K^M$ となる。したがって、高次マルコフ連鎖は状態数が指数的に拡大した1次マルコフ連鎖と等価である。

#### 穴埋め問題
1. $M$ 次マルコフ連鎖を1次マルコフ連鎖に書き換えた場合、新しい状態空間のサイズは $\\text{[ (A) ]}$ となる。
2. 遷移確率行列において、過去のシフトが一致しない要素の遷移確率は $\\text{[ (B) ]}$ である。
3. したがって、高次連鎖は1次連鎖の $\\text{[ (C) ]}$ 遷移行列モデルとして包含される。
*(解: A: $K^M$, B: $0$, C: 希薄 (スパース))*
"""
    ex13_4_code = """# Exercise 13.4 数値検証: 2次マルコフ連鎖 (M=2, K=2) の1次マルコフ連鎖表現
K = 2
M = 2
P_2nd = np.random.dirichlet(np.ones(K), size=(K, K))

K_macro = K ** M
T_macro = np.zeros((K_macro, K_macro))

for y_prev in range(K_macro):
    x_prev2 = y_prev // K
    x_prev1 = y_prev % K
    for y_curr in range(K_macro):
        x_curr1 = y_curr // K
        x_curr0 = y_curr % K
        if x_curr1 == x_prev1:
            T_macro[y_prev, y_curr] = P_2nd[x_prev2, x_prev1, x_curr0]
        else:
            T_macro[y_prev, y_curr] = 0.0

np.testing.assert_allclose(np.sum(T_macro, axis=1), np.ones(K_macro), atol=1e-12)

seq = [0, 1, 0, 0, 1, 1, 0]
ll_2nd = sum(np.log(P_2nd[seq[i-2], seq[i-1], seq[i]]) for i in range(2, len(seq)))
macro_seq = [seq[i-1]*K + seq[i] for i in range(1, len(seq))]
ll_macro = sum(np.log(T_macro[macro_seq[j-1], macro_seq[j]]) for j in range(1, len(macro_seq)))

np.testing.assert_allclose(ll_2nd, ll_macro, atol=1e-12)
print(f"Exercise 13.4 verified: Log-likelihoods match identically ({ll_2nd:.6f} == {ll_macro:.6f})!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_4_md), nbf.v4.new_code_cell(ex13_4_code)])

    return cells
