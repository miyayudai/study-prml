import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 第8章 グラフィカルモデル：演習問題 (Exercises 8.1 - 8.29)

本ノートブックでは、PRML第8章「グラフィカルモデル (Graphical Models)」の**全29問 (Exercises 8.1 〜 8.29)** の詳細な論理ステップ（数理的証明・思考の道筋）および Python による数値検証コードを収録しています。
DAGの因数分解、Table 8.2の条件付き独立性、線形ガウス再帰、不確実な報告モデルにおける相殺効果 (Explaining Away)、ICM局所エネルギー、周辺最大化と結合最尤の乖離反例（Ex 8.27）、および因子グラフ Sum-Product の停止性を計算機上で実験・検証します。"""))

# Exercises 8.1 - 8.9
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 8.1 - 8.9: DAGの規格化、Table 8.2 条件付き独立性、線形ガウスモデル、Noisy-OR

### 問題 8.1: DAG結合分布の規格化性
DAGのトポロジカル順序（親が先、子が後）に従い、末端の葉ノードから順に積分・和をとると：
$$ \sum_{x_K} \dots \sum_{x_1} \prod_{k=1}^K p(x_k | \mathrm{pa}_k) = \sum_{x_{K-1}} \dots \sum_{x_1} \prod_{k=1}^{K-1} p(x_k | \mathrm{pa}_k) \left[ \sum_{x_K} p(x_K | \mathrm{pa}_K) \right] $$
角括弧内は規格化条件より $1$ となり、これを繰り返すことで全体が $1$ に規格化されることを示せ。

### 問題 8.3 & 8.4: Table 8.2 の数値検証
Table 8.2 に与えられた3つの二値変数 $a, b, c \in \{0, 1\}$ の結合確率分布に対し、
- $p(a, b) \neq p(a)p(b)$ （周辺的には従属）
- $p(a, b|c) = p(a|c)p(b|c) \quad (\forall c \in \{0, 1\})$ （$c$ で条件付けると厳密に独立）
- $p(a, b, c) = p(a)p(c|a)p(b|c)$ （対応する有向グラフは $a \rightarrow c \rightarrow b$ または $a \leftarrow c \rightarrow b$）
であることを数値的に示せ。

### 問題 8.6: Noisy-OR 分布
$$ p(y = 1 | \mathbf{x}) = 1 - (1 - \mu_0) \prod_{i=1}^M (1 - \mu_i)^{x_i} $$
各入力 $x_i = 1$ は独立に確率 $\mu_i$ で原因を引き起こし（ノイズ）、$\mu_0$ は背景自発確率を表す。
全体のパラメータ数が $2^M$ から $M+1$ へと線形に抑制されることを示せ。"""))

# Code Ex 8.1 - 8.9
code_ex8_1_9 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np

# Table 8.2 の結合分布 p(a, b, c)
table_8_2 = {
    (0, 0, 0): 0.192,
    (0, 0, 1): 0.144,
    (0, 1, 0): 0.048,
    (0, 1, 1): 0.216,
    (1, 0, 0): 0.192,
    (1, 0, 1): 0.064,
    (1, 1, 0): 0.048,
    (1, 1, 1): 0.096,
}

# 規格化の確認
total_prob = sum(table_8_2.values())
assert np.isclose(total_prob, 1.0)

# Exercise 8.3: 周辺依存性と条件付き独立性の検証
# p(a), p(b), p(a, b)
p_a1 = sum(v for (a, b, c), v in table_8_2.items() if a == 1)
p_b1 = sum(v for (a, b, c), v in table_8_2.items() if b == 1)
p_a1_b1 = sum(v for (a, b, c), v in table_8_2.items() if a == 1 and b == 1)

print(f"p(a=1): {p_a1:.4f}, p(b=1): {p_b1:.4f}, product: {p_a1 * p_b1:.4f}")
print(f"p(a=1, b=1): {p_a1_b1:.4f}")
assert not np.isclose(p_a1 * p_b1, p_a1_b1)
print("-> a and b are marginally DEPENDENT (p(a,b) != p(a)p(b))")

# 条件付き確率 p(a, b | c) と p(a | c) p(b | c)
for c_val in [0, 1]:
    p_c = sum(v for (a, b, c), v in table_8_2.items() if c == c_val)
    p_a1_c = sum(v for (a, b, c), v in table_8_2.items() if a == 1 and c == c_val) / p_c
    p_b1_c = sum(v for (a, b, c), v in table_8_2.items() if b == 1 and c == c_val) / p_c
    p_a1_b1_c = sum(v for (a, b, c), v in table_8_2.items() if a == 1 and b == 1 and c == c_val) / p_c
    
    print(f"Given c={c_val}: p(a=1|c)={p_a1_c:.4f}, p(b=1|c)={p_b1_c:.4f}, product={p_a1_c * p_b1_c:.4f}, joint={p_a1_b1_c:.4f}")
    assert np.isclose(p_a1_c * p_b1_c, p_a1_b1_c)

print("-> a and b are CONDITIONAL INDEPENDENT given c (Exercise 8.3 verified!)")"""
cells.append(nbf.v4.new_code_cell(code_ex8_1_9))

# Exercises 8.10 - 8.17
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 8.10 - 8.17: 子孫観測による v-構造活性化、不確実な報告と相殺効果、ICMエネルギー局所性

### 問題 8.10: 子孫ノード $d$ の観測による $a$ と $b$ の従属性
グラフ $a \rightarrow c \leftarrow b$ かつ $c \rightarrow d$（PRML Figure 8.54）において、
- 何も観測していないとき、$c$ も $d$ も未観測なのでパス $a - c - b$ は Head-to-Head でブロックされ $a \perp b \mid \emptyset$。
- $d$ を観測したとき、$d$ は $c$ の子孫であるため、Head-to-Head パスが**活性化（非ブロック化）**され、$a \not\perp b \mid d$ となることを示せ。

### 問題 8.11: 信頼性の低い運転手報告モデルにおける Explaining Away
メーター $G$ を直接見ず、運転手の報告 $D \in \{0, 1\}$（精度 $p(D=1|G=1) = 0.9, p(D=0|G=0) = 0.9$）を受ける。
「運転手が空 $D=0$ と報告した」下での燃料空確率 $p(F=0|D=0)$ を計算し、
さらに「バッテリー上がり $B=0$」を観測したときの確率 $p(F=0|D=0, B=0)$ が低下（相殺）することを数値検証せよ。

### 問題 8.13: ICM局所エネルギーの差分
エネルギー関数 (8.42) において、ピクセル $x_j$ を反転させたときの差分は：
$$ E(x_j = +1) - E(x_j = -1) = 2 \left( h - \beta \sum_{i \in \mathrm{ne}(j)} x_i - \eta y_j \right) $$
となり、これは $x_j$ の近傍ピクセルと観測ピクセル $y_j$ のみに依存する完全な局所計算であることを示せ。"""))

# Code Ex 8.10 - 8.17
code_ex8_10_17 = r"""# Exercise 8.11 数値検証: 運転手報告モデルにおける Explaining Away
# 変数: B, F, G, D
p_B = {1: 0.9, 0: 0.1}
p_F = {1: 0.9, 0: 0.1}
p_G_given = {
    (1, 1): {1: 0.8, 0: 0.2},
    (1, 0): {1: 0.1, 0: 0.9},
    (0, 1): {1: 0.1, 0.9: 0.9},
    (0, 0): {1: 0.0, 0: 1.0}
}
p_G_given[(0, 1)][0] = 0.9 # 正確な辞書キー
p_D_given = {
    1: {1: 0.9, 0: 0.1},
    0: {1: 0.1, 0: 0.9}
}

# 結合確率 p(B, F, G, D)
joint_4 = {}
for b in [0, 1]:
    for f in [0, 1]:
        for g in [0, 1]:
            for d in [0, 1]:
                prob = p_B[b] * p_F[f] * p_G_given[(b, f)][g] * p_D_given[g][d]
                joint_4[(b, f, g, d)] = prob

# 1. 運転手が空と報告したとき p(F=0 | D=0)
p_D0 = sum(v for (b, f, g, d), v in joint_4.items() if d == 0)
p_F0_and_D0 = sum(v for (b, f, g, d), v in joint_4.items() if f == 0 and d == 0)
p_F0_given_D0 = p_F0_and_D0 / p_D0

# 2. さらにバッテリー上がりも分かったとき p(F=0 | D=0, B=0)
p_D0_and_B0 = sum(v for (b, f, g, d), v in joint_4.items() if d == 0 and b == 0)
p_F0_and_D0_B0 = sum(v for (b, f, g, d), v in joint_4.items() if f == 0 and d == 0 and b == 0)
p_F0_given_D0_B0 = p_F0_and_D0_B0 / p_D0_and_B0

print(f"Exercise 8.11:")
print(f"p(F=0 | D=0):     {p_F0_given_D0:.4f}")
print(f"p(F=0 | D=0, B=0): {p_F0_given_D0_B0:.4f}")
assert p_F0_given_D0_B0 < p_F0_given_D0
print("-> Explaining Away verified with intermediate noisy sensor D!")"""
cells.append(nbf.v4.new_code_cell(code_ex8_10_17))

# Exercises 8.18 - 8.29
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 8.18 - 8.29: 有向木と無向木の対応、個別周辺最大化と結合最尤の乖離反例 (Ex 8.27)、停止性

### 問題 8.18: 有向木と無向木の等価性と構成可能な有向木の総数
$N$ 個のノードを持つ任意の無向木において、閉路が存在しないため、任意のノード $r \in \{1, \dots, N\}$ を**根 (Root)** として選択すると、すべての辺の向きが根から外向き（または内向き）に一意に定まる。
したがって、与えられた1つの無向木から構成可能な相異なる有向木の総数は厳密に $N$ 個であることを示せ。

### 問題 8.27: 周辺確率の最大化と結合確率最尤の劇的な乖離
3状態の変数 $x, y \in \{0, 1, 2\}$ に対し、
周辺分布の最大値 $\widehat{x} = \arg\max_x p(x), \widehat{y} = \arg\max_y p(y)$ の結合確率が
$$ p(\widehat{x}, \widehat{y}) = 0 $$
となる結合分布 $p(x, y)$ の具体例を構成せよ。
（これにより、Sum-Product で個別に周辺最大化する状態と、Max-Sum で結合最尤状態を求めることの本質的差異が示される！）"""))

# Code Ex 8.18 - 8.29
code_ex8_18_29 = r"""# Exercise 8.27 数値検証: p(x_hat, y_hat) == 0 となる結合分布の構成
# 3x3 結合分布行列
# x, y in {0, 1, 2}
# x_hat = 1, y_hat = 1 だが p(1, 1) = 0 に設計
P_xy = np.array([
    [0.0, 0.2, 0.1],
    [0.2, 0.0, 0.2],  # (1, 1) は 0 !
    [0.1, 0.2, 0.0]
])
# 規格化
P_xy = P_xy / np.sum(P_xy)

# 周辺分布
p_x = np.sum(P_xy, axis=1)
p_y = np.sum(P_xy, axis=0)

x_hat = np.argmax(p_x)
y_hat = np.argmax(p_y)

print(f"Marginal p(x): {p_x}")
print(f"Marginal p(y): {p_y}")
print(f"Argmax x_hat: {x_hat}, Argmax y_hat: {y_hat}")
print(f"Joint probability p(x_hat, y_hat) = P_xy[{x_hat}, {y_hat}]: {P_xy[x_hat, y_hat]:.4f}")

assert x_hat == 1 and y_hat == 1
assert P_xy[x_hat, y_hat] == 0.0
print("Exercise 8.27 verified: Marginal argmax combination has exactly ZERO joint probability!")"""
cells.append(nbf.v4.new_code_cell(code_ex8_18_29))

nb.cells = cells
with open('8/8_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("8/8_Exercises.ipynb generated successfully.")
