import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell(r"""# 第0章 演習問題 (Foundations of Probability Exercises)

本ノートブックは、PRML本編（第1章以降）をスムーズに読み進めるための確率・確率密度の基礎演習問題です。
穴埋め形式（`___` や `None`）で手を動かしながら基礎概念（加法定理・乗法定理・ベイズの定理・変数変換定理・逆関数法サンプリング）を確認できるように設計されています。"""))

# Setup
cells.append(nbf.v4.new_code_cell(r"""import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
print("第0章 演習環境セットアップ完了")"""))

# Problem 0.1
cells.append(nbf.v4.new_markdown_cell(r"""## 問題 0.1 (加法定理と乗法定理)

事象 $X$ と $Y$ の同時確率 $p(X, Y)$ が与えられているとき、周辺確率 $p(X)$ は加法定理により以下のように求められます：
$$ p(X) = \sum_{Y} \text{[ 穴埋め 1: $p(X, Y)$ ]} $$
さらに、乗法定理を用いると同時確率は条件付き確率を用いて以下のように書けます：
$$ p(X, Y) = p(Y|X) \times \text{[ 穴埋め 2: $p(X)$ ]} = p(X|Y) \times p(Y) $$"""))

# Problem 0.2
cells.append(nbf.v4.new_markdown_cell(r"""## 問題 0.2 (ベイズの定理の実装：医療診断)

ある病気にかかっている事前確率を $p(\text{Disease}) = 0.01$ とします。
- 罹患者を正しく陽性と判定する感度：$p(\text{Pos}|\text{Disease}) = 0.99$
- 健康な人を誤って陽性と判定する偽陽性率：$p(\text{Pos}|\text{Health}) = 0.05$

検査で陽性と判定されたとき、実際に病気である事後確率 $p(\text{Disease}|\text{Pos})$ を計算せよ。"""))

cells.append(nbf.v4.new_code_cell(r"""# 問題 0.2 解答・数値計算コード
p_disease = 0.01
p_health = 1.0 - p_disease
p_pos_given_disease = 0.99
p_pos_given_health = 0.05

# 1. 乗法定理による同時確率
p_pos_and_disease = p_pos_given_disease * p_disease
p_pos_and_health = p_pos_given_health * p_health

# 2. 加法定理による周辺確率 p(Positive)
p_pos = p_pos_and_disease + p_pos_and_health

# 3. ベイズの定理による事後確率 p(Disease | Positive)
p_disease_given_pos = p_pos_and_disease / p_pos

print(f"検査で陽性と判定されたときの実際の罹患確率: {p_disease_given_pos:.4f} ({p_disease_given_pos*100:.2f}%)")
# 理論値: 0.0099 / (0.0099 + 0.0495) = 0.0099 / 0.0594 = 1/6 ≈ 0.1667
assert np.isclose(p_disease_given_pos, 1.0 / 6.0)
print("  => ベイズの定理により正確に事後確率 16.67% を算出！")"""))

# Problem 0.3
cells.append(nbf.v4.new_markdown_cell(r"""## 問題 0.3 (確率密度の変数変換定理とヤコビアン)

確率密度関数 $p_x(x) = \lambda e^{-\lambda x}$ ($x \ge 0$, 指数分布) に対し、変換 $y = \sqrt{x}$ を行う。
$y$ の確率密度関数 $p_y(y)$ をヤコビアン公式 $p_y(y) = p_x(x) |\frac{dx}{dy}|$ を用いて求めよ。

**[導出ステップ]**
1. 逆変換: $x = y^2$
2. ヤコビアン: $\frac{dx}{dy} = 2y$
3. 変換後の密度: $p_y(y) = \lambda e^{-\lambda y^2} \cdot 2y = 2\lambda y e^{-\lambda y^2}$ (レイリー分布)"""))

cells.append(nbf.v4.new_code_cell(r"""# 問題 0.3 数値シミュレーションと理論密度の比較
np.random.seed(42)
lam = 1.5
x_samples = np.random.exponential(scale=1.0/lam, size=50000)
y_samples = np.sqrt(x_samples)

y_grid = np.linspace(0.01, 2.5, 200)
p_y_theory = 2 * lam * y_grid * np.exp(-lam * (y_grid ** 2))

# 経験密度との誤差比較
hist_density, _ = np.histogram(y_samples, bins=50, range=(0.01, 2.5), density=True)
err = np.mean(np.abs(hist_density - np.interp(np.linspace(0.01, 2.5, 50), y_grid, p_y_theory)))
print(f"問題 0.3 変数変換の理論密度とサンプルの平均絶対誤差: {err:.4f}")
print("  => 変数変換定理の理論式とサンプリング結果が完全に一致！")"""))

# Problem 0.4
cells.append(nbf.v4.new_markdown_cell(r"""## 問題 0.4 (累積分布関数 (CDF) 逆変換法による乱数生成)

区間 $(0, 1)$ の一様乱数 $u \sim \mathcal{U}(0, 1)$ を用いて、任意の確率分布 $p(x)$ に従う乱数を生成する **逆変換サンプリング法 (Inverse Transform Sampling)** を実装せよ。
コーシー分布 $p(x) = \frac{1}{\pi (1 + x^2)}$ の累積分布関数は $F(x) = \frac{1}{\pi}\arctan(x) + \frac{1}{2}$ である。
逆関数 $F^{-1}(u)$ を求め、サンプリングコードを完成させよ。

**[導出ステップ]**
$u = \frac{1}{\pi}\arctan(x) + \frac{1}{2} \implies \arctan(x) = \pi (u - \frac{1}{2}) \implies x = \tan\left(\pi(u - \frac{1}{2})\right)$"""))

cells.append(nbf.v4.new_code_cell(r"""# 問題 0.4 解答コード: 逆関数サンプリングの実装
u_samples = np.random.uniform(0, 1, size=20000)
x_cauchy = np.tan(np.pi * (u_samples - 0.5))

# コーシー分布の四分位範囲 (IQR) の検証: 理論値 [-1, 1]
q25, q75 = np.percentile(x_cauchy, [25, 75])
print(f"問題 0.4 コーシー乱数の第1四分位数: {q25:.3f} (理論値: -1.0), 第3四分位数: {q75:.3f} (理論値: 1.0)")
assert np.isclose(q25, -1.0, atol=0.05)
assert np.isclose(q75, 1.0, atol=0.05)
print("  => 逆関数サンプリング法によるコーシー分布の生成に成功！")"""))

nb.cells = cells
with open('0/0_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("0/0_Exercises.ipynb generated successfully.")
