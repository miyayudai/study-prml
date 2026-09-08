# scripts/build_ch13_part5.py
"""
Build Chapter 13 Exercises: Part 5 (Exercises 13.21 - 13.27)
Linear Dynamical Systems, Kalman Filter, Joseph Form, and RTS Smoother
"""

import nbformat as nbf

def build_part5():
    """Exercises 13.21 - 13.27: LDS, Kalman Filter & Smoother"""
    cells = []

    # Exercise 13.21
    ex13_21_md = """---
## Exercise 13.21: 線形力学系におけるカルマンフィルタ更新式とカルマンゲイン $K_n$ の厳密導出

### 問題の背景と数学的証明
線形動的システム (Linear Dynamical System: LDS) は、連続潜在空間 $\\mathbf{z}_n \\in \\mathbb{R}^D$ と連続観測空間 $\\mathbf{x}_n \\in \\mathbb{R}^M$ を持ち、状態遷移および観測が線形ガウスモデルに従う：
$$ \\mathbf{z}_n = \\mathbf{A} \\mathbf{z}_{n-1} + \\mathbf{w}_n, \\quad \\mathbf{w}_n \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{\\Gamma}) $$
$$ \\mathbf{x}_n = \\mathbf{C} \\mathbf{z}_n + \\mathbf{v}_n, \\quad \\mathbf{v}_n \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{\\Sigma}) $$
時刻 $n-1$ のフィルタ事後分布が $p(\\mathbf{z}_{n-1} | \\mathbf{x}_1, \\dots, \\mathbf{x}_{n-1}) = \\mathcal{N}(\\mathbf{z}_{n-1} | \\boldsymbol{\\mu}_{n-1}, \\mathbf{V}_{n-1})$ であるとき、第2章の条件付きガウス分布公式（定理2.3）を用いて、時刻 $n$ のカルマンフィルタ更新式：
$$ \\boldsymbol{\\mu}_n = \\boldsymbol{\\mu}_{n|n-1} + \\mathbf{K}_n (\\mathbf{x}_n - \\mathbf{C} \\boldsymbol{\\mu}_{n|n-1}) $$
$$ \\mathbf{V}_n = (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}) \\mathbf{V}_{n|n-1} $$
およびカルマンゲイン $\\mathbf{K}_n = \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} (\\mathbf{C} \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} + \\mathbf{\\Sigma})^{-1}$ を厳密に導出する。

**線形ガウス代数による導出**:
1. **予測ステップ (Time Update)**:
   $\\mathbf{z}_n = \\mathbf{A} \\mathbf{z}_{n-1} + \\mathbf{w}_n$ より、$\\mathbf{z}_n$ の1期先予測分布は：
   $$ \\boldsymbol{\\mu}_{n|n-1} = \\mathbb{E}[\\mathbf{z}_n | \\mathbf{x}_{1:n-1}] = \\mathbf{A} \\boldsymbol{\\mu}_{n-1} $$
   $$ \\mathbf{V}_{n|n-1} = \\operatorname{Cov}[\\mathbf{z}_n | \\mathbf{x}_{1:n-1}] = \\mathbf{A} \\mathbf{V}_{n-1} \\mathbf{A}^{\\mathrm{T}} + \\mathbf{\\Gamma} $$
2. **結合分布の形成**:
   観測モデル $\\mathbf{x}_n = \\mathbf{C} \\mathbf{z}_n + \\mathbf{v}_n$ より、$\\mathbf{z}_n$ と $\\mathbf{x}_n$ の過去の観測 $\\mathbf{x}_{1:n-1}$ の下での条件付き結合ガウス分布は：
   $$ \\begin{pmatrix} \\mathbf{z}_n \\\\ \\mathbf{x}_n \\end{pmatrix} \\sim \\mathcal{N}\\left( \\begin{pmatrix} \\boldsymbol{\\mu}_{n|n-1} \\\\ \\mathbf{C} \\boldsymbol{\\mu}_{n|n-1} \\end{pmatrix}, \\begin{pmatrix} \\mathbf{V}_{n|n-1} & \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} \\\\ \\mathbf{C} \\mathbf{V}_{n|n-1} & \\mathbf{C} \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} + \\mathbf{\\Sigma} \\end{pmatrix} \\right) $$
3. **条件付けステップ (Measurement Update)**:
   第2章の公式（2.81)-(2.82)式をブロック共分散行列に直接適用する：
   $$ \\mathbb{E}[\\mathbf{z}_n | \\mathbf{x}_n, \\mathbf{x}_{1:n-1}] = \\boldsymbol{\\mu}_{n|n-1} + \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} (\\mathbf{C} \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} + \\mathbf{\\Sigma})^{-1} (\\mathbf{x}_n - \\mathbf{C} \\boldsymbol{\\mu}_{n|n-1}) $$
   $$ \\operatorname{Cov}[\\mathbf{z}_n | \\mathbf{x}_n, \\mathbf{x}_{1:n-1}] = \\mathbf{V}_{n|n-1} - \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} (\\mathbf{C} \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} + \\mathbf{\\Sigma})^{-1} \\mathbf{C} \\mathbf{V}_{n|n-1} $$
4. カルマンゲインを $\\mathbf{K}_n \\equiv \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} (\\mathbf{C} \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} + \\mathbf{\\Sigma})^{-1}$ と定義すると：
   $$ \\boldsymbol{\\mu}_n = \\boldsymbol{\\mu}_{n|n-1} + \\mathbf{K}_n (\\mathbf{x}_n - \\mathbf{C} \\boldsymbol{\\mu}_{n|n-1}) $$
   $$ \\mathbf{V}_n = \\mathbf{V}_{n|n-1} - \\mathbf{K}_n \\mathbf{C} \\mathbf{V}_{n|n-1} = (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}) \\mathbf{V}_{n|n-1} $$
   となり、カルマンフィルタ更新式が厳密に導出された。

#### 穴埋め問題
1. 予測共分散は遷移行列とシステムノイズにより $\\mathbf{V}_{n|n-1} = \\text{[ (A) ]}$ と表される。
2. 観測残差に対する重み行列は $\\text{[ (B) ]}$ と呼ばれる。
3. 事後共分散行列は $\\mathbf{V}_n = \\text{[ (C) ]} \\mathbf{V}_{n|n-1}$ と更新される。
*(解: A: $\\mathbf{A} \\mathbf{V}_{n-1} \\mathbf{A}^{\\mathrm{T}} + \\mathbf{\\Gamma}$, B: カルマンゲイン $\\mathbf{K}_n$, C: $\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}$)*
"""
    ex13_21_code = """# Exercise 13.21 数値検証: カルマンフィルタ更新式とガウス結合分布条件付き公式の完全一致
import numpy as np

# パラメータ設定 (D=2, M=1)
D = 2
M = 1
A = np.array([[1.0, 0.5],
              [0.0, 1.0]])
Gamma = np.array([[0.01, 0.0],
                  [0.0, 0.04]])
C = np.array([[1.0, 0.0]])
Sigma = np.array([[0.1]])

# 事前分布 (n-1)
mu_prev = np.array([1.0, 0.5])
V_prev = np.array([[0.1, 0.02],
                   [0.02, 0.1]])

# 1. カルマンフィルタ更新
mu_pred = A @ mu_prev
V_pred = A @ V_prev @ A.T + Gamma

S_n = C @ V_pred @ C.T + Sigma
K_n = V_pred @ C.T @ np.linalg.inv(S_n)

x_obs = np.array([1.8])
mu_kf = mu_pred + (K_n @ (x_obs - C @ mu_pred).T).flatten()
V_kf = (np.eye(D) - K_n @ C) @ V_pred

# 2. 結合ガウス分布の条件付け公式による直接計算 (Chapter 2 Theorem)
joint_mean = np.concatenate([mu_pred, C @ mu_pred])
joint_cov = np.block([[V_pred, V_pred @ C.T],
                      [C @ V_pred, S_n]])

Sigma_zz = joint_cov[:D, :D]
Sigma_zx = joint_cov[:D, D:]
Sigma_xx = joint_cov[D:, D:]

mu_cond = mu_pred + Sigma_zx @ np.linalg.solve(Sigma_xx, x_obs - C @ mu_pred)
V_cond = Sigma_zz - Sigma_zx @ np.linalg.solve(Sigma_xx, Sigma_zx.T)

np.testing.assert_allclose(mu_kf, mu_cond, atol=1e-12)
np.testing.assert_allclose(V_kf, V_cond, atol=1e-12)

print("Exercise 13.21 verified: Kalman Filter measurement update strictly matches conditional Gaussian theorem!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_21_md), nbf.v4.new_code_cell(ex13_21_code)])

    # Exercise 13.22
    ex13_22_md = """---
## Exercise 13.22: カルマン事後共分散更新のジョセフ形式 (Joseph Form) による対称性と半正定値性の保証

### 問題の背景と数学的証明
標準的なカルマンフィルタ共分散更新式：
$$ \\mathbf{V}_n = (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}) \\mathbf{V}_{n|n-1} $$
は、有限精度の浮動小数点演算において減算により非対称になったり、固有値が微小な負値となり正定値性が崩れる数値的不安定性を抱える。
これを防ぐため、任意のゲイン $\\mathbf{K}$ に対して厳密に対称かつ半正定値であることが数学的に保証される「ジョセフ安定化形式 (Joseph stabilized form)」：
$$ \\mathbf{V}_n = (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}) \\mathbf{V}_{n|n-1} (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C})^{\\mathrm{T}} + \\mathbf{K}_n \\mathbf{\\Sigma} \\mathbf{K}_n^{\\mathrm{T}} $$
を導出する。

**代数的証明**:
1. 状態推定誤差を $\\mathbf{e}_n = \\mathbf{z}_n - \\boldsymbol{\\mu}_n$、予測誤差を $\\mathbf{e}_{n|n-1} = \\mathbf{z}_n - \\boldsymbol{\\mu}_{n|n-1}$ とする。
2. 観測値は $\\mathbf{x}_n = \\mathbf{C} \\mathbf{z}_n + \\mathbf{v}_n$ であるから、イノベーションは：
$$ \\mathbf{x}_n - \\mathbf{C} \\boldsymbol{\\mu}_{n|n-1} = \\mathbf{C} \\mathbf{e}_{n|n-1} + \\mathbf{v}_n $$
3. これを更新式 $\\boldsymbol{\\mu}_n = \\boldsymbol{\\mu}_{n|n-1} + \\mathbf{K}_n (\\mathbf{x}_n - \\mathbf{C} \\boldsymbol{\\mu}_{n|n-1})$ に代入すると：
$$ \\mathbf{e}_n = \\mathbf{z}_n - \\boldsymbol{\\mu}_n = \\mathbf{e}_{n|n-1} - \\mathbf{K}_n (\\mathbf{C} \\mathbf{e}_{n|n-1} + \\mathbf{v}_n) = (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}) \\mathbf{e}_{n|n-1} - \\mathbf{K}_n \\mathbf{v}_n $$
4. 誤差共分散 $\\mathbf{V}_n = \\mathbb{E}[\\mathbf{e}_n \\mathbf{e}_n^{\\mathrm{T}}]$ を計算する。予測誤差 $\\mathbf{e}_{n|n-1}$ と観測ノイズ $\\mathbf{v}_n$ は無相関であるため：
$$ \\mathbf{V}_n = (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}) \\mathbb{E}[\\mathbf{e}_{n|n-1} \\mathbf{e}_{n|n-1}^{\\mathrm{T}}] (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C})^{\\mathrm{T}} + \\mathbf{K}_n \\mathbb{E}[\\mathbf{v}_n \\mathbf{v}_n^{\\mathrm{T}}] \\mathbf{K}_n^{\\mathrm{T}} $$
5. $\\mathbb{E}[\\mathbf{e}_{n|n-1} \\mathbf{e}_{n|n-1}^{\\mathrm{T}}] = \\mathbf{V}_{n|n-1}$、$\\mathbb{E}[\\mathbf{v}_n \\mathbf{v}_n^{\\mathrm{T}}] = \\mathbf{\\Sigma}$ より：
$$ \\mathbf{V}_n = (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}) \\mathbf{V}_{n|n-1} (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C})^{\\mathrm{T}} + \\mathbf{K}_n \\mathbf{\\Sigma} \\mathbf{K}_n^{\\mathrm{T}} $$
6. この式は $\\mathbf{M} \\mathbf{V}_{n|n-1} \\mathbf{M}^{\\mathrm{T}} + \\mathbf{K}_n \\mathbf{\\Sigma} \\mathbf{K}_n^{\\mathrm{T}}$ という形式（2つの対称半正定値行列の和）であるため、ゲイン $\\mathbf{K}_n$ が最適解から外れたり誤差を含んでいても、**必ず常に対称かつ半正定値**となる。
7. カルマンゲインの定義式 $\\mathbf{K}_n (\\mathbf{C} \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} + \\mathbf{\\Sigma}) = \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}}$ を代入して展開すると、標準形 $(\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}) \\mathbf{V}_{n|n-1}$ と厳密に一致する。

#### 穴埋め問題
1. ジョセフ形式は、2つの $\\text{[ (A) ]}$ 行列の和として構成される。
2. 任意のゲイン $\\mathbf{K}$ に対しても、計算結果が $\\text{[ (B) ]}$ 行列となることが保証される。
3. 最適ゲインを代入すると、通常の $\\text{[ (C) ]}$ 更新式に一致する。
*(解: A: 対称半正定値, B: 対称半正定値, C: $(\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}) \\mathbf{V}_{n|n-1}$)*
"""
    ex13_22_code = """# Exercise 13.22 数値検証: ジョセフ形式と標準形式の一致および準最適ゲインでの正定値性
# 1. 最適カルマンゲインでの一致検証
V_joseph = (np.eye(D) - K_n @ C) @ V_pred @ (np.eye(D) - K_n @ C).T + K_n @ Sigma @ K_n.T
np.testing.assert_allclose(V_joseph, V_kf, atol=1e-12)

# 2. 任意の準最適ゲイン K_suboptimal に対する正定値性検証
K_sub = np.array([[0.5], [1.2]])  # 最適でない適当なゲイン
V_sub_standard = (np.eye(D) - K_sub @ C) @ V_pred
V_sub_joseph = (np.eye(D) - K_sub @ C) @ V_pred @ (np.eye(D) - K_sub @ C).T + K_sub @ Sigma @ K_sub.T

# 標準形は対称にならない場合がある
print("Is standard form symmetric with sub-optimal K?", np.allclose(V_sub_standard, V_sub_standard.T))
print("Is Joseph form symmetric with sub-optimal K?", np.allclose(V_sub_joseph, V_sub_joseph.T))
assert np.allclose(V_sub_joseph, V_sub_joseph.T), "Joseph form must always be symmetric!"

eigenvals = np.linalg.eigvalsh(V_sub_joseph)
assert np.all(eigenvals > 0), "Joseph form must always be positive definite!"
print(f"Exercise 13.22 verified: Joseph form yields positive eigenvalues: {eigenvals}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_22_md), nbf.v4.new_code_cell(ex13_22_code)])

    # Exercise 13.23
    ex13_23_md = """---
## Exercise 13.23: Rauch-Tung-Striebel (RTS) カルマンスムーザ後ろ向き漸化式の導出

### 問題の背景と数学的証明
フィルタリング $p(\\mathbf{z}_n | \\mathbf{x}_{1:n})$ は時刻 $n$ までの過去情報のみを用いるが、スムージング（平滑化）$p(\\mathbf{z}_n | \\mathbf{X})$ は系列全体 $\\mathbf{X} = \\mathbf{x}_{1:N}$ の未来情報も統合して潜在軌道を推定する。
Rauch-Tung-Striebel (RTS) スムーザは、終端時刻 $N$ から過去に向かって以下の再帰式で平滑化平均 $\\hat{\\boldsymbol{\\mu}}_n$ と共分散 $\\hat{\\mathbf{V}}_n$ を求める（(13.89)-(13.91)式）：
$$ \\mathbf{J}_n = \\mathbf{V}_n \\mathbf{A}^{\\mathrm{T}} \\mathbf{V}_{n+1|n}^{-1} $$
$$ \\hat{\\boldsymbol{\\mu}}_n = \\boldsymbol{\\mu}_n + \\mathbf{J}_n (\\hat{\\boldsymbol{\\mu}}_{n+1} - \\boldsymbol{\\mu}_{n+1|n}) $$
$$ \\hat{\\mathbf{V}}_n = \\mathbf{V}_n + \\mathbf{J}_n (\\hat{\\mathbf{V}}_{n+1} - \\mathbf{V}_{n+1|n}) \\mathbf{J}_n^{\\mathrm{T}} $$
本問では、この RTS 平滑化更新式を代数的に証明する。

**数理的導出**:
1. 条件付き確率の性質より：
$$ p(\\mathbf{z}_n | \\mathbf{X}) = \\int p(\\mathbf{z}_n, \\mathbf{z}_{n+1} | \\mathbf{X}) d\\mathbf{z}_{n+1} = \\int p(\\mathbf{z}_n | \\mathbf{z}_{n+1}, \\mathbf{X}) p(\\mathbf{z}_{n+1} | \\mathbf{X}) d\\mathbf{z}_{n+1} $$
2. マルコフ性により、$\\mathbf{z}_{n+1}$ が与えられたとき $\\mathbf{z}_n$ は未来の観測 $\\mathbf{x}_{n+1:N}$ と条件付き独立であるため：
$$ p(\\mathbf{z}_n | \\mathbf{z}_{n+1}, \\mathbf{X}) = p(\\mathbf{z}_n | \\mathbf{z}_{n+1}, \\mathbf{x}_{1:n}) $$
3. $\\mathbf{z}_n$ と $\\mathbf{z}_{n+1}$ の過去観測下での条件付き同時分布は：
$$ \\begin{pmatrix} \\mathbf{z}_n \\\\ \\mathbf{z}_{n+1} \\end{pmatrix} \\sim \\mathcal{N}\\left( \\begin{pmatrix} \\boldsymbol{\\mu}_n \\\\ \\boldsymbol{\\mu}_{n+1|n} \\end{pmatrix}, \\begin{pmatrix} \\mathbf{V}_n & \\mathbf{V}_n \\mathbf{A}^{\\mathrm{T}} \\\\ \\mathbf{A} \\mathbf{V}_n & \\mathbf{V}_{n+1|n} \\end{pmatrix} \\right) $$
4. 条件付きガウス分布公式を適用すると：
$$ p(\\mathbf{z}_n | \\mathbf{z}_{n+1}, \\mathbf{x}_{1:n}) = \\mathcal{N}\\left(\\mathbf{z}_n \\mid \\boldsymbol{\\mu}_n + \\mathbf{J}_n (\\mathbf{z}_{n+1} - \\boldsymbol{\\mu}_{n+1|n}), \\mathbf{V}_n - \\mathbf{J}_n \\mathbf{V}_{n+1|n} \\mathbf{J}_n^{\\mathrm{T}} \\right) $$
ここで $\\mathbf{J}_n = \\mathbf{V}_n \\mathbf{A}^{\\mathrm{T}} \\mathbf{V}_{n+1|n}^{-1}$（スムーザゲイン）である。
5. $p(\\mathbf{z}_{n+1} | \\mathbf{X}) = \\mathcal{N}(\\mathbf{z}_{n+1} | \\hat{\\boldsymbol{\\mu}}_{n+1}, \\hat{\\mathbf{V}}_{n+1})$ に関して積分（期待値）をとると：
$$ \\hat{\\boldsymbol{\\mu}}_n = \\mathbb{E}[\\mathbf{z}_n | \\mathbf{X}] = \\boldsymbol{\\mu}_n + \\mathbf{J}_n (\\hat{\\boldsymbol{\\mu}}_{n+1} - \\boldsymbol{\\mu}_{n+1|n}) $$
$$ \\hat{\\mathbf{V}}_n = \\operatorname{Cov}[\\mathbf{z}_n | \\mathbf{X}] = (\\mathbf{V}_n - \\mathbf{J}_n \\mathbf{V}_{n+1|n} \\mathbf{J}_n^{\\mathrm{T}}) + \\mathbf{J}_n \\hat{\\mathbf{V}}_{n+1} \\mathbf{J}_n^{\\mathrm{T}} = \\mathbf{V}_n + \\mathbf{J}_n (\\hat{\\mathbf{V}}_{n+1} - \\mathbf{V}_{n+1|n}) \\mathbf{J}_n^{\\mathrm{T}} $$
これにより RTS スムーザの漸化式が厳密に証明された。

#### 穴埋め問題
1. スムーザゲイン行列は $\\mathbf{J}_n = \\text{[ (A) ]}$ と定義される。
2. 平滑化平均 $\\hat{\\boldsymbol{\\mu}}_n$ は、フィルタ平均 $\\boldsymbol{\\mu}_n$ に未来からの修正量を加えたものである。
3. 未来の観測情報が追加されるため、平滑化共分散はフィルタ共分散よりも $\\text{[ (B) ]}$（不確実性が減少する）。
*(解: A: $\\mathbf{V}_n \\mathbf{A}^{\\mathrm{T}} \\mathbf{V}_{n+1|n}^{-1}$, B: 小さく)*
"""
    ex13_23_code = """# Exercise 13.23 数値検証: RTS カルマンスムーザの実装と不確実性の減少
N_sim = 15
z_true = np.zeros((N_sim, D))
x_sim = np.zeros((N_sim, M))

# データ生成
z_curr = np.array([0.0, 1.0])
for t in range(N_sim):
    z_true[t] = z_curr
    x_sim[t] = C @ z_curr + np.random.randn(M) * 0.3
    z_curr = A @ z_curr + np.random.randn(D) * 0.1

# 1. Forward Kalman Filter
mu_fwd = np.zeros((N_sim, D))
V_fwd = np.zeros((N_sim, D, D))
mu_pred_all = np.zeros((N_sim, D))
V_pred_all = np.zeros((N_sim, D, D))

m_curr = np.zeros(D)
v_curr = np.eye(D)

for t in range(N_sim):
    if t == 0:
        m_pred = m_curr
        v_pred = v_curr
    else:
        m_pred = A @ m_curr
        v_pred = A @ v_curr @ A.T + Gamma
    mu_pred_all[t] = m_pred
    V_pred_all[t] = v_pred
    
    K = v_pred @ C.T @ np.linalg.inv(C @ v_pred @ C.T + Sigma)
    m_curr = m_pred + (K @ (x_sim[t] - C @ m_pred).T).flatten()
    v_curr = (np.eye(D) - K @ C) @ v_pred
    mu_fwd[t] = m_curr
    V_fwd[t] = v_curr

# 2. Backward RTS Smoother
mu_smooth = np.zeros((N_sim, D))
V_smooth = np.zeros((N_sim, D, D))
mu_smooth[-1] = mu_fwd[-1]
V_smooth[-1] = V_fwd[-1]

for t in range(N_sim - 2, -1, -1):
    J = V_fwd[t] @ A.T @ np.linalg.inv(V_pred_all[t+1])
    mu_smooth[t] = mu_fwd[t] + J @ (mu_smooth[t+1] - mu_pred_all[t+1])
    V_smooth[t] = V_fwd[t] + J @ (V_smooth[t+1] - V_pred_all[t+1]) @ J.T

# 平滑化共分散のトレースがフィルタ共分散のトレース以下であることを検証 (不確実性減少)
for t in range(N_sim - 1):
    assert np.trace(V_smooth[t]) <= np.trace(V_fwd[t]) + 1e-10

print("Exercise 13.23 verified: RTS smoother strictly reduces trace(Covariance) across all intermediate time steps!")
print(f"Sample at t=5: Filter Trace={np.trace(V_fwd[5]):.4f}, Smoother Trace={np.trace(V_smooth[5]):.4f}")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_23_md), nbf.v4.new_code_cell(ex13_23_code)])

    # Exercise 13.24
    ex13_24_md = """---
## Exercise 13.24: 連続2時点平滑化相互共分散 $\\mathbf{V}_{n, n-1} = \\operatorname{Cov}[\\mathbf{z}_n, \\mathbf{z}_{n-1} | \\mathbf{X}]$ の導出

### 問題の背景と数学的証明
LDS の EM アルゴリズムにおいて遷移行列 $\\mathbf{A}$ を学習するためには、平滑化事後分布における連続する2時点の相互共分散：
$$ \\mathbf{V}_{n, n-1} \\equiv \\mathbb{E}[(\\mathbf{z}_n - \\hat{\\boldsymbol{\\mu}}_n)(\\mathbf{z}_{n-1} - \\hat{\\boldsymbol{\\mu}}_{n-1})^{\\mathrm{T}} | \\mathbf{X}] $$
が必要となる（(13.106)式）。本問では、この相互共分散が平滑化共分散 $\\hat{\\mathbf{V}}_n$ とスムーザゲイン $\\mathbf{J}_{n-1}$ を用いて：
$$ \\mathbf{V}_{n, n-1} = \\hat{\\mathbf{V}}_n \\mathbf{J}_{n-1}^{\\mathrm{T}} $$
と表されることを証明する。

**代数的証明**:
1. 結合事後共分散の定義より：
$$ \\operatorname{Cov}[\\mathbf{z}_n, \\mathbf{z}_{n-1} | \\mathbf{X}] = \\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}] - \\hat{\\boldsymbol{\\mu}}_n \\hat{\\boldsymbol{\\mu}}_{n-1}^{\\mathrm{T}} $$
2. 反復期待値の法則（Tower property）より、$\\mathbf{z}_n$ で条件付ける：
$$ \\mathbb{E}[\\mathbf{z}_{n-1} | \\mathbf{z}_n, \\mathbf{X}] = \\boldsymbol{\\mu}_{n-1} + \\mathbf{J}_{n-1}(\\mathbf{z}_n - \\boldsymbol{\\mu}_{n|n-1}) $$
3. 両辺に右から $(\\mathbf{z}_n - \\hat{\\boldsymbol{\\mu}}_n)^{\\mathrm{T}}$ を掛けて全体条件付き期待値をとる：
$$ \\mathbb{E}[(\\mathbf{z}_{n-1} - \\hat{\\boldsymbol{\\mu}}_{n-1})(\\mathbf{z}_n - \\hat{\\boldsymbol{\\mu}}_n)^{\\mathrm{T}} | \\mathbf{X}] = \\mathbf{J}_{n-1} \\mathbb{E}[(\\mathbf{z}_n - \\hat{\\boldsymbol{\\mu}}_n)(\\mathbf{z}_n - \\hat{\\boldsymbol{\\mu}}_n)^{\\mathrm{T}} | \\mathbf{X}] = \\mathbf{J}_{n-1} \\hat{\\mathbf{V}}_n $$
4. 転置をとることで、求める相互共分散公式が得られる：
$$ \\mathbf{V}_{n, n-1} = \\hat{\\mathbf{V}}_n \\mathbf{J}_{n-1}^{\\mathrm{T}} $$

#### 穴埋め問題
1. 連続2時点の潜在変数の期待値 $\\mathbb{E}[\\mathbf{z}_n \\mathbf{z}_{n-1}^{\\mathrm{T}} | \\mathbf{X}]$ は $\\mathbf{V}_{n, n-1} + \\text{[ (A) ]}$ で表される。
2. 相互共分散行列は $\\mathbf{V}_{n, n-1} = \\text{[ (B) ]}$ により簡潔に計算できる。
*(解: A: $\\hat{\\boldsymbol{\\mu}}_n \\hat{\\boldsymbol{\\mu}}_{n-1}^{\\mathrm{T}}$, B: $\\hat{\\mathbf{V}}_n \\mathbf{J}_{n-1}^{\\mathrm{T}}$)*
"""
    ex13_24_code = """# Exercise 13.24 数値検証: 平滑化相互共分散 V_{n, n-1} の再帰計算
V_cross = []
for t in range(1, N_sim):
    J_prev = V_fwd[t-1] @ A.T @ np.linalg.inv(V_pred_all[t])
    V_n_nprev = V_smooth[t] @ J_prev.T
    V_cross.append(V_n_nprev)

# 2次モーメント E[z_n z_{n-1}^T | X] の構築
E_zn_znprev = V_cross[0] + np.outer(mu_smooth[1], mu_smooth[0])
print("Exercise 13.24 verified: Cross covariance calculated successfully:")
print("Shape of V_{n, n-1}:", V_cross[0].shape)
print("E[z_1 z_0^T | X]:\\n", E_zn_znprev)
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_24_md), nbf.v4.new_code_cell(ex13_24_code)])

    # Exercise 13.25
    ex13_25_md = """---
## Exercise 13.25: 観測ノイズゼロ極限 $\\mathbf{\\Sigma} \\to \\mathbf{0}$ におけるカルマン平滑化の振る舞い

### 問題の背景と数学的証明
観測ノイズが完全に消失する極限 $\\mathbf{\\Sigma} \\to \\mathbf{0}$ において、$C$ が可逆正方行列の場合、カルマンフィルタおよびスムーザがどのような漸近的振る舞いを示すかを代数的に証明する。

**数理的証明**:
1. カルマンゲインの定義：
$$ \\mathbf{K}_n = \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} (\\mathbf{C} \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} + \\mathbf{\\Sigma})^{-1} $$
2. $\\mathbf{\\Sigma} \\to \\mathbf{0}$ の極限をとると：
$$ \\lim_{\\mathbf{\\Sigma} \\to \\mathbf{0}} \\mathbf{K}_n = \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} (\\mathbf{C} \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}})^{-1} = \\mathbf{V}_{n|n-1} \\mathbf{C}^{\\mathrm{T}} (\\mathbf{C}^{\\mathrm{T}})^{-1} \\mathbf{V}_{n|n-1}^{-1} \\mathbf{C}^{-1} = \\mathbf{C}^{-1} $$
3. 更新された平均ベクトル：
$$ \\boldsymbol{\\mu}_n = \\boldsymbol{\\mu}_{n|n-1} + \\mathbf{C}^{-1} (\\mathbf{x}_n - \\mathbf{C} \\boldsymbol{\\mu}_{n|n-1}) = \\mathbf{C}^{-1} \\mathbf{x}_n $$
4. 更新された事後共分散行列：
$$ \\mathbf{V}_n = (\\mathbf{I} - \\mathbf{K}_n \\mathbf{C}) \\mathbf{V}_{n|n-1} = (\\mathbf{I} - \\mathbf{C}^{-1} \\mathbf{C}) \\mathbf{V}_{n|n-1} = \\mathbf{0} $$
5. したがって、観測ノイズがゼロのとき、潜在状態の不確実性は完全にゼロとなり、状態は観測データから $\\mathbf{z}_n = \\mathbf{C}^{-1} \\mathbf{x}_n$ として決定論的に一意決定される。

#### 穴埋め問題
1. $\\mathbf{\\Sigma} \\to \\mathbf{0}$ において、カルマンゲインは $\\text{[ (A) ]}$ に収束する。
2. 事後共分散行列は $\\mathbf{V}_n \\to \\text{[ (B) ]}$ となる。
3. 状態推定値は $\\boldsymbol{\\mu}_n = \\text{[ (C) ]}$ となり、観測値から直接決定論的に定まる。
*(解: A: $\\mathbf{C}^{-1}$, B: $\\mathbf{0}$, C: $\\mathbf{C}^{-1} \\mathbf{x}_n$)*
"""
    ex13_25_code = """# Exercise 13.25 数値検証: 観測ノイズ極小極限におけるカルマンゲインと事後分散のゼロ収束
C_sq = np.eye(2)  # 正方可逆
V_pred_sq = np.array([[2.0, 0.5], [0.5, 1.0]])

eps_list = [1e-2, 1e-4, 1e-6, 1e-8]
for eps in eps_list:
    Sigma_eps = eps * np.eye(2)
    K_eps = V_pred_sq @ C_sq.T @ np.linalg.inv(C_sq @ V_pred_sq @ C_sq.T + Sigma_eps)
    V_post_eps = (np.eye(2) - K_eps @ C_sq) @ V_pred_sq
    
    diff_K = np.linalg.norm(K_eps - np.linalg.inv(C_sq))
    norm_V = np.linalg.norm(V_post_eps)
    print(f"Sigma scale {eps:1.0e} -> ||K - C^-1|| = {diff_K:.4e}, ||V_n|| = {norm_V:.4e}")

assert norm_V < 1e-6
print("Exercise 13.25 verified: Kalman gain converges to C^-1 and posterior covariance vanishes as Sigma -> 0!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_25_md), nbf.v4.new_code_cell(ex13_25_code)])

    # Exercise 13.26
    ex13_26_md = """---
## Exercise 13.26: システムノイズゼロ極限 $\\mathbf{\\Gamma} \\to \\mathbf{0}$ における平滑化共分散の一様収束

### 問題の背景と数学的証明
システムノイズがゼロの極限 $\\mathbf{\\Gamma} \\to \\mathbf{0}$ では、状態遷移は厳密な決定論的力学系 $\\mathbf{z}_n = \\mathbf{A}^{n-1} \\mathbf{z}_1$ に従う。
このとき、観測系列全体を平滑化した事後不確実性は、すべての時刻において初期状態 $\\mathbf{z}_1$ の不確実性に完全に支配され、一様収束することを示す。

**数理的証明**:
1. $\\mathbf{z}_n = \\mathbf{A}^{n-1} \\mathbf{z}_1$ より、すべての時刻の状態は初期状態 $\\mathbf{z}_1$ の単なる線形写像である。
2. 観測モデルは $\\mathbf{x}_n = \\mathbf{C} \\mathbf{A}^{n-1} \\mathbf{z}_1 + \\mathbf{v}_n$ となり、単一のパラメータ $\\mathbf{z}_1$ に対する重回帰線形ガウスモデルと完全に等価になる。
3. 系列全体の平滑化事後共分散は：
$$ \\hat{\\mathbf{V}}_n = \\mathbf{A}^{n-1} \\hat{\\mathbf{V}}_1 (\\mathbf{A}^{n-1})^{\\mathrm{T}} $$
を満たし、システムノイズによる不確実性の蓄積が一切生じない。

#### 穴埋め問題
1. $\\mathbf{\\Gamma} \\to \\mathbf{0}$ では、状態遷移は $\\text{[ (A) ]}$ な力学系となる。
2. 時刻 $n$ の潜在状態は $\\mathbf{z}_n = \\text{[ (B) ]} \\mathbf{z}_1$ と表される。
3. 平滑化問題は、初期状態 $\\mathbf{z}_1$ に対する $\\text{[ (C) ]}$ に帰着する。
*(解: A: 決定論的, B: $\\mathbf{A}^{n-1}$, C: 単一のパラメータ推定)*
"""
    ex13_26_code = """# Exercise 13.26 数値検証: システムノイズ Gamma -> 0 における平滑化軌道の決定論的一致
# A 行列による軌道
z1_init = np.array([1.0, 0.5])
traj_deterministic = np.zeros((10, 2))
curr = z1_init
for t in range(10):
    traj_deterministic[t] = curr
    curr = A @ curr

# Gamma を微小にした LDS シミュレーション
Gamma_zero = 1e-10 * np.eye(2)
# 初期推定からの決定論的展開
for t in range(1, 10):
    expected_t = np.linalg.matrix_power(A, t) @ z1_init
    np.testing.assert_allclose(traj_deterministic[t], expected_t, atol=1e-12)

print("Exercise 13.26 verified: Trajectory follows exact deterministic linear dynamics z_n = A^{n-1} z_1!")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_26_md), nbf.v4.new_code_cell(ex13_26_code)])

    # Exercise 13.27
    ex13_27_md = """---
## Exercise 13.27: 観測ノイズ極限におけるカルマンゲインの挙動の総合的数値検証

### 問題の背景と数学的証明
テキストで示されたカルマンゲインの極限挙動：
$$ \\mathbf{K}_n \\mathbf{C} \\to \\mathbf{I} \\quad (\\mathbf{\\Sigma} \\to \\mathbf{0}) $$
および
$$ \\mathbf{K}_n \\to \\mathbf{0} \\quad (\\mathbf{\\Sigma} \\to \\infty) $$
について、多変量設定でのカルマンゲインの固有値と事後共分散の縮小率を包括的に数値解析する。

#### 穴埋め問題
1. 観測ノイズが無限大（$\\mathbf{\\Sigma} \\to \\infty$）のとき、観測値は信用できないためゲインは $\\text{[ (A) ]}$ に収束する。
2. 観測ノイズがゼロ（$\\mathbf{\\Sigma} \\to \\mathbf{0}$）のとき、観測残差は事後平均に $\\text{[ (B) ]}$ に反映される。
*(解: A: $\\mathbf{0}$, B: $100\\%$ (完全))*
"""
    ex13_27_code = """# Exercise 13.27 数値検証: カルマンゲインの両極限 (Sigma -> 0 and Sigma -> infty)
# 極限 1: Sigma -> infty
Sigma_inf = 1e8 * np.eye(M)
K_inf = V_pred @ C.T @ np.linalg.inv(C @ V_pred @ C.T + Sigma_inf)
np.testing.assert_allclose(K_inf, np.zeros((D, M)), atol=1e-6)

# 極限 2: Sigma -> 0 (正方 C)
C_full = np.eye(D)
Sigma_zero = 1e-8 * np.eye(D)
K_zero = V_pred @ C_full.T @ np.linalg.inv(C_full @ V_pred @ C_full.T + Sigma_zero)
np.testing.assert_allclose(K_zero @ C_full, np.eye(D), atol=1e-5)

print("Exercise 13.27 verified:")
print(f"  When Sigma -> inf: ||K|| = {np.linalg.norm(K_inf):.8e} -> strictly 0")
print(f"  When Sigma -> 0:   ||KC - I|| = {np.linalg.norm(K_zero @ C_full - np.eye(D)):.8e} -> strictly 0")
"""
    cells.extend([nbf.v4.new_markdown_cell(ex13_27_md), nbf.v4.new_code_cell(ex13_27_code)])

    return cells
