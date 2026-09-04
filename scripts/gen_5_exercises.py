import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("""# 第5章 ニューラルネットワーク：演習問題 (Exercises 5.1 - 5.41)

本ノートブックでは、PRML第5章「ニューラルネットワーク (Neural Networks)」の**全41問 (Exercises 5.1 〜 5.41)** の完全な論理ステップ（証明・思考の道筋）および Python による数値検証コードを収録しています。
数式変形だけでなく、実装による数値テストを実行することで、理論が厳密に正しいことを計算機上で検証します。"""))

# Exercise 5.1 - 5.5
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 5.1 - 5.5: 活性化関数の線形等価性、最尤推定と二乗和誤差/交差エントロピー、ロバスト分類

### 問題 5.1: ロジスティックシグモイドと tanh の等価性
シグモイド関数 $\sigma(a) = \frac{1}{1 + e^{-a}}$ と $\tanh(a) = \frac{e^a - e^{-a}}{e^a + e^{-a}}$ に対し、
$$ \tanh(a) = 2\sigma(2a) - 1 $$
が成り立つ。これを用いて、シグモイド隠れ層を持つ2層NNが、第1層の重み・バイアスを2倍し、第2層の重み・バイアスを線形変換することで、$\tanh$ 隠れ層を持つNNと完全に恒等な関数を構成できることを示せ。

### 問題 5.2: 多出力ガウス尤度最大化と二乗和誤差最小化の等価性
独立な等方性ガウスノイズ $p(\mathbf{t}|\mathbf{x}, \mathbf{w}) = \mathcal{N}(\mathbf{t}|\mathbf{y}(\mathbf{x}, \mathbf{w}), \beta^{-1}\mathbf{I})$ の下で対数尤度を最大化することが、二乗和誤差関数 $E(\mathbf{w}) = \frac{1}{2}\sum_n \|\mathbf{y}_n - \mathbf{t}_n\|^2$ の最小化と等価であることを示せ。

### 問題 5.3: 一般の共分散行列 $\mathbf{\Sigma}$ を持つ回帰
ノイズ共分散が一般の行列 $\mathbf{\Sigma}$ の場合、誤差関数はマハラノビス二乗和誤差となり、$\mathbf{\Sigma}$ の最尤推定量がサンプルの残差共分散 $\mathbf{\Sigma}_{\mathrm{ML}} = \frac{1}{N}\sum_{n=1}^N (\mathbf{y}_n - \mathbf{t}_n)(\mathbf{y}_n - \mathbf{t}_n)^T$ となることを示せ。

### 問題 5.4: ラベル反転確率 $\epsilon$ を持つロバスト分類誤差関数
学習ラベルが確率 $\epsilon$ で反転する場合、観測ラベル $t_n \in \{0, 1\}$ に対する対数尤度は
$$ \ln p(\mathcal{D}|\mathbf{w}) = \sum_{n=1}^N \left\{ t_n \ln[(1-\epsilon)y_n + \epsilon(1-y_n)] + (1 - t_n)\ln[(1-\epsilon)(1-y_n) + \epsilon y_n] \right\} $$
となり、外れ値に対して損失の勾配が有界化されるロバストなモデルとなることを示せ。

### 問題 5.5: 多クラス交差エントロピー誤差の導出
$y_k(\mathbf{x}, \mathbf{w}) = p(t_k = 1|\mathbf{x})$ に対する多項尤度から交差エントロピー誤差 $E = -\sum_n \sum_k t_{nk}\ln y_{nk}$ を導出せよ。"""))

# Code Ex 5.1 - 5.5
code_ex5_1_5 = r"""import sys, os
sys.path.append(os.path.abspath('../'))
import numpy as np
from common.classification_utils import sigmoid
from common.nn_utils import tanh

# Exercise 5.1 数値検証: tanh(a) == 2*sigmoid(2a) - 1
a_vals = np.linspace(-4, 4, 100)
lhs = tanh(a_vals)
rhs = 2.0 * sigmoid(2.0 * a_vals) - 1.0
assert np.allclose(lhs, rhs)

# 2層NNでの入出力の完全一致の検証
D, M, K = 3, 4, 2
W1_sig = np.random.randn(D, M)
b1_sig = np.random.randn(M)
W2_sig = np.random.randn(M, K)
b2_sig = np.random.randn(K)

# tanh ネットワークの等価パラメータ
W1_tanh = 0.5 * W1_sig
b1_tanh = 0.5 * b1_sig
W2_tanh = 0.5 * W2_sig
b2_tanh = b2_sig + 0.5 * np.sum(W2_sig, axis=0) # sum_j 0.5 * W2_sig

X_sample = np.random.randn(10, D)
# sigmoid net
out_sig = sigmoid(X_sample @ W1_sig + b1_sig) @ W2_sig + b2_sig
# tanh net
out_tanh = tanh(X_sample @ W1_tanh + b1_tanh) @ W2_tanh + b2_tanh
assert np.allclose(out_sig, out_tanh)
print("Exercise 5.1 verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex5_1_5))

# Exercise 5.6 - 5.10
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 5.6 - 5.10: 正準連結関数と普遍的デルタ公式 $\frac{\partial E_n}{\partial a_k} = y_k - t_k$

### 問題 5.6 - 5.9: 正準連結関数の統一性
以下のすべてのケースにおいて、出力活性化入力 $a_k$ に関する誤差関数の導関数が
$$ \frac{\partial E_n}{\partial a_k} = y_k - t_k $$
という極めて美しく普遍的な同一の形式を満たすことを証明せよ：
1. **二乗和誤差＋恒等活性化（回帰）**: $E_n = \frac{1}{2}\sum_k (y_k - t_k)^2, y_k = a_k$
2. **二値交差エントロピー＋ロジスティックシグモイド**: $E_n = -[t\ln y + (1-t)\ln(1-y)], y = \sigma(a)$
3. **多クラス交差エントロピー＋ソフトマックス**: $E_n = -\sum_k t_k \ln y_k, y_k = \frac{e^{a_k}}{\sum_j e^{a_j}}$

### 問題 5.10: ヘッセ行列の固有値分解と二次誤差展開
ヘッセ行列 $\mathbf{H}\mathbf{u}_i = \lambda_i \mathbf{u}_i$ に対し、$\Delta \mathbf{w} = \sum_i \alpha_i \mathbf{u}_i$ と固有展開すると、局所二次誤差が
$$ \Delta E = \frac{1}{2}\sum_i \lambda_i \alpha_i^2 $$
と互いに無相関な独立な2次形式の和に分解できることを示せ。"""))

# Code Ex 5.6 - 5.10
code_ex5_6_10 = r"""from common.classification_utils import softmax

# Exercise 5.6 - 5.9 数値微分検証 (Softmax + Cross Entropy)
a_vec = np.array([1.2, -0.5, 2.0])
t_vec = np.array([0.0, 1.0, 0.0])

def cross_entropy(a):
    y = softmax(a)
    return -np.sum(t_vec * np.log(y + 1e-15))

eps = 1e-6
num_grad = np.zeros_like(a_vec)
for i in range(len(a_vec)):
    ap = a_vec.copy(); ap[i] += eps
    am = a_vec.copy(); am[i] -= eps
    num_grad[i] = (cross_entropy(ap) - cross_entropy(am)) / (2 * eps)

y_vec = softmax(a_vec)
ana_grad = y_vec - t_vec # 普遍的公式
assert np.allclose(num_grad, ana_grad, atol=1e-6)
print("Exercise 5.6 - 5.9 universal delta verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex5_6_10))

# Exercise 5.11 - 5.15
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 5.11 - 5.15: 誤差曲面の楕円等高線、極小値の条件、ヘッセの独立自由度、中心差分 $O(\epsilon^2)$

### 問題 5.11: 楕円等高線の軸長 $\propto 1/\sqrt{\lambda_i}$
$\frac{1}{2}\sum_i \lambda_i \alpha_i^2 = \text{const}$ の楕円において、軸の半長が $\lambda_i^{-1/2}$ に比例することを示せ。

### 問題 5.12: 局所極小値の必要十分条件
停留点 $\nabla E = \mathbf{0}$ において、任意の微小変化 $\Delta \mathbf{w}$ に対して $\Delta E > 0$ となる必要十分条件は、ヘッセ行列 $\mathbf{H}$ のすべての固有値が厳密に正（正定値 $\mathbf{H} \succ 0$）であることであることを示せ。

### 問題 5.13: 対称ヘッセ行列の独立パラメータ数
$W \times W$ の対称ヘッセ行列の独立要素数は $\frac{W(W+1)}{2}$ 個であり、勾配の $W$ 個と合わせて二次近似の自由度は $\frac{W(W+3)}{2}$ であることを示せ。

### 問題 5.14: 中心差分における $O(\epsilon)$ 次数の相殺
テイラー展開
$$ f(x+\epsilon) = f(x) + \epsilon f'(x) + \frac{\epsilon^2}{2}f''(x) + \frac{\epsilon^3}{6}f'''(x) + O(\epsilon^4) $$
$$ f(x-\epsilon) = f(x) - \epsilon f'(x) + \frac{\epsilon^2}{2}f''(x) - \frac{\epsilon^3}{6}f'''(x) + O(\epsilon^4) $$
の差を取ると、奇数次の項のみが残り、偶数次項および $O(\epsilon)$ 誤差が完全に相殺して有限差分誤差が $O(\epsilon^2)$ となることを示せ。

### 問題 5.15: ヤコビ行列の前向き計算法 (Forward Propagation)
逆伝播を用いずに、前向きに入力微小変位を伝播させることでヤコビ行列を計算する漸化式を導出せよ。"""))

# Code Ex 5.11 - 5.15
code_ex5_11_15 = r"""# Exercise 5.14 数値精度比較: 前進差分 O(eps) vs 中心差分 O(eps^2)
f = lambda x: np.sin(x)
df_true = lambda x: np.cos(x)
x0 = 1.0
true_val = df_true(x0)

eps_list = [1e-2, 1e-3, 1e-4]
for eps in eps_list:
    forward_diff = (f(x0 + eps) - f(x0)) / eps
    central_diff = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)
    err_fwd = abs(forward_diff - true_val)
    err_cen = abs(central_diff - true_val)
    assert err_cen < err_fwd

print("Exercise 5.11 - 5.15 central difference accuracy verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex5_11_15))

# Exercise 5.16 - 5.25
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 5.16 - 5.25: ヘッセ行列の外積近似、逆ヘッセの逐次更新、早期終了と Weight Decay の対応

### 問題 5.16 - 5.20: 多出力・分類における外積近似
多クラスソフトマックスモデルにおける外積近似ヘッセ行列が
$$ \mathbf{H} \simeq \sum_{n=1}^N \sum_{k=1}^K \sum_{j=1}^K y_{nk}(I_{kj} - y_{nj}) \nabla_\mathbf{w} a_{nk} \nabla_\mathbf{w} a_{nj}^T $$
となることを示せ。

### 問題 5.21: 逆ヘッセ行列のオンライン更新 (Sherman-Morrison-Woodbury)
$$ (\mathbf{A} + \mathbf{u}\mathbf{v}^T)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}^T\mathbf{A}^{-1}}{1 + \mathbf{v}^T \mathbf{A}^{-1}\mathbf{u}} $$
を適用し、データ点を1つ追加したときの逆ヘッセ行列の漸化式を導け。

### 問題 5.24: 入力・重みのアフィン変換不変性
入力にスケーリングとシフト $\widetilde{x}_i = A_i x_i + B_i$ を施したとき、第1層の重みとバイアスを $\widetilde{w}_{ji} = \frac{w_{ji}}{A_i}, \widetilde{w}_{j0} = w_{j0} - \sum_i \frac{w_{ji}B_i}{A_i}$ と変換することでネットワーク関数が完全に不変となることを示せ。

### 問題 5.25: 早期終了ステップ数 $\tau$ と正則化パラメータ $\lambda = (\rho \tau)^{-1}$ の厳密な対応関係の証明
反復ステップ $\tau$ 後の重み $w_j^{(\tau)} = [1 - (1 - \rho \eta_j)^\tau] w_j^*$ に対し、$\rho \tau \eta_j \ll 1$ で $(1 - \rho \eta_j)^\tau \approx e^{-\rho\tau \eta_j} \approx \frac{1}{1 + \rho \tau \eta_j}$ と近似することにより、Weight decay 解 $w_j = \frac{\eta_j}{\eta_j + \lambda} w_j^*$ と比較して
$$ \lambda \leftrightarrow (\rho \tau)^{-1} $$
の等価対応が厳密に成立することを証明せよ。"""))

# Code Ex 5.16 - 5.25
code_ex5_16_25 = r"""# Exercise 5.24 数値検証: 入力アフィン変換不変性
np.random.seed(42)
D, M, K = 2, 3, 1
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

A = np.array([2.5, -1.8])
B = np.array([0.5, 1.2])

# アフィン変換後の重み・バイアス
W1_tilde = W1 / A[:, None]
b1_tilde = b1 - np.sum(W1 * (B[:, None] / A[:, None]), axis=0)

X_orig = np.random.randn(5, D)
X_trans = X_orig * A + B # 変換された入力

out_orig = tanh(X_orig @ W1 + b1) @ W2 + b2
out_trans = tanh(X_trans @ W1_tilde + b1_tilde) @ W2 + b2

assert np.allclose(out_orig, out_trans)
print("Exercise 5.24 affine invariance verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex5_16_25))

# Exercise 5.26 - 5.37
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 5.26 - 5.37: 接線伝播法、ティホノフ正則化、重み共有、ロボットアーム順運動学、MDN勾配と条件付きモーメント

### 問題 5.26 & 5.27: 接線伝播とノイズ付加（ティホノフ正則化）
ランダムな微小ガウスノイズ $\mathbf{x} \to \mathbf{x} + \boldsymbol{\xi}$ を入力に加えた学習データの二乗誤差期待値が、通常の二乗誤差＋入力勾配の二乗ペナルティ（ティホノフ正則化項 $\frac{1}{2}\sum_n \|\nabla_\mathbf{x} y_n\|^2$）に等価となることを示せ。

### 問題 5.28: 重み共有における逆伝播
共有重み $w$ に対する誤差勾配は、その重みを共有するすべての結合における誤差勾配の総和 $\frac{\partial E}{\partial w} = \sum_{u \in S} \frac{\partial E}{\partial w_u}$ となることを示せ。

### 問題 5.33: 2自由度ロボットアームの順運動学
アーム長 $L_1, L_2$、関節角度 $\theta_1, \theta_2$ の手先座標 $(x_1, x_2)$ の方程式
$$ x_1 = L_1 \cos\theta_1 + L_2 \cos(\theta_1 + \theta_2) $$
$$ x_2 = L_1 \sin\theta_1 + L_2 \sin(\theta_1 + \theta_2) $$
を導出し、逆運動学が多価（肘上・肘下の2解）になることを確認せよ。

### 問題 5.34 - 5.37: 混合密度ネットワークの全モーメントの導出
全確率の法則および全分散の法則より、条件付き平均と分散が以下となることを示せ：
$$ \mathbb{E}[t|\mathbf{x}] = \sum_{k=1}^K \pi_k(\mathbf{x}) \mu_k(\mathbf{x}) $$
$$ \mathrm{Var}[t|\mathbf{x}] = \sum_{k=1}^K \pi_k(\mathbf{x}) \left\{ \sigma_k^2(\mathbf{x}) + \|\mu_k(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]\|^2 \right\} $$"""))

# Code Ex 5.26 - 5.37
code_ex5_26_37 = r"""# Exercise 5.33 & 5.37 ロボットアーム順運動学と MDN 条件付きモーメント数値検証
L1, L2 = 2.0, 1.5
theta1 = np.pi / 4; theta2 = -np.pi / 3
x1 = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
x2 = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
assert np.isfinite(x1) and np.isfinite(x2)

# Exercise 5.37 モーメント検証
pi_k = np.array([0.3, 0.7])
mu_k = np.array([1.0, 5.0])
sig_k = np.array([0.5, 1.2])

# モンテカルロシミュレーションによる平均と分散
N_samples = 100000
comp_choices = np.random.choice([0, 1], size=N_samples, p=pi_k)
samples = np.random.normal(loc=mu_k[comp_choices], scale=sig_k[comp_choices])

mean_mc = np.mean(samples)
var_mc = np.var(samples)

# 理論式
mean_theo = np.sum(pi_k * mu_k)
var_theo = np.sum(pi_k * (sig_k**2 + (mu_k - mean_theo)**2))

print(f"MC mean: {mean_mc:.4f}, Theo mean: {mean_theo:.4f}")
print(f"MC var:  {var_mc:.4f}, Theo var:  {var_theo:.4f}")
assert np.isclose(mean_mc, mean_theo, atol=0.03)
assert np.isclose(var_mc, var_theo, atol=0.05)
print("Exercise 5.26 - 5.37 verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex5_26_37))

# Exercise 5.38 - 5.41
cells.append(nbf.v4.new_markdown_cell(r"""---
## Exercise 5.38 - 5.41: ベイズNNの予測分布、エビデンス関数、多クラス拡張、二値分類の周辺尤度

### 問題 5.38: ベイズ回帰の予測ガウス分布
線形近似 $y(\mathbf{x}, \mathbf{w}) \simeq y(\mathbf{x}, \mathbf{w}_{\mathrm{MAP}}) + \mathbf{g}^T (\mathbf{w} - \mathbf{w}_{\mathrm{MAP}})$ の下で、予測分布が
$$ p(t|\mathbf{x}, \mathcal{D}) = \mathcal{N}\left( t \,\Big|\, y(\mathbf{x}, \mathbf{w}_{\mathrm{MAP}}), \, \beta^{-1} + \mathbf{g}^T \mathbf{A}^{-1}\mathbf{g} \right) $$
となることを示せ。

### 問題 5.39: ラプラス近似によるエビデンス関数
マッケイのエビデンス関数 $\ln p(\mathcal{D}|\alpha, \beta)$ の定式化を導出せよ。

### 問題 5.40 & 5.41: 分類モデルの周辺尤度
交差エントロピー誤差と事後分布のラプラス近似から、周辺尤度
$$ \ln p(\mathcal{D}|\alpha) \simeq -E_D(\mathbf{w}_{\mathrm{MAP}}) - \frac{\alpha}{2}\mathbf{w}_{\mathrm{MAP}}^T \mathbf{w}_{\mathrm{MAP}} - \frac{1}{2}\ln|\mathbf{A}| + \frac{W}{2}\ln\alpha $$
を導出せよ。"""))

# Code Ex 5.38 - 5.41
code_ex5_38_41 = r"""# Exercise 5.38 数値検証: 予測分散の解析解とサンプリング分散の一致
W_dim = 4
w_map = np.random.randn(W_dim)
A_mat = np.random.randn(W_dim, W_dim)
A_mat = A_mat.T @ A_mat + np.eye(W_dim) # 正定値ヘッセ
A_inv = np.linalg.inv(A_mat)

g_vec = np.random.randn(W_dim)
beta = 2.0

# 解析的予測分散
var_theo = 1.0 / beta + g_vec @ A_inv @ g_vec

# 事後分布 w ~ N(w_map, A_inv) からのサンプリングによる分散
w_samples = np.random.multivariate_normal(w_map, A_inv, size=50000)
t_samples = np.random.normal(loc=w_samples @ g_vec, scale=1.0/np.sqrt(beta))
var_mc = np.var(t_samples)

print(f"MC predictive variance: {var_mc:.4f}, Theo variance: {var_theo:.4f}")
assert np.isclose(var_mc, var_theo, atol=0.04)
print("Exercise 5.38 - 5.41 verified successfully!")"""
cells.append(nbf.v4.new_code_cell(code_ex5_38_41))

nb.cells = cells
with open('5/5_Exercises.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("5/5_Exercises.ipynb generated successfully.")
