import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbformat.read('0/0_Exercises.ipynb', as_version=4)

q8_md = r"""---
## 問題 0.8 (正定値カーネルのグラム行列半正定値性とガウス過程の周辺化整合性)

PRML 第6章（カーネル法）の基盤理論となる正定値カーネル（再生核ヒルベルト空間の再生核）と、ガウス過程の確率論的支柱であるコルモゴロフの拡張定理（周辺化整合性）について以下の問いに答えよ。

1. **特徴空間とグラム行列の半正定値性**:
   特徴写像 $\boldsymbol{\phi}(\mathbf{x}) \in \mathcal{H}$ による内積で定義されるカーネル $k(\mathbf{x}, \mathbf{x}') = \langle \boldsymbol{\phi}(\mathbf{x}), \boldsymbol{\phi}(\mathbf{x}') \rangle_{\mathcal{H}}$ に対し、任意の有限データ集合 $\{\mathbf{x}_n\}_{n=1}^N$ から構成されるグラム行列 $\mathbf{K} \in \mathbb{R}^{N \times N}$ ($K_{nm} = k(\mathbf{x}_n, \mathbf{x}_m)$) が半正定値 ($\mathbf{K} \succeq 0$) であることを示せ。

   **【解答の要点】**
   任意の非ゼロ実数ベクトル $\mathbf{v} = (v_1, \dots, v_N)^T \in \mathbb{R}^N$ に対し、二次形式を計算すると：
   $$
   \mathbf{v}^T \mathbf{K} \mathbf{v} = \sum_{n=1}^N \sum_{m=1}^N v_n v_m k(\mathbf{x}_n, \mathbf{x}_m) = \sum_{n=1}^N \sum_{m=1}^N v_n v_m \langle \boldsymbol{\phi}(\mathbf{x}_n), \boldsymbol{\phi}(\mathbf{x}_m) \rangle = \left\| \sum_{n=1}^N v_n \boldsymbol{\phi}(\mathbf{x}_n) \right\|_{\mathcal{H}}^2 \ge 0
   $$
   ノルムの非負性より、任意の有限標本に対して $\mathbf{v}^T \mathbf{K} \mathbf{v} \ge 0$ が恒等的に成立するため、$\mathbf{K}$ は常に半正定値行列となる。

2. **ガウス過程の周辺化整合性 (Marginalization Consistency)**:
   コルモゴロフの拡張定理により、無限次元の確率過程（ガウス過程）が存在するための必要十分条件は、任意の有限部分集合に対する同時分布が周辺化操作に関して無矛盾であることである。
   $N$ 点の同時ガウス分布 $p(\mathbf{y}_N) = \mathcal{N}(\mathbf{y}_N | \mathbf{0}, \mathbf{K}_N)$ と、$N+1$ 点の同時ガウス分布
   $$
   p(\mathbf{y}_N, y_{N+1}) = \mathcal{N}\left( \begin{pmatrix} \mathbf{y}_N \\ y_{N+1} \end{pmatrix} \middle| \begin{pmatrix} \mathbf{0} \\ 0 \end{pmatrix}, \begin{pmatrix} \mathbf{K}_N & \mathbf{k} \\ \mathbf{k}^T & c \end{pmatrix} \right)
   $$
   において、$y_{N+1}$ を積分消去した周辺分布 $\int p(\mathbf{y}_N, y_{N+1}) dy_{N+1}$ が厳密に $p(\mathbf{y}_N)$ と一致することをガウス分布のブロック分割定理を用いて証明せよ。

   **【解答の要点】**
   多変量ガウス分布のブロック分割性質（PRML 式 2.85）より、同時正規分布に従う確率変数の部分ベクトルに対する周辺分布は、単に対応する共分散行列のブロック小行列を共分散として持ち、他の変数に関するブロック（$\mathbf{k}, c$）に一切依存しない。
   したがって、$\int p(\mathbf{y}_N, y_{N+1}) dy_{N+1} = \mathcal{N}(\mathbf{y}_N | \mathbf{0}, \mathbf{K}_N)$ が厳密に成立し、新たな観測点の追加や削除に対して一貫した確率測度が保証される。
"""

q8_code = r"""# 問題 0.8 数値検証: グラム行列の半正定値性とGPの周辺化整合性
import numpy as np

np.random.seed(42)

# 1. グラム行列の半正定値性 (Gram matrix PSD check)
N = 8
X = np.random.uniform(-3, 3, size=(N, 1))

# RBF (Squared Exponential) カーネル
def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
    dists_sq = (X1 - X2.T) ** 2
    return variance * np.exp(-0.5 * dists_sq / (length_scale ** 2))

K = rbf_kernel(X, X)
eigvals = np.linalg.eigvalsh(K)

print("グラム行列の固有値:", eigvals)
assert np.all(eigvals >= -1e-12), "グラム行列の固有値が負（半正定値性に反する）です"
print("検証1成功: すべての固有値が非負（最小固有値: {:.2e}）".format(np.min(eigvals)))

# 2. 周辺化整合性のモンテカルロ検証
# x_{N+1} を追加
x_new = np.array([[0.5]])
X_plus = np.vstack([X, x_new])
K_plus = rbf_kernel(X_plus, X_plus) + 1e-6 * np.eye(N + 1)

# N+1 次元の同時分布から大量サンプリング
n_mc_samples = 50000
samples_plus = np.random.multivariate_normal(np.zeros(N + 1), K_plus, size=n_mc_samples)

# y_{N+1} を単に無視（周辺化）して最初の N 点の経験共分散を計算
samples_N_marginalized = samples_plus[:, :N]
cov_empirical = np.cov(samples_N_marginalized, rowvar=False)

# N 点の理論共分散 K[:N, :N]
cov_true = rbf_kernel(X, X)

max_diff = np.max(np.abs(cov_empirical - cov_true))
print(f"N点の理論共分散とN+1点同時サンプル周辺化共分散の最大誤差: {max_diff:.4f}")
assert max_diff < 0.05, "周辺化整合性の経験値が理論値から逸脱しています"
print("検証2成功: 周辺化整合性（Marginalization Consistency）が数値的に確認されました！")
"""

nb.cells.append(nbformat.v4.new_markdown_cell(q8_md))
nb.cells.append(nbformat.v4.new_code_cell(q8_code))

# Execute all cells
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': '0/'}})
nbformat.write(nb, '0/0_Exercises.ipynb')
print("Successfully appended and executed 問題 0.8 in 0/0_Exercises.ipynb!")
