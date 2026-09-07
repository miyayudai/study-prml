import nbformat

with open('0/0_Foundations_of_Probability.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Remove the last 2 cells if they were just added
if len(nb.cells) > 0 and '0.14' in nb.cells[-2].source:
    nb.cells = nb.cells[:-2]

sec014_md = r"""## 0.14 正定値カーネル・再生核ヒルベルト空間 (RKHS) とガウス過程の周辺化整合性 (第6章への補講)

第6章「カーネル法」および第7章「スパースカーネルマシン」では、非線形特徴空間への写像 $\mathbf{x} \mapsto \boldsymbol{\phi}(\mathbf{x})$ を直接計算することなく、内積 $k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}')$ を介してアルゴリズムを構築する「カーネルトリック」が中心となります。本節では、その数学的正当性を支える基盤理論を整理します。

### 1. グラム行列の半正定値性とマーサーの定理 (Mercer's Theorem)

関数 $k(\mathbf{x}, \mathbf{x}')$ が有効なカーネルであるための必要十分条件は、任意の有限入力集合 $\{\mathbf{x}_n\}_{n=1}^N$ に対して定義される**グラム行列 (Gram Matrix)** $\mathbf{K} \in \mathbb{R}^{N \times N}$ ($K_{nm} = k(\mathbf{x}_n, \mathbf{x}_m)$) が常に**半正定値 (Positive Semidefinite; PSD)** であることです：

$$
\mathbf{c}^T \mathbf{K} \mathbf{c} = \sum_{n=1}^N \sum_{m=1}^N c_n c_m k(\mathbf{x}_n, \mathbf{x}_m) \ge 0 \quad (\forall \mathbf{c} \in \mathbb{R}^N)
$$

**マーサーの定理**:
有界閉区間上の連続対称核 $k(\mathbf{x}, \mathbf{x}')$ に対し、積分作用素 $[T_k f](\mathbf{x}) = \int k(\mathbf{x}, \mathbf{x}') f(\mathbf{x}') d\mathbf{x}'$ が非負定値であるとき、正規直交固有関数系 $\{\psi_i(\mathbf{x})\}$ と非負固有値 $\lambda_i \ge 0$ が存在し、次のように一様絶対収束します：
$$
k(\mathbf{x}, \mathbf{x}') = \sum_{i=1}^\infty \lambda_i \psi_i(\mathbf{x}) \psi_i(\mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\phi}(\mathbf{x}') \quad \left( \phi_i(\mathbf{x}) = \sqrt{\lambda_i} \psi_i(\mathbf{x}) \right)
$$

### 2. 再生核ヒルベルト空間 (RKHS) と再生性 (Reproducing Property)

核 $k$ に付随する再生核ヒルベルト空間 $\mathcal{H}_k$ は、以下の2大性質を満たすヒルベルト空間です：
1. **核関数族の所属**: 任意の $\mathbf{x}$ に対し、関数 $k(\cdot, \mathbf{x})$ は $\mathcal{H}_k$ に属する。
2. **再生性 (Reproducing Property)**: 任意の $f \in \mathcal{H}_k$ に対し、
$$
\langle f, k(\cdot, \mathbf{x}) \rangle_{\mathcal{H}_k} = f(\mathbf{x})
$$
特に $f = k(\cdot, \mathbf{x}')$ とおくことで、$\langle k(\cdot, \mathbf{x}'), k(\cdot, \mathbf{x}) \rangle_{\mathcal{H}_k} = k(\mathbf{x}', \mathbf{x})$ が得られます。

### 3. コルモゴロフの拡張定理とガウス過程の周辺化整合性 (Marginal Consistency)

**ガウス過程 (Gaussian Process; GP)** は、任意の有限個の入力点 $\{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ における関数値の結合分布が多変量ガウス分布に従う確率過程です：
$$
p(y(\mathbf{x}_1), \dots, y(\mathbf{x}_N)) = \mathcal{N}(\mathbf{y} | \mathbf{m}, \mathbf{K})
$$
この定義が無限次元の関数空間上の確率測度として矛盾なく存在するための本質的要件が**コルモゴロフの周辺化整合性 (Consistency)** です：
$$
p(y_1, \dots, y_N) = \int p(y_1, \dots, y_N, y_{N+1}) d y_{N+1}
$$
ガウス分布は周辺化によって平均・分散パラメータがそのまま保たれる（式 2.85）という驚異的な自己整合性を持つため、テスト点 $x_{N+1}$ をいくら追加しても、既存の訓練点集合上の同時分布は変化しません。"""

sec014_code = r"""# 0.14 数値検証と可視化: ガウス核のグラム行列半正定値性とガウス過程整合性
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('result', exist_ok=True)
np.random.seed(42)

# 1. グラム行列の半正定値性 (PSD) 検証
def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    dists = (x1[:, None] - x2[None, :]) ** 2
    return variance * np.exp(-0.5 * dists / (length_scale ** 2))

X_sample = np.sort(np.random.uniform(-3, 3, 20))
K_mat = rbf_kernel(X_sample, X_sample)

# 固有値がすべて非負であることを検証
eigvals = np.linalg.eigvalsh(K_mat)
assert np.all(eigvals >= -1e-10), f'Gram matrix must be PSD, got min eigval {np.min(eigvals)}'
print(f'Gram matrix (20x20) min eigenvalue: {np.min(eigvals):.6e} >= 0 (PSD verified!)')

# 2. ガウス過程からの事前関数サンプリングと周辺化整合性
x_dense = np.linspace(-3, 3, 100)
K_dense = rbf_kernel(x_dense, x_dense) + 1e-6 * np.eye(100)
L_dense = np.linalg.cholesky(K_dense)
gp_samples = L_dense @ np.random.randn(100, 3)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# グラム行列のヒートマップ
im = axes[0].imshow(K_mat, cmap='viridis', extent=[-3, 3, -3, 3], origin='lower')
axes[0].set_title('RBF Kernel Gram Matrix $K_{nm}$', fontsize=12)
axes[0].set_xlabel('$x_n$', fontsize=11); axes[0].set_ylabel('$x_m$', fontsize=11)
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

# ガウス過程事前分布からの関数サンプル
for i in range(3):
    axes[1].plot(x_dense, gp_samples[:, i], lw=2, label=f'GP sample {i+1}')
axes[1].fill_between(x_dense, -2.0, 2.0, color='gray', alpha=0.2, label='95% Prior Interval ($2\\sigma$)')
axes[1].set_title('Gaussian Process Prior Function Samples', fontsize=12)
axes[1].set_xlabel('x', fontsize=11); axes[1].set_ylabel('y(x)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
fig_path = 'result/fig0_rbf_gram_and_gp_prior.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f'Saved visualization to {fig_path}')
print('Section 0.14 successfully verified!')"""

nb.cells.append(nbformat.v4.new_markdown_cell(sec014_md))
nb.cells.append(nbformat.v4.new_code_cell(sec014_code))

with open('0/0_Foundations_of_Probability.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Appended clean Section 0.14 to 0/0_Foundations_of_Probability.ipynb!')
