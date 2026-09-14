## 演習 2.16: 共分散行列の正定値性

**問題:**
共分散行列 $\boldsymbol{\Sigma}$ に対し、任意の非ゼロ実ベクトル $\mathbf{v}$ に対して $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} > 0$ であること（正定値性）と、行列 $\boldsymbol{\Sigma}$ のすべての固有値 $\lambda_i$ が正（$\lambda_i > 0$）であることが同値であることを示せ。

**解答と解説:**

共分散行列 $\boldsymbol{\Sigma}$ は実対称行列であるため、直交行列 $\mathbf{U}$ を用いて対角化可能です。すなわち、
$$ \boldsymbol{\Sigma} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T $$
と表せます。ここで、$\mathbf{U}$ はその列が $\boldsymbol{\Sigma}$ の正規直交な固有ベクトル $\mathbf{u}_i$ からなる行列（$\mathbf{U}^T \mathbf{U} = \mathbf{I}$）、$\boldsymbol{\Lambda}$ は対応する固有値 $\lambda_i$ を対角成分に持つ対角行列です。

任意のベクトル $\mathbf{v}$ による二次形式を考えます。$\mathbf{v}$ を $\boldsymbol{\Sigma}$ の固有ベクトルを基底として展開します。$\mathbf{c} = \mathbf{U}^T \mathbf{v}$ とおくと、$\mathbf{c}$ の第 $i$ 成分は $c_i = \mathbf{u}_i^T \mathbf{v}$ であり、ベクトル $\mathbf{v}$ は $\mathbf{v} = \mathbf{U} \mathbf{c} = \sum_{i=1}^D c_i \mathbf{u}_i$ と表せます。

これを二次形式 $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v}$ に代入します。

$$
\begin{aligned}
\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} &= \mathbf{v}^T (\mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T) \mathbf{v} \\
&= (\mathbf{U}^T \mathbf{v})^T \boldsymbol{\Lambda} (\mathbf{U}^T \mathbf{v}) \\
&= \mathbf{c}^T \boldsymbol{\Lambda} \mathbf{c} \\
&= \sum_{i=1}^D \lambda_i c_i^2
\end{aligned}
$$

**1. $\lambda_i > 0$ ならば $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} > 0$ の証明:**
すべての固有値が $\lambda_i > 0$ であると仮定します。
任意の非ゼロベクトル $\mathbf{v} \neq \mathbf{0}$ に対して、$\mathbf{U}$ は正則な直交行列なので、$\mathbf{c} = \mathbf{U}^T \mathbf{v}$ も非ゼロベクトルとなります。したがって、少なくとも1つの $c_i$ は非ゼロであり、$c_i^2 > 0$ となります。
$\lambda_i > 0$ および $c_i^2 \geq 0$ であることから、和 $\sum_{i=1}^D \lambda_i c_i^2$ は必ず正になります。
よって、$\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} > 0$ が成り立ちます。

**2. $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} > 0$ ならば $\lambda_i > 0$ の証明:**
任意の非ゼロベクトル $\mathbf{v}$ に対して $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} > 0$ が成り立つと仮定します。
特定の $j$ について、$\mathbf{v}$ として固有ベクトル $\mathbf{u}_j$ を選びます。$\mathbf{u}_j$ はノルムが1の非ゼロベクトルです。
このとき、$\mathbf{c} = \mathbf{U}^T \mathbf{u}_j$ は、第 $j$ 成分が 1 で他の成分が 0 のベクトルになります（すなわち $c_j = 1$, $c_i = 0$ for $i \neq j$）。
これを二次形式の式に代入すると、
$$ \mathbf{u}_j^T \boldsymbol{\Sigma} \mathbf{u}_j = \sum_{i=1}^D \lambda_i c_i^2 = \lambda_j (1)^2 = \lambda_j $$
となります。
仮定より、任意の非ゼロベクトルに対して二次形式は正なので、$\mathbf{u}_j^T \boldsymbol{\Sigma} \mathbf{u}_j > 0$ です。
したがって、$\lambda_j > 0$ となります。これがすべての $j=1, \dots, D$ について成り立つため、すべての固有値は正です。

以上により、共分散行列の正定値性と全固有値が正であることは同値であることが示されました。
