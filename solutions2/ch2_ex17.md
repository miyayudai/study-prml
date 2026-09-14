## 演習 2.17: 精度行列の固有値と固有ベクトル

**問題:**
実対称な共分散行列 $\boldsymbol{\Sigma}$ が固有ベクトル $\mathbf{u}_i$ と対応する固有値 $\lambda_i$ を持つとする。このとき、逆行列である精度行列 $\boldsymbol{\Sigma}^{-1}$ の固有ベクトルは $\boldsymbol{\Sigma}$ と同一の $\mathbf{u}_i$ であり、対応する固有値は $1/\lambda_i$ となることを示せ。

**解答と解説:**

共分散行列 $\boldsymbol{\Sigma}$ の固有値方程式は以下のように定義されます。
$$ \boldsymbol{\Sigma} \mathbf{u}_i = \lambda_i \mathbf{u}_i $$

ここで、$\boldsymbol{\Sigma}$ は正定値対称行列であると仮定します（演習 2.16 参照）。したがって、すべての固有値 $\lambda_i$ は正（$\lambda_i > 0$）であり、0にはなりません。そのため、$\boldsymbol{\Sigma}$ は正則（逆行列を持つ）となります。

上記の固有値方程式の両辺に、左から精度行列 $\boldsymbol{\Sigma}^{-1}$ を掛けます。
$$ \boldsymbol{\Sigma}^{-1} (\boldsymbol{\Sigma} \mathbf{u}_i) = \boldsymbol{\Sigma}^{-1} (\lambda_i \mathbf{u}_i) $$

左辺は行列の積の結合法則より $\boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma} = \mathbf{I}$ （単位行列）となるため、
$$ \mathbf{I} \mathbf{u}_i = \mathbf{u}_i $$
となります。

右辺については、$\lambda_i$ はスカラーであるため、行列の前に出すことができます。
$$ \lambda_i (\boldsymbol{\Sigma}^{-1} \mathbf{u}_i) $$

したがって、式は以下のようになります。
$$ \mathbf{u}_i = \lambda_i (\boldsymbol{\Sigma}^{-1} \mathbf{u}_i) $$

$\lambda_i \neq 0$ であるため、両辺を $\lambda_i$ で割ることができます。
$$ \frac{1}{\lambda_i} \mathbf{u}_i = \boldsymbol{\Sigma}^{-1} \mathbf{u}_i $$

左右を入れ替えると、
$$ \boldsymbol{\Sigma}^{-1} \mathbf{u}_i = \frac{1}{\lambda_i} \mathbf{u}_i $$
となります。

この方程式は、ベクトル $\mathbf{u}_i$ が行列 $\boldsymbol{\Sigma}^{-1}$ の固有ベクトルであり、その対応する固有値が $1/\lambda_i$ であることを示しています。

**物理的・幾何学的解釈:**
多変量ガウス分布において、共分散行列 $\boldsymbol{\Sigma}$ は分布の「広がり」を表し、その固有ベクトル $\mathbf{u}_i$ は等高線（楕円体）の主軸の方向、固有値 $\lambda_i$ はその方向への分散（広がりの大きさ）を表します。
一方、精度行列 $\boldsymbol{\Sigma}^{-1}$ は「確信度」を表します。分散が大きい方向（$\lambda_i$ が大）は確信度が低く（$1/\lambda_i$ が小）、分散が小さい方向（$\lambda_i$ が小）は確信度が高く（$1/\lambda_i$ が大）なります。主軸の方向は同一のまま、スケールが逆数になるという直感的な性質がこの証明から確認できます。
