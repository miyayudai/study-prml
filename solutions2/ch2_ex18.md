## 演習 2.18: 多変量ガウス分布の規格化の直接証明

**問題:**
$D$ 次元多変量ガウス分布の確率密度関数は次式で与えられる。
$$ \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left\{ -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right\} $$
直交変換 $\mathbf{y} = \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu})$ を用いて、この分布が全空間で積分すると 1 になること（規格化条件）を直接証明せよ。

**解答と解説:**

多変量ガウス分布の全空間における積分 $I$ を考えます。
$$ I = \int_{-\infty}^{\infty} \dots \int_{-\infty}^{\infty} \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) \, \mathrm{d}\mathbf{x} $$

共分散行列 $\boldsymbol{\Sigma}$ を直交行列 $\mathbf{U}$ で固有値分解します。$\boldsymbol{\Sigma} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T$ であり、その逆行列（精度行列）は $\boldsymbol{\Sigma}^{-1} = \mathbf{U} \boldsymbol{\Lambda}^{-1} \mathbf{U}^T$ です。

指数部にあるマハラノビス距離の二乗を展開します。
$$ (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{U} \boldsymbol{\Lambda}^{-1} \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu}) $$

ここで、新しい変数 $\mathbf{y}$ を次のように定義します。
$$ \mathbf{y} = \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu}) $$
この変換により、指数部は以下のようにシンプルになります。
$$ (\mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu}))^T \boldsymbol{\Lambda}^{-1} \mathbf{y} = \mathbf{y}^T \boldsymbol{\Lambda}^{-1} \mathbf{y} = \sum_{i=1}^D \frac{y_i^2}{\lambda_i} $$

次に、変数変換 $\mathbf{x} \to \mathbf{y}$ に伴う微小体積要素の変換（ヤコビアン）を計算します。
$\mathbf{x} = \mathbf{U} \mathbf{y} + \boldsymbol{\mu}$ であるため、ヤコビ行列は $\frac{\partial \mathbf{x}}{\partial \mathbf{y}} = \mathbf{U}$ となります。
直交行列の行列式の絶対値は 1（$|\det(\mathbf{U})| = 1$）であるため、
$$ \mathrm{d}\mathbf{x} = |\det(\mathbf{U})| \, \mathrm{d}\mathbf{y} = \mathrm{d}\mathbf{y} = \mathrm{d}y_1 \mathrm{d}y_2 \dots \mathrm{d}y_D $$
となります。

また、共分散行列の行列式 $|\boldsymbol{\Sigma}|$ は、その固有値の積で表されます。
$$ |\boldsymbol{\Sigma}| = |\mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T| = |\mathbf{U}| |\boldsymbol{\Lambda}| |\mathbf{U}^T| = 1 \cdot \left( \prod_{i=1}^D \lambda_i \right) \cdot 1 = \prod_{i=1}^D \lambda_i $$

これらの結果を積分 $I$ に代入します。
$$
\begin{aligned}
I &= \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \int \exp \left\{ -\frac{1}{2} \mathbf{y}^T \boldsymbol{\Lambda}^{-1} \mathbf{y} \right\} \mathrm{d}\mathbf{y} \\
&= \frac{1}{(2\pi)^{D/2} \left( \prod_{i=1}^D \lambda_i \right)^{1/2}} \int \dots \int \exp \left\{ -\frac{1}{2} \sum_{i=1}^D \frac{y_i^2}{\lambda_i} \right\} \mathrm{d}y_1 \dots \mathrm{d}y_D
\end{aligned}
$$

指数関数の性質 $\exp(\sum A_i) = \prod \exp(A_i)$ を用いると、多重積分は $D$ 個の1次元積分の積に分解されます。
$$ I = \prod_{i=1}^D \left[ \frac{1}{(2\pi \lambda_i)^{1/2}} \int_{-\infty}^{\infty} \exp \left( -\frac{y_i^2}{2\lambda_i} \right) \mathrm{d}y_i \right] $$

括弧の中は、平均 0、分散 $\lambda_i$ の1次元ガウス分布の全空間積分そのものであり、1次元ガウス積分の公式 $\int_{-\infty}^{\infty} \exp(-a z^2) \mathrm{d}z = \sqrt{\pi / a}$ より、その値は $(2\pi \lambda_i)^{1/2}$ となります。したがって括弧内は 1 になります。
$$ I = \prod_{i=1}^D [ 1 ] = 1 $$

以上により、多変量ガウス分布が正しく規格化されていることが証明されました。
