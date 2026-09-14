## Exercise 2.40 (多変量 Student's t分布)

多変量スチューデントのt分布について、その共分散行列が $\frac{\nu}{\nu - 2}\boldsymbol{\Lambda}^{-1}$ となることを確認します。1変量の場合（Exercise 2.38）と同様の「無限混合ガウス分布」の枠組みを多変量に拡張して証明します。

### 多変量t分布の無限混合表現
$D$ 次元の多変量t分布も、多変量ガウス分布の精度をスカラー変数 $\eta$ によってスケールさせ、$\eta$ についてガンマ分布で周辺化することで得られます。
$$
\mathrm{St}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Lambda}, \nu) = \int_0^\infty \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, (\eta \boldsymbol{\Lambda})^{-1}) \mathrm{Gam}\left(\eta \middle| \frac{\nu}{2}, \frac{\nu}{2}\right) \, d\eta
$$
ここで、
- 条件付き分布 $p(\mathbf{x} | \eta) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \eta^{-1} \boldsymbol{\Lambda}^{-1})$ は平均 $\boldsymbol{\mu}$、共分散行列 $\eta^{-1} \boldsymbol{\Lambda}^{-1}$ を持つ $D$ 次元ガウス分布です。
- 事前分布 $p(\eta) = \mathrm{Gam}(\eta | \nu/2, \nu/2)$ は形状パラメータ $\nu/2$、尺度パラメータ $\nu/2$ のガンマ分布です。

### 平均ベクトルの導出
全期待値の法則を用います。
$$
\mathbb{E}[\mathbf{x}] = \mathbb{E}_\eta [ \mathbb{E}_{\mathbf{x}} [\mathbf{x} | \eta] ]
$$
条件付きガウス分布の平均は $\boldsymbol{\mu}$ であり、$\eta$ に依存しません。
$$
\mathbb{E}_{\mathbf{x}} [\mathbf{x} | \eta] = \boldsymbol{\mu}
$$
これを外側の期待値に代入します。
$$
\mathbb{E}[\mathbf{x}] = \mathbb{E}_\eta [\boldsymbol{\mu}] = \boldsymbol{\mu}
$$
（ただし $\nu > 1$ の場合）

### 共分散行列の導出
多変量における全分散の法則（あるいは条件付き共分散の公式）を用います。
$$
\mathrm{cov}[\mathbf{x}] = \mathbb{E}_\eta [ \mathrm{cov}_{\mathbf{x}}[\mathbf{x} | \eta] ] + \mathrm{cov}_\eta [ \mathbb{E}_{\mathbf{x}}[\mathbf{x} | \eta] ]
$$
第2項は、$\mathbb{E}_{\mathbf{x}}[\mathbf{x} | \eta] = \boldsymbol{\mu}$（定数ベクトル）であるため、その共分散はゼロ行列 $\mathbf{0}$ になります。
$$
\mathrm{cov}_\eta [ \boldsymbol{\mu} ] = \mathbf{0}
$$
第1項について、条件付きガウス分布の共分散は $\eta^{-1} \boldsymbol{\Lambda}^{-1}$ です。
$$
\mathrm{cov}[\mathbf{x}] = \mathbb{E}_\eta [ \eta^{-1} \boldsymbol{\Lambda}^{-1} ] = \mathbb{E}_\eta [ \eta^{-1} ] \boldsymbol{\Lambda}^{-1}
$$
$\eta \sim \mathrm{Gam}(\nu/2, \nu/2)$ について、逆数 $\eta^{-1}$ の期待値は Exercise 2.38 で計算した通りです（$a = \nu/2, b = \nu/2$ を代入）。
$$
\mathbb{E}_\eta [ \eta^{-1} ] = \frac{b}{a - 1} = \frac{\nu/2}{\nu/2 - 1} = \frac{\nu}{\nu - 2}
$$
これを代入すると、多変量t分布の共分散行列が得られます。
$$
\mathrm{cov}[\mathbf{x}] = \frac{\nu}{\nu - 2} \boldsymbol{\Lambda}^{-1}
$$
スカラーの場合と全く同様に、ガンマ重みの混合による性質がベクトルや行列にそのまま引き継がれるため、この結果が成立することが確認できました。ただし、共分散が定義されるためには $\nu > 2$ である必要があります。
