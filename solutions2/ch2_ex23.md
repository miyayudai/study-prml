## Exercise 2.23 (周辺ガウス分布の導出)

**問題:**
指数部の平方完成により、$\mathbf{x}_b$ を積分消去して $p(\mathbf{x}_a) = \mathcal{N}(\mathbf{x}_a | \boldsymbol{\mu}_a, \boldsymbol{\Sigma}_{aa})$ を導出せよ。

**解答と解説:**

同時確率分布 $p(\mathbf{x}_a, \mathbf{x}_b)$ が多変量ガウス分布であるとき、$\mathbf{x}_a$ の周辺分布 $p(\mathbf{x}_a)$ は $\mathbf{x}_b$ に関して積分することで得られます。
$$ p(\mathbf{x}_a) = \int p(\mathbf{x}_a, \mathbf{x}_b) d\mathbf{x}_b $$

同時ガウス分布の指数部（$-\frac{1}{2}$ を除いた部分）を $\mathbf{x}_a$ と $\mathbf{x}_b$ について展開します。簡単のために、変数を平均からの差 $\mathbf{y}_a = \mathbf{x}_a - \boldsymbol{\mu}_a$, $\mathbf{y}_b = \mathbf{x}_b - \boldsymbol{\mu}_b$ と置きます。
$$ Q = \begin{pmatrix} \mathbf{y}_a \\ \mathbf{y}_b \end{pmatrix}^T \begin{pmatrix} \boldsymbol{\Lambda}_{aa} & \boldsymbol{\Lambda}_{ab} \\ \boldsymbol{\Lambda}_{ba} & \boldsymbol{\Lambda}_{bb} \end{pmatrix} \begin{pmatrix} \mathbf{y}_a \\ \mathbf{y}_b \end{pmatrix} $$
$$ Q = \mathbf{y}_a^T \boldsymbol{\Lambda}_{aa} \mathbf{y}_a + 2\mathbf{y}_a^T \boldsymbol{\Lambda}_{ab} \mathbf{y}_b + \mathbf{y}_b^T \boldsymbol{\Lambda}_{bb} \mathbf{y}_b $$
（ここで $\boldsymbol{\Lambda}_{ba}^T = \boldsymbol{\Lambda}_{ab}$ を使用しました。）

積分を計算するために、$\mathbf{y}_b$ について平方完成を行います。
$$ Q = (\mathbf{y}_b - \mathbf{m})^T \boldsymbol{\Lambda}_{bb} (\mathbf{y}_b - \mathbf{m}) + C(\mathbf{y}_a) $$
となるような $\mathbf{m}$ と $C(\mathbf{y}_a)$ を見つけます。展開して係数を比較すると、
$$ (\mathbf{y}_b - \mathbf{m})^T \boldsymbol{\Lambda}_{bb} (\mathbf{y}_b - \mathbf{m}) = \mathbf{y}_b^T \boldsymbol{\Lambda}_{bb} \mathbf{y}_b - 2\mathbf{m}^T \boldsymbol{\Lambda}_{bb} \mathbf{y}_b + \mathbf{m}^T \boldsymbol{\Lambda}_{bb} \mathbf{m} $$
$\mathbf{y}_b$ の1次の項を比較すると、
$$ -2\mathbf{m}^T \boldsymbol{\Lambda}_{bb} \mathbf{y}_b = 2\mathbf{y}_a^T \boldsymbol{\Lambda}_{ab} \mathbf{y}_b \implies \mathbf{m}^T \boldsymbol{\Lambda}_{bb} = -\mathbf{y}_a^T \boldsymbol{\Lambda}_{ab} \implies \mathbf{m} = -\boldsymbol{\Lambda}_{bb}^{-1} \boldsymbol{\Lambda}_{ba} \mathbf{y}_a $$
これを用いると、残りの項 $C(\mathbf{y}_a)$ は、
$$ C(\mathbf{y}_a) = \mathbf{y}_a^T \boldsymbol{\Lambda}_{aa} \mathbf{y}_a - \mathbf{m}^T \boldsymbol{\Lambda}_{bb} \mathbf{m} = \mathbf{y}_a^T \boldsymbol{\Lambda}_{aa} \mathbf{y}_a - (-\mathbf{y}_a^T \boldsymbol{\Lambda}_{ab}) \boldsymbol{\Lambda}_{bb}^{-1} (-\boldsymbol{\Lambda}_{ba} \mathbf{y}_a) $$
$$ C(\mathbf{y}_a) = \mathbf{y}_a^T (\boldsymbol{\Lambda}_{aa} - \boldsymbol{\Lambda}_{ab}\boldsymbol{\Lambda}_{bb}^{-1}\boldsymbol{\Lambda}_{ba}) \mathbf{y}_a $$

これで $p(\mathbf{x}_a, \mathbf{x}_b)$ の指数部が $\mathbf{y}_b$ に関する二次形式と、$\mathbf{y}_a$ だけに依存する項に分離されました。$\mathbf{x}_b$ に関する積分（$\mathbf{y}_b$ の積分に等しい）を実行すると、第一項は正規化定数の一部となり消去されます。
$$ \int \exp\left( -\frac{1}{2} (\mathbf{y}_b - \mathbf{m})^T \boldsymbol{\Lambda}_{bb} (\mathbf{y}_b - \mathbf{m}) \right) d\mathbf{y}_b = (2\pi)^{M/2} |\boldsymbol{\Lambda}_{bb}|^{-1/2} $$
（$M$ は $\mathbf{x}_b$ の次元）

したがって、周辺分布 $p(\mathbf{x}_a)$ は残った項 $\exp\left(-\frac{1}{2} C(\mathbf{y}_a)\right)$ に比例します。
$$ p(\mathbf{x}_a) \propto \exp\left( -\frac{1}{2} (\mathbf{x}_a - \boldsymbol{\mu}_a)^T (\boldsymbol{\Lambda}_{aa} - \boldsymbol{\Lambda}_{ab}\boldsymbol{\Lambda}_{bb}^{-1}\boldsymbol{\Lambda}_{ba}) (\mathbf{x}_a - \boldsymbol{\mu}_a) \right) $$

この分布は明らかに平均 $\boldsymbol{\mu}_a$ のガウス分布です。その共分散行列 $\boldsymbol{\Sigma}_{aa}$ は精度行列の逆行列なので、
$$ \boldsymbol{\Sigma}_{aa} = (\boldsymbol{\Lambda}_{aa} - \boldsymbol{\Lambda}_{ab}\boldsymbol{\Lambda}_{bb}^{-1}\boldsymbol{\Lambda}_{ba})^{-1} $$
Schur補行列の公式から、これは元のブロック共分散行列の $(a,a)$ ブロックに等しくなります。したがって、周辺分布は、
$$ p(\mathbf{x}_a) = \mathcal{N}(\mathbf{x}_a | \boldsymbol{\mu}_a, \boldsymbol{\Sigma}_{aa}) $$
となることが証明されました。
