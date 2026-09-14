## Exercise 2.22 (条件付きガウス分布の導出)

**問題:**
Schur補行列の公式を用いて、条件付き共分散 $\boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Lambda}_{aa}^{-1}$ を導出せよ。

**解答と解説:**

多変量ガウス分布において、確率変数ベクトル $\mathbf{x}$ が二つの部分ベクトル $\mathbf{x}_a$ と $\mathbf{x}_b$ に分割されているとします。
$$ \mathbf{x} = \begin{pmatrix} \mathbf{x}_a \\ \mathbf{x}_b \end{pmatrix} $$
共分散行列 $\boldsymbol{\Sigma}$ とその逆行列である精度行列 $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ も同様にブロック分割されます。
$$ \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ \boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb} \end{pmatrix}, \quad \boldsymbol{\Lambda} = \begin{pmatrix} \boldsymbol{\Lambda}_{aa} & \boldsymbol{\Lambda}_{ab} \\ \boldsymbol{\Lambda}_{ba} & \boldsymbol{\Lambda}_{bb} \end{pmatrix} $$

$\mathbf{x}_b$ が与えられたときの $\mathbf{x}_a$ の条件付き分布 $p(\mathbf{x}_a | \mathbf{x}_b)$ もガウス分布となり、その共分散を $\boldsymbol{\Sigma}_{a|b}$ とします。

ガウス分布の指数部を展開すると、$\mathbf{x}_a$ に関する二次形式の係数行列が条件付き精度行列 $\boldsymbol{\Sigma}_{a|b}^{-1}$ となります。指数部の二次項は次のようになります。
$$ -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Lambda} (\mathbf{x} - \boldsymbol{\mu}) $$
これを $\mathbf{x}_a$ について整理すると、$\mathbf{x}_a^T \boldsymbol{\Lambda}_{aa} \mathbf{x}_a$ の項が現れます。したがって、条件付き精度行列は $\boldsymbol{\Lambda}_{aa}$ であり、条件付き共分散行列はその逆行列になります。
$$ \boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Lambda}_{aa}^{-1} $$

ここで、ブロック行列の逆行列に関するSchur（シューア）補行列の公式を用います。
$\boldsymbol{\Sigma}$ と $\boldsymbol{\Lambda}$ は互いに逆行列なので、
$$ \begin{pmatrix} \boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ \boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb} \end{pmatrix} \begin{pmatrix} \boldsymbol{\Lambda}_{aa} & \boldsymbol{\Lambda}_{ab} \\ \boldsymbol{\Lambda}_{ba} & \boldsymbol{\Lambda}_{bb} \end{pmatrix} = \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{0} & \mathbf{I} \end{pmatrix} $$
この行列の積の(1, 1)ブロックと(1, 2)ブロックの要素から以下の連立方程式が得られます。
1) $\boldsymbol{\Sigma}_{aa}\boldsymbol{\Lambda}_{aa} + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Lambda}_{ba} = \mathbf{I}$
2) $\boldsymbol{\Sigma}_{aa}\boldsymbol{\Lambda}_{ab} + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Lambda}_{bb} = \mathbf{0}$

同様に(2, 1)と(2, 2)からも得られます。
3) $\boldsymbol{\Sigma}_{ba}\boldsymbol{\Lambda}_{aa} + \boldsymbol{\Sigma}_{bb}\boldsymbol{\Lambda}_{ba} = \mathbf{0}$

式(3)より、$\boldsymbol{\Lambda}_{ba}$ を解くと、
$$ \boldsymbol{\Lambda}_{ba} = -\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}\boldsymbol{\Lambda}_{aa} $$
これを式(1)に代入します。
$$ \boldsymbol{\Sigma}_{aa}\boldsymbol{\Lambda}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}\boldsymbol{\Lambda}_{aa} = \mathbf{I} $$
$$ (\boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba})\boldsymbol{\Lambda}_{aa} = \mathbf{I} $$
したがって、両辺に逆行列をとることで、
$$ \boldsymbol{\Lambda}_{aa}^{-1} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba} $$
が導かれます。右辺は $\boldsymbol{\Sigma}$ の $\boldsymbol{\Sigma}_{bb}$ に関するSchur補行列です。

これにより、条件付き共分散行列 $\boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Lambda}_{aa}^{-1}$ は元の共分散行列のブロックを用いて $\boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}$ と表現できることが示されました。
