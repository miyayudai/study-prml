# 演習問題 2.24

## 問題設定
分割されたガウス分布において、条件付き分布 $p(\mathbf{x}_a | \mathbf{x}_b)$ がガウス分布であることを示し、その平均と共分散行列を導出せよ。

## 解答と証明
同時分布の二次形式の指数部分を展開して平方完成を行うことで、条件付き分布を求める。
指数部分は以下のように書ける。
$$ -\frac{1}{2} \begin{pmatrix} \mathbf{x}_a - \boldsymbol{\mu}_a \\ \mathbf{x}_b - \boldsymbol{\mu}_b \end{pmatrix}^T \begin{pmatrix} \mathbf{\Lambda}_{aa} & \mathbf{\Lambda}_{ab} \\ \mathbf{\Lambda}_{ba} & \mathbf{\Lambda}_{bb} \end{pmatrix} \begin{pmatrix} \mathbf{x}_a - \boldsymbol{\mu}_a \\ \mathbf{x}_b - \boldsymbol{\mu}_b \end{pmatrix} $$

$\mathbf{x}_a$ に関する項を取り出し、平方完成を行うと、
$$ -\frac{1}{2} \left( \mathbf{x}_a^T \mathbf{\Lambda}_{aa} \mathbf{x}_a - 2\mathbf{x}_a^T (\mathbf{\Lambda}_{aa}\boldsymbol{\mu}_a - \mathbf{\Lambda}_{ab}(\mathbf{x}_b - \boldsymbol{\mu}_b)) + \text{const} \right) $$
となる。これから、条件付き分布の共分散行列と平均を読み取ることができる。

条件付き共分散行列は精度行列の逆行列となる。
$$ \boldsymbol{\Sigma}_{a|b} = \mathbf{\Lambda}_{aa}^{-1} $$
ここで、ブロック行列の逆行列の公式を用いると、$\mathbf{\Lambda}_{aa}^{-1} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} \boldsymbol{\Sigma}_{ba}$ である。

条件付き平均は以下のようになる。
$$ \boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a - \mathbf{\Lambda}_{aa}^{-1} \mathbf{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$
再び逆行列の公式より、$-\mathbf{\Lambda}_{aa}^{-1} \mathbf{\Lambda}_{ab} = \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1}$ が成り立つので、
$$ \boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$
を得る。これらはガウス分布のパラメータを構成する。
(証明終)
