# 演習問題 2.26

## 問題設定
連続確率変数の線形変換における確率密度関数の変換則を用いて、ある確率変数 $\mathbf{x}$ が平均 $\boldsymbol{\mu}$、共分散行列 $\boldsymbol{\Sigma}$ のガウス分布に従うとき、線形変換 $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{b}$ を施した確率変数 $\mathbf{y}$ もガウス分布に従うことを示せ。

## 解答と証明
確率変数 $\mathbf{y}$ の確率密度関数 $p(\mathbf{y})$ を求める。
ヤコビアンを用いて変数変換を行う。$\mathbf{x} = \mathbf{A}^{-1}(\mathbf{y} - \mathbf{b})$ とすると、微小体積要素の比はヤコビアンの行列式の絶対値 $|\mathbf{A}^{-1}|$ で与えられる。
$$ p(\mathbf{y}) = p(\mathbf{x}) |\text{det}(\mathbf{A}^{-1})| $$

$\mathbf{x}$ の分布は $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma})$ であるため、
$$ p(\mathbf{x}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right) $$

これを変数変換すると、指数部分は以下のようになる。
$$ -\frac{1}{2} (\mathbf{A}^{-1}(\mathbf{y} - \mathbf{b}) - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{A}^{-1}(\mathbf{y} - \mathbf{b}) - \boldsymbol{\mu}) $$
式を整理すると、
$$ = -\frac{1}{2} (\mathbf{y} - (\mathbf{A}\boldsymbol{\mu} + \mathbf{b}))^T (\mathbf{A}^{-1})^T \boldsymbol{\Sigma}^{-1} \mathbf{A}^{-1} (\mathbf{y} - (\mathbf{A}\boldsymbol{\mu} + \mathbf{b})) $$
$$ = -\frac{1}{2} (\mathbf{y} - \boldsymbol{\mu}_y)^T (\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)^{-1} (\mathbf{y} - \boldsymbol{\mu}_y) $$

ここで、$\boldsymbol{\mu}_y = \mathbf{A}\boldsymbol{\mu} + \mathbf{b}$ とおいた。
共分散行列を $\boldsymbol{\Sigma}_y = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T$ とおくと、行列式の部分は、
$$ |\boldsymbol{\Sigma}_y| = |\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T| = |\mathbf{A}|^2 |\boldsymbol{\Sigma}| $$
となる。
これを係数部分に代入すると、正しく正規化された $\mathbf{y}$ に関するガウス分布 $\mathcal{N}(\mathbf{y} | \boldsymbol{\mu}_y, \boldsymbol{\Sigma}_y)$ が得られる。
(証明終)
