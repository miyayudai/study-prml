# 演習問題 2.27

## 問題設定
ベイズの定理を用いて、線形ガウスモデルにおける事後分布を求める公式（線形ガウスモデルの結合則）を導出せよ。事前分布を $p(\mathbf{x}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \mathbf{\Lambda}^{-1})$、条件付き分布を $p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y} | \mathbf{A}\mathbf{x} + \mathbf{b}, \mathbf{L}^{-1})$ とする。

## 解答と証明
結合分布 $p(\mathbf{x}, \mathbf{y}) = p(\mathbf{y}|\mathbf{x})p(\mathbf{x})$ の対数をとると、以下の二次形式が得られる。
$$ \ln p(\mathbf{x}, \mathbf{y}) = -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Lambda} (\mathbf{x} - \boldsymbol{\mu}) - \frac{1}{2} (\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b})^T \mathbf{L} (\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b}) + \text{const} $$

$\mathbf{x}$ と $\mathbf{y}$ の同時分布もガウス分布となるため、これを $\mathbf{z} = \begin{pmatrix} \mathbf{x} \\ \mathbf{y} \end{pmatrix}$ の二次形式に整理する。
二次項は、
$$ -\frac{1}{2} \mathbf{x}^T (\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A}) \mathbf{x} - \frac{1}{2} \mathbf{y}^T \mathbf{L} \mathbf{y} + \frac{1}{2} \mathbf{x}^T \mathbf{A}^T \mathbf{L} \mathbf{y} + \frac{1}{2} \mathbf{y}^T \mathbf{L} \mathbf{A} \mathbf{x} $$
となる。
これから、精度行列 $\mathbf{R}$ は、
$$ \mathbf{R} = \begin{pmatrix} \mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A} & -\mathbf{A}^T\mathbf{L} \\ -\mathbf{L}\mathbf{A} & \mathbf{L} \end{pmatrix} $$
と読み取れる。

共分散行列 $\boldsymbol{\Sigma} = \mathbf{R}^{-1}$ は、ブロック行列の逆行列公式を用いて求めることができる。
周辺分布 $p(\mathbf{y})$ と事後分布 $p(\mathbf{x}|\mathbf{y})$ も、同時分布のパラメータからガウス分布の公式を用いて即座に得られる。
事後分布の精度行列は $\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A}$ となり、平均は平方完成における一次の項から計算される。
(証明終)
