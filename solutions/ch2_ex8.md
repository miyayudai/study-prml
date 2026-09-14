# 演習問題 2.8 (Exercise 2.8)

## 問題の目的
多変量ガウス分布が与えられたとき、その同時確率分布 $p(\mathbf{x}_a, \mathbf{x}_b)$ から、変数の一部を積分消去した周辺分布 $p(\mathbf{x}_a)$ がどのような分布になるかを導出します。結果として周辺分布もガウス分布になり、その平均と分散が直感的な形（それぞれ全体平均の対応するブロック、全体共分散の対応するブロック）になることを示します。

## 導出のステップ

多変量ガウス分布の確率密度関数は以下のように定義されます。
$$ \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}} \frac{1}{|\boldsymbol{\Sigma}|^{1/2}} \exp\left\{ -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right\} $$

ここで、確率変数ベクトル $\mathbf{x}$、平均ベクトル $\boldsymbol{\mu}$、共分散行列 $\boldsymbol{\Sigma}$ を以下のようにブロック分割します。
$$ \mathbf{x} = \begin{pmatrix} \mathbf{x}_a \\ \mathbf{x}_b \end{pmatrix}, \quad \boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ \boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb} \end{pmatrix} $$

また、共分散行列の逆行列である精度行列 $\boldsymbol{\Lambda} \equiv \boldsymbol{\Sigma}^{-1}$ も同様に分割します。
$$ \boldsymbol{\Lambda} = \begin{pmatrix} \boldsymbol{\Lambda}_{aa} & \boldsymbol{\Lambda}_{ab} \\ \boldsymbol{\Lambda}_{ba} & \boldsymbol{\Lambda}_{bb} \end{pmatrix} $$

周辺分布 $p(\mathbf{x}_a)$ は、同時確率 $p(\mathbf{x}_a, \mathbf{x}_b)$ を $\mathbf{x}_b$ について積分することで得られます。
$$ p(\mathbf{x}_a) = \int p(\mathbf{x}_a, \mathbf{x}_b) d\mathbf{x}_b $$

指数部分（二次形式）のみに注目し、これを $-\frac{1}{2} \Delta$ とおきます。
$$ \Delta = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Lambda} (\mathbf{x} - \boldsymbol{\mu}) $$

$\mathbf{x}_a$ と $\mathbf{x}_b$ に分けて展開すると、
$$ \Delta = (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{aa} (\mathbf{x}_a - \boldsymbol{\mu}_a) + (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$
$$ + (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Lambda}_{ba} (\mathbf{x}_a - \boldsymbol{\mu}_a) + (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Lambda}_{bb} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$

この式を $\mathbf{x}_b$ について平方完成します。$\mathbf{x}_b$ に依存する項を取り出すと、
$$ \Delta = (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Lambda}_{bb} (\mathbf{x}_b - \boldsymbol{\mu}_b) + 2 (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Lambda}_{ba} (\mathbf{x}_a - \boldsymbol{\mu}_a) + \text{const} $$

ここで $\mathbf{m} = \boldsymbol{\mu}_b - \boldsymbol{\Lambda}_{bb}^{-1} \boldsymbol{\Lambda}_{ba} (\mathbf{x}_a - \boldsymbol{\mu}_a)$ とおくと、
$$ \Delta = (\mathbf{x}_b - \mathbf{m})^T \boldsymbol{\Lambda}_{bb} (\mathbf{x}_b - \mathbf{m}) - (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{ab} \boldsymbol{\Lambda}_{bb}^{-1} \boldsymbol{\Lambda}_{ba} (\mathbf{x}_a - \boldsymbol{\mu}_a) + (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{aa} (\mathbf{x}_a - \boldsymbol{\mu}_a) $$

$\mathbf{x}_b$ についての積分は、正規化されていないガウス積分の形になり、$\mathbf{x}_b$ に依存しない項だけが残ります。残った $\mathbf{x}_a$ に関する二次形式は、
$$ (\mathbf{x}_a - \boldsymbol{\mu}_a)^T (\boldsymbol{\Lambda}_{aa} - \boldsymbol{\Lambda}_{ab} \boldsymbol{\Lambda}_{bb}^{-1} \boldsymbol{\Lambda}_{ba}) (\mathbf{x}_a - \boldsymbol{\mu}_a) $$

ブロック行列の逆行列の公式により、$\boldsymbol{\Sigma}_{aa} = (\boldsymbol{\Lambda}_{aa} - \boldsymbol{\Lambda}_{ab} \boldsymbol{\Lambda}_{bb}^{-1} \boldsymbol{\Lambda}_{ba})^{-1}$ となるため、周辺分布もガウス分布となり、そのパラメータは以下で与えられます。
$$ p(\mathbf{x}_a) = \mathcal{N}(\mathbf{x}_a | \boldsymbol{\mu}_a, \boldsymbol{\Sigma}_{aa}) $$
