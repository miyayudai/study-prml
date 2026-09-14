# 演習問題 2.19: マハラノビス距離とガウス分布

多変量ガウス分布において、指数部に現れる二次形式（マハラノビス距離の2乗）が特定の確率分布に従うことを示します。

## 1. 問題の設定

$D$ 次元の確率変数ベクトル $\mathbf{x}$ が多変量ガウス分布 $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma})$ に従うとします。マハラノビス距離の2乗 $\Delta^2$ は以下で定義されます。

$$
\Delta^2 = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})
$$

この $\Delta^2$ が自由度 $D$ のカイ二乗分布（$\chi^2$ 分布）に従うことを導出します。

## 2. 変数変換による標準化

共分散行列 $\boldsymbol{\Sigma}$ は実対称行列であり、正定値行列であるため、固有値分解（またはコレスキー分解）を用いて $\boldsymbol{\Sigma} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$ と書くことができます。ここで $\mathbf{U}$ は直交行列、$\mathbf{\Lambda}$ は正の固有値 $\lambda_i$ を対角成分に持つ対角行列です。

新しい確率変数ベクトル $\mathbf{y}$ を次のように定義します。
$$
\mathbf{y} = \boldsymbol{\Lambda}^{-1/2} \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu})
$$

この変換により、$\mathbf{x} - \boldsymbol{\mu} = \mathbf{U} \boldsymbol{\Lambda}^{1/2} \mathbf{y}$ となります。これを $\Delta^2$ の式に代入します。

$$
\Delta^2 = (\mathbf{U} \boldsymbol{\Lambda}^{1/2} \mathbf{y})^T (\mathbf{U} \mathbf{\Lambda} \mathbf{U}^T)^{-1} (\mathbf{U} \boldsymbol{\Lambda}^{1/2} \mathbf{y})
$$

転置と逆行列の性質を用いて展開します。
$$
\Delta^2 = \mathbf{y}^T \boldsymbol{\Lambda}^{1/2} \mathbf{U}^T (\mathbf{U} \boldsymbol{\Lambda}^{-1} \mathbf{U}^T) \mathbf{U} \boldsymbol{\Lambda}^{1/2} \mathbf{y}
$$

直交行列の性質 $\mathbf{U}^T \mathbf{U} = \mathbf{I}$ を利用して整理すると、
$$
\Delta^2 = \mathbf{y}^T \boldsymbol{\Lambda}^{1/2} \boldsymbol{\Lambda}^{-1} \boldsymbol{\Lambda}^{1/2} \mathbf{y} = \mathbf{y}^T \mathbf{y} = \sum_{i=1}^{D} y_i^2
$$
となります。

## 3. $\mathbf{y}$ の分布

変数 $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ をアフィン変換した $\mathbf{y}$ もまたガウス分布に従います。その平均と共分散を計算します。

平均：
$$ \mathbb{E}[\mathbf{y}] = \boldsymbol{\Lambda}^{-1/2} \mathbf{U}^T (\mathbb{E}[\mathbf{x}] - \boldsymbol{\mu}) = \mathbf{0} $$

共分散：
$$ \text{cov}[\mathbf{y}] = \boldsymbol{\Lambda}^{-1/2} \mathbf{U}^T \text{cov}[\mathbf{x}] (\boldsymbol{\Lambda}^{-1/2} \mathbf{U}^T)^T = \boldsymbol{\Lambda}^{-1/2} \mathbf{U}^T \boldsymbol{\Sigma} \mathbf{U} \boldsymbol{\Lambda}^{-1/2} $$
$\boldsymbol{\Sigma} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T$ より $\mathbf{U}^T \boldsymbol{\Sigma} \mathbf{U} = \mathbf{\Lambda}$ であるため、
$$ \text{cov}[\mathbf{y}] = \boldsymbol{\Lambda}^{-1/2} \mathbf{\Lambda} \boldsymbol{\Lambda}^{-1/2} = \mathbf{I} $$

したがって、$\mathbf{y} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ となり、各成分 $y_i$ は互いに独立な標準正規分布 $\mathcal{N}(0, 1)$ に従います。

## 4. 結論

$\Delta^2$ は独立な標準正規分布に従う $D$ 個の確率変数 $y_i$ の平方和として表されます。
$$ \Delta^2 = \sum_{i=1}^{D} y_i^2 $$

標準正規分布の平方和の分布は、定義により自由度 $D$ の**カイ二乗分布**となります。
これより、マハラノビス距離の2乗は $\chi^2(D)$ に従うことが証明されました。
