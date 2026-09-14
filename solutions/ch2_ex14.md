# 演習問題 2.14 (Exercise 2.14)

## 問題の目的
多変量ガウス分布における共分散行列 $\boldsymbol{\Sigma}$ の対数尤度関数を微分し、最尤推定量の導出過程において、行列の微分公式を用いる方法を確認します。

## 尤度関数と対数尤度

データセット $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_N)$ が独立同分布 (i.i.d.) で $D$ 次元のガウス分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ から生成されているとします。
対数尤度関数は以下のようになります。
$$ \ln p(\mathbf{X} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{ND}{2} \ln(2\pi) - \frac{N}{2} \ln |\boldsymbol{\Sigma}| - \frac{1}{2} \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}) $$

簡略化のため、共分散の最尤推定を行うために $\boldsymbol{\mu}$ はすでに最尤推定量 $\boldsymbol{\mu}_{ML} = \frac{1}{N}\sum \mathbf{x}_n$ に設定されているとし、精度行列 $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ で微分することを考えます。

## 行列微分を用いた最大化

対数尤度関数を $\boldsymbol{\Lambda}$ について書き換えます。
$$ \ln p(\mathbf{X} | \boldsymbol{\Lambda}) = \frac{N}{2} \ln |\boldsymbol{\Lambda}| - \frac{1}{2} \text{Tr}(\boldsymbol{\Lambda} \mathbf{S}) + \text{const} $$
ここで $\mathbf{S}$ は非正規化の散布行列 $\mathbf{S} = \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T$ です。

行列の微分公式 $\frac{\partial}{\partial \boldsymbol{\Lambda}} \ln |\boldsymbol{\Lambda}| = \boldsymbol{\Lambda}^{-1}$ （すなわち $\boldsymbol{\Sigma}$）および $\frac{\partial}{\partial \boldsymbol{\Lambda}} \text{Tr}(\boldsymbol{\Lambda} \mathbf{S}) = \mathbf{S}$ を用います。

対数尤度を $\boldsymbol{\Lambda}$ で微分し、ゼロとおきます。
$$ \frac{\partial \ln p}{\partial \boldsymbol{\Lambda}} = \frac{N}{2} \boldsymbol{\Sigma} - \frac{1}{2} \mathbf{S} = \mathbf{0} $$

これを解くことで、共分散行列の最尤推定量 $\boldsymbol{\Sigma}_{ML}$ が直ちに得られます。
$$ \boldsymbol{\Sigma}_{ML} = \frac{1}{N} \mathbf{S} = \frac{1}{N} \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu}_{ML})(\mathbf{x}_n - \boldsymbol{\mu}_{ML})^T $$

この結果から、ガウス分布の分散の最尤推定量が標本分散（不偏分散ではない）となることが行列微分の観点から示されました。
