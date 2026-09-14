## Exercise 2.26 (ガウス分布の最尤推定)

**【問題】**
対数尤度を偏微分して、標本平均と標本共分散行列の最尤推定量が導かれることを示せ。

**【解答】**
多変量ガウス分布における単一の観測データ $\mathbf{x}$ の対数尤度は次のように与えられます。
$$ \ln \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{D}{2}\ln(2\pi) - \frac{1}{2}\ln|\boldsymbol{\Sigma}| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) $$

したがって、$N$ 個の独立な観測データセット $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_N)^T$ に関する対数尤度関数は以下のようになります。
$$ \ln p(\mathbf{X} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{ND}{2}\ln(2\pi) - \frac{N}{2}\ln|\boldsymbol{\Sigma}| - \frac{1}{2}\sum_{n=1}^N (\mathbf{x}_n-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}_n-\boldsymbol{\mu}) $$

**1. 標本平均 $\boldsymbol{\mu}_{\mathrm{ML}}$ の導出**
対数尤度関数を $\boldsymbol{\mu}$ に関して偏微分します。二次形式の微分公式 $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}-\mathbf{a})^T \mathbf{A} (\mathbf{x}-\mathbf{a}) = 2\mathbf{A}(\mathbf{x}-\mathbf{a})$ （$\mathbf{A}$ が対称行列の場合）を用いると、
$$ \frac{\partial}{\partial \boldsymbol{\mu}} \ln p(\mathbf{X} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \sum_{n=1}^N \boldsymbol{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}) $$
となります。これを $0$ とおいて最尤推定量を求めます。
$$ \sum_{n=1}^N \boldsymbol{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}_{\mathrm{ML}}) = \mathbf{0} $$
$\boldsymbol{\Sigma}^{-1}$ は正則であるため両辺に左から $\boldsymbol{\Sigma}$ を掛けて整理します。
$$ \sum_{n=1}^N \mathbf{x}_n - N\boldsymbol{\mu}_{\mathrm{ML}} = \mathbf{0} \quad \Rightarrow \quad \boldsymbol{\mu}_{\mathrm{ML}} = \frac{1}{N}\sum_{n=1}^N \mathbf{x}_n $$
これにより、平均の最尤推定量が標本平均と一致することが示されました。

**2. 標本共分散行列 $\boldsymbol{\Sigma}_{\mathrm{ML}}$ の導出**
次に、$\boldsymbol{\Sigma}$ についての最尤推定量を求めます。行列の微分のために精度行列 $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ を導入して対数尤度関数を書き換えます。
$$ \ln p(\mathbf{X} | \boldsymbol{\mu}, \boldsymbol{\Lambda}) = -\frac{ND}{2}\ln(2\pi) + \frac{N}{2}\ln|\boldsymbol{\Lambda}| - \frac{1}{2}\sum_{n=1}^N (\mathbf{x}_n-\boldsymbol{\mu})^T\boldsymbol{\Lambda}(\mathbf{x}_n-\boldsymbol{\mu}) $$
トレースの巡回性質 $\mathbf{x}^T \mathbf{A} \mathbf{x} = \mathrm{Tr}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = \mathrm{Tr}(\mathbf{A} \mathbf{x}\mathbf{x}^T)$ を用いると、右辺の最後の項は
$$ -\frac{1}{2}\sum_{n=1}^N \mathrm{Tr}\left(\boldsymbol{\Lambda}(\mathbf{x}_n-\boldsymbol{\mu})(\mathbf{x}_n-\boldsymbol{\mu})^T\right) $$
と表せます。この対数尤度を $\boldsymbol{\Lambda}$ で偏微分します。行列の微分公式 $\frac{\partial}{\partial \mathbf{A}}\ln|\mathbf{A}| = (\mathbf{A}^{-1})^T$ と $\frac{\partial}{\partial \mathbf{A}}\mathrm{Tr}(\mathbf{A}\mathbf{B}) = \mathbf{B}^T$ を利用します（$\boldsymbol{\Lambda}$ は対称行列であることを考慮）。
$$ \frac{\partial}{\partial \boldsymbol{\Lambda}} \ln p(\mathbf{X} | \boldsymbol{\mu}_{\mathrm{ML}}, \boldsymbol{\Lambda}) = \frac{N}{2}\boldsymbol{\Lambda}^{-1} - \frac{1}{2}\sum_{n=1}^N (\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})^T $$
これを $0$ とおき、$\boldsymbol{\Sigma}_{\mathrm{ML}} = \boldsymbol{\Lambda}^{-1}$ とすると、
$$ N\boldsymbol{\Sigma}_{\mathrm{ML}} - \sum_{n=1}^N (\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})^T = \mathbf{0} $$
$$ \boldsymbol{\Sigma}_{\mathrm{ML}} = \frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})^T $$
以上より、共分散行列の最尤推定量が標本共分散と一致することが導出されました。
