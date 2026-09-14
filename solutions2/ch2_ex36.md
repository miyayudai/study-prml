## Exercise 2.36 (ガウス-ウィシャート分布の事後分布更新)

多変量ガウス分布において、平均 $\boldsymbol{\mu}$ と精度行列 $\boldsymbol{\Lambda}$ の両方が未知である場合に対する共役事前分布であるガウス-ウィシャート分布（Gaussian-Wishart distribution）を考え、ベイズ更新によって事後分布も同じくガウス-ウィシャート分布となることを証明します。

### 尤度関数
$N$ 個の独立な観測データ $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_N)^T$ が与えられたとき、尤度関数は以下のようになります。
$$
p(\mathbf{X} | \boldsymbol{\mu}, \boldsymbol{\Lambda}) = \prod_{n=1}^N \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1}) \propto |\boldsymbol{\Lambda}|^{N/2} \exp\left\{ -\frac{1}{2} \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})^T \boldsymbol{\Lambda} (\mathbf{x}_n - \boldsymbol{\mu}) \right\}
$$
ここで、データの標本平均を $\mathbf{\bar{x}} = \frac{1}{N}\sum_{n=1}^N \mathbf{x}_n$ とすると、指数部の和は次のように変形できます。
$$
\sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})^T \boldsymbol{\Lambda} (\mathbf{x}_n - \boldsymbol{\mu}) = \sum_{n=1}^N (\mathbf{x}_n - \mathbf{\bar{x}})^T \boldsymbol{\Lambda} (\mathbf{x}_n - \mathbf{\bar{x}}) + N(\mathbf{\bar{x}} - \boldsymbol{\mu})^T \boldsymbol{\Lambda} (\mathbf{\bar{x}} - \boldsymbol{\mu})
$$
さらに、第一項にトレースの巡回不変性を用いると、
$$
\sum_{n=1}^N (\mathbf{x}_n - \mathbf{\bar{x}})^T \boldsymbol{\Lambda} (\mathbf{x}_n - \mathbf{\bar{x}}) = \mathrm{Tr}\left( \boldsymbol{\Lambda} \sum_{n=1}^N (\mathbf{x}_n - \mathbf{\bar{x}})(\mathbf{x}_n - \mathbf{\bar{x}})^T \right)
$$
したがって、
$$
p(\mathbf{X} | \boldsymbol{\mu}, \boldsymbol{\Lambda}) \propto |\boldsymbol{\Lambda}|^{N/2} \exp\left\{ -\frac{N}{2} (\boldsymbol{\mu} - \mathbf{\bar{x}})^T \boldsymbol{\Lambda} (\boldsymbol{\mu} - \mathbf{\bar{x}}) - \frac{1}{2} \mathrm{Tr}(\boldsymbol{\Lambda} \mathbf{S}) \right\}
$$
ここで、$\mathbf{S} = \sum_{n=1}^N (\mathbf{x}_n - \mathbf{\bar{x}})(\mathbf{x}_n - \mathbf{\bar{x}})^T$ は非正規化の標本共分散行列（散布行列）です。

### ガウス-ウィシャート事前分布
ガウス-ウィシャート分布 $p(\boldsymbol{\mu}, \boldsymbol{\Lambda}) = p(\boldsymbol{\mu} | \boldsymbol{\Lambda}) p(\boldsymbol{\Lambda})$ は以下のように定義されます。
$$
p(\boldsymbol{\mu}, \boldsymbol{\Lambda}) \propto \left( |\boldsymbol{\Lambda}|^{1/2} \exp\left\{ -\frac{\beta_0}{2}(\boldsymbol{\mu} - \mathbf{m}_0)^T \boldsymbol{\Lambda} (\boldsymbol{\mu} - \mathbf{m}_0) \right\} \right) \left( |\boldsymbol{\Lambda}|^{(\nu_0 - D - 1)/2} \exp\left\{ -\frac{1}{2}\mathrm{Tr}(\mathbf{W}_0^{-1} \boldsymbol{\Lambda}) \right\} \right)
$$
$$
p(\boldsymbol{\mu}, \boldsymbol{\Lambda}) \propto |\boldsymbol{\Lambda}|^{(\nu_0 - D)/2} \exp\left\{ -\frac{\beta_0}{2}(\boldsymbol{\mu} - \mathbf{m}_0)^T \boldsymbol{\Lambda} (\boldsymbol{\mu} - \mathbf{m}_0) - \frac{1}{2}\mathrm{Tr}(\mathbf{W}_0^{-1} \boldsymbol{\Lambda}) \right\}
$$

### 事後分布の導出
ベイズの定理より、事後分布は尤度と事前分布の積に比例します。
$$
p(\boldsymbol{\mu}, \boldsymbol{\Lambda} | \mathbf{X}) \propto p(\mathbf{X} | \boldsymbol{\mu}, \boldsymbol{\Lambda}) p(\boldsymbol{\mu}, \boldsymbol{\Lambda})
$$
$$
\propto |\boldsymbol{\Lambda}|^{\frac{\nu_0 + N - D}{2}} \exp\left\{ -\frac{1}{2} \left[ \beta_0(\boldsymbol{\mu} - \mathbf{m}_0)^T \boldsymbol{\Lambda} (\boldsymbol{\mu} - \mathbf{m}_0) + N(\boldsymbol{\mu} - \mathbf{\bar{x}})^T \boldsymbol{\Lambda} (\boldsymbol{\mu} - \mathbf{\bar{x}}) + \mathrm{Tr}\left((\mathbf{W}_0^{-1} + \mathbf{S}) \boldsymbol{\Lambda}\right) \right] \right\}
$$
ここで $\boldsymbol{\mu}$ に関する二次形式を平方完成します。
$$
\beta_0(\boldsymbol{\mu} - \mathbf{m}_0)^T \boldsymbol{\Lambda} (\boldsymbol{\mu} - \mathbf{m}_0) + N(\boldsymbol{\mu} - \mathbf{\bar{x}})^T \boldsymbol{\Lambda} (\boldsymbol{\mu} - \mathbf{\bar{x}}) = (\beta_0 + N)(\boldsymbol{\mu} - \mathbf{m}_N)^T \boldsymbol{\Lambda} (\boldsymbol{\mu} - \mathbf{m}_N) + \frac{\beta_0 N}{\beta_0 + N}(\mathbf{\bar{x}} - \mathbf{m}_0)^T \boldsymbol{\Lambda} (\mathbf{\bar{x}} - \mathbf{m}_0)
$$
ただし、$\mathbf{m}_N = \frac{\beta_0 \mathbf{m}_0 + N \mathbf{\bar{x}}}{\beta_0 + N}$ としました。
また、余った項をトレースに組み込みます。
$$
(\mathbf{\bar{x}} - \mathbf{m}_0)^T \boldsymbol{\Lambda} (\mathbf{\bar{x}} - \mathbf{m}_0) = \mathrm{Tr}\left( \boldsymbol{\Lambda} (\mathbf{\bar{x}} - \mathbf{m}_0)(\mathbf{\bar{x}} - \mathbf{m}_0)^T \right)
$$
これらをまとめると、事後分布の指数部は以下のようになります。
$$
-\frac{1}{2} \left[ (\beta_0 + N)(\boldsymbol{\mu} - \mathbf{m}_N)^T \boldsymbol{\Lambda} (\boldsymbol{\mu} - \mathbf{m}_N) + \mathrm{Tr}\left( \left( \mathbf{W}_0^{-1} + \mathbf{S} + \frac{\beta_0 N}{\beta_0 + N}(\mathbf{\bar{x}} - \mathbf{m}_0)(\mathbf{\bar{x}} - \mathbf{m}_0)^T \right) \boldsymbol{\Lambda} \right) \right]
$$
事前分布と同じ関数形になっていることが確認できます。したがって事後分布もガウス-ウィシャート分布 $\mathcal{NW}(\boldsymbol{\mu}, \boldsymbol{\Lambda} | \mathbf{m}_N, \beta_N, \mathbf{W}_N, \nu_N)$ となり、パラメータの更新則は以下のようになります。
$$
\beta_N = \beta_0 + N
$$
$$
\mathbf{m}_N = \frac{\beta_0 \mathbf{m}_0 + N \mathbf{\bar{x}}}{\beta_0 + N}
$$
$$
\mathbf{W}_N^{-1} = \mathbf{W}_0^{-1} + \mathbf{S} + \frac{\beta_0 N}{\beta_0 + N}(\mathbf{\bar{x}} - \mathbf{m}_0)(\mathbf{\bar{x}} - \mathbf{m}_0)^T
$$
$$
\nu_N = \nu_0 + N
$$
この結果は、散布行列 $\mathbf{S}$ が尺度行列の逆行列に加算され、さらに事前平均と標本平均の差に基づく項が追加されることを示しています。
