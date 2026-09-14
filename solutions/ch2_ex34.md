# 演習問題 2.34

## 問題
多変量ガウス分布の共分散行列の最尤解を見つけるために、対数尤度関数 (2.118) を $\mathbf{\Sigma}$ について最大化する必要がある。このとき、共分散行列は対称かつ正定値でなければならないという制約があるが、ここではこれらの制約を無視して直接最大化を行う。付録Cの公式 (C.21)、(C.26)、(C.28) を用いて、対数尤度関数 (2.118) を最大化する共分散行列 $\mathbf{\Sigma}$ が標本共分散 (2.122) で与えられることを示せ。

## 解答

データ集合 $\mathbf{X}$ に対する多変量ガウス分布の対数尤度関数 (2.118) は次のように与えられます。

$$
\ln p(\mathbf{X} | \boldsymbol{\mu}, \mathbf{\Sigma}) = -\frac{ND}{2}\ln(2\pi) - \frac{N}{2}\ln|\mathbf{\Sigma}| - \frac{1}{2}\sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu})
$$

対数尤度関数の第3項の総和の中身はスカラーであるため、跡（トレース）の性質 $\text{Tr}(c) = c$ および $\text{Tr}(\mathbf{A}\mathbf{B}) = \text{Tr}(\mathbf{B}\mathbf{A})$ を用いて次のように変形できます。
$$
(\mathbf{x}_n - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}) = \text{Tr} \left( (\mathbf{x}_n - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}) \right)
= \text{Tr} \left( \mathbf{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T \right)
$$

これにより、対数尤度関数は次のように書き直せます。
$$
L(\mathbf{\Sigma}) = -\frac{N}{2}\ln|\mathbf{\Sigma}| - \frac{1}{2} \text{Tr} \left( \mathbf{\Sigma}^{-1} \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T \right)
$$
ここで、非正規化された散布行列を $\mathbf{S}_{unscaled} = \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T$ と置きます。
$$
L(\mathbf{\Sigma}) = -\frac{N}{2}\ln|\mathbf{\Sigma}| - \frac{1}{2} \text{Tr} (\mathbf{\Sigma}^{-1} \mathbf{S}_{unscaled})
$$

これを $\mathbf{\Sigma}$ について偏微分します。付録Cの公式を用います。
まず、公式 (C.28) により、行列式 $\ln|\mathbf{\Sigma}|$ の微分は次のようになります。
$$
\frac{\partial}{\partial \mathbf{\Sigma}} \ln|\mathbf{\Sigma}| = (\mathbf{\Sigma}^{-1})^T
$$

次に、第2項の $\text{Tr} (\mathbf{\Sigma}^{-1} \mathbf{S}_{unscaled})$ の微分です。微小変化 $d\mathbf{\Sigma}$ を考えると、公式 (C.26) より $d(\mathbf{\Sigma}^{-1}) = -\mathbf{\Sigma}^{-1}(d\mathbf{\Sigma})\mathbf{\Sigma}^{-1}$ です。
これと公式 (C.21) $\frac{\partial}{\partial \mathbf{A}} \text{Tr}(\mathbf{A}\mathbf{B}) = \mathbf{B}^T$ のようなトレースの性質を用いて計算します。
$$
d\text{Tr}(\mathbf{\Sigma}^{-1} \mathbf{S}_{unscaled}) = \text{Tr}(d(\mathbf{\Sigma}^{-1}) \mathbf{S}_{unscaled}) = \text{Tr}(-\mathbf{\Sigma}^{-1}(d\mathbf{\Sigma})\mathbf{\Sigma}^{-1} \mathbf{S}_{unscaled}) = -\text{Tr}(\mathbf{\Sigma}^{-1}\mathbf{S}_{unscaled}\mathbf{\Sigma}^{-1} d\mathbf{\Sigma})
$$
したがって、微分は行列の成分比較により次のようになります（ここでは対称性を仮定せずに微分しています）。
$$
\frac{\partial}{\partial \mathbf{\Sigma}} \text{Tr} (\mathbf{\Sigma}^{-1} \mathbf{S}_{unscaled}) = -(\mathbf{\Sigma}^{-1}\mathbf{S}_{unscaled}\mathbf{\Sigma}^{-1})^T = -(\mathbf{\Sigma}^{-1})^T \mathbf{S}_{unscaled}^T (\mathbf{\Sigma}^{-1})^T
$$

これらを合わせて、対数尤度関数の $\mathbf{\Sigma}$ に関する勾配を $0$ と置きます。
$$
\frac{\partial L}{\partial \mathbf{\Sigma}} = -\frac{N}{2} (\mathbf{\Sigma}^{-1})^T + \frac{1}{2} (\mathbf{\Sigma}^{-1})^T \mathbf{S}_{unscaled}^T (\mathbf{\Sigma}^{-1})^T = 0
$$

全体に右から $( (\mathbf{\Sigma}^{-1})^T )^{-1} = \mathbf{\Sigma}^T$ を掛け、さらに左から $\mathbf{\Sigma}^T$ を掛けます。
$$
-\frac{N}{2} \mathbf{\Sigma}^T + \frac{1}{2} \mathbf{S}_{unscaled}^T = 0
$$
$$
N \mathbf{\Sigma}^T = \mathbf{S}_{unscaled}^T
$$
両辺の転置を取ると、
$$
N \mathbf{\Sigma} = \mathbf{S}_{unscaled} = \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T
$$
$$
\mathbf{\Sigma} = \frac{1}{N} \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T
$$

これが最大化を与える共分散行列 $\mathbf{\Sigma}_{ML}$ であり、式 (2.122) の標本共分散と一致します。最終的な結果が対称行列かつ正定値行列になっているため、事前の制約を満たす妥当な解であることが確認できます。
