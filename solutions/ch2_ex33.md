# 演習問題 2.33

## 問題
演習問題 2.32 と同じ同時分布を考える。今度は平方完成の手法を用いて、条件付き分布 $p(\mathbf{x}|\mathbf{y})$ の平均と共分散の表式を求めよ。さらに、これらが対応する式 (2.111) と (2.112) に一致することを確認せよ。

## 解答

条件付き分布 $p(\mathbf{x}|\mathbf{y})$ を求めるために、$\mathbf{y}$ が与えられた下での $\mathbf{x}$ の分布を考えます。これは同時分布 $p(\mathbf{x}, \mathbf{y})$ を $\mathbf{x}$ の関数として見なすことに等しいです。同時分布の対数の指数部は以下のようになります。

$$
\ln p(\mathbf{x}, \mathbf{y}) = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\mathbf{\Lambda}(\mathbf{x} - \boldsymbol{\mu}) - \frac{1}{2}(\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b})^T\mathbf{L}(\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b}) + \text{const}
$$

この式を展開し、$\mathbf{x}$ に依存する項だけを抽出します。
$$
= -\frac{1}{2}\mathbf{x}^T\mathbf{\Lambda}\mathbf{x} + \mathbf{x}^T\mathbf{\Lambda}\boldsymbol{\mu} - \frac{1}{2}\mathbf{x}^T\mathbf{A}^T\mathbf{L}\mathbf{A}\mathbf{x} + \mathbf{x}^T\mathbf{A}^T\mathbf{L}(\mathbf{y} - \mathbf{b}) + \text{const}'
$$

$\mathbf{x}$ に関する二次項と一次項をそれぞれまとめます。

**二次項:**
$$
-\frac{1}{2}\mathbf{x}^T(\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})\mathbf{x}
$$

**一次項:**
$$
\mathbf{x}^T(\mathbf{\Lambda}\boldsymbol{\mu} + \mathbf{A}^T\mathbf{L}(\mathbf{y} - \mathbf{b}))
$$

一般に、ガウス分布 $\mathcal{N}(\mathbf{x}|\mathbf{m}, \mathbf{\Sigma})$ の対数は次のような形をしています。
$$
\ln \mathcal{N}(\mathbf{x}|\mathbf{m}, \mathbf{\Sigma}) = -\frac{1}{2}(\mathbf{x} - \mathbf{m})^T\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{m}) + \text{const}
$$
$$
= -\frac{1}{2}\mathbf{x}^T\mathbf{\Sigma}^{-1}\mathbf{x} + \mathbf{x}^T\mathbf{\Sigma}^{-1}\mathbf{m} + \text{const}
$$

この一般的な形と、先ほどまとめた $\mathbf{x}$ についての式を係数比較します。

まず、二次項を比較することで、条件付き分布の共分散行列（の逆行列である精度行列）が得られます。
$$
\mathbf{\Sigma}^{-1} = \mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A}
$$
したがって、共分散は次のようになります。
$$
\text{cov}[\mathbf{x}|\mathbf{y}] = (\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})^{-1}
$$

次に、一次項を比較することで平均が得られます。
$$
\mathbf{\Sigma}^{-1}\mathbf{m} = \mathbf{\Lambda}\boldsymbol{\mu} + \mathbf{A}^T\mathbf{L}(\mathbf{y} - \mathbf{b})
$$
両辺に左から $\mathbf{\Sigma}$ （上で求めた共分散行列）を掛けると、平均 $\mathbf{m}$ は次のようになります。
$$
\mathbb{E}[\mathbf{x}|\mathbf{y}] = \mathbf{m} = (\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})^{-1} \{ \mathbf{\Lambda}\boldsymbol{\mu} + \mathbf{A}^T\mathbf{L}(\mathbf{y} - \mathbf{b}) \}
$$

以上の結果をまとめると、条件付き分布 $p(\mathbf{x}|\mathbf{y})$ はガウス分布となり、その平均と共分散は以下のようになります。
$$
\mathbb{E}[\mathbf{x}|\mathbf{y}] = \mathbf{\Sigma}_{x|y} \{ \mathbf{A}^T\mathbf{L}(\mathbf{y} - \mathbf{b}) + \mathbf{\Lambda}\boldsymbol{\mu} \}
$$
$$
\text{cov}[\mathbf{x}|\mathbf{y}] = \mathbf{\Sigma}_{x|y} = (\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})^{-1}
$$

これはまさに教科書の式 (2.111) と (2.112) と一致しており、結果が確かめられました。
