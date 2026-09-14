# 演習問題 2.31

## 問題
多次元確率変数 $\mathbf{x}$ と $\mathbf{z}$ がそれぞれ独立なガウス分布 $p(\mathbf{x}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_x, \mathbf{\Sigma}_x)$ と $p(\mathbf{z}) = \mathcal{N}(\mathbf{z}|\boldsymbol{\mu}_z, \mathbf{\Sigma}_z)$ に従うとする。これらの和 $\mathbf{y} = \mathbf{x} + \mathbf{z}$ の周辺分布 $p(\mathbf{y})$ を、周辺分布 $p(\mathbf{x})$ と条件付き分布 $p(\mathbf{y}|\mathbf{x})$ の積で構成される線形ガウスモデルを考えることによって求めよ。ここで、結果 (2.109) と (2.110) を用いよ。

## 解答

与えられた条件から、$\mathbf{y} = \mathbf{x} + \mathbf{z}$ であり、$\mathbf{x}$ と $\mathbf{z}$ は独立です。$\mathbf{x}$ が与えられたときの $\mathbf{y}$ の分布、すなわち条件付き分布 $p(\mathbf{y}|\mathbf{x})$ を考えます。
$\mathbf{x}$ が固定されているとき、$\mathbf{y}$ の不確実性は $\mathbf{z}$ のみに由来します。したがって、$\mathbf{y}$ の条件付き分布は、$\mathbf{z}$ の分布を $\mathbf{x}$ だけ平行移動させたものになります。
$$
p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y} | \mathbf{x} + \boldsymbol{\mu}_z, \mathbf{\Sigma}_z)
$$

これで、線形ガウスモデルの標準的な形が揃いました。
$$
p(\mathbf{x}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_x, \mathbf{\Sigma}_x)
$$
$$
p(\mathbf{y}|\mathbf{x}) = \mathcal{N}(\mathbf{y}|\mathbf{A}\mathbf{x} + \mathbf{b}, \mathbf{L}^{-1})
$$

教科書の線形ガウスモデル（式 (2.99), (2.100)）のパラメータと対応させると、以下のようになります。
- $\boldsymbol{\mu} = \boldsymbol{\mu}_x$
- $\mathbf{\Lambda}^{-1} = \mathbf{\Sigma}_x$
- $\mathbf{A} = \mathbf{I}$ （単位行列）
- $\mathbf{b} = \boldsymbol{\mu}_z$
- $\mathbf{L}^{-1} = \mathbf{\Sigma}_z$

式 (2.109) と (2.110) によれば、周辺分布 $p(\mathbf{y})$ は平均 $\mathbb{E}[\mathbf{y}]$ と共分散 $\text{cov}[\mathbf{y}]$ を持つガウス分布 $\mathcal{N}(\mathbf{y}|\mathbb{E}[\mathbf{y}], \text{cov}[\mathbf{y}])$ となり、それぞれ次のように与えられます。

$$
\mathbb{E}[\mathbf{y}] = \mathbf{A}\boldsymbol{\mu} + \mathbf{b}
$$
$$
\text{cov}[\mathbf{y}] = \mathbf{L}^{-1} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T
$$

これらの式に先ほどの対応関係を代入します。

**平均の計算:**
$$
\mathbb{E}[\mathbf{y}] = \mathbf{I}\boldsymbol{\mu}_x + \boldsymbol{\mu}_z = \boldsymbol{\mu}_x + \boldsymbol{\mu}_z
$$

**共分散の計算:**
$$
\text{cov}[\mathbf{y}] = \mathbf{\Sigma}_z + \mathbf{I}\mathbf{\Sigma}_x\mathbf{I}^T = \mathbf{\Sigma}_z + \mathbf{\Sigma}_x = \mathbf{\Sigma}_x + \mathbf{\Sigma}_z
$$

以上より、$\mathbf{y} = \mathbf{x} + \mathbf{z}$ の周辺分布は次のガウス分布になることが分かります。
$$
p(\mathbf{y}) = \mathcal{N}(\mathbf{y} | \boldsymbol{\mu}_x + \boldsymbol{\mu}_z, \mathbf{\Sigma}_x + \mathbf{\Sigma}_z)
$$

この結果は、独立なガウス変数の和の分布が、それぞれの平均の和を平均とし、共分散の和を共分散とするガウス分布になるというよく知られた性質を、線形ガウスモデルの一般的な枠組みから導出できることを示しています。
