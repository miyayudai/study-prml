# 演習問題 2.28

## 問題設定
ディリクレ分布が多項分布の共役事前分布であることを示せ。

## 解答と証明
多項分布は、パラメータ $\boldsymbol{\mu} = (\mu_1, \dots, \mu_K)^T$ に対して、データ $\mathbf{x} = (x_1, \dots, x_K)^T$ (ただし $\sum x_k = N$) が観測されたときの確率を以下のように与える。
$$ p(\mathbf{x} | \boldsymbol{\mu}) = \frac{N!}{x_1! \dots x_K!} \prod_{k=1}^K \mu_k^{x_k} $$

事前分布としてディリクレ分布を仮定する。ディリクレ分布の定義は以下の通りである。
$$ p(\boldsymbol{\mu} | \boldsymbol{\alpha}) = \text{Dir}(\boldsymbol{\mu} | \boldsymbol{\alpha}) = \frac{\Gamma(\sum \alpha_k)}{\prod \Gamma(\alpha_k)} \prod_{k=1}^K \mu_k^{\alpha_k - 1} $$
ここで $\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_K)^T$ は事前分布のパラメータ（ハイパーパラメータ）である。

ベイズの定理より、事後分布 $p(\boldsymbol{\mu} | \mathbf{x}, \boldsymbol{\alpha})$ は尤度と事前分布の積に比例する。
$$ p(\boldsymbol{\mu} | \mathbf{x}, \boldsymbol{\alpha}) \propto p(\mathbf{x} | \boldsymbol{\mu}) p(\boldsymbol{\mu} | \boldsymbol{\alpha}) $$

式を代入して整理すると、
$$ p(\boldsymbol{\mu} | \mathbf{x}, \boldsymbol{\alpha}) \propto \left( \prod_{k=1}^K \mu_k^{x_k} \right) \left( \prod_{k=1}^K \mu_k^{\alpha_k - 1} \right) = \prod_{k=1}^K \mu_k^{x_k + \alpha_k - 1} $$

この式は、パラメータ $\boldsymbol{\alpha}' = \boldsymbol{\alpha} + \mathbf{x}$ を持つ新しいディリクレ分布の関数形と完全に一致する。
したがって、正規化定数を補うことで事後分布もディリクレ分布になることがわかる。
$$ p(\boldsymbol{\mu} | \mathbf{x}, \boldsymbol{\alpha}) = \text{Dir}(\boldsymbol{\mu} | \boldsymbol{\alpha} + \mathbf{x}) $$

以上より、ディリクレ分布が多項分布の共役事前分布であることが示された。
(証明終)
