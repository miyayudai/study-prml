# Exercise 2.50 (共役事前分布の事後更新の一般証明)

## 課題
指数型分布族の一般共役事前分布を導入し、観測データに基づいてベイズ更新を行った際の事後分布が事前分布と同じ関数形（共役性）を保ち、そのパラメータがどのように更新されるかを導出せよ。

## 解答と解説
指数型分布族に属する確率分布のモデル（尤度関数）は次のように表されます。
$$ p(\mathbf{x} | \boldsymbol{\eta}) = h(\mathbf{x}) g(\boldsymbol{\eta}) \exp\left( \boldsymbol{\eta}^T \mathbf{u}(\mathbf{x}) \right) $$
ここで、$g(\boldsymbol{\eta}) = \exp(-A(\boldsymbol{\eta}))$ とすることもできます。

独立同分布 (i.i.d.) に従う $N$ 個のデータ点 $\mathbf{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ が与えられたときの尤度関数は以下のようになります。
$$ p(\mathbf{X} | \boldsymbol{\eta}) = \prod_{n=1}^N h(\mathbf{x}_n) g(\boldsymbol{\eta}) \exp\left( \boldsymbol{\eta}^T \mathbf{u}(\mathbf{x}_n) \right) $$
$$ = \left( \prod_{n=1}^N h(\mathbf{x}_n) \right) g(\boldsymbol{\eta})^N \exp\left( \boldsymbol{\eta}^T \sum_{n=1}^N \mathbf{u}(\mathbf{x}_n) \right) $$

### 共役事前分布の定義
この尤度関数に対して共役となる事前分布 $p(\boldsymbol{\eta})$ は、尤度関数と同じ $\boldsymbol{\eta}$ に関する依存性を持つように設計されます。一般に次のような形をとります。
$$ p(\boldsymbol{\eta} | \boldsymbol{\chi}, \nu) = f(\boldsymbol{\chi}, \nu) g(\boldsymbol{\eta})^\nu \exp\left( \boldsymbol{\eta}^T \boldsymbol{\chi} \right) $$
ここで：
- $\nu$ は事前分布における「擬似的な観測データ数」に対応するスカラーパラメータ
- $\boldsymbol{\chi}$ は十分統計量の「擬似的な和」に対応するベクトルパラメータ
- $f(\boldsymbol{\chi}, \nu)$ は確率分布の総和（積分）を 1 にするための規格化定数です。

### ベイズ更新と事後分布の導出
ベイズの定理を用いて、データ $\mathbf{X}$ を観測した後の事後分布 $p(\boldsymbol{\eta} | \mathbf{X}, \boldsymbol{\chi}, \nu)$ を求めます。事後分布は、尤度関数と事前分布の積に比例します。
$$ p(\boldsymbol{\eta} | \mathbf{X}, \boldsymbol{\chi}, \nu) \propto p(\mathbf{X} | \boldsymbol{\eta}) p(\boldsymbol{\eta} | \boldsymbol{\chi}, \nu) $$

尤度関数と事前分布の式を代入し、$\boldsymbol{\eta}$ に依存する項のみを取り出して次のように整理できます。
$$ p(\boldsymbol{\eta} | \mathbf{X}, \boldsymbol{\chi}, \nu) \propto \left[ g(\boldsymbol{\eta})^N \exp\left( \boldsymbol{\eta}^T \sum_{n=1}^N \mathbf{u}(\mathbf{x}_n) \right) \right] \times \left[ g(\boldsymbol{\eta})^\nu \exp\left( \boldsymbol{\eta}^T \boldsymbol{\chi} \right) \right] $$

指数と $g(\boldsymbol{\eta})$ のべき乗をまとめます。
$$ p(\boldsymbol{\eta} | \mathbf{X}, \boldsymbol{\chi}, \nu) \propto g(\boldsymbol{\eta})^{N + \nu} \exp\left( \boldsymbol{\eta}^T \left( \boldsymbol{\chi} + \sum_{n=1}^N \mathbf{u}(\mathbf{x}_n) \right) \right) $$

この式を事前分布の定義式 $p(\boldsymbol{\eta} | \boldsymbol{\chi}, \nu) \propto g(\boldsymbol{\eta})^\nu \exp(\boldsymbol{\eta}^T \boldsymbol{\chi})$ と見比べると、全く同じ関数形になっていることがわかります。

これにより、事前分布が共役であることが示され、更新された新しいパラメータ $\boldsymbol{\chi}', \nu'$ は以下のようになることが分かります。
$$ \nu' = \nu + N $$
$$ \boldsymbol{\chi}' = \boldsymbol{\chi} + \sum_{n=1}^N \mathbf{u}(\mathbf{x}_n) $$

### 結論
指数型分布族の一般共役事前分布において、事後分布のパラメータは、
1. 観測データの数 $N$ を事前分布のパラメータ $\nu$ に足し合わせる ($\nu' = \nu + N$)
2. 観測された十分統計量の和 $\sum_{n=1}^N \mathbf{u}(\mathbf{x}_n)$ を事前分布のパラメータ $\boldsymbol{\chi}$ に足し合わせる ($\boldsymbol{\chi}' = \boldsymbol{\chi} + \sum_{n=1}^N \mathbf{u}(\mathbf{x}_n)$)
ことによって更新されます。これにより、パラメータ $\nu$ と $\boldsymbol{\chi}$ が「擬似的な観測数」および「擬似的な十分統計量の和」として解釈できることが数式から直感的に示されました。
