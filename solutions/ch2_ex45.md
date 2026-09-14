# 演習問題 2.45

## 問題の概要
混合ガウス分布（Gaussian Mixture Model, GMM）において、負担率（responsibilities）の総和が1になることを示し、その確率論的解釈を与える。

## 解答

混合ガウスモデルは、複数のガウス分布の線形結合として次のように定義されます。

$$
p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

ここで、$\pi_k$ は混合係数（事前確率）であり、$\sum_{k=1}^K \pi_k = 1$ かつ $0 \le \pi_k \le 1$ を満たします。

### 負担率の定義
データ点 $\mathbf{x}_n$ が与えられたとき、それがコンポーネント $k$ から生成された事後確率は「負担率」と呼ばれ、$\gamma(z_{nk})$ で表されます。ベイズの定理より、

$$
\gamma(z_{nk}) \equiv p(z_k=1|\mathbf{x}_n) = \frac{p(z_k=1) p(\mathbf{x}_n|z_k=1)}{p(\mathbf{x}_n)}
$$
$$
\gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

### 総和が1になることの証明
与えられたデータ点 $\mathbf{x}_n$ について、すべてのコンポーネント $k$ に関する負担率の総和をとります。

$$
\sum_{k=1}^K \gamma(z_{nk}) = \sum_{k=1}^K \frac{\pi_k \mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

分母は $k$ に依存しないため、和の外に出すことができます（または和を分子に適用します）。

$$
\sum_{k=1}^K \gamma(z_{nk}) = \frac{\sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

分子と分母は全く同じ式になっているため、この値は1になります。

$$
\sum_{k=1}^K \gamma(z_{nk}) = 1
$$

### 確率論的解釈
潜在変数 $\mathbf{z}_n$ は 1-of-K 符号化表現を用いており、$z_{nk} \in \{0, 1\}$ かつ $\sum_{k=1}^K z_{nk} = 1$ を満たします。
負担率 $\gamma(z_{nk})$ は事後確率 $p(z_{nk}=1|\mathbf{x}_n)$ を表しています。確率の公理に従い、互いに排反で網羅的な事象（データ点 $\mathbf{x}_n$ がコンポーネント $1, \dots, K$ のいずれか1つから生成された）の事後確率の和は必ず1になります。これは直感的に「データ点 $\mathbf{x}_n$ を説明する責任（responsibility）を、$K$ 個のコンポーネントで総和が1になるように分割して担っている」と解釈できます。
