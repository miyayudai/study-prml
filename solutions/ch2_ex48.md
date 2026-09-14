# 演習問題 2.48

## 問題の概要
ベルヌーイ分布の混合モデルについて、パラメータ $\boldsymbol{\mu}_k$ の最尤推定におけるEMアルゴリズムのMステップでの更新式を導出する。

## 解答

演習問題 2.47に引き続き、ベルヌーイ分布の混合モデルの対数尤度を最大化します。
データ点 $\mathbf{x} = (x_1, \dots, x_D)^T$ （ただし $x_i \in \{0, 1\}$）に対するベルヌーイ分布は以下のように表されます。

$$
p(\mathbf{x}|\boldsymbol{\mu}_k) = \prod_{i=1}^D \mu_{ki}^{x_i} (1 - \mu_{ki})^{1 - x_i}
$$

対数尤度関数は以下の通りです。

$$
\ln p(\mathbf{X}|\boldsymbol{\mu}, \boldsymbol{\pi}) = \sum_{n=1}^N \ln \left\{ \sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j) \right\}
$$

### パラメータ $\mu_{ki}$ についての微分
対数尤度を特定のパラメータ $\mu_{ki}$ について偏微分します。対数の中身の和に対する微分の連鎖律を適用します。

$$
\frac{\partial}{\partial \mu_{ki}} \ln p(\mathbf{X}|\boldsymbol{\mu}, \boldsymbol{\pi}) = \sum_{n=1}^N \frac{1}{\sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j)} \frac{\partial}{\partial \mu_{ki}} \left( \sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j) \right)
$$

和の中の $j=k$ の項のみが $\mu_{ki}$ に依存するため、微分は以下のようになります。

$$
= \sum_{n=1}^N \frac{\pi_k}{\sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j)} \frac{\partial}{\partial \mu_{ki}} p(\mathbf{x}_n|\boldsymbol{\mu}_k)
$$

ここで負担率 $\gamma(z_{nk}) = \frac{\pi_k p(\mathbf{x}_n|\boldsymbol{\mu}_k)}{\sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j)}$ を用いるために、分母分子に $p(\mathbf{x}_n|\boldsymbol{\mu}_k)$ を補います。

$$
= \sum_{n=1}^N \frac{\pi_k p(\mathbf{x}_n|\boldsymbol{\mu}_k)}{\sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j)} \frac{1}{p(\mathbf{x}_n|\boldsymbol{\mu}_k)} \frac{\partial}{\partial \mu_{ki}} p(\mathbf{x}_n|\boldsymbol{\mu}_k)
$$

$$
= \sum_{n=1}^N \gamma(z_{nk}) \frac{\partial}{\partial \mu_{ki}} \ln p(\mathbf{x}_n|\boldsymbol{\mu}_k)
$$

### 対数ベルヌーイ分布の微分
ベルヌーイ分布の対数は次のようになります。

$$
\ln p(\mathbf{x}_n|\boldsymbol{\mu}_k) = \sum_{i=1}^D \left[ x_{ni} \ln \mu_{ki} + (1 - x_{ni}) \ln (1 - \mu_{ki}) \right]
$$

これを $\mu_{ki}$ について偏微分します。

$$
\frac{\partial}{\partial \mu_{ki}} \ln p(\mathbf{x}_n|\boldsymbol{\mu}_k) = \frac{x_{ni}}{\mu_{ki}} - \frac{1 - x_{ni}}{1 - \mu_{ki}} = \frac{x_{ni}(1 - \mu_{ki}) - \mu_{ki}(1 - x_{ni})}{\mu_{ki}(1 - \mu_{ki})} = \frac{x_{ni} - \mu_{ki}}{\mu_{ki}(1 - \mu_{ki})}
$$

### 最尤方程式の導出
これを元の式に代入し、0とおきます。

$$
\sum_{n=1}^N \gamma(z_{nk}) \frac{x_{ni} - \mu_{ki}}{\mu_{ki}(1 - \mu_{ki})} = 0
$$

分子が0になればよいので、

$$
\sum_{n=1}^N \gamma(z_{nk}) (x_{ni} - \mu_{ki}) = 0
$$
$$
\sum_{n=1}^N \gamma(z_{nk}) x_{ni} = \sum_{n=1}^N \gamma(z_{nk}) \mu_{ki}
$$

$\mu_{ki}$ は $n$ に依存しないため和の外に出すことができます。

$$
\mu_{ki} \sum_{n=1}^N \gamma(z_{nk}) = \sum_{n=1}^N \gamma(z_{nk}) x_{ni}
$$

ここで $N_k = \sum_{n=1}^N \gamma(z_{nk})$ とおくと、

$$
\mu_{ki} = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) x_{ni}
$$

ベクトル記法でまとめると、

$$
\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^N \gamma(z_{nk}) \mathbf{x}_n
$$

これがMステップにおける $\boldsymbol{\mu}_k$ の更新式です。この結果は、各コンポーネントの平均ベクトルが、そのコンポーネントに対する負担率で重み付けされたデータ点の平均となることを示しています。
