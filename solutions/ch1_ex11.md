# 演習問題 1.11

## 問題設定
未知の平均 $\mu$ と分散 $\sigma^2$ を持つ 1 次元ガウス分布から、独立同分布 (i.i.d.) に抽出されたデータ集合 $\mathbf{x} = (x_1, \dots, x_N)^T$ が与えられたとする。このとき、最尤推定を用いて得られる標本平均 $\mu_{ML}$ と標本分散 $\sigma^2_{ML}$ の式を導出する。

## 解答
ガウス分布から得られた $N$ 個の独立な観測値に基づく対数尤度関数は次のように定義される：
$$ \ln p(\mathbf{x} | \mu, \sigma^2) = \sum_{n=1}^N \ln \mathcal{N}(x_n | \mu, \sigma^2) $$
$$ = \sum_{n=1}^N \ln \left( \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left\{ -\frac{1}{2\sigma^2}(x_n - \mu)^2 \right\} \right) $$
$$ = -\frac{1}{2\sigma^2} \sum_{n=1}^N (x_n - \mu)^2 - \frac{N}{2} \ln(2\pi) - \frac{N}{2} \ln(\sigma^2) $$

### 1. 平均 $\mu$ の最尤推定量 $\mu_{ML}$
対数尤度関数を $\mu$ に関して微分し、それを 0 とおいて最大化するパラメータを見つける。
$$ \frac{\partial}{\partial \mu} \ln p(\mathbf{x} | \mu, \sigma^2) = \frac{\partial}{\partial \mu} \left( -\frac{1}{2\sigma^2} \sum_{n=1}^N (x_n - \mu)^2 \right) $$
$$ = -\frac{1}{2\sigma^2} \sum_{n=1}^N 2(x_n - \mu) \cdot (-1) = \frac{1}{\sigma^2} \sum_{n=1}^N (x_n - \mu) $$
これを 0 とおく：
$$ \frac{1}{\sigma^2} \sum_{n=1}^N (x_n - \mu_{ML}) = 0 $$
$\sigma^2 > 0$ であるため、
$$ \sum_{n=1}^N x_n - \sum_{n=1}^N \mu_{ML} = 0 $$
$$ \sum_{n=1}^N x_n - N \mu_{ML} = 0 $$
よって、平均の最尤推定量は標本平均と一致する：
$$ \mu_{ML} = \frac{1}{N} \sum_{n=1}^N x_n $$

### 2. 分散 $\sigma^2$ の最尤推定量 $\sigma^2_{ML}$
次に、対数尤度関数を $\sigma^2$ に関して微分し、0 とおく。計算の便宜上、$\sigma^2 = v$ とおいて $v$ について微分する：
$$ \frac{\partial}{\partial v} \ln p(\mathbf{x} | \mu_{ML}, v) = \frac{\partial}{\partial v} \left( -\frac{1}{2v} \sum_{n=1}^N (x_n - \mu_{ML})^2 - \frac{N}{2} \ln(2\pi) - \frac{N}{2} \ln(v) \right) $$
$$ = \frac{1}{2v^2} \sum_{n=1}^N (x_n - \mu_{ML})^2 - \frac{N}{2v} $$
これを 0 とおく：
$$ \frac{1}{2v_{ML}^2} \sum_{n=1}^N (x_n - \mu_{ML})^2 - \frac{N}{2v_{ML}} = 0 $$
両辺に $2v_{ML}^2$ を掛ける（$v_{ML} > 0$ と仮定）：
$$ \sum_{n=1}^N (x_n - \mu_{ML})^2 - N v_{ML} = 0 $$
よって、分散の最尤推定量は：
$$ \sigma^2_{ML} = v_{ML} = \frac{1}{N} \sum_{n=1}^N (x_n - \mu_{ML})^2 $$

以上により、ガウス分布のパラメータの最尤推定量はそれぞれ、データの標本平均と、その標本平均からの偏差の二乗平均（標本分散）であることが導かれた。
