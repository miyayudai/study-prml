# 演習問題 2.43

## 問題の概要
Studentのt分布の平均と分散を導出する。

## 解答

Studentのt分布は、精度 $\tau$ に関してガンマ分布を事前分布としたガウス分布の周辺化として以下のように定義されます。

$$
\text{St}(x|\mu, \lambda, \nu) = \int_0^\infty \mathcal{N}(x|\mu, (\tau\lambda)^{-1}) \text{Gam}(\tau|\nu/2, \nu/2) d\tau
$$

### 1. 平均の導出
期待値の線形性と反復期待値の法則（Law of Iterated Expectations）を用いると、平均は次のように計算できます。

$$
\mathbb{E}[x] = \mathbb{E}_\tau [\mathbb{E}_x [x|\tau]]
$$

内側の期待値は、与えられた $\tau$ に対するガウス分布の平均なので、$\mathbb{E}_x [x|\tau] = \mu$ です。これは $\tau$ に依存しません。
したがって、

$$
\mathbb{E}[x] = \mathbb{E}_\tau [\mu] = \mu
$$

ただし、この期待値が存在するためには、分布の裾が十分に早く減衰する必要があります。t分布の場合、$\nu > 1$ のときに平均が定義されます。

### 2. 分散の導出
同様に、分散に関する全分散の法則（Law of Total Variance）を用います。

$$
\text{Var}[x] = \mathbb{E}_\tau [\text{Var}_x[x|\tau]] + \text{Var}_\tau[\mathbb{E}_x[x|\tau]]
$$

第2項について、$\mathbb{E}_x[x|\tau] = \mu$ であり定数なので、$\text{Var}_\tau[\mu] = 0$ となります。
第1項について、ガウス分布の分散は $\text{Var}_x[x|\tau] = (\tau\lambda)^{-1}$ です。したがって、

$$
\text{Var}[x] = \mathbb{E}_\tau [(\tau\lambda)^{-1}] = \frac{1}{\lambda} \mathbb{E}_\tau [\tau^{-1}]
$$

ここで、$\tau \sim \text{Gam}(\nu/2, \nu/2)$ です。ガンマ分布 $x \sim \text{Gam}(a, b)$ における $x^{-1}$ の期待値は、逆ガンマ分布の性質から $\mathbb{E}[x^{-1}] = \frac{b}{a-1}$ （ただし $a > 1$）となります。
今回の場合、$a = \nu/2, b = \nu/2$ なので、

$$
\mathbb{E}_\tau [\tau^{-1}] = \frac{\nu/2}{\nu/2 - 1} = \frac{\nu}{\nu - 2}
$$

これを元の式に代入すると、

$$
\text{Var}[x] = \frac{1}{\lambda} \frac{\nu}{\nu - 2}
$$

ただし、この分散が定義されるためには $\nu > 2$ である必要があります。
以上より、平均は $\mu$ （$\nu > 1$）、分散は $\frac{\nu}{\lambda(\nu-2)}$ （$\nu > 2$）であることが示されました。
