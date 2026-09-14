# 演習問題 2.2

## 問題設定
確率変数 $m$ がパラメータ $N, \mu$ の二項分布 (Binomial distribution) に従うとする。
$$ \text{Bin}(m | N, \mu) = \binom{N}{m} \mu^m (1-\mu)^{N-m} $$
$m$ を $N$ 個の独立なベルヌーイ変数 $x_i \in \{0, 1\}$ の和 $m = \sum_{i=1}^N x_i$ として表すことにより、二項分布の期待値と分散がそれぞれ以下で与えられることを示せ。
$$ \mathbb{E}[m] = N\mu, \quad \text{var}[m] = N\mu(1-\mu) $$

## 解答と導出

二項分布は、$N$ 回の独立なベルヌーイ試行において成功（$x_i = 1$）した回数 $m$ が従う分布である。したがって、$m$ は各試行の結果を表す確率変数 $x_i \sim \text{Bern}(x | \mu)$ を用いて次のように書ける。
$$ m = \sum_{i=1}^N x_i $$
ここで、各 $x_i$ は互いに独立に同じベルヌーイ分布に従うため、演習問題2.1の結果から以下が成り立つ。
$$ \mathbb{E}[x_i] = \mu $$
$$ \text{var}[x_i] = \mu(1 - \mu) $$

### 1. 期待値の導出
期待値の線形性により、和の期待値は期待値の和となる。この性質は確率変数が独立でなくても成り立つが、今回のケースでも当然適用できる。

$$
\begin{aligned}
\mathbb{E}[m] &= \mathbb{E}\left[ \sum_{i=1}^N x_i \right] \\
&= \sum_{i=1}^N \mathbb{E}[x_i] \\
&= \sum_{i=1}^N \mu \\
&= N\mu
\end{aligned}
$$
よって、二項分布の期待値が $\mathbb{E}[m] = N\mu$ であることが示された。

### 2. 分散の導出
互いに独立な確率変数の和の分散は、それぞれの分散の和に等しいという性質を持つ。これは確率変数 $x_i$ と $x_j$ ($i \neq j$) の共分散がゼロになるためである。

$$
\begin{aligned}
\text{var}[m] &= \text{var}\left[ \sum_{i=1}^N x_i \right] \\
&= \sum_{i=1}^N \text{var}[x_i] \quad (\because x_i \text{ は互いに独立}) \\
&= \sum_{i=1}^N \mu(1 - \mu) \\
&= N\mu(1 - \mu)
\end{aligned}
$$

以上により、二項分布の分散が $\text{var}[m] = N\mu(1 - \mu)$ であることが証明された。独立性を用いたこのアプローチは、複雑な和の計算を避けることができるため非常に強力である。
