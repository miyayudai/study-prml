# 演習問題 2.1

## 問題設定
ベルヌーイ分布 (Bernoulli distribution) に従う確率変数 $x \in \{0, 1\}$ が与えられているとする。ここで、$x = 1$ となる確率は $\mu$ であり、$x = 0$ となる確率は $1 - \mu$ である。この分布は次のように書ける。
$$ \text{Bern}(x | \mu) = \mu^x (1 - \mu)^{1-x} $$
このとき、$x$ の期待値（平均）と分散を求めよ。

## 解答と導出

### 1. 期待値の導出
確率変数 $x$ は離散変数であり、取り得る値は $x = 0$ または $x = 1$ のみである。したがって、期待値 $\mathbb{E}[x]$ は定義より以下のように計算できる。

$$
\begin{aligned}
\mathbb{E}[x] &= \sum_{x \in \{0, 1\}} x \, p(x) \\
&= 1 \cdot p(x=1) + 0 \cdot p(x=0) \\
&= 1 \cdot \mu + 0 \cdot (1 - \mu) \\
&= \mu
\end{aligned}
$$

よって、期待値は $\mathbb{E}[x] = \mu$ であることが示された。

### 2. 分散の導出
分散 $\text{var}[x]$ は以下の公式を用いて計算することができる。
$$ \text{var}[x] = \mathbb{E}[x^2] - (\mathbb{E}[x])^2 $$

まず、$\mathbb{E}[x^2]$ を計算する。
$$
\begin{aligned}
\mathbb{E}[x^2] &= \sum_{x \in \{0, 1\}} x^2 \, p(x) \\
&= 1^2 \cdot p(x=1) + 0^2 \cdot p(x=0) \\
&= 1 \cdot \mu + 0 \cdot (1 - \mu) \\
&= \mu
\end{aligned}
$$

これと先に求めた $\mathbb{E}[x] = \mu$ を分散の公式に代入する。
$$
\begin{aligned}
\text{var}[x] &= \mathbb{E}[x^2] - (\mathbb{E}[x])^2 \\
&= \mu - \mu^2 \\
&= \mu(1 - \mu)
\end{aligned}
$$

以上により、ベルヌーイ分布に従う確率変数 $x$ の分散は $\text{var}[x] = \mu(1 - \mu)$ であることが証明された。
