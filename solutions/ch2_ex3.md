# 演習問題 2.3

## 問題設定
二項分布が正しく正規化されていること（すなわち、すべての可能な $m$ の値にわたって和をとると 1 になること）を示せ。
また、この正規化の条件式の両辺を $\mu$ について微分することにより、二項分布の期待値と分散を導出せよ。

二項分布の定義：
$$ \text{Bin}(m | N, \mu) = \binom{N}{m} \mu^m (1-\mu)^{N-m} $$

## 解答と導出

### 1. 二項分布の正規化の証明
二項定理 (Binomial theorem) は次のように与えられる。
$$ (a + b)^N = \sum_{m=0}^N \binom{N}{m} a^m b^{N-m} $$

この式において、$a = \mu$、$b = 1 - \mu$ とおくと、
$$
\begin{aligned}
\sum_{m=0}^N \binom{N}{m} \mu^m (1-\mu)^{N-m} &= (\mu + (1 - \mu))^N \\
&= 1^N \\
&= 1
\end{aligned}
$$
したがって、すべての $m \in \{0, 1, \dots, N\}$ にわたる確率の和が 1 となり、二項分布が正しく正規化されていることが示された。

### 2. 微分を用いた期待値の導出
正規化の式を少し一般化して、二項定理の両辺を $\mu$ について微分することを考える。ここでは、簡単のため $(x + y)^N = \sum \binom{N}{m} x^m y^{N-m}$ を用いる。
両辺を $x$ で微分すると：
$$ N(x + y)^{N-1} = \sum_{m=0}^N \binom{N}{m} m x^{m-1} y^{N-m} $$

両辺に $x$ を掛けると：
$$ N x (x + y)^{N-1} = \sum_{m=0}^N \binom{N}{m} m x^m y^{N-m} $$

ここで、$x = \mu, y = 1-\mu$ を代入すると $x + y = 1$ となるため、左辺は $N\mu$ になる。
$$ N \mu (1)^{N-1} = \sum_{m=0}^N m \binom{N}{m} \mu^m (1-\mu)^{N-m} $$
右辺は期待値の定義 $\sum m \, p(m)$ そのものである。
よって、
$$ \mathbb{E}[m] = N\mu $$
が得られる。

### 3. 微分を用いた分散の導出
先ほど得られた式 $N(x+y)^{N-1} = \sum \binom{N}{m} m x^{m-1} y^{N-m}$ の両辺をさらに $x$ で微分する。
$$ N(N-1)(x+y)^{N-2} = \sum_{m=0}^N \binom{N}{m} m(m-1) x^{m-2} y^{N-m} $$

両辺に $x^2$ を掛けると：
$$ N(N-1) x^2 (x+y)^{N-2} = \sum_{m=0}^N \binom{N}{m} (m^2 - m) x^m y^{N-m} $$

ここでも $x = \mu, y = 1-\mu$ を代入する。
$$ N(N-1) \mu^2 = \mathbb{E}[m^2 - m] = \mathbb{E}[m^2] - \mathbb{E}[m] $$

これより $\mathbb{E}[m^2]$ を求める。$\mathbb{E}[m] = N\mu$ であるから、
$$
\begin{aligned}
\mathbb{E}[m^2] &= N(N-1)\mu^2 + \mathbb{E}[m] \\
&= N^2\mu^2 - N\mu^2 + N\mu
\end{aligned}
$$

分散の公式 $\text{var}[m] = \mathbb{E}[m^2] - (\mathbb{E}[m])^2$ に代入する。
$$
\begin{aligned}
\text{var}[m] &= (N^2\mu^2 - N\mu^2 + N\mu) - (N\mu)^2 \\
&= N^2\mu^2 - N\mu^2 + N\mu - N^2\mu^2 \\
&= N\mu - N\mu^2 \\
&= N\mu(1 - \mu)
\end{aligned}
$$

以上により、微分を用いる手法でも二項分布の期待値が $N\mu$、分散が $N\mu(1-\mu)$ となることが正しく導出された。
