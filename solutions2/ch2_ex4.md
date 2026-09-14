# 演習問題 2.4 (Exercise 2.4)

## 問題の概要 (Problem Statement)
規格化条件 $\sum_{m=0}^N \binom{N}{m} \mu^m (1-\mu)^{N-m} = 1$ の両辺を $\mu$ について微分することで、二項分布の平均 $\mathbb{E}[m] = N\mu$ を導出せよ。さらに2階微分を用いて分散 $\mathrm{var}[m] = N\mu(1-\mu)$ を導出せよ。

## 解答と証明 (Solution and Proof)

### 1. 1階微分による平均の導出 (Derivation of the Mean via 1st Derivative)
規格化条件：
$$ \sum_{m=0}^N \binom{N}{m} \mu^m (1-\mu)^{N-m} = 1 $$
この両辺を $\mu$ で微分します（積の微分法則を用います）：
$$ \sum_{m=0}^N \binom{N}{m} \left[ m \mu^{m-1} (1-\mu)^{N-m} - (N-m) \mu^m (1-\mu)^{N-m-1} \right] = 0 $$
両辺に $\mu(1-\mu)$ を掛けます：
$$ \sum_{m=0}^N \binom{N}{m} \left[ m \mu^m (1-\mu)^{N-m+1} - (N-m) \mu^{m+1} (1-\mu)^{N-m} \right] = 0 $$
各項の共通部分である $\binom{N}{m} \mu^m (1-\mu)^{N-m}$ を括り出します：
$$ \sum_{m=0}^N \binom{N}{m} \mu^m (1-\mu)^{N-m} [m(1-\mu) - (N-m)\mu] = 0 $$
中括弧の中を整理すると：
$$ m - m\mu - N\mu + m\mu = m - N\mu $$
したがって、方程式は次のようになります：
$$ \sum_{m=0}^N (m - N\mu) \binom{N}{m} \mu^m (1-\mu)^{N-m} = 0 $$
ここで $p(m|N, \mu) = \binom{N}{m} \mu^m (1-\mu)^{N-m}$ より、
$$ \sum_{m=0}^N m p(m|N, \mu) - \sum_{m=0}^N N\mu p(m|N, \mu) = 0 $$
$$ \mathbb{E}[m] - N\mu (1) = 0 \implies \mathbb{E}[m] = N\mu $$
これにより、二項分布の平均が $N\mu$ であることが示されました。

### 2. 2階微分による分散の導出 (Derivation of Variance via 2nd Derivative)
先ほど得られた1階微分の方程式の左辺の一部から始めます：
$$ \mathbb{E}[m] = \sum_{m=0}^N m \binom{N}{m} \mu^m (1-\mu)^{N-m} = N\mu $$
この式の両辺を再度 $\mu$ で微分します：
$$ \frac{\partial}{\partial \mu} \sum_{m=0}^N m \binom{N}{m} \mu^m (1-\mu)^{N-m} = \frac{\partial}{\partial \mu} (N\mu) = N $$
左辺の微分は、積の微分法則により、
$$ \sum_{m=0}^N m \binom{N}{m} \left( m \mu^{m-1}(1-\mu)^{N-m} - (N-m)\mu^m(1-\mu)^{N-m-1} \right) = N $$
両辺に $\mu(1-\mu)$ を掛けると、
$$ \sum_{m=0}^N m \binom{N}{m} \mu^m (1-\mu)^{N-m} [m(1-\mu) - (N-m)\mu] = N\mu(1-\mu) $$
先ほどと同じように括弧内は $m - N\mu$ となるので、
$$ \sum_{m=0}^N m (m - N\mu) p(m|N, \mu) = N\mu(1-\mu) $$
和を展開すると、
$$ \sum_{m=0}^N m^2 p(m|N, \mu) - N\mu \sum_{m=0}^N m p(m|N, \mu) = N\mu(1-\mu) $$
$$ \mathbb{E}[m^2] - N\mu \cdot \mathbb{E}[m] = N\mu(1-\mu) $$
$\mathbb{E}[m] = N\mu$ を代入すると、
$$ \mathbb{E}[m^2] - (N\mu)^2 = N\mu(1-\mu) $$
分散の定義より $\mathrm{var}[m] = \mathbb{E}[m^2] - (\mathbb{E}[m])^2 = \mathbb{E}[m^2] - (N\mu)^2$ なので、
$$ \mathrm{var}[m] = N\mu(1-\mu) $$
となり、分散が導出されました。
