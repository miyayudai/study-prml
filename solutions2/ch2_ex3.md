# 演習問題 2.3 (Exercise 2.3)

## 問題の概要 (Problem Statement)
二項係数の関係式 $\binom{N}{m} = \binom{N-1}{m} + \binom{N-1}{m-1}$ を証明し、数学的帰納法または二項定理を用いて二項分布が規格化されていること（$\sum_{m=0}^N \binom{N}{m} \mu^m (1-\mu)^{N-m} = 1$）を示せ。

## 解答と証明 (Solution and Proof)

### 1. 二項係数の漸化式の証明 (Proof of the Recurrence Relation)
組み合わせの定義 $\binom{N}{m} = \frac{N!}{m!(N-m)!}$ を用いて、右辺を変形します：
$$ \binom{N-1}{m} + \binom{N-1}{m-1} = \frac{(N-1)!}{m!(N-1-m)!} + \frac{(N-1)!}{(m-1)!(N-m)!} $$
共通の分母を持つように通分します。第一項の分母分子に $(N-m)$ を掛け、第二項の分母分子に $m$ を掛けます：
$$ = \frac{(N-1)!(N-m)}{m!(N-m)!} + \frac{(N-1)!m}{m!(N-m)!} $$
$$ = \frac{(N-1)! \{ (N-m) + m \}}{m!(N-m)!} $$
分子の括弧内を計算すると $(N-m) + m = N$ になります。
$$ = \frac{(N-1)! \cdot N}{m!(N-m)!} = \frac{N!}{m!(N-m)!} = \binom{N}{m} $$
これにより、漸化式 $\binom{N}{m} = \binom{N-1}{m} + \binom{N-1}{m-1}$ が証明されました。

### 2. 二項分布の規格化 (Normalization of the Binomial Distribution)
二項分布は $p(m|N, \mu) = \binom{N}{m} \mu^m (1-\mu)^{N-m}$ で定義されます。これが確率分布として規格化されていること、つまり
$$ \sum_{m=0}^N \binom{N}{m} \mu^m (1-\mu)^{N-m} = 1 $$
であることを二項定理を用いて示します。

二項定理は以下のように表されます：
$$ (a + b)^N = \sum_{m=0}^N \binom{N}{m} a^m b^{N-m} $$
この式において、$a = \mu$、$b = 1-\mu$ と置きます：
$$ (\mu + 1 - \mu)^N = \sum_{m=0}^N \binom{N}{m} \mu^m (1-\mu)^{N-m} $$
左辺を計算すると、
$$ (\mu + 1 - \mu)^N = 1^N = 1 $$
となります。したがって、
$$ \sum_{m=0}^N \binom{N}{m} \mu^m (1-\mu)^{N-m} = 1 $$
が成立し、二項分布が正しく規格化されていることが示されました。
