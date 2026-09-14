# 演習問題 2.1 (Exercise 2.1)

## 問題の概要 (Problem Statement)
ベルヌーイ分布 $p(x|\mu) = \mu^x (1 - \mu)^{1 - x}$ ($x \in \{0, 1\}$) について、以下の性質を証明せよ。
1. $\sum_{x=0}^1 p(x|\mu) = 1$
2. $\mathbb{E}[x] = \mu$
3. $\mathrm{var}[x] = \mu(1 - \mu)$
4. エントロピー $\mathrm{H}[x] = -\mu \ln \mu - (1 - \mu) \ln (1 - \mu)$

## 解答と証明 (Solution and Proof)

### 1. 規格化条件 (Normalization)
ベルヌーイ分布が確率分布として規格化されていることを示します。変数 $x$ は $0$ または $1$ の値をとるため、すべての可能な $x$ について確率の和を計算します：
$$ \sum_{x=0}^1 p(x|\mu) = p(x=0|\mu) + p(x=1|\mu) $$
それぞれの確率を代入すると、
$$ p(0|\mu) = \mu^0 (1-\mu)^{1-0} = 1-\mu $$
$$ p(1|\mu) = \mu^1 (1-\mu)^{1-1} = \mu $$
となります。したがって、
$$ \sum_{x=0}^1 p(x|\mu) = (1-\mu) + \mu = 1 $$
となり、規格化されていることが示されました。

### 2. 期待値 (Expected Value)
期待値の定義 $\mathbb{E}[x] = \sum_{x} x p(x)$ に従って計算します：
$$ \mathbb{E}[x] = \sum_{x=0}^1 x p(x|\mu) = 0 \cdot p(0|\mu) + 1 \cdot p(1|\mu) $$
それぞれの項を計算すると、
$$ \mathbb{E}[x] = 0 \cdot (1-\mu) + 1 \cdot \mu = \mu $$
となり、平均は $\mu$ であることがわかります。

### 3. 分散 (Variance)
分散は $\mathrm{var}[x] = \mathbb{E}[x^2] - (\mathbb{E}[x])^2$ を用いて計算します。まず、2次のモーメント $\mathbb{E}[x^2]$ を求めます：
$$ \mathbb{E}[x^2] = \sum_{x=0}^1 x^2 p(x|\mu) = 0^2 \cdot p(0|\mu) + 1^2 \cdot p(1|\mu) $$
$$ \mathbb{E}[x^2] = 0 \cdot (1-\mu) + 1 \cdot \mu = \mu $$
先ほど求めた $\mathbb{E}[x] = \mu$ を分散の公式に代入すると、
$$ \mathrm{var}[x] = \mathbb{E}[x^2] - (\mathbb{E}[x])^2 = \mu - \mu^2 = \mu(1-\mu) $$
となり、分散が導出されました。

### 4. エントロピー (Entropy)
離散変数 $x$ のエントロピーの定義 $\mathrm{H}[x] = -\sum_{x} p(x) \ln p(x)$ に従って計算します：
$$ \mathrm{H}[x] = -\sum_{x=0}^1 p(x|\mu) \ln p(x|\mu) $$
和を展開すると、
$$ \mathrm{H}[x] = - \left[ p(0|\mu) \ln p(0|\mu) + p(1|\mu) \ln p(1|\mu) \right] $$
$$ \mathrm{H}[x] = - \left[ (1-\mu) \ln (1-\mu) + \mu \ln \mu \right] $$
順番を整理して、
$$ \mathrm{H}[x] = -\mu \ln \mu - (1-\mu) \ln (1-\mu) $$
となり、エントロピーの式が導出されました。
