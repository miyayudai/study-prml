# 演習問題 2.2 (Exercise 2.2)

## 問題の概要 (Problem Statement)
変数 $x \in \{-1, +1\}$ に対し、分布を以下のように定義する。
$$ p(x|\mu) = \left( \frac{1+\mu}{2} \right)^{\frac{1+x}{2}} \left( \frac{1-\mu}{2} \right)^{\frac{1-x}{2}} \quad (\mu \in [-1, 1]) $$
この分布が規格化されていることを示し、平均、分散、エントロピーを求めよ。

## 解答と証明 (Solution and Proof)

### 1. 規格化条件 (Normalization)
変数 $x$ は $-1$ または $+1$ の値をとるため、それぞれの確率を計算して和が1になるか確認します。
まず、$x = +1$ の場合：
$$ \frac{1+x}{2} = \frac{2}{2} = 1, \quad \frac{1-x}{2} = 0 $$
$$ p(x=1|\mu) = \left( \frac{1+\mu}{2} \right)^1 \left( \frac{1-\mu}{2} \right)^0 = \frac{1+\mu}{2} $$
次に、$x = -1$ の場合：
$$ \frac{1+x}{2} = 0, \quad \frac{1-x}{2} = \frac{2}{2} = 1 $$
$$ p(x=-1|\mu) = \left( \frac{1+\mu}{2} \right)^0 \left( \frac{1-\mu}{2} \right)^1 = \frac{1-\mu}{2} $$
したがって、すべての状態の確率の和は、
$$ \sum_{x \in \{-1, 1\}} p(x|\mu) = p(1|\mu) + p(-1|\mu) = \frac{1+\mu}{2} + \frac{1-\mu}{2} = \frac{2}{2} = 1 $$
となり、分布は規格化されています。

### 2. 平均 (Expected Value)
期待値の定義に従って計算します：
$$ \mathbb{E}[x] = \sum_{x \in \{-1, 1\}} x p(x|\mu) = (+1) \cdot p(1|\mu) + (-1) \cdot p(-1|\mu) $$
それぞれの確率を代入すると、
$$ \mathbb{E}[x] = 1 \cdot \frac{1+\mu}{2} - 1 \cdot \frac{1-\mu}{2} = \frac{1+\mu - (1-\mu)}{2} = \frac{2\mu}{2} = \mu $$
となり、平均は $\mu$ になります。

### 3. 分散 (Variance)
分散を求めるために、まず2次のモーメント $\mathbb{E}[x^2]$ を計算します。$x \in \{-1, 1\}$ より、常に $x^2 = 1$ が成り立ちます。したがって：
$$ \mathbb{E}[x^2] = \sum_{x \in \{-1, 1\}} x^2 p(x|\mu) = 1 \cdot p(1|\mu) + 1 \cdot p(-1|\mu) = 1 \cdot \left( \frac{1+\mu}{2} + \frac{1-\mu}{2} \right) = 1 $$
これを用いて分散を計算します：
$$ \mathrm{var}[x] = \mathbb{E}[x^2] - (\mathbb{E}[x])^2 = 1 - \mu^2 $$
よって、分散は $1 - \mu^2$ となります。

### 4. エントロピー (Entropy)
エントロピーの定義に従い、計算します：
$$ \mathrm{H}[x] = -\sum_{x \in \{-1, 1\}} p(x|\mu) \ln p(x|\mu) $$
$$ \mathrm{H}[x] = - \left[ p(1|\mu) \ln p(1|\mu) + p(-1|\mu) \ln p(-1|\mu) \right] $$
確率の式を代入すると、
$$ \mathrm{H}[x] = - \left[ \frac{1+\mu}{2} \ln \left(\frac{1+\mu}{2}\right) + \frac{1-\mu}{2} \ln \left(\frac{1-\mu}{2}\right) \right] $$
これが求めるエントロピーの式になります。
