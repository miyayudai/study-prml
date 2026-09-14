# 演習問題 1.10

## 問題設定
2つの互いに独立な確率変数 $x$ と $y$ について、その和の期待値と分散が、それぞれ個別の期待値と分散の和に等しいことを証明する。

## 解答
確率変数 $x, y$ の同時確率分布を $p(x, y)$ とする。 $x, y$ が独立であるという仮定から、同時分布は周辺分布の積に分解できる：
$$ p(x, y) = p_x(x) p_y(y) $$

### 1. 期待値の加法性
和 $x + y$ の期待値は定義より以下のように書ける：
$$ \mathbb{E}[x + y] = \iint (x + y) p(x, y) dx dy $$
独立性を用いて分解し、展開する：
$$ \mathbb{E}[x + y] = \iint (x + y) p_x(x) p_y(y) dx dy $$
$$ = \int x p_x(x) dx \int p_y(y) dy + \int p_x(x) dx \int y p_y(y) dy $$
確率密度関数の性質より、$\int p_y(y) dy = 1$ および $\int p_x(x) dx = 1$ であるから、
$$ \mathbb{E}[x + y] = \mathbb{E}[x] \times 1 + 1 \times \mathbb{E}[y] = \mathbb{E}[x] + \mathbb{E}[y] $$
これは $x, y$ が独立でなくても成り立つ一般的な性質であるが、ここでは独立性を用いても同様に示された。

### 2. 分散の加法性
分散の定義より、 $x+y$ の分散は以下のように表される：
$$ var[x + y] = \mathbb{E}[(x + y)^2] - (\mathbb{E}[x + y])^2 $$
まず、第1項 $\mathbb{E}[(x + y)^2]$ を展開する：
$$ \mathbb{E}[(x + y)^2] = \mathbb{E}[x^2 + 2xy + y^2] = \mathbb{E}[x^2] + 2\mathbb{E}[xy] + \mathbb{E}[y^2] $$
ここで、$x$ と $y$ が独立であるため、積の期待値は期待値の積になる：
$$ \mathbb{E}[xy] = \iint xy p_x(x) p_y(y) dx dy = \left( \int x p_x(x) dx \right) \left( \int y p_y(y) dy \right) = \mathbb{E}[x]\mathbb{E}[y] $$
よって、
$$ \mathbb{E}[(x + y)^2] = \mathbb{E}[x^2] + 2\mathbb{E}[x]\mathbb{E}[y] + \mathbb{E}[y^2] $$

次に、第2項を展開する（先ほど示した期待値の加法性を用いる）：
$$ (\mathbb{E}[x + y])^2 = (\mathbb{E}[x] + \mathbb{E}[y])^2 = (\mathbb{E}[x])^2 + 2\mathbb{E}[x]\mathbb{E}[y] + (\mathbb{E}[y])^2 $$

これらを分散の式に代入して整理する：
$$ var[x + y] = \left\{ \mathbb{E}[x^2] + 2\mathbb{E}[x]\mathbb{E}[y] + \mathbb{E}[y^2] \right\} - \left\{ (\mathbb{E}[x])^2 + 2\mathbb{E}[x]\mathbb{E}[y] + (\mathbb{E}[y])^2 \right\} $$
$2\mathbb{E}[x]\mathbb{E}[y]$ の項が相殺されるため、
$$ var[x + y] = \left( \mathbb{E}[x^2] - (\mathbb{E}[x])^2 \right) + \left( \mathbb{E}[y^2] - (\mathbb{E}[y])^2 \right) $$
それぞれの括弧の中は $x$ と $y$ の分散そのものである。したがって：
$$ var[x + y] = var[x] + var[y] $$

以上により、独立な確率変数の和の分散は、それぞれの分散の和に等しいことが証明された。
