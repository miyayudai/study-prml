# 演習問題 1.6

## 問題
二つの変数 $x$ と $y$ が独立であるならば、それらの共分散がゼロになることを示せ。

## 解答・解説

二つの変数 $x, y$ の共分散 $\text{cov}[x, y]$ は次のように定義されます。
$$ \text{cov}[x, y] = \mathbb{E}_{x, y} [ \{x - \mathbb{E}[x]\}\{y - \mathbb{E}[y]\} ] $$

この式を展開し、期待値の線形性を用いると、次のよく知られた形になります。
$$ \text{cov}[x, y] = \mathbb{E}_{x, y} [xy - x\mathbb{E}[y] - y\mathbb{E}[x] + \mathbb{E}[x]\mathbb{E}[y]] $$
$$ \text{cov}[x, y] = \mathbb{E}_{x, y}[xy] - \mathbb{E}[x]\mathbb{E}[y] - \mathbb{E}[y]\mathbb{E}[x] + \mathbb{E}[x]\mathbb{E}[y] $$
$$ \text{cov}[x, y] = \mathbb{E}_{x, y}[xy] - \mathbb{E}[x]\mathbb{E}[y] $$

変数 $x$ と $y$ が独立であるという条件は、それらの同時確率分布 $p(x, y)$ が周辺確率分布の積として表せることを意味します。
$$ p(x, y) = p(x)p(y) $$

このとき、積 $xy$ の期待値 $\mathbb{E}_{x, y}[xy]$ は次のように計算されます（連続変数の場合として積分で記述しますが、離散変数でも和に置き換えるだけで同様に成立します）。
$$ \mathbb{E}_{x, y}[xy] = \iint xy \cdot p(x, y) dx dy $$
独立性の条件 $p(x, y) = p(x)p(y)$ を代入します。
$$ \mathbb{E}_{x, y}[xy] = \iint xy \cdot p(x)p(y) dx dy $$
積分を $x$ と $y$ について分離します。
$$ \mathbb{E}_{x, y}[xy] = \left( \int x \cdot p(x) dx \right) \left( \int y \cdot p(y) dy \right) $$

カッコ内の積分はそれぞれ $\mathbb{E}[x]$ と $\mathbb{E}[y]$ の定義そのものです。したがって、
$$ \mathbb{E}_{x, y}[xy] = \mathbb{E}[x]\mathbb{E}[y] $$
となります。独立な変数の場合、「積の期待値は期待値の積に等しい」ことが示されました。

この結果を先の共分散の式に代入します。
$$ \text{cov}[x, y] = \mathbb{E}[x]\mathbb{E}[y] - \mathbb{E}[x]\mathbb{E}[y] = 0 $$

以上より、変数 $x$ と $y$ が独立であるならば、それらの共分散はゼロになることが示されました。
（注意：逆は一般には成り立ちません。共分散がゼロであっても、変数が独立であるとは限りません。共分散は線形な相関の強さのみを測る指標だからです。）
