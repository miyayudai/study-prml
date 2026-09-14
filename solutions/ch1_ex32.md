# 演習問題 1.32

## 問題の概要
結合確率分布 $p(x, y)$ を持つ2つの確率変数 $x, y$ において、結合エントロピーについて次の不等式が成り立つことを証明する。
$$
H[x, y] \le H[x] + H[y]
$$
また、等号が成立するための必要十分条件が $x$ と $y$ が独立であること（すなわち $p(x, y) = p(x)p(y)$）であることを示す。

## 証明
カルバック・ライブラー情報量（KL divergence）の非負性を利用する。
任意の2つの確率分布 $p(\mathbf{z})$ と $q(\mathbf{z})$ に対して、$\text{KL}(p||q) \ge 0$ が成り立ち、等号は $p(\mathbf{z}) = q(\mathbf{z})$ のときに限って成立することが知られている（ギブスの不等式）。

ここで、結合分布 $p(x, y)$ と、それぞれの周辺分布の積 $p(x)p(y)$ を考え、これらの間のKL情報量（相互情報量 $I(x, y)$）を計算する。
$$
I(x, y) = \text{KL}(p(x,y)||p(x)p(y)) = \iint p(x, y) \ln \frac{p(x, y)}{p(x)p(y)} dx dy
$$
対数の性質を用いて分解すると、
$$
I(x, y) = \iint p(x, y) \ln p(x, y) dx dy - \iint p(x, y) \ln p(x) dx dy - \iint p(x, y) \ln p(y) dx dy
$$
第1項は $-H[x, y]$ である。第2項と第3項はそれぞれ周辺分布の性質を用いて以下のように計算できる。
$$
- \iint p(x, y) \ln p(x) dx dy = - \int \left( \int p(x, y) dy \right) \ln p(x) dx = - \int p(x) \ln p(x) dx = H[x]
$$
$$
- \iint p(x, y) \ln p(y) dx dy = - \int \left( \int p(x, y) dx \right) \ln p(y) dy = - \int p(y) \ln p(y) dy = H[y]
$$
よって、
$$
I(x, y) = -H[x, y] + H[x] + H[y]
$$
となる。KL情報量の非負性 $\text{KL}(p||q) \ge 0$ より $I(x, y) \ge 0$ であるため、
$$
-H[x, y] + H[x] + H[y] \ge 0
$$
$$
H[x, y] \le H[x] + H[y]
$$
が導かれる。

さらに、等号が成立するのはKL情報量が $0$ になるとき、すなわち
$$
p(x, y) = p(x)p(y)
$$
がすべての $x, y$ について成り立つときのみである。これは $x$ と $y$ が互いに独立であることの定義に他ならない。
以上より、題意は示された。
