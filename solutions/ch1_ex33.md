# 演習問題 1.33

## 問題の概要
2つの確率変数 $x$ と $y$ の相互情報量 $I(x, y)$ が、エントロピーと条件付きエントロピーを用いて次のように表されることを証明する。
$$
I(x, y) = H[x] - H[x|y] = H[y] - H[y|x]
$$

## 証明
相互情報量 $I(x, y)$ の定義は以下の通りである。
$$
I(x, y) = \iint p(x, y) \ln \frac{p(x, y)}{p(x)p(y)} dx dy
$$
対数の性質を用いて、対数部分を展開する。
$$
\ln \frac{p(x, y)}{p(x)p(y)} = \ln p(x, y) - \ln p(x) - \ln p(y)
$$
また、確率の乗法定理 $p(x, y) = p(y|x)p(x) = p(x|y)p(y)$ より、対数部分は次のように変形することもできる。
$$
\ln \frac{p(x, y)}{p(x)p(y)} = \ln \frac{p(x|y)p(y)}{p(x)p(y)} = \ln \frac{p(x|y)}{p(x)} = \ln p(x|y) - \ln p(x)
$$
これを相互情報量の定義式に代入する。
$$
I(x, y) = \iint p(x, y) \left( \ln p(x|y) - \ln p(x) \right) dx dy
$$
積分を分けると、
$$
I(x, y) = \iint p(x, y) \ln p(x|y) dx dy - \iint p(x, y) \ln p(x) dx dy
$$
第1項は、定義より $-H[x|y]$（条件付きエントロピーの負）である。
第2項は、$y$ について周辺化することで $-H[x]$ となる。
$$
-\iint p(x, y) \ln p(x) dx dy = -\int \left( \int p(x, y) dy \right) \ln p(x) dx = -\int p(x) \ln p(x) dx = H[x]
$$
したがって、
$$
I(x, y) = -H[x|y] + H[x] = H[x] - H[x|y]
$$
が成り立つ。

全く同様にして、対数部分を $p(x, y) = p(y|x)p(x)$ を用いて変形すると、
$$
\ln \frac{p(x, y)}{p(x)p(y)} = \ln \frac{p(y|x)p(x)}{p(x)p(y)} = \ln \frac{p(y|x)}{p(y)} = \ln p(y|x) - \ln p(y)
$$
となるため、これを用いて積分すると、
$$
I(x, y) = \iint p(x, y) \left( \ln p(y|x) - \ln p(y) \right) dx dy
$$
$$
= \iint p(x, y) \ln p(y|x) dx dy - \iint p(x, y) \ln p(y) dx dy = -H[y|x] + H[y] = H[y] - H[y|x]
$$
となる。
以上より、相互情報量が周辺エントロピーから条件付きエントロピーを引いたものに等しいことが証明された。
