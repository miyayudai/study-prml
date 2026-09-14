## Exercise 1.40: 相加平均と相乗平均の関係 (AM-GM Inequality)

### 問題
関数 $f(x) = \ln x$ にイェンセンの不等式 (1.115) を適用することにより、実数の集合の相加平均（算術平均）が、相乗平均（幾何平均）を下回ることはないことを示せ。

### 解答と解説

$N$ 個の正の実数 $x_1, x_2, \ldots, x_N$ を考えます。これらに対する相加平均 (Arithmetic Mean) と相乗平均 (Geometric Mean) は次のように定義されます：
- 相加平均： $A = \frac{1}{N} \sum_{i=1}^N x_i$
- 相乗平均： $G = \left( \prod_{i=1}^N x_i \right)^{1/N}$

証明すべき不等式は $A \geq G$ です。

まず、問題文で指定された関数 $f(x) = \ln x$ の性質を確認します。
この関数の2階微分は $f''(x) = -1/x^2 < 0$ となり、常に負です。これは $f(x) = \ln x$ が **上に凸な関数 (concave function)** であることを意味します。

イェンセンの不等式 (1.115) は通常、下に凸な関数 (convex function) $g(x)$ に対して $g(E[x]) \leq E[g(x)]$ として定義されます。
ここでは $g(x) = - \ln x$ と置くと、$g''(x) = 1/x^2 > 0$ となり下に凸な関数になるため、イェンセンの不等式を適用できます。
等確率の係数として $\lambda_i = \frac{1}{N}$ （ここで $\sum_{i=1}^N \lambda_i = 1$）を設定します。

イェンセンの不等式より：
$$ g\left( \sum_{i=1}^N \lambda_i x_i \right) \leq \sum_{i=1}^N \lambda_i g(x_i) $$

これに $g(x) = -\ln x$ と $\lambda_i = \frac{1}{N}$ を代入します：
$$ - \ln \left( \frac{1}{N} \sum_{i=1}^N x_i \right) \leq \sum_{i=1}^N \frac{1}{N} (-\ln x_i) $$

両辺に $-1$ を掛けます。このとき不等号の向きが反転することに注意します（これは上に凸な関数 $\ln x$ に直接イェンセンの不等式を適用した形になります）：
$$ \ln \left( \frac{1}{N} \sum_{i=1}^N x_i \right) \geq \frac{1}{N} \sum_{i=1}^N \ln x_i $$

右辺の対数の和を整理します。対数の性質 $a \ln b = \ln(b^a)$ と $\sum \ln b_i = \ln (\prod b_i)$ を用います：
$$ \frac{1}{N} \sum_{i=1}^N \ln x_i = \sum_{i=1}^N \ln (x_i^{1/N}) = \ln \left( \prod_{i=1}^N x_i^{1/N} \right) = \ln \left( \left( \prod_{i=1}^N x_i \right)^{1/N} \right) $$

したがって、不等式は次のようになります：
$$ \ln \left( \frac{1}{N} \sum_{i=1}^N x_i \right) \geq \ln \left( \left( \prod_{i=1}^N x_i \right)^{1/N} \right) $$

自然対数関数 $\ln(x)$ は単調増加関数（$x_1 > x_2 \iff \ln x_1 > \ln x_2$）であるため、対数を外しても不等号の関係はそのまま保たれます：
$$ \frac{1}{N} \sum_{i=1}^N x_i \geq \left( \prod_{i=1}^N x_i \right)^{1/N} $$

左辺は相加平均、右辺は相乗平均を表しており、これで相加平均が相乗平均を下回らないこと（相加相乗平均の不等式）が証明されました。
