# 演習問題 2.59

## 問題文（要約）
確率密度関数が $p(x|\sigma) = \frac{1}{\sigma} f\left(\frac{x}{\sigma}\right)$ で与えられるとき、この分布が正規化条件を満たす（積分すると1になる）ことを示せ。ただし、$f(u)$ は正当な確率密度関数（すなわち $\int f(u) du = 1$）であると仮定する。

## 解答と解説

確率密度関数の全区間での積分が1になることを示せばよいので、以下のように $x$ について積分を行います。
$$ \int_{-\infty}^{\infty} p(x|\sigma) dx = \int_{-\infty}^{\infty} \frac{1}{\sigma} f\left(\frac{x}{\sigma}\right) dx $$

ここで、変数を $u = \frac{x}{\sigma}$ と置換して積分を評価します（置換積分法）。
$u$ で微分すると $\frac{du}{dx} = \frac{1}{\sigma}$ 、すなわち $dx = \sigma du$ となります。
また、積分範囲は $x$ が $-\infty \to \infty$ のとき（$\sigma > 0$ を前提とすると）、$u$ も $-\infty \to \infty$ となります。

これを元の積分式に代入すると：
$$ \int_{-\infty}^{\infty} \frac{1}{\sigma} f(u) (\sigma du) = \int_{-\infty}^{\infty} f(u) du $$

問題の仮定より、$f(u)$ は正規化された正当な確率密度関数であるため、その積分は1になります。
$$ \int_{-\infty}^{\infty} f(u) du = 1 $$

したがって、スケールパラメータ $\sigma$ を導入した分布 $p(x|\sigma)$ もまた、全区間で積分すると1になり、正当な確率分布であることが示されました。
$$ \int_{-\infty}^{\infty} p(x|\sigma) dx = 1 $$
