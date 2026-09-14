## Exercise 1.36: 凸関数と2階微分 (Convex Functions and Second Derivatives)

### 問題
狭義の凸関数 (strictly convex function) は、関数上の任意の2点を結ぶ弦 (chord) が、常にその関数のグラフより上側に位置するような関数として定義される。この条件が、関数の2階微分が常に正であることと等価であることを示せ。

### 解答と解説

1変数関数 $f(x)$ が2回連続微分可能であると仮定します。

**1. 弦が関数の上にある $\implies$ 2階微分が正**

凸関数の定義から、任意の $a, b$ (ただし $a \neq b$) と $0 < \lambda < 1$ に対して次が成り立ちます：
$$ f(\lambda a + (1 - \lambda)b) < \lambda f(a) + (1 - \lambda)f(b) $$

ここで、$b = x$, $a = x + \Delta x$ と置きます。
$$ f(x + \lambda \Delta x) < \lambda f(x + \Delta x) + (1 - \lambda)f(x) $$

関数 $f(x)$ を $x$ の周りで2次までテイラー展開します：
$$ f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{\epsilon^2}{2} f''(x) + O(\epsilon^3) $$

左辺 $f(x + \lambda \Delta x)$ に適用すると：
$$ f(x + \lambda \Delta x) = f(x) + \lambda \Delta x f'(x) + \frac{\lambda^2 (\Delta x)^2}{2} f''(x) + O((\Delta x)^3) $$

右辺の $f(x + \Delta x)$ に適用すると：
$$ \lambda f(x + \Delta x) + (1 - \lambda)f(x) = \lambda \left( f(x) + \Delta x f'(x) + \frac{(\Delta x)^2}{2} f''(x) + O((\Delta x)^3) \right) + (1 - \lambda)f(x) $$
$$ = f(x) + \lambda \Delta x f'(x) + \frac{\lambda (\Delta x)^2}{2} f''(x) + O((\Delta x)^3) $$

これを元の不等式に代入して整理すると：
$$ \frac{\lambda^2 (\Delta x)^2}{2} f''(x) < \frac{\lambda (\Delta x)^2}{2} f''(x) $$
$$ \lambda^2 f''(x) < \lambda f''(x) $$

$0 < \lambda < 1$ より $\lambda - \lambda^2 > 0$ であるため、
$$ \lambda(1 - \lambda) f''(x) > 0 \implies f''(x) > 0 $$
となり、2階微分が正であることが導かれます。

**2. 2階微分が正 $\implies$ 弦が関数の上にある**

逆に $f''(x) > 0$ と仮定します。平均値の定理の拡張（テイラーの定理の剰余項）を利用します。
$z = \lambda a + (1 - \lambda)b$ とします。
$f(a)$ と $f(b)$ を $z$ の周りで展開すると：
$$ f(a) = f(z) + (a - z)f'(z) + \frac{(a - z)^2}{2}f''(c_1) \quad (c_1 \text{ は } a, z \text{ の間}) $$
$$ f(b) = f(z) + (b - z)f'(z) + \frac{(b - z)^2}{2}f''(c_2) \quad (c_2 \text{ は } b, z \text{ の間}) $$

これらの式にそれぞれ $\lambda$ と $(1 - \lambda)$ を掛けて足し合わせます：
$$ \lambda f(a) + (1 - \lambda)f(b) = f(z) + (\lambda a + (1 - \lambda)b - z)f'(z) + \lambda \frac{(a - z)^2}{2}f''(c_1) + (1 - \lambda)\frac{(b - z)^2}{2}f''(c_2) $$

ここで $z = \lambda a + (1 - \lambda)b$ より、1次微分の項はゼロになります。
また、$f''(x) > 0$ より、剰余項は両方とも正になります。したがって、
$$ \lambda f(a) + (1 - \lambda)f(b) > f(z) = f(\lambda a + (1 - \lambda)b) $$
となり、狭義の凸関数の定義を満たします。

以上より、2つの条件は等価であることが証明されました。
