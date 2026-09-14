# 演習問題 1.27

## 問題文
ミンコフスキー損失関数に基づく期待損失
$$ \mathbb{E}[L_q] = \iint |y(\mathbf{x}) - t|^q p(\mathbf{x}, t) \mathrm{d}\mathbf{x} \mathrm{d}t $$
について考える。
(a) $q=1$ の場合、これを最小化する $y(\mathbf{x})$ が条件付き中央値（conditional median）になることを示せ。
(b) $q \to 0$ の場合、これを最小化する $y(\mathbf{x})$ が条件付き最頻値（conditional mode）になることを示せ。

## 解答と解説
### (a) $q=1$ の場合（絶対値損失と中央値）
$q=1$ のとき、期待損失の $y(\mathbf{x})$ に関する最小化は、各 $\mathbf{x}$ ごとに次の関数 $f(y)$ を最小化することに等しいです（ここで $\mathbf{x}$ への依存関係を省略して $y$ と書きます）。
$$ f(y) = \int_{-\infty}^{\infty} |y - t| p(t | \mathbf{x}) \mathrm{d}t $$
絶対値を外すため、積分区間を $t < y$ と $t \ge y$ に分割します。
$$ f(y) = \int_{-\infty}^{y} (y - t) p(t | \mathbf{x}) \mathrm{d}t - \int_{y}^{\infty} (y - t) p(t | \mathbf{x}) \mathrm{d}t $$
$y$ で微分するためにライプニッツの積分法則を適用します。
$$ \frac{\partial f(y)}{\partial y} = \int_{-\infty}^{y} p(t | \mathbf{x}) \mathrm{d}t + (y - y) p(y | \mathbf{x}) - \int_{y}^{\infty} p(t | \mathbf{x}) \mathrm{d}t - (- (y - y) p(y | \mathbf{x})) $$
$$ = \int_{-\infty}^{y} p(t | \mathbf{x}) \mathrm{d}t - \int_{y}^{\infty} p(t | \mathbf{x}) \mathrm{d}t = 0 $$
これを満たすためには、以下が成り立つ必要があります。
$$ \int_{-\infty}^{y} p(t | \mathbf{x}) \mathrm{d}t = \int_{y}^{\infty} p(t | \mathbf{x}) \mathrm{d}t $$
確率密度の全体積分は 1 であるため、この等式は双方が $1/2$ になることを意味します。すなわち、確率の累積がちょうど 50% になる点であり、これは**条件付き中央値**の定義そのものです。

### (b) $q \to 0$ の場合（0-1近似と最頻値）
期待損失を次のように変形します。
$$ \mathbb{E}[L_q] = \int p(\mathbf{x}) \left( \int |y(\mathbf{x}) - t|^q p(t | \mathbf{x}) \mathrm{d}t \right) \mathrm{d}\mathbf{x} $$
内側の積分 $\int |y(\mathbf{x}) - t|^q p(t | \mathbf{x}) \mathrm{d}t$ を最小化することを考えます。
$q \to 0$ の極限では、関数 $|y - t|^q$ は次のように振る舞います。
$$ \lim_{q \to 0} |y - t|^q = \begin{cases} 0 & (t = y) \\ 1 & (t \neq y) \end{cases} $$
厳密な極限では積分値の差が見えづらいため、$t = y$ の非常に小さな $\epsilon$ 近傍 $\mathcal{U} = [y-\epsilon, y+\epsilon]$ を考えます。$q$ が非常に0に近いとき、
$$ \int |y - t|^q p(t | \mathbf{x}) \mathrm{d}t \approx \int_{t \notin \mathcal{U}} 1 \cdot p(t | \mathbf{x}) \mathrm{d}t + \int_{t \in \mathcal{U}} 0 \cdot p(t | \mathbf{x}) \mathrm{d}t = 1 - \int_{y-\epsilon}^{y+\epsilon} p(t | \mathbf{x}) \mathrm{d}t $$
この値を最小化することは、第二項である積分 $\int_{y-\epsilon}^{y+\epsilon} p(t | \mathbf{x}) \mathrm{d}t$ を最大化することと同義です。
$\epsilon$ が十分に小さいとき、この積分は $2\epsilon \cdot p(y | \mathbf{x})$ で近似されます。したがって、この値を最大化する $y$ は、確率密度関数 $p(t | \mathbf{x})$ が最大となる点、すなわち**条件付き最頻値（conditional mode）**になります。
