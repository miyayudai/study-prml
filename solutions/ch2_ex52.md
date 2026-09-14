# PRML Exercise 2.52

## 問題
集中度パラメータ $m$ が大きい極限（$m \to \infty$）において、von Mises分布（フォン・ミーゼス分布）がガウス分布に帰着することを示せ。

## 解答と解説
von Mises分布は円周上の分布であり、以下の式で与えられます。
$$ p(\theta|\theta_0, m) = \frac{1}{2\pi I_0(m)} \exp(m \cos(\theta - \theta_0)) $$
ここで、$I_0(m)$ は第1種0次変形Bessel関数です。
$m \to \infty$ のとき、この分布は $\theta \approx \theta_0$ の付近で鋭いピークを持ちます。したがって、$\theta - \theta_0$ が小さいとしてテイラー展開を用いて近似を行います。

### 1. 指数部分の展開
コサイン関数 $\cos(\theta - \theta_0)$ を $\theta_0$ の周りでテイラー展開すると、
$$ \cos(\theta - \theta_0) \approx 1 - \frac{(\theta - \theta_0)^2}{2} + \mathcal{O}((\theta - \theta_0)^4) $$
となります。これを指数部に代入すると、
$$ \exp(m \cos(\theta - \theta_0)) \approx \exp\left( m \left( 1 - \frac{(\theta - \theta_0)^2}{2} \right) \right) = \exp(m) \exp\left( -\frac{m}{2} (\theta - \theta_0)^2 \right) $$
となります。
$m$ が十分に大きい場合、高次項 $\exp(-m \mathcal{O}((\theta - \theta_0)^4))$ の影響は無視できるため、上記の近似が成り立ちます。

### 2. 正規化定数の評価
次に、分布全体が積分して1になるように正規化定数を考えます。
元のvon Mises分布の分母にある $2\pi I_0(m)$ は、分子 $\exp(m \cos(\theta - \theta_0))$ の $[0, 2\pi)$ における積分に等しいです。
$$ 2\pi I_0(m) = \int_0^{2\pi} \exp(m \cos(\theta - \theta_0)) d\theta $$
前述の近似を用いると、被積分関数は $\theta = \theta_0$ の周りでのみ大きな値を持ち、その他の領域では急速に $0$ に減衰します（$m \to \infty$ のため）。したがって、積分範囲を $(-\infty, \infty)$ に拡張しても値はほとんど変わりません。
$$ \int_0^{2\pi} \exp(m \cos(\theta - \theta_0)) d\theta \approx \exp(m) \int_{-\infty}^{\infty} \exp\left( -\frac{m}{2} (\theta - \theta_0)^2 \right) d\theta $$
右辺の積分は、分散が $1/m$ のガウス積分の形になっているため、
$$ \int_{-\infty}^{\infty} \exp\left( -\frac{m}{2} (\theta - \theta_0)^2 \right) d\theta = \sqrt{\frac{2\pi}{m}} $$
となります。したがって、正規化定数に関連する変形Bessel関数の漸近形は
$$ 2\pi I_0(m) \approx \exp(m) \sqrt{\frac{2\pi}{m}} $$
となります。

### 3. 極限における分布の形
1.と2.の結果を元のvon Mises分布の式に代入します。
$$ p(\theta|\theta_0, m) \approx \frac{\exp(m) \exp\left( -\frac{m}{2} (\theta - \theta_0)^2 \right)}{\exp(m) \sqrt{\frac{2\pi}{m}}} $$
$\exp(m)$ が相殺され、式を整理すると、
$$ p(\theta|\theta_0, m) \approx \sqrt{\frac{m}{2\pi}} \exp\left( -\frac{m}{2} (\theta - \theta_0)^2 \right) $$
が得られます。

これは、平均 $\theta_0$、精度（分散の逆数）$m$ を持つ1次元のガウス分布 $\mathcal{N}(\theta | \theta_0, m^{-1})$ と全く同じ形式です。
よって、$m \to \infty$ の極限において、von Mises分布はガウス分布に帰着することが示されました。
