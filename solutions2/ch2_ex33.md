## Exercise 2.33 (ガンマ分布のモーメントと最頻値)

ガンマ分布 $\mathrm{Gam}(\tau|a, b)$ の平均 $a/b$ と最頻値 $(a-1)/b$ を導出せよ。

**【解答】**

ガンマ分布の定義は以下の通りである。
$$ \mathrm{Gam}(\tau | a, b) = \frac{1}{\Gamma(a)} b^a \tau^{a-1} \exp(-b \tau) \quad (\tau > 0) $$

### 1. 平均 (期待値) の導出
確率変数 $\tau$ の期待値 $\mathbb{E}[\tau]$ は定義より以下のように計算される。
$$ \mathbb{E}[\tau] = \int_0^\infty \tau \cdot \mathrm{Gam}(\tau | a, b) d\tau $$
$$ \mathbb{E}[\tau] = \int_0^\infty \tau \frac{1}{\Gamma(a)} b^a \tau^{a-1} \exp(-b \tau) d\tau = \frac{b^a}{\Gamma(a)} \int_0^\infty \tau^{(a+1)-1} \exp(-b \tau) d\tau $$
ここで、積分部分はパラメータが $(a+1, b)$ である未規格化のガンマ分布の形をしている。ガンマ分布の規格化条件から、
$$ \int_0^\infty \tau^{\alpha-1} \exp(-b \tau) d\tau = \frac{\Gamma(\alpha)}{b^\alpha} $$
であるため、$\alpha = a+1$ を代入すると、
$$ \int_0^\infty \tau^{(a+1)-1} \exp(-b \tau) d\tau = \frac{\Gamma(a+1)}{b^{a+1}} $$
となる。これを期待値の式に代入する。
$$ \mathbb{E}[\tau] = \frac{b^a}{\Gamma(a)} \frac{\Gamma(a+1)}{b^{a+1}} $$
ガンマ関数の性質 $\Gamma(a+1) = a\Gamma(a)$ を用いると、
$$ \mathbb{E}[\tau] = \frac{b^a}{\Gamma(a)} \frac{a\Gamma(a)}{b^{a+1}} = \frac{a}{b} $$
となり、平均が $a/b$ であることが示された。

### 2. 最頻値 (Mode) の導出
確率密度関数が最大となる $\tau$ （最頻値）を求めるには、密度関数の対数をとり、それを $\tau$ で微分して $0$ となる点を求めればよい。
対数密度は以下のようになる。
$$ \ln \mathrm{Gam}(\tau | a, b) = (a-1)\ln \tau - b \tau + a \ln b - \ln \Gamma(a) $$
これを $\tau$ で微分すると、
$$ \frac{d}{d\tau} \ln \mathrm{Gam}(\tau | a, b) = \frac{a-1}{\tau} - b $$
この微分を $0$ と置いて $\tau$ について解く。
$$ \frac{a-1}{\tau} - b = 0 \implies \tau = \frac{a-1}{b} $$
また、2階微分を計算すると、
$$ \frac{d^2}{d\tau^2} \ln \mathrm{Gam}(\tau | a, b) = -\frac{a-1}{\tau^2} $$
となり、$a > 1$ の場合には常に負となるため、この停留点は極大値であり、全体における最大値（最頻値）を与えることがわかる。
（なお、$a \le 1$ の場合、密度は $\tau \to 0$ で発散、または単調減少となるため、最頻値は $\tau = 0$ とされる）。
以上より、最頻値が $(a-1)/b$ であることが示された。
