# 演習問題 1.8

## 問題設定
1次元ガウス分布（正規分布）の期待値が $\mu$ になることを、変数変換を用いて証明する。
ガウス分布は以下のように定義される：
$$ \mathcal{N}(x | \mu, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left\{ -\frac{1}{2\sigma^2}(x - \mu)^2 \right\} $$

## 解答
連続型確率変数 $x$ の期待値（平均）は次のように定義される：
$$ \mathbb{E}[x] = \int_{-\infty}^{\infty} x \mathcal{N}(x | \mu, \sigma^2) dx $$
$$ \mathbb{E}[x] = \int_{-\infty}^{\infty} x \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left\{ -\frac{1}{2\sigma^2}(x - \mu)^2 \right\} dx $$

ここで積分を計算しやすくするために、$y = x - \mu$ と変数変換を行う。
このとき、$x = y + \mu$ であり、微小要素は $dx = dy$ となる。積分範囲は変わらず $-\infty$ から $\infty$ である。
これを式に代入すると：
$$ \mathbb{E}[x] = \int_{-\infty}^{\infty} (y + \mu) \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left( -\frac{1}{2\sigma^2}y^2 \right) dy $$
$$ = \int_{-\infty}^{\infty} y \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left( -\frac{1}{2\sigma^2}y^2 \right) dy + \mu \int_{-\infty}^{\infty} \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left( -\frac{1}{2\sigma^2}y^2 \right) dy $$

第1項について考察する。被積分関数のうち $y \exp\left( -\frac{1}{2\sigma^2}y^2 \right)$ は $y$ の奇関数（$f(-y) = -f(y)$）である。
奇関数を原点に対して対称な区間 $[-\infty, \infty]$ で積分すると、その値はゼロになることが知られている：
$$ \int_{-\infty}^{\infty} y \exp\left( -\frac{1}{2\sigma^2}y^2 \right) dy = 0 $$

第2項については、積分部分が「平均 0、分散 $\sigma^2$ のガウス分布を全区間で積分したもの」に他ならない。確率密度関数の規格化条件により、その積分値は 1 となる：
$$ \int_{-\infty}^{\infty} \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left( -\frac{1}{2\sigma^2}y^2 \right) dy = 1 $$

したがって、上記の2つの結果を足し合わせると：
$$ \mathbb{E}[x] = 0 + \mu \times 1 = \mu $$

以上より、1次元ガウス分布の期待値が $\mu$ であることが示された。この性質により、パラメータ $\mu$ はガウス分布の「平均」として自然に解釈される。
