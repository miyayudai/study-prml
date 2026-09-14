## Exercise 2.37 (ガンマ分布の周辺化によるStudent's t分布の導出)

精度パラメータ $\tau$ をガンマ分布で積分消去（周辺化）することにより、1次元のスチューデントのt分布（Student's t-distribution）が導出されることを示します。これは、ガウス分布の分散（精度）に対する不確実性を考慮した無限混合分布としてt分布が解釈できることを意味します。

### 設定
平均 $\mu$、精度 $\tau$ のガウス分布を仮定します。
$$
p(x | \mu, \tau) = \mathcal{N}(x | \mu, \tau^{-1}) = \left( \frac{\tau}{2\pi} \right)^{1/2} \exp\left\{ -\frac{\tau}{2} (x - \mu)^2 \right\}
$$
精度 $\tau$ に対する事前分布として、パラメータ $a, b$ のガンマ分布を考えます。
$$
p(\tau | a, b) = \mathrm{Gam}(\tau | a, b) = \frac{b^a}{\Gamma(a)} \tau^{a-1} \exp(-b\tau)
$$

### 周辺化の計算
同時分布 $p(x, \tau) = p(x | \tau)p(\tau)$ から、$\tau$ について積分することで周辺分布 $p(x)$ を求めます。
$$
p(x) = \int_0^\infty p(x | \mu, \tau) p(\tau | a, b) \, d\tau
$$
$$
p(x) = \int_0^\infty \left( \frac{\tau}{2\pi} \right)^{1/2} \exp\left\{ -\frac{\tau}{2} (x - \mu)^2 \right\} \frac{b^a}{\Gamma(a)} \tau^{a-1} \exp(-b\tau) \, d\tau
$$
定数を積分の外に出し、指数部と $\tau$ のべき乗をまとめます。
$$
p(x) = \frac{b^a}{\Gamma(a)} \frac{1}{(2\pi)^{1/2}} \int_0^\infty \tau^{a - 1/2} \exp\left\{ -\tau \left( b + \frac{(x - \mu)^2}{2} \right) \right\} \, d\tau
$$
ここで、ガンマ関数の定義（$\int_0^\infty t^{z-1} e^{-ct} dt = \frac{\Gamma(z)}{c^z}$）を用います。
今回の積分では、$z = a + 1/2$、$c = b + \frac{(x - \mu)^2}{2}$ に対応します。したがって、積分の結果は
$$
\int_0^\infty \tau^{a - 1/2} \exp(-c\tau) \, d\tau = \frac{\Gamma(a + 1/2)}{\left( b + \frac{(x - \mu)^2}{2} \right)^{a + 1/2}}
$$
となります。これを元の式に代入すると、
$$
p(x) = \frac{b^a}{\Gamma(a)} \frac{1}{(2\pi)^{1/2}} \frac{\Gamma(a + 1/2)}{\left( b + \frac{(x - \mu)^2}{2} \right)^{a + 1/2}}
$$
$$
p(x) = \frac{\Gamma(a + 1/2)}{\Gamma(a)} \frac{1}{(2\pi b)^{1/2}} \left( 1 + \frac{(x - \mu)^2}{2b} \right)^{-(a + 1/2)}
$$

### t分布の標準形への対応
ここで、新しいパラメータ $\nu = 2a$、$\lambda = a/b$ を導入します。これにより $a = \nu/2$、$b = \nu / (2\lambda)$ となります。
これらを上の式に代入します。
$$
2b = \frac{\nu}{\lambda} \implies \frac{(x - \mu)^2}{2b} = \frac{\lambda(x - \mu)^2}{\nu}
$$
$$
a + 1/2 = \frac{\nu + 1}{2}
$$
また、正規化定数の部分は、
$$
\frac{1}{(2\pi b)^{1/2}} = \left( \frac{\lambda}{\pi \nu} \right)^{1/2}
$$
これらをまとめると、
$$
p(x) = \frac{\Gamma((\nu + 1)/2)}{\Gamma(\nu/2)} \left( \frac{\lambda}{\pi \nu} \right)^{1/2} \left[ 1 + \frac{\lambda(x - \mu)^2}{\nu} \right]^{-(\nu + 1)/2}
$$
これは、自由度 $\nu$、位置パラメータ $\mu$、精度パラメータ $\lambda$（分散の逆数に対応）を持つスチューデントのt分布 $\mathrm{St}(x | \mu, \lambda, \nu)$ と完全に一致します。
ガンマ積分を実行することで、ガウス分布の指数関数的減衰ではなく、t分布特有の「べき乗減衰」が得られることがわかります。
