## Exercise 2.39 (Student's t分布のガウス分布への収束)

自由度 $\nu$ が無限大（$\nu \to \infty$）になる極限において、スチューデントのt分布がガウス分布に収束することを示します。

### t分布の式
1次元のt分布の定義は以下の通りです。
$$
\mathrm{St}(x | \mu, \lambda, \nu) = \frac{\Gamma((\nu + 1)/2)}{\Gamma(\nu/2)} \left( \frac{\lambda}{\pi \nu} \right)^{1/2} \left[ 1 + \frac{\lambda(x - \mu)^2}{\nu} \right]^{-(\nu + 1)/2}
$$

### 指数部（べき乗部分）の極限
式の中で $x$ に依存する部分は以下の形をしています。
$$
\left[ 1 + \frac{\lambda(x - \mu)^2}{\nu} \right]^{-(\nu + 1)/2}
$$
ここで、ネイピア数 $e$ の定義（あるいは極限公式）
$$
\lim_{\nu \to \infty} \left( 1 + \frac{z}{\nu} \right)^\nu = e^z
$$
を用います。$z = \lambda(x - \mu)^2$ と置きます。
$$
\lim_{\nu \to \infty} \left[ 1 + \frac{\lambda(x - \mu)^2}{\nu} \right]^{-(\nu + 1)/2} = \lim_{\nu \to \infty} \left( \left[ 1 + \frac{\lambda(x - \mu)^2}{\nu} \right]^\nu \right)^{-1/2} \left[ 1 + \frac{\lambda(x - \mu)^2}{\nu} \right]^{-1/2}
$$
$$
= (e^{\lambda(x - \mu)^2})^{-1/2} \cdot 1 = \exp\left\{ -\frac{\lambda}{2} (x - \mu)^2 \right\}
$$
これは、ガウス分布の指数部と正確に一致します。

### 正規化定数の極限
次に、正規化定数の極限を調べます。スターリングの近似公式（$\Gamma(z+1) \approx \sqrt{2\pi z} (z/e)^z$）を用いて $\frac{\Gamma((\nu + 1)/2)}{\Gamma(\nu/2) \nu^{1/2}}$ の極限を直接計算することもできますが、より簡単な方法は、確率密度関数は積分が $1$ にならなければならないという性質（規格化条件）を利用することです。

先ほどの計算により、$p(x)$ は非正規化ガウス分布に収束することが分かっています。
$$
\lim_{\nu \to \infty} \mathrm{St}(x | \mu, \lambda, \nu) \propto \exp\left\{ -\frac{\lambda}{2} (x - \mu)^2 \right\}
$$
ガウス分布の性質から、この指数部を積分して $1$ にするためには、正規化定数は
$$
\left( \frac{\lambda}{2\pi} \right)^{1/2}
$$
とならなければなりません。
実際に元の式の係数部分を厳密に評価すると、
$$
\lim_{\nu \to \infty} \frac{\Gamma((\nu + 1)/2)}{\Gamma(\nu/2)} \left( \frac{\lambda}{\pi \nu} \right)^{1/2} = \left( \frac{\lambda}{2\pi} \right)^{1/2}
$$
となります。
したがって、全体として
$$
\lim_{\nu \to \infty} \mathrm{St}(x | \mu, \lambda, \nu) = \left( \frac{\lambda}{2\pi} \right)^{1/2} \exp\left\{ -\frac{\lambda}{2} (x - \mu)^2 \right\} = \mathcal{N}(x | \mu, \lambda^{-1})
$$
となり、自由度 $\nu \to \infty$ のときt分布は平均 $\mu$、精度 $\lambda$ のガウス分布に完全に収束することが示されました。
