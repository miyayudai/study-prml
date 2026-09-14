## Exercise 2.32 (ガンマ事前分布による精度の事後分布)

ガウス分布の精度に対する共役事前分布がガンマ分布になることを示せ。

**【解答】**

分散の逆数である精度（precision）を $\lambda \equiv 1/\sigma^2$ と定義する。平均 $\mu$ が既知であるとし、精度 $\lambda$ を推論する問題を考える。
独立同分布（i.i.d.）に従う $N$ 個の観測データ $\mathbf{x} = (x_1, \dots, x_N)^T$ に対する尤度関数は、1次元ガウス分布の定義から以下のようになる。
$$ p(\mathbf{x} | \lambda) = \prod_{n=1}^N \mathcal{N}(x_n | \mu, \lambda^{-1}) = \prod_{n=1}^N \left[ \frac{\lambda^{1/2}}{(2\pi)^{1/2}} \exp\left( -\frac{\lambda}{2}(x_n - \mu)^2 \right) \right] $$
これを $\lambda$ の関数として整理すると、
$$ p(\mathbf{x} | \lambda) = \frac{\lambda^{N/2}}{(2\pi)^{N/2}} \exp\left( -\frac{\lambda}{2} \sum_{n=1}^N (x_n - \mu)^2 \right) $$
となる。尤度関数は $\lambda$ について次のような関数形を持っていることがわかる：
$$ p(\mathbf{x} | \lambda) \propto \lambda^{N/2} \exp(- \lambda \times \text{const}) $$

共役事前分布を構成するためには、事前分布 $p(\lambda)$ も尤度関数と同じ関数形を持つ必要がある。したがって、次のような形を仮定する。
$$ p(\lambda) \propto \lambda^{a_0 - 1} \exp(-b_0 \lambda) $$
これはまさにガンマ分布（Gamma distribution）$\mathrm{Gam}(\lambda | a_0, b_0)$ の関数形である。
$$ \mathrm{Gam}(\lambda | a_0, b_0) = \frac{1}{\Gamma(a_0)} b_0^{a_0} \lambda^{a_0 - 1} \exp(-b_0 \lambda) $$
ここで、$a_0 > 0$ は形状パラメータ（shape parameter）、$b_0 > 0$ は尺度パラメータ（rate parameter）である。

このガンマ事前分布を用いて、ベイズの定理により事後分布 $p(\lambda | \mathbf{x})$ を計算する。
$$ p(\lambda | \mathbf{x}) \propto p(\mathbf{x} | \lambda) p(\lambda) $$
$$ p(\lambda | \mathbf{x}) \propto \left[ \lambda^{N/2} \exp\left( -\frac{\lambda}{2} \sum_{n=1}^N (x_n - \mu)^2 \right) \right] \left[ \lambda^{a_0 - 1} \exp(-b_0 \lambda) \right] $$
指数部とべき乗部をそれぞれまとめると、
$$ p(\lambda | \mathbf{x}) \propto \lambda^{a_0 + N/2 - 1} \exp\left( - \lambda \left( b_0 + \frac{1}{2} \sum_{n=1}^N (x_n - \mu)^2 \right) \right) $$
となる。
この関数形を見ると、事後分布も再びガンマ分布の未規格化の形をしていることがわかる。すなわち、
$$ p(\lambda | \mathbf{x}) = \mathrm{Gam}(\lambda | a_N, b_N) $$
であり、そのパラメータは以下のように更新される：
$$ a_N = a_0 + \frac{N}{2} $$
$$ b_N = b_0 + \frac{1}{2} \sum_{n=1}^N (x_n - \mu)^2 $$

これにより、ガウス分布の精度に対する共役事前分布がガンマ分布であることが示された。
また、パラメータの更新式から、$N$ 個のデータが観測されると、パラメータ $a$ は $N/2$ だけ増加し、パラメータ $b$ はデータの分散に比例する量 $\frac{1}{2} \sum (x_n - \mu)^2$ だけ増加することがわかる。これは事前分布の $2a_0$ を「有効観測数」と解釈できることを示唆している。
