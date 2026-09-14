## Exercise 2.35 (ガウス-ガンマ分布の事後分布更新)

平均と精度の双方が未知のときの共役事前分布の事後更新式をまとめよ。

**【解答】**

1次元のガウス分布において、平均 $\mu$ と精度 $\lambda$ （分散の逆数）の双方が未知である場合を考える。データ $\mathbf{x} = (x_1, \dots, x_N)^T$ が観測されたときの尤度関数は以下のようになる。
$$ p(\mathbf{x} | \mu, \lambda) = \prod_{n=1}^N \left( \frac{\lambda}{2\pi} \right)^{1/2} \exp\left( -\frac{\lambda}{2}(x_n - \mu)^2 \right) = \left( \frac{\lambda}{2\pi} \right)^{N/2} \exp\left( -\frac{\lambda}{2} \sum_{n=1}^N (x_n - \mu)^2 \right) $$

この尤度関数の指数部を変形する。標本平均 $\bar{x} = \frac{1}{N}\sum_{n=1}^N x_n$ と標本分散に比例する項を用いて、和を分割する：
$$ \sum_{n=1}^N (x_n - \mu)^2 = \sum_{n=1}^N (x_n - \bar{x} + \bar{x} - \mu)^2 = \sum_{n=1}^N (x_n - \bar{x})^2 + N(\bar{x} - \mu)^2 $$
となるため、尤度関数は次のように書ける。
$$ p(\mathbf{x} | \mu, \lambda) \propto \lambda^{N/2} \exp\left( -\frac{\lambda}{2} \sum_{n=1}^N (x_n - \bar{x})^2 -\frac{N\lambda}{2}(\mu - \bar{x})^2 \right) $$

この尤度に対する共役事前分布 $p(\mu, \lambda)$ は、$\mu$ と $\lambda$ について尤度と同じ関数形を持たなければならない。これを実現する分布が**ガウス-ガンマ分布 (Gaussian-Gamma distribution)** である：
$$ p(\mu, \lambda) = \mathcal{N}(\mu | \mu_0, (\beta_0 \lambda)^{-1}) \mathrm{Gam}(\lambda | a_0, b_0) $$
展開すると、
$$ p(\mu, \lambda) \propto \lambda^{1/2} \exp\left( -\frac{\beta_0 \lambda}{2} (\mu - \mu_0)^2 \right) \lambda^{a_0 - 1} \exp(-b_0 \lambda) = \lambda^{a_0 - 1/2} \exp\left( -\frac{\beta_0 \lambda}{2} (\mu - \mu_0)^2 - b_0 \lambda \right) $$

ベイズの定理により、事後分布 $p(\mu, \lambda | \mathbf{x}) \propto p(\mathbf{x} | \mu, \lambda) p(\mu, \lambda)$ を計算する。尤度と事前分布を掛け合わせると、
$$ p(\mu, \lambda | \mathbf{x}) \propto \lambda^{a_0 + N/2 - 1/2} \exp\left( - \lambda \left[ \frac{\beta_0}{2}(\mu - \mu_0)^2 + \frac{N}{2}(\mu - \bar{x})^2 + \frac{1}{2} \sum_{n=1}^N (x_n - \bar{x})^2 + b_0 \right] \right) $$
ここで、指数部における $\mu$ に関する二次式（大括弧内の最初の2項）を平方完成する。
$$ \frac{\beta_0}{2}(\mu - \mu_0)^2 + \frac{N}{2}(\mu - \bar{x})^2 = \frac{1}{2} \left( \beta_0(\mu^2 - 2\mu\mu_0 + \mu_0^2) + N(\mu^2 - 2\mu\bar{x} + \bar{x}^2) \right) $$
$$ = \frac{1}{2} \left( (\beta_0 + N)\mu^2 - 2(\beta_0 \mu_0 + N\bar{x})\mu + \beta_0 \mu_0^2 + N\bar{x}^2 \right) $$
$$ = \frac{\beta_0 + N}{2} \left( \mu - \frac{\beta_0 \mu_0 + N\bar{x}}{\beta_0 + N} \right)^2 + \frac{1}{2} \left( \beta_0 \mu_0^2 + N\bar{x}^2 - \frac{(\beta_0 \mu_0 + N\bar{x})^2}{\beta_0 + N} \right) $$
残りの定数項（$\mu$ に依存しない項）を整理すると、
$$ \beta_0 \mu_0^2 + N\bar{x}^2 - \frac{(\beta_0 \mu_0 + N\bar{x})^2}{\beta_0 + N} = \frac{\beta_0(\beta_0+N)\mu_0^2 + N(\beta_0+N)\bar{x}^2 - (\beta_0^2 \mu_0^2 + 2\beta_0 N \mu_0 \bar{x} + N^2 \bar{x}^2)}{\beta_0 + N} $$
$$ = \frac{\beta_0 N \mu_0^2 + \beta_0 N \bar{x}^2 - 2\beta_0 N \mu_0 \bar{x}}{\beta_0 + N} = \frac{\beta_0 N (\bar{x} - \mu_0)^2}{\beta_0 + N} $$
となる。

これらを元の事後分布の式に戻すと、事後分布も再びガウス-ガンマ分布の形をしていることがわかる。
$$ p(\mu, \lambda | \mathbf{x}) = \mathcal{N}(\mu | \mu_N, (\beta_N \lambda)^{-1}) \mathrm{Gam}(\lambda | a_N, b_N) $$
そして、パラメータの更新式は以下のように導出される。

1. **$\mu$ の精度倍率 $\beta$ の更新:**
$$ \beta_N = \beta_0 + N $$
2. **$\mu$ の平均 $\mu_N$ の更新:**
$$ \mu_N = \frac{\beta_0 \mu_0 + N\bar{x}}{\beta_0 + N} = \frac{\beta_0}{\beta_0 + N}\mu_0 + \frac{N}{\beta_0 + N}\bar{x} $$
3. **ガンマ分布の形状パラメータ $a$ の更新:**
$$ a_N = a_0 + \frac{N}{2} $$
4. **ガンマ分布の尺度パラメータ $b$ の更新:**
指数部の $\mu$ に依存しない項を全てまとめることで得られる。
$$ b_N = b_0 + \frac{1}{2} \sum_{n=1}^N (x_n - \bar{x})^2 + \frac{\beta_0 N (\bar{x} - \mu_0)^2}{2(\beta_0 + N)} $$
ここで、第2項はデータの分散（各データと標本平均の差の二乗和）に由来し、第3項は事前平均 $\mu_0$ と標本平均 $\bar{x}$ の差に起因する追加の分散成分である。

以上より、平均と精度の双方が未知のときの共役事前分布であるガウス-ガンマ分布の事後分布の更新式が導出された。
