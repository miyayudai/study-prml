# 演習問題 1.9

## 問題設定
1次元ガウス分布の分散を求める。具体的には、二次モーメント $\mathbb{E}[x^2]$ を計算し、そこから分散 $var[x]$ が $\sigma^2$ となることを示す。

## 解答
確率変数 $x$ の二次モーメント $\mathbb{E}[x^2]$ は次のように定義される：
$$ \mathbb{E}[x^2] = \int_{-\infty}^{\infty} x^2 \mathcal{N}(x | \mu, \sigma^2) dx $$
$$ \mathbb{E}[x^2] = \int_{-\infty}^{\infty} x^2 \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left\{ -\frac{1}{2\sigma^2}(x - \mu)^2 \right\} dx $$

演習問題1.8と同様に、$y = x - \mu$ （すなわち $x = y + \mu$, $dx = dy$）と変数変換を行う：
$$ \mathbb{E}[x^2] = \int_{-\infty}^{\infty} (y + \mu)^2 \frac{1}{(2\pi\sigma^2)^{1/2}} \exp\left( -\frac{1}{2\sigma^2}y^2 \right) dy $$
展開すると、
$$ (y + \mu)^2 = y^2 + 2\mu y + \mu^2 $$
となるので、積分は以下の3つの項に分割される：
$$ \mathbb{E}[x^2] = \int y^2 \mathcal{N}(y | 0, \sigma^2) dy + 2\mu \int y \mathcal{N}(y | 0, \sigma^2) dy + \mu^2 \int \mathcal{N}(y | 0, \sigma^2) dy $$

各項を評価していく：
1. **第2項**：被積分関数は奇関数であるため、積分値は 0 となる（演習1.8で確認済み）。
2. **第3項**：確率密度関数の全空間での積分は 1 であるため、この項は $\mu^2 \times 1 = \mu^2$ となる。
3. **第1項**：これは平均 0、分散 $\sigma^2$ のガウス分布の分散の定義そのものであるが、部分積分を用いて直接評価する：
   $$ I = \frac{1}{(2\pi\sigma^2)^{1/2}} \int_{-\infty}^{\infty} y \cdot \left( y \exp\left( -\frac{y^2}{2\sigma^2} \right) \right) dy $$
   ここで $u = y$、 $dv = y \exp(-y^2 / 2\sigma^2) dy$ とおくと、 $du = dy$、$v = -\sigma^2 \exp(-y^2 / 2\sigma^2)$ となる。
   部分積分 $\int u dv = [uv] - \int v du$ を適用すると：
   $$ I = \frac{1}{(2\pi\sigma^2)^{1/2}} \left[ -y\sigma^2 \exp\left( -\frac{y^2}{2\sigma^2} \right) \right]_{-\infty}^{\infty} + \frac{1}{(2\pi\sigma^2)^{1/2}} \int_{-\infty}^{\infty} \sigma^2 \exp\left( -\frac{y^2}{2\sigma^2} \right) dy $$
   第1項目の境界値評価では、指数関数の減衰が多項式の増大を上回るため、極限は 0 になる。第2項目は：
   $$ \sigma^2 \int_{-\infty}^{\infty} \mathcal{N}(y | 0, \sigma^2) dy = \sigma^2 \times 1 = \sigma^2 $$

したがって、3つの項をすべて足し合わせると、
$$ \mathbb{E}[x^2] = \sigma^2 + 0 + \mu^2 = \mu^2 + \sigma^2 $$
が得られる。

最後に、分散の定義 $var[x] = \mathbb{E}[x^2] - (\mathbb{E}[x])^2$ を用いる。演習1.8で $\mathbb{E}[x] = \mu$ を示しているので：
$$ var[x] = (\mu^2 + \sigma^2) - \mu^2 = \sigma^2 $$

これにより、パラメータ $\sigma^2$ がこの分布の分散を表すことが厳密に証明された。
