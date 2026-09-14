# 演習問題 2.5

## 問題設定
ベータ分布 (Beta distribution) が次のように定義されている。
$$ \text{Beta}(\mu | a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \mu^{a-1} (1-\mu)^{b-1} $$
ここで、パラメータが $a > 1, b > 1$ のとき、この分布の最頻値（モード）が次で与えられることを示せ。
$$ \mu_{\text{mode}} = \frac{a - 1}{a + b - 2} $$

## 解答と導出

最頻値（モード）とは、確率密度関数 $\text{Beta}(\mu | a, b)$ が最大となる $\mu$ の値である。これを求めるためには、分布の対数をとり、それを $\mu$ で微分して 0 になる点を求めればよい（対数関数は単調増加関数であるため、元の関数を最大化する $\mu$ と対数を最大化する $\mu$ は一致する）。

### 1. 対数密度の計算
分布の対数 $\ln \text{Beta}(\mu | a, b)$ を求める。
定数部分（$\mu$ に依存しない部分）を $C = \ln \left( \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \right)$ とおくと、

$$
\begin{aligned}
\ln \text{Beta}(\mu | a, b) &= C + \ln \left( \mu^{a-1} (1-\mu)^{b-1} \right) \\
&= C + (a-1)\ln \mu + (b-1)\ln (1-\mu)
\end{aligned}
$$

### 2. $\mu$ による微分
この対数密度関数を $\mu$ で微分する。

$$
\begin{aligned}
\frac{d}{d\mu} \ln \text{Beta}(\mu | a, b) &= \frac{d}{d\mu} \left( (a-1)\ln \mu + (b-1)\ln (1-\mu) \right) \\
&= \frac{a-1}{\mu} + \frac{b-1}{1-\mu} \cdot (-1) \\
&= \frac{a-1}{\mu} - \frac{b-1}{1-\mu}
\end{aligned}
$$

### 3. 微分を 0 と置いて最頻値を導出
最大値を与える極値点を求めるため、微分の結果を 0 に等しいと置く。

$$ \frac{a-1}{\mu} - \frac{b-1}{1-\mu} = 0 $$

これを $\mu$ について解く。
$$ \frac{a-1}{\mu} = \frac{b-1}{1-\mu} $$

両辺に $\mu(1-\mu)$ を掛けて分母を払う。
$$ (a-1)(1-\mu) = (b-1)\mu $$
$$ a - 1 - (a-1)\mu = b\mu - \mu $$
$$ a - 1 = (b-1)\mu + (a-1)\mu $$
$$ a - 1 = (b - 1 + a - 1)\mu $$
$$ a - 1 = (a + b - 2)\mu $$

したがって、$\mu$ について解くと次のようになる。
$$ \mu = \frac{a - 1}{a + b - 2} $$

### 4. 最大値かどうかの確認（2階微分）
この極値が確かに最大値を与えているか（つまり最頻値であるか）を確認するため、2階微分を調べる。
$$
\begin{aligned}
\frac{d^2}{d\mu^2} \ln \text{Beta}(\mu | a, b) &= \frac{d}{d\mu} \left( \frac{a-1}{\mu} - \frac{b-1}{1-\mu} \right) \\
&= -\frac{a-1}{\mu^2} - \frac{b-1}{(1-\mu)^2}
\end{aligned}
$$
$a > 1, b > 1$ という条件より、$a-1 > 0$ かつ $b-1 > 0$ である。
$\mu \in (0, 1)$ の範囲において、$\mu^2 > 0$ かつ $(1-\mu)^2 > 0$ であるため、
$$ -\frac{a-1}{\mu^2} - \frac{b-1}{(1-\mu)^2} < 0 $$
となり、2階微分は常に負である。
これは対数密度関数が上に凸であることを意味し、導出された極値点が唯一の最大値（モード）であることが保証される。

よって、ベータ分布の最頻値は
$$ \mu_{\text{mode}} = \frac{a - 1}{a + b - 2} $$
となることが証明された。
