## Exercise 2.6 (ベータ分布の平均、分散、最頻値)

ベータ分布は区間 $[0, 1]$ 上で定義される連続確率分布であり、パラメータ $\mu$ の事後分布として重要です。確率密度関数は次のように定義されます：
$$ \mathrm{Beta}(\mu|a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \mu^{a-1} (1-\mu)^{b-1} = \frac{1}{B(a, b)} \mu^{a-1} (1-\mu)^{b-1} $$
ここで、$B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$ はベータ関数です。ガンマ関数の重要な性質である $\Gamma(x+1) = x\Gamma(x)$ を活用して、平均、分散、および最頻値を導出します。

### 1. 平均（期待値）の導出

ベータ分布に従う確率変数 $\mu$ の期待値 $\mathbb{E}[\mu]$ は、定義に従って積分を計算します：
$$ \mathbb{E}[\mu] = \int_0^1 \mu \, \mathrm{Beta}(\mu|a, b) d\mu $$
$$ \mathbb{E}[\mu] = \int_0^1 \mu \frac{1}{B(a, b)} \mu^{a-1} (1-\mu)^{b-1} d\mu = \frac{1}{B(a, b)} \int_0^1 \mu^a (1-\mu)^{b-1} d\mu $$

積分部分はパラメータが $a+1$ と $b$ のベータ関数の積分形と同じですので、次のように書き換えられます：
$$ \mathbb{E}[\mu] = \frac{B(a+1, b)}{B(a, b)} $$

ここで、ベータ関数とガンマ関数の関係 $B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$ を用います：
$$ \mathbb{E}[\mu] = \frac{\Gamma(a+1)\Gamma(b)}{\Gamma(a+b+1)} \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} $$

ガンマ関数の性質 $\Gamma(x+1) = x\Gamma(x)$ を用いて簡略化します。$\Gamma(a+1) = a\Gamma(a)$ であり、$\Gamma(a+b+1) = (a+b)\Gamma(a+b)$ です：
$$ \mathbb{E}[\mu] = \frac{a\Gamma(a)\Gamma(b)}{(a+b)\Gamma(a+b)} \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} = \frac{a}{a+b} $$

### 2. 分散の導出

分散は $\mathrm{var}[\mu] = \mathbb{E}[\mu^2] - (\mathbb{E}[\mu])^2$ を用いて計算します。まず、2次のモーメント $\mathbb{E}[\mu^2]$ を求めます：
$$ \mathbb{E}[\mu^2] = \int_0^1 \mu^2 \mathrm{Beta}(\mu|a, b) d\mu = \frac{1}{B(a, b)} \int_0^1 \mu^{a+1} (1-\mu)^{b-1} d\mu = \frac{B(a+2, b)}{B(a, b)} $$

平均の導出と同様に、ガンマ関数で表して展開します：
$$ \mathbb{E}[\mu^2] = \frac{\Gamma(a+2)\Gamma(b)}{\Gamma(a+b+2)} \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} = \frac{(a+1)a\Gamma(a)\Gamma(b)}{(a+b+1)(a+b)\Gamma(a+b)} \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} = \frac{a(a+1)}{(a+b)(a+b+1)} $$

したがって、分散は：
$$ \mathrm{var}[\mu] = \mathbb{E}[\mu^2] - (\mathbb{E}[\mu])^2 = \frac{a(a+1)}{(a+b)(a+b+1)} - \left(\frac{a}{a+b}\right)^2 $$
通分して計算します：
$$ \mathrm{var}[\mu] = \frac{a(a+1)(a+b) - a^2(a+b+1)}{(a+b)^2(a+b+1)} = \frac{a(a^2+ab+a+b) - a^3-a^2b-a^2}{(a+b)^2(a+b+1)} $$
$$ \mathrm{var}[\mu] = \frac{a^3+a^2b+a^2+ab - a^3-a^2b-a^2}{(a+b)^2(a+b+1)} = \frac{ab}{(a+b)^2(a+b+1)} $$

### 3. 最頻値（mode）の導出

最頻値は、確率密度関数 $\mathrm{Beta}(\mu|a, b)$ を最大化する $\mu$ の値です。対数尤度を最大化する方が計算が簡単なため、対数密度 $\ln p(\mu)$ を $\mu$ で微分して 0 と置きます。
$$ \ln p(\mu) = \ln \left( \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \right) + (a-1)\ln\mu + (b-1)\ln(1-\mu) $$

$\mu$ で微分します（$a, b > 1$ と仮定）：
$$ \frac{d}{d\mu}\ln p(\mu) = \frac{a-1}{\mu} - \frac{b-1}{1-\mu} = 0 $$
$$ \frac{a-1}{\mu} = \frac{b-1}{1-\mu} \implies (a-1)(1-\mu) = (b-1)\mu $$
$$ a - 1 - (a - 1)\mu = b\mu - \mu $$
$$ a - 1 = (a - 1 + b - 1)\mu $$
$$ a - 1 = (a + b - 2)\mu \implies \mu = \frac{a-1}{a+b-2} $$

このように、最頻値は $\frac{a-1}{a+b-2}$ となることが導かれます。
