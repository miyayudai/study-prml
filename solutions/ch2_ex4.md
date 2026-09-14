# 演習問題 2.4

## 問題設定
ベータ分布 (Beta distribution) は次のように定義される。
$$ \text{Beta}(\mu | a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \mu^{a-1} (1-\mu)^{b-1} $$
ここで $\mu \in [0, 1]$ であり、$a > 0, b > 0$ である。
このベータ分布の期待値と分散が以下で与えられることを示せ。
$$ \mathbb{E}[\mu] = \frac{a}{a+b}, \quad \text{var}[\mu] = \frac{ab}{(a+b)^2(a+b+1)} $$

## 解答と導出

ベータ分布が正しく正規化されていること、すなわち
$$ \int_0^1 \mu^{a-1} (1-\mu)^{b-1} d\mu = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)} $$
であることを前提として利用する。また、ガンマ関数の基本性質である $\Gamma(x+1) = x\Gamma(x)$ も多用する。

### 1. 期待値の導出
期待値の定義に従い積分を計算する。

$$
\begin{aligned}
\mathbb{E}[\mu] &= \int_0^1 \mu \, \text{Beta}(\mu | a, b) d\mu \\
&= \int_0^1 \mu \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \mu^{a-1} (1-\mu)^{b-1} d\mu \\
&= \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \int_0^1 \mu^{(a+1)-1} (1-\mu)^{b-1} d\mu
\end{aligned}
$$

ここで、積分の部分はパラメータが $a+1$ と $b$ のベータ関数の定義そのものであるため、次のように置き換えられる。
$$ \int_0^1 \mu^{(a+1)-1} (1-\mu)^{b-1} d\mu = \frac{\Gamma(a+1)\Gamma(b)}{\Gamma(a+1+b)} $$

これを元の式に代入し、$\Gamma(x+1) = x\Gamma(x)$ を用いて整理する。

$$
\begin{aligned}
\mathbb{E}[\mu] &= \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \frac{\Gamma(a+1)\Gamma(b)}{\Gamma(a+b+1)} \\
&= \frac{\Gamma(a+b)}{\Gamma(a)} \frac{a\Gamma(a)}{(a+b)\Gamma(a+b)} \\
&= \frac{a}{a+b}
\end{aligned}
$$
よって、期待値が $\frac{a}{a+b}$ であることが示された。

### 2. 分散の導出
分散を求めるために、まず $\mathbb{E}[\mu^2]$ を計算する。

$$
\begin{aligned}
\mathbb{E}[\mu^2] &= \int_0^1 \mu^2 \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \mu^{a-1} (1-\mu)^{b-1} d\mu \\
&= \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \int_0^1 \mu^{(a+2)-1} (1-\mu)^{b-1} d\mu \\
&= \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} \frac{\Gamma(a+2)\Gamma(b)}{\Gamma(a+b+2)}
\end{aligned}
$$

再び $\Gamma(x+1) = x\Gamma(x)$ を繰り返し用いて展開する。
$$ \Gamma(a+2) = (a+1)\Gamma(a+1) = (a+1)a\Gamma(a) $$
$$ \Gamma(a+b+2) = (a+b+1)\Gamma(a+b+1) = (a+b+1)(a+b)\Gamma(a+b) $$

これらを代入して整理する。
$$
\begin{aligned}
\mathbb{E}[\mu^2] &= \frac{\Gamma(a+b)}{\Gamma(a)} \frac{(a+1)a\Gamma(a)}{(a+b+1)(a+b)\Gamma(a+b)} \\
&= \frac{a(a+1)}{(a+b)(a+b+1)}
\end{aligned}
$$

最後に分散の公式 $\text{var}[\mu] = \mathbb{E}[\mu^2] - (\mathbb{E}[\mu])^2$ に代入する。

$$
\begin{aligned}
\text{var}[\mu] &= \frac{a(a+1)}{(a+b)(a+b+1)} - \left( \frac{a}{a+b} \right)^2 \\
&= \frac{a(a+1)(a+b) - a^2(a+b+1)}{(a+b)^2(a+b+1)} \\
&= \frac{a(a^2 + ab + a + b) - (a^3 + a^2b + a^2)}{(a+b)^2(a+b+1)} \\
&= \frac{a^3 + a^2b + a^2 + ab - a^3 - a^2b - a^2}{(a+b)^2(a+b+1)} \\
&= \frac{ab}{(a+b)^2(a+b+1)}
\end{aligned}
$$

以上により、ベータ分布の分散が正しく導出された。
