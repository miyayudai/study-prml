# 演習問題 1.30

## 問題の概要
2つの1次元ガウス分布 $p(x) = \mathcal{N}(x|\mu, \sigma^2)$ と $q(x) = \mathcal{N}(x|m, s^2)$ の間のカルバック・ライブラー (Kullback-Leibler) 情報量 $\text{KL}(p||q)$ を計算する。

## 証明
カルバック・ライブラー情報量は次のように定義される。
$$
\text{KL}(p||q) = \int_{-\infty}^{\infty} p(x) \ln \frac{p(x)}{q(x)} dx = \int_{-\infty}^{\infty} p(x) \ln p(x) dx - \int_{-\infty}^{\infty} p(x) \ln q(x) dx
$$

ここで、$p(x)$ と $q(x)$ はそれぞれ次のガウス分布である。
$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right)
$$
$$
q(x) = \frac{1}{\sqrt{2\pi s^2}} \exp\left( -\frac{(x-m)^2}{2s^2} \right)
$$

第1項は $-H[p]$ （エントロピーの負の値）に等しい。1次元ガウス分布 $p(x)$ のエントロピーは既知の結果を用いて以下のように表される。
$$
\int p(x) \ln p(x) dx = -H[p] = -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{1}{2}
$$

次に、第2項を計算する。
$$
\ln q(x) = -\frac{1}{2}\ln(2\pi s^2) - \frac{(x-m)^2}{2s^2}
$$
これを $p(x)$ で期待値をとる。
$$
\int p(x) \ln q(x) dx = \int p(x) \left( -\frac{1}{2}\ln(2\pi s^2) - \frac{(x-m)^2}{2s^2} \right) dx
$$
$$
= -\frac{1}{2}\ln(2\pi s^2) - \frac{1}{2s^2} \int p(x) (x-m)^2 dx
$$

積分部分を展開して期待値を計算する。$(x-m)^2 = (x-\mu + \mu-m)^2 = (x-\mu)^2 + 2(x-\mu)(\mu-m) + (\mu-m)^2$ と分解できる。
$p(x)$ において、$\mathbb{E}[x-\mu] = 0$ および $\mathbb{E}[(x-\mu)^2] = \sigma^2$ であるから、
$$
\int p(x) (x-m)^2 dx = \sigma^2 + (\mu-m)^2
$$
となる。したがって、
$$
\int p(x) \ln q(x) dx = -\frac{1}{2}\ln(2\pi s^2) - \frac{\sigma^2 + (\mu-m)^2}{2s^2}
$$

以上をまとめて KL 情報量を計算する。
$$
\text{KL}(p||q) = \left( -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{1}{2} \right) - \left( -\frac{1}{2}\ln(2\pi s^2) - \frac{\sigma^2 + (\mu-m)^2}{2s^2} \right)
$$
整理すると、
$$
\text{KL}(p||q) = \frac{1}{2} \ln \frac{s^2}{\sigma^2} + \frac{\sigma^2 + (\mu-m)^2}{2s^2} - \frac{1}{2}
$$
これが求める結果である。
