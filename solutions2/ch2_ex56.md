# Exercise 2.56

## 問題
分散 $\sigma^2$ に対する Jeffreys 事前分布（Jeffreys prior）が $p(\sigma^2) \propto 1/\sigma^2$ になることを示せ。

## 解答と解説

Jeffreys 事前分布は、パラメータの変換に対して不変な事前分布として知られており、フィッシャー情報量 $I(\theta)$ を用いて次のように定義されます。
$$ p(\theta) \propto \sqrt{I(\theta)} $$
ここで、フィッシャー情報量 $I(\theta)$ は対数尤度関数の二階微分の期待値の符号を反転させたものです。
$$ I(\theta) = -\mathbb{E} \left[ \frac{\partial^2}{\partial \theta^2} \ln p(x | \theta) \right] $$

本問では、平均 $\mu$ が既知のガウス分布 $\mathcal{N}(x | \mu, \sigma^2)$ における、分散パラメータ $\theta = \sigma^2$ に対する事前分布を考えます。

まず、1つのデータ点 $x$ に対する尤度関数を書き下し、その対数を取ります。
$$ p(x | \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2} \right) $$
$$ \ln p(x | \sigma^2) = -\frac{1}{2} \ln(2\pi) - \frac{1}{2} \ln(\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2} $$

次に、この対数尤度を $\sigma^2$ について1階微分します。
$$ \frac{\partial}{\partial (\sigma^2)} \ln p(x | \sigma^2) = -\frac{1}{2\sigma^2} + \frac{(x-\mu)^2}{2(\sigma^2)^2} $$

さらに、もう一度 $\sigma^2$ について微分（2階微分）します。
$$ \frac{\partial^2}{\partial (\sigma^2)^2} \ln p(x | \sigma^2) = \frac{1}{2(\sigma^2)^2} - \frac{(x-\mu)^2}{(\sigma^2)^3} $$

ここで、フィッシャー情報量 $I(\sigma^2)$ を求めるために期待値を取ります。データ $x$ は真の分布 $\mathcal{N}(\mu, \sigma^2)$ に従うと仮定しているため、分散の定義より $\mathbb{E}[(x-\mu)^2] = \sigma^2$ となります。
したがって、期待値は次のように計算されます。
$$ \mathbb{E} \left[ \frac{\partial^2}{\partial (\sigma^2)^2} \ln p(x | \sigma^2) \right] = \frac{1}{2\sigma^4} - \frac{\mathbb{E}[(x-\mu)^2]}{\sigma^6} = \frac{1}{2\sigma^4} - \frac{\sigma^2}{\sigma^6} = \frac{1}{2\sigma^4} - \frac{1}{\sigma^4} = -\frac{1}{2\sigma^4} $$

この結果に負号をかけたものがフィッシャー情報量です。
$$ I(\sigma^2) = \frac{1}{2\sigma^4} $$

最後に、Jeffreys 事前分布の定義に従って平方根を取ります。
$$ p(\sigma^2) \propto \sqrt{I(\sigma^2)} = \sqrt{\frac{1}{2\sigma^4}} \propto \frac{1}{\sigma^2} $$

以上より、分散 $\sigma^2$ に対する Jeffreys 事前分布は $p(\sigma^2) \propto 1/\sigma^2$ となることが示されました。これは、スケールパラメータに対する無情報事前分布（対数を取ると一様分布になる）と一致します。
