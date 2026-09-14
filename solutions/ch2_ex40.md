# 演習問題 2.40

## 問題設定
事前分布における無情報事前分布（Jeffreys事前分布）の性質について証明せよ。特に、位置パラメータと尺度パラメータに対する事前分布を導出せよ。

## 解答と証明

Jeffreys事前分布は、パラメータの変換に対して不変な事前分布として以下のように定義されます。
$$ p(	heta) \propto \sqrt{	ext{det} \, \mathcal{I}(	heta)} $$
ここで、$\mathcal{I}(	heta)$ はフィッシャー情報行列であり、
$$ \mathcal{I}(	heta) = \mathbb{E}\left[ \left( rac{\partial \ln p(x|	heta)}{\partial 	heta} ight)^2 ight] = -\mathbb{E}\left[ rac{\partial^2 \ln p(x|	heta)}{\partial 	heta^2} ight] $$
で与えられます。

**1. 位置パラメータの場合:**
分布が $p(x|\mu) = f(x-\mu)$ の形で与えられる場合を考えます。
$$ \ln p(x|\mu) = \ln f(x-\mu) $$
このとき、
$$ rac{\partial \ln p(x|\mu)}{\partial \mu} = -rac{f'(x-\mu)}{f(x-\mu)} $$
フィッシャー情報量は、
$$ \mathcal{I}(\mu) = \int \left( -rac{f'(x-\mu)}{f(x-\mu)} ight)^2 f(x-\mu) dx $$
変数変換 $y = x - \mu$ を行うと、$dy = dx$ となり、
$$ \mathcal{I}(\mu) = \int \left( rac{f'(y)}{f(y)} ight)^2 f(y) dy $$
この積分は $\mu$ に依存しない定数となります。したがって、Jeffreys事前分布は
$$ p(\mu) \propto 	ext{const} $$
となり、一様事前分布が導かれます。

**2. 尺度パラメータの場合:**
分布が $p(x|\sigma) = rac{1}{\sigma}f\left(rac{x}{\sigma}ight)$ の形で与えられる場合を考えます。
$$ \ln p(x|\sigma) = -\ln \sigma + \ln f\left(rac{x}{\sigma}ight) $$
微分の連鎖律を用いて計算し、同様にフィッシャー情報量を求めると、$\mathcal{I}(\sigma) \propto rac{1}{\sigma^2}$ となることが示されます。
したがって、尺度パラメータに対するJeffreys事前分布は
$$ p(\sigma) \propto rac{1}{\sigma} $$
となります。これは $\ln \sigma$ に対して一様分布を仮定することと等価です。
