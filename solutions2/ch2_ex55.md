## Exercise 2.55 (ガウス平均に対する Jeffreys 事前分布)

1次元ガウス分布（正規分布）において、分散 $\sigma^2$ が既知であり、平均 $\mu$ のみが未知パラメータである場合を考えます。このとき、平均 $\mu$ に対するジェフリーズ事前分布 (Jeffreys prior) が一様分布になることを導出します。

### ガウス分布の対数尤度とスコア関数
分散 $\sigma^2$ が既知の1次元ガウス分布の確率密度関数は次のように与えられます：
$$ p(x|\mu) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) $$

この対数尤度を $\mu$ の関数として計算します：
$$ \ln p(x|\mu) = -\frac{1}{2} \ln(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2} $$

対数尤度をパラメータ $\mu$ で偏微分し、スコア関数を求めます：
$$ \frac{\partial}{\partial \mu} \ln p(x|\mu) = - \frac{2(x - \mu)}{2\sigma^2} \cdot (-1) = \frac{x - \mu}{\sigma^2} $$

### フィッシャー情報量の計算
ジェフリーズ事前分布を求めるためには、まずフィッシャー情報量 $I(\mu)$ を計算する必要があります。フィッシャー情報量はスコア関数の二乗期待値として定義されます：
$$ I(\mu) = \mathbb{E}_{x} \left[ \left( \frac{\partial}{\partial \mu} \ln p(x|\mu) \right)^2 \right] $$

先ほど求めたスコア関数を代入します：
$$ I(\mu) = \mathbb{E}_{x} \left[ \left( \frac{x - \mu}{\sigma^2} \right)^2 \right] = \frac{1}{\sigma^4} \mathbb{E}_{x} \left[ (x - \mu)^2 \right] $$

ここで、$\mathbb{E}_{x}[(x - \mu)^2]$ は真の分布 $p(x|\mu)$ の下での $x$ の分散そのものであり、$\sigma^2$ に等しいことが分かります：
$$ \mathbb{E}_{x} \left[ (x - \mu)^2 \right] = \sigma^2 $$

これを代入すると、フィッシャー情報量は次のように計算されます：
$$ I(\mu) = \frac{1}{\sigma^4} \cdot \sigma^2 = \frac{1}{\sigma^2} $$

### ジェフリーズ事前分布の導出
ジェフリーズ事前分布は、フィッシャー情報量の平方根に比例するように定義されます：
$$ p(\mu) \propto \sqrt{I(\mu)} $$

今回導出した $I(\mu) = \frac{1}{\sigma^2}$ は $\mu$ に依存しない定数です。したがって、ジェフリーズ事前分布は：
$$ p(\mu) \propto \sqrt{\frac{1}{\sigma^2}} = \frac{1}{\sigma} \propto \text{const} $$

### 結論
以上より、ガウス分布の平均 $\mu$ に対するジェフリーズ事前分布は定数関数となり、**一様分布** (improper uniform prior) になることが示されました。
この結果は、Exercise 2.51 で議論された「位置パラメータに対する平行移動不変性に基づく一様事前分布」の結論と完全に一致しています。
