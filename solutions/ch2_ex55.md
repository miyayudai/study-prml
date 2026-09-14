# PRML Exercise 2.55

## 問題
von Mises分布の集中度パラメータ $m$ に対する最尤推定量を導出する過程で得られる式（PRML式 2.185）
$$ A(m_{\text{ML}}) = \frac{1}{N} \sum_{n=1}^N \cos(\theta_n - \theta_0^{\text{ML}}) $$
を変形し、これが平均ベクトルの長さ $\bar{r}$ と等しくなること、すなわち $A(m_{\text{ML}}) = \bar{r}$ を示せ。ここで、$\bar{r}$ はPRML式(2.167), (2.168)を用いて定義されている。

## 解答と解説
まず、三角関数の加法定理 $\cos(A - B) = \cos A \cos B + \sin A \sin B$ を用いて、与えられた式の右辺を展開します。

$$ A(m_{\text{ML}}) = \frac{1}{N} \sum_{n=1}^N \left( \cos \theta_n \cos \theta_0^{\text{ML}} + \sin \theta_n \sin \theta_0^{\text{ML}} \right) $$

和の演算を分割し、$\theta_0^{\text{ML}}$ に依存する項を和の外に括り出します。
$$ A(m_{\text{ML}}) = \left( \frac{1}{N} \sum_{n=1}^N \cos \theta_n \right) \cos \theta_0^{\text{ML}} + \left( \frac{1}{N} \sum_{n=1}^N \sin \theta_n \right) \sin \theta_0^{\text{ML}} $$

ここで、PRMLで導入されている以下の定義（式 2.167 および 2.168）を用います。観測データの平均的な方向 $\bar{\theta}$ とそのベクトル長 $\bar{r}$ は次のように表されます。
$$ \bar{r} \cos \bar{\theta} = \frac{1}{N} \sum_{n=1}^N \cos \theta_n $$
$$ \bar{r} \sin \bar{\theta} = \frac{1}{N} \sum_{n=1}^N \sin \theta_n $$

これを先ほどの式に代入します。
$$ A(m_{\text{ML}}) = (\bar{r} \cos \bar{\theta}) \cos \theta_0^{\text{ML}} + (\bar{r} \sin \bar{\theta}) \sin \theta_0^{\text{ML}} $$
$\bar{r}$ で括ると、
$$ A(m_{\text{ML}}) = \bar{r} \left( \cos \bar{\theta} \cos \theta_0^{\text{ML}} + \sin \bar{\theta} \sin \theta_0^{\text{ML}} \right) $$
再び三角関数の加法定理 $\cos A \cos B + \sin A \sin B = \cos(A - B)$ を逆に用いると、
$$ A(m_{\text{ML}}) = \bar{r} \cos(\bar{\theta} - \theta_0^{\text{ML}}) $$
となります。

Exercise 2.53（またはPRML本文の議論）より、方向の最尤推定量 $\theta_0^{\text{ML}}$ はデータの平均方向 $\bar{\theta}$ に等しいことが分かっています。すなわち、
$$ \theta_0^{\text{ML}} = \bar{\theta} $$
です。

これを代入すると、$\bar{\theta} - \theta_0^{\text{ML}} = 0$ となり、$\cos(0) = 1$ であるため、
$$ A(m_{\text{ML}}) = \bar{r} \cos(0) = \bar{r} $$
が得られます。

### 結論
最尤推定量 $m_{\text{ML}}$ を決定する関数 $A(m)$ は、観測されたデータ群から作られる合成ベクトルの平均長 $\bar{r}$ に一致します。$\bar{r}$ が大きい（データが1つの方向に集中している）ほど、$m_{\text{ML}}$ も大きくなり、逆にデータが散らばって $\bar{r}$ が $0$ に近いときは $m_{\text{ML}}$ も $0$ に近づくという直感的な性質と合致しています。
