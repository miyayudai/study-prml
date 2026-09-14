## Exercise 2.38 (Student's t分布の平均と分散)

スチューデントのt分布の平均と分散を、精度 $\tau$ に関する無限混合ガウス分布としての性質を利用して導出します。
t分布は以下のようにガウス分布をガンマ分布で重み付けして積分したものです。
$$
\mathrm{St}(x | \mu, \lambda, \nu) = \int_0^\infty \mathcal{N}(x | \mu, (\tau\lambda)^{-1}) \mathrm{Gam}\left(\tau \middle| \frac{\nu}{2}, \frac{\nu}{2}\right) \, d\tau
$$

### 平均の導出
期待値の線形性と全期待値の法則（Tower property）を用います。
$$
\mathbb{E}[x] = \mathbb{E}_\tau [ \mathbb{E}_x [x | \tau] ]
$$
条件付き分布 $p(x | \tau)$ は平均 $\mu$ のガウス分布であるため、$\mathbb{E}_x [x | \tau] = \mu$ です。
$$
\mathbb{E}[x] = \mathbb{E}_\tau [\mu] = \mu
$$
したがって、t分布の平均は $\mu$ となります。ただし、期待値が存在するためには $\nu > 1$ である必要があります（$\nu \le 1$ の場合、裾が重すぎて平均は定義されません。例えば $\nu=1$ はコーシー分布に対応します）。

### 分散の導出
全分散の法則（Law of total variance）を用います。
$$
\mathrm{var}[x] = \mathbb{E}_\tau [ \mathrm{var}_x[x | \tau] ] + \mathrm{var}_\tau [ \mathbb{E}_x[x | \tau] ]
$$
まず、第2項について考えます。$\mathbb{E}_x[x | \tau] = \mu$ は $\tau$ に依存しない定数であるため、その分散は $0$ です。
$$
\mathrm{var}_\tau [ \mathbb{E}_x[x | \tau] ] = \mathrm{var}_\tau [ \mu ] = 0
$$
次に第1項について考えます。条件付きガウス分布 $p(x | \tau)$ の分散は $(\tau\lambda)^{-1}$ です。
$$
\mathrm{var}[x] = \mathbb{E}_\tau [ (\tau\lambda)^{-1} ] = \frac{1}{\lambda} \mathbb{E}_\tau [ \tau^{-1} ]
$$
ここで、$\tau \sim \mathrm{Gam}(\tau | a, b)$（$a = \nu/2, b = \nu/2$）について $\mathbb{E}[\tau^{-1}]$ を計算します。
ガンマ分布の定義より、
$$
\mathbb{E}[\tau^{-1}] = \int_0^\infty \tau^{-1} \frac{b^a}{\Gamma(a)} \tau^{a-1} e^{-b\tau} \, d\tau = \frac{b^a}{\Gamma(a)} \int_0^\infty \tau^{(a-1)-1} e^{-b\tau} \, d\tau
$$
積分部分はパラメータ $a-1, b$ のガンマ分布の未正規化の形に等しいため、その値は $\frac{\Gamma(a-1)}{b^{a-1}}$ となります。
$$
\mathbb{E}[\tau^{-1}] = \frac{b^a}{\Gamma(a)} \frac{\Gamma(a-1)}{b^{a-1}} = \frac{b}{a-1}
$$
$a = \nu/2, b = \nu/2$ を代入すると、
$$
\mathbb{E}[\tau^{-1}] = \frac{\nu/2}{\nu/2 - 1} = \frac{\nu}{\nu - 2}
$$
したがって、t分布の分散は以下のように求まります。
$$
\mathrm{var}[x] = \frac{1}{\lambda} \frac{\nu}{\nu - 2} = \frac{\nu}{\nu - 2}\lambda^{-1}
$$
ただし、$\tau^{-1}$ の期待値が存在するためには $a > 1$ すなわち $\nu > 2$ である必要があります。$\nu \le 2$ の場合は分散は無限大（または未定義）となります。
