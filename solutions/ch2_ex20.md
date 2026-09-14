# 演習問題 2.20: スチューデントのt分布とガウス分布の無限混合

スチューデントのt分布が、分散にガンマ分布の事前分布を置いたガウス分布の周辺化（無限混合）として導出できることを示します。これは機械学習においてロバストなモデルを構築する際の重要な性質です。

## 1. 問題の設定

条件付きガウス分布 $p(x | \tau)$ を考えます。ここで精度（分散の逆数） $\tau$ がガンマ分布 $Gam(\tau | a, b)$ に従うとします。
$$
p(x | \tau) = \mathcal{N}(x | \mu, \tau^{-1}) = \sqrt{\frac{\tau}{2\pi}} \exp\left( -\frac{\tau}{2}(x - \mu)^2 \right)
$$
$$
Gam(\tau | a, b) = \frac{b^a}{\Gamma(a)} \tau^{a-1} \exp(-b\tau)
$$

精度 $\tau$ を周辺化して積分消去することで、$x$ の周辺分布 $p(x)$ を求めます。
$$
p(x) = \int_0^\infty p(x | \tau) Gam(\tau | a, b) d\tau
$$

## 2. 積分計算

積分の中に確率密度の式を代入します。
$$
p(x) = \int_0^\infty \left( \frac{\tau}{2\pi} \right)^{1/2} \exp\left( -\frac{\tau}{2}(x - \mu)^2 \right) \frac{b^a}{\Gamma(a)} \tau^{a-1} \exp(-b\tau) d\tau
$$

$\tau$ に関する項をまとめます。
$$
p(x) = \frac{b^a}{\Gamma(a) \sqrt{2\pi}} \int_0^\infty \tau^{a - \frac{1}{2}} \exp\left( -\tau \left( b + \frac{(x - \mu)^2}{2} \right) \right) d\tau
$$

この積分は、未正規化のガンマ分布の形をしています。ガンマ関数の定義 $\int_0^\infty z^{k-1} e^{-cz} dz = \frac{\Gamma(k)}{c^k}$ を用いて積分を評価します。
ここで、$k = a + \frac{1}{2}$、$c = b + \frac{(x - \mu)^2}{2}$ です。

$$
\int_0^\infty \tau^{a - \frac{1}{2}} e^{-c\tau} d\tau = \frac{\Gamma(a + 1/2)}{\left( b + \frac{(x - \mu)^2}{2} \right)^{a + 1/2}}
$$

これを元の式に代入します。
$$
p(x) = \frac{b^a \Gamma(a + 1/2)}{\Gamma(a) \sqrt{2\pi}} \left( b + \frac{(x - \mu)^2}{2} \right)^{-(a + 1/2)}
$$

## 3. パラメータの再定義とt分布への変形

得られた分布をスチューデントのt分布の標準形に合わせるため、パラメータを再定義します。
$\nu = 2a$、$\lambda = a/b$ と置きます。これにより $a = \nu/2$、$b = \nu / (2\lambda)$ となります。

先ほどの式の項を整理します。
括弧の中身を変形します：
$$
b + \frac{(x - \mu)^2}{2} = \frac{\nu}{2\lambda} + \frac{(x - \mu)^2}{2} = \frac{\nu}{2\lambda} \left( 1 + \frac{\lambda(x - \mu)^2}{\nu} \right)
$$

これを代入すると、
$$
p(x) = \frac{(\nu / (2\lambda))^{\nu/2} \Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2}) \sqrt{2\pi}} \left( \frac{\nu}{2\lambda} \right)^{-(\nu+1)/2} \left( 1 + \frac{\lambda(x - \mu)^2}{\nu} \right)^{-(\nu+1)/2}
$$

係数の部分の $\frac{\nu}{2\lambda}$ を相殺させます。
$$
\left( \frac{\nu}{2\lambda} \right)^{\nu/2} \left( \frac{\nu}{2\lambda} \right)^{-(\nu+1)/2} = \left( \frac{\nu}{2\lambda} \right)^{-1/2} = \sqrt{\frac{2\lambda}{\nu}}
$$

これを係数に掛けると、
$$
p(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2}) \sqrt{2\pi}} \sqrt{\frac{2\lambda}{\nu}} \left( 1 + \frac{\lambda(x - \mu)^2}{\nu} \right)^{-(\nu+1)/2} = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})} \left( \frac{\lambda}{\pi\nu} \right)^{1/2} \left( 1 + \frac{\lambda(x - \mu)^2}{\nu} \right)^{-(\nu+1)/2}
$$

## 4. 結論

導出された $p(x)$ は、自由度 $\nu$、平均 $\mu$、精度パラメータ $\lambda$ を持つ**スチューデントのt分布** $St(x | \mu, \lambda, \nu)$ の定義式と完全に一致します。
これにより、t分布は「精度がガンマ分布に従うようなガウス分布の無限混合」として解釈できることが示されました。この性質が、t分布が外れ値に対してロバストである（裾が重い）幾何学的な理由です。
