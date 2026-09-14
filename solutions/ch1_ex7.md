# 演習問題 1.7

## 問題
一次元ガウス分布 (1.46)
$$ \mathcal{N}(x | \mu, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{1/2}} \exp \left\{ -\frac{1}{2\sigma^2} (x - \mu)^2 \right\} $$
が正しく正規化されていること（全区間で積分すると 1 になること）を示せ。

## 解答・解説

ガウス分布が正規化されていることを示すには、次の積分が 1 になることを示せば十分です。
$$ \int_{-\infty}^{\infty} \mathcal{N}(x | \mu, \sigma^2) dx = 1 $$

まず、変数変換を行います。$y = x - \mu$ とおくと、$dy = dx$ であり、積分範囲は変わらず $(-\infty, \infty)$ です。
$$ \int_{-\infty}^{\infty} \mathcal{N}(x | \mu, \sigma^2) dx = \frac{1}{(2\pi\sigma^2)^{1/2}} \int_{-\infty}^{\infty} \exp \left( -\frac{1}{2\sigma^2} y^2 \right) dy $$

この積分の値（ガウス積分）を $I$ とおきます。
$$ I = \int_{-\infty}^{\infty} \exp \left( -\frac{1}{2\sigma^2} y^2 \right) dy $$
目標は、$I = (2\pi\sigma^2)^{1/2}$ であることを示すことです。

$I$ を直接計算することは難しいため、二乗 $I^2$ を考え、2次元の重積分に拡張する有名なテクニック（ポアソン積分）を用います。
$$ I^2 = \left( \int_{-\infty}^{\infty} \exp \left( -\frac{1}{2\sigma^2} y^2 \right) dy \right) \left( \int_{-\infty}^{\infty} \exp \left( -\frac{1}{2\sigma^2} z^2 \right) dz \right) $$
積分変数は任意に選べるため、2つ目の積分変数を $z$ としました。これらを1つの重積分としてまとめます。
$$ I^2 = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \exp \left( -\frac{1}{2\sigma^2} (y^2 + z^2) \right) dy dz $$

ここで、デカルト座標 $(y, z)$ から極座標 $(r, \theta)$ への変数変換を行います。
$$ y = r \cos \theta, \quad z = r \sin \theta $$
このとき、$y^2 + z^2 = r^2$ となります。
ヤコビアンの行列式は $r$ となるため、微小面積要素は $dy dz = r dr d\theta$ と変換されます。
積分範囲は、全平面を覆うために $r$ は $0$ から $\infty$、$\theta$ は $0$ から $2\pi$ となります。

極座標での重積分に書き換えます。
$$ I^2 = \int_0^{2\pi} \int_0^{\infty} \exp \left( -\frac{1}{2\sigma^2} r^2 \right) r dr d\theta $$

積分変数が分離できるため、$\theta$ についての積分を先に計算します。
$$ I^2 = \left( \int_0^{2\pi} d\theta \right) \left( \int_0^{\infty} \exp \left( -\frac{1}{2\sigma^2} r^2 \right) r dr \right) $$
$$ I^2 = 2\pi \int_0^{\infty} r \exp \left( -\frac{1}{2\sigma^2} r^2 \right) dr $$

次に $r$ についての積分を行います。$u = \frac{1}{2\sigma^2} r^2$ と置換積分します。$du = \frac{1}{\sigma^2} r dr$、すなわち $r dr = \sigma^2 du$ となります。
積分範囲は $r$ が $0 \to \infty$ のとき、$u$ も $0 \to \infty$ です。
$$ I^2 = 2\pi \int_0^{\infty} \exp(-u) \sigma^2 du $$
$$ I^2 = 2\pi\sigma^2 \left[ -\exp(-u) \right]_0^{\infty} $$
$$ I^2 = 2\pi\sigma^2 (0 - (-1)) = 2\pi\sigma^2 $$

$I$ は被積分関数が常に正であることから $I > 0$ なので、
$$ I = (2\pi\sigma^2)^{1/2} $$
が得られます。

これを最初の式に代入します。
$$ \int_{-\infty}^{\infty} \mathcal{N}(x | \mu, \sigma^2) dx = \frac{1}{(2\pi\sigma^2)^{1/2}} \times I = \frac{1}{(2\pi\sigma^2)^{1/2}} \times (2\pi\sigma^2)^{1/2} = 1 $$

以上より、ガウス分布は全区間で積分すると 1 となり、正しく正規化されていることが示されました。
