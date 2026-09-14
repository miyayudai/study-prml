# 演習問題 2.5 (Exercise 2.5)

## 問題の概要 (Problem Statement)
ガンマ関数の定義 $\Gamma(x) = \int_0^\infty t^{x-1} e^{-t} dt$ を用いて、
$$ \Gamma(a)\Gamma(b) = \Gamma(a+b) \int_0^1 \mu^{a-1} (1-\mu)^{b-1} d\mu $$
を示し、ベータ分布の規格化係数が $\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}$ となることを証明せよ。

## 解答と証明 (Solution and Proof)

### 1. ガンマ関数の積の積分表現 (Integral Representation of the Product of Gamma Functions)
まず、$\Gamma(a)$ と $\Gamma(b)$ の積を考えます：
$$ \Gamma(a)\Gamma(b) = \left( \int_0^\infty x^{a-1} e^{-x} dx \right) \left( \int_0^\infty y^{b-1} e^{-y} dy \right) $$
積分の変数が独立であるため、二重積分として一つにまとめることができます：
$$ \Gamma(a)\Gamma(b) = \int_0^\infty \int_0^\infty x^{a-1} y^{b-1} e^{-(x+y)} dx dy $$

### 2. 変数変換 (Change of Variables)
ここで、新しい変数 $t$ と $\mu$ を次のように導入します：
$$ t = x + y, \quad \mu = \frac{x}{x+y} $$
これらを逆に解くと、
$$ x = t\mu, \quad y = t(1-\mu) $$
となります。積分範囲は、$x \in [0, \infty), y \in [0, \infty)$ のとき、$t \in [0, \infty)$、そして $\mu \in [0, 1]$ となります。

次に、変数変換のためのヤコビアン $J$ を計算します：
$$ J = \det \begin{pmatrix} \frac{\partial x}{\partial t} & \frac{\partial x}{\partial \mu} \\ \frac{\partial y}{\partial t} & \frac{\partial y}{\partial \mu} \end{pmatrix} = \det \begin{pmatrix} \mu & t \\ 1-\mu & -t \end{pmatrix} $$
行列式を計算すると、
$$ J = \mu(-t) - t(1-\mu) = -t\mu - t + t\mu = -t $$
ヤコビアンの絶対値は $|J| = t$ となります。したがって、微分要素は $dx dy = t \, dt d\mu$ と変換されます。

### 3. 積分の計算とベータ分布の規格化係数の導出 (Evaluation of the Integral)
変数変換を二重積分に適用します：
$$ \Gamma(a)\Gamma(b) = \int_0^1 \int_0^\infty (t\mu)^{a-1} (t(1-\mu))^{b-1} e^{-t} t \, dt d\mu $$
項を整理して $t$ と $\mu$ でまとめます：
$$ = \int_0^1 \int_0^\infty t^{a-1} \mu^{a-1} t^{b-1} (1-\mu)^{b-1} e^{-t} t \, dt d\mu $$
$$ = \int_0^1 \int_0^\infty t^{a+b-1} e^{-t} \mu^{a-1} (1-\mu)^{b-1} dt d\mu $$
この積分は $t$ に関する部分と $\mu$ に関する部分に分離できます：
$$ = \left( \int_0^\infty t^{a+b-1} e^{-t} dt \right) \left( \int_0^1 \mu^{a-1} (1-\mu)^{b-1} d\mu \right) $$
第1の括弧内の積分は、まさにガンマ関数の定義より $\Gamma(a+b)$ です。したがって、
$$ \Gamma(a)\Gamma(b) = \Gamma(a+b) \int_0^1 \mu^{a-1} (1-\mu)^{b-1} d\mu $$
が示されました。

### 4. ベータ分布の規格化 (Normalization of the Beta Distribution)
ベータ関数の定義 $B(a, b) = \int_0^1 \mu^{a-1} (1-\mu)^{b-1} d\mu$ を用いると、先ほどの式は以下のように書けます：
$$ B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)} $$
ベータ分布は $p(\mu|a, b) = C \cdot \mu^{a-1} (1-\mu)^{b-1}$ の形をしており、$\int_0^1 p(\mu|a, b) d\mu = 1$ を満たす必要があります。したがって規格化係数 $C$ は、
$$ C = \frac{1}{\int_0^1 \mu^{a-1} (1-\mu)^{b-1} d\mu} = \frac{1}{B(a, b)} = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} $$
となり、ベータ分布の規格化係数が $\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}$ であることが証明されました。
