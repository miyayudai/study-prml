# 演習問題 1.18

## 問題設定
$D$次元空間において、半径 $r$ の球の体積 $V_D(r)$ と表面積 $S_D(r)$ の関係を明らかにし、単位球の体積 $V_D(1)$ と表面積 $S_D(1)$ を求めよ。

## 解答
$D$次元球の体積 $V_D(r)$ は、半径 $r$ の $D$ 乗に比例することが知られている。
単位球の体積を $V_D$ とおくと、
$$ V_D(r) = V_D r^D $$
と書ける。

球の表面積 $S_D(r)$ は、体積 $V_D(r)$ を半径 $r$ で微分したものに等しい。
$$ S_D(r) = \frac{d}{dr} V_D(r) = D V_D r^{D-1} $$
単位球（$r=1$）の表面積を $S_D$ とすると、
$$ S_D = D V_D $$
となる。

次に、$V_D$ を具体的に求めるために、以下の $D$重積分を考える。
$$ I = \int_{-\infty}^\infty \cdots \int_{-\infty}^\infty \exp\left(-\sum_{i=1}^D x_i^2\right) dx_1 \cdots dx_D $$
この積分は各次元について独立に計算でき、ガウス積分の公式により、
$$ I = \left( \int_{-\infty}^\infty \exp(-x^2) dx \right)^D = (\sqrt{\pi})^D = \pi^{D/2} $$

一方で、この積分を極座標系で評価する。
微小体積要素は $S_D(r) dr$ であるから、
$$ I = \int_0^\infty \exp(-r^2) S_D(r) dr = S_D \int_0^\infty r^{D-1} \exp(-r^2) dr $$
ここで $u = r^2$ と変数変換を行うと、$du = 2r dr$ より $dr = \frac{du}{2r}$ となり、
$$ I = S_D \int_0^\infty u^{(D-1)/2} \exp(-u) \frac{du}{2 u^{1/2}} = \frac{S_D}{2} \int_0^\infty u^{D/2 - 1} \exp(-u) du $$
これはガンマ関数の定義そのものであり、
$$ I = \frac{S_D}{2} \Gamma\left(\frac{D}{2}\right) $$
となる。

2つの $I$ の表現を等置して $S_D$ について解くと、
$$ S_D = \frac{2 \pi^{D/2}}{\Gamma(D/2)} $$
さらに、$V_D = S_D / D$ の関係と $\Gamma(D/2 + 1) = (D/2)\Gamma(D/2)$ を用いると、
$$ V_D = \frac{\pi^{D/2}}{(D/2) \Gamma(D/2)} = \frac{\pi^{D/2}}{\Gamma(D/2 + 1)} $$
以上により、$D$次元単位球の表面積と体積が求められた。
