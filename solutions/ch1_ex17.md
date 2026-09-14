# 演習問題 1.17

## 問題設定
ガンマ関数 $\Gamma(x)$ について、以下の性質を示せ。
1. $\Gamma(x+1) = x\Gamma(x)$
2. $\Gamma(1) = 1$

## 解答

### 1. $\Gamma(x+1) = x\Gamma(x)$ の証明
ガンマ関数は次のように定義される。
$$ \Gamma(x) = \int_0^\infty u^{x-1} e^{-u} du $$

$\Gamma(x+1)$ を定義に従って書き直すと、
$$ \Gamma(x+1) = \int_0^\infty u^x e^{-u} du $$
となる。
これを部分積分を用いて計算する。
$f(u) = u^x$, $g'(u) = e^{-u}$ とおくと、
$f'(u) = x u^{x-1}$, $g(u) = -e^{-u}$ であるから、
$$ \Gamma(x+1) = \left[ -u^x e^{-u} \right]_0^\infty - \int_0^\infty \left( x u^{x-1} \right) (-e^{-u}) du $$

第1項は $u \to \infty$ で $e^{-u}$ が支配的となるため $0$ に収束し、$u=0$ でも $0$ となる（$x>0$のとき）。
したがって、
$$ \Gamma(x+1) = x \int_0^\infty u^{x-1} e^{-u} du = x \Gamma(x) $$
が示された。

### 2. $\Gamma(1) = 1$ の証明
定義に $x=1$ を代入する。
$$ \Gamma(1) = \int_0^\infty u^{1-1} e^{-u} du = \int_0^\infty e^{-u} du $$
これを積分すると、
$$ \Gamma(1) = \left[ -e^{-u} \right]_0^\infty = (0) - (-1) = 1 $$
よって、$\Gamma(1) = 1$ が示された。
