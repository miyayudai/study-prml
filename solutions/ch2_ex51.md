# PRML Exercise 2.51

## 問題
オイラーの公式 $\exp(iA) = \cos A + i\sin A$ を用いて、次の三角関数の加法定理と恒等式を証明せよ。
1. $\cos^2 A + \sin^2 A = 1$
2. $\cos(A - B) = \cos A \cos B + \sin A \sin B$
3. $\sin(A - B) = \sin A \cos B - \cos A \sin B$

## 解答と解説
複素指数関数に関するオイラーの公式を出発点として、各等式を証明します。

### 1. $\cos^2 A + \sin^2 A = 1$ の証明
オイラーの公式より、$\exp(iA)$ とその複素共役 $\exp(-iA)$ は以下のように表せます。
$$ \exp(iA) = \cos A + i\sin A $$
$$ \exp(-iA) = \cos(-A) + i\sin(-A) = \cos A - i\sin A $$
ここで、$\exp(iA) \cdot \exp(-iA)$ を計算します。指数法則より左辺は
$$ \exp(iA) \cdot \exp(-iA) = \exp(iA - iA) = \exp(0) = 1 $$
となります。一方、右辺同士の積を計算すると
$$ (\cos A + i\sin A)(\cos A - i\sin A) = \cos^2 A - (i\sin A)^2 = \cos^2 A - i^2 \sin^2 A = \cos^2 A + \sin^2 A $$
となります（$i^2 = -1$ より）。
したがって、両辺を比較することで
$$ \cos^2 A + \sin^2 A = 1 $$
が導かれます。

### 2. $\cos(A - B) = \cos A \cos B + \sin A \sin B$ の証明
$\exp(i(A - B))$ を考えます。オイラーの公式を適用すると
$$ \exp(i(A - B)) = \cos(A - B) + i\sin(A - B) $$
となります。したがって、$\cos(A - B)$ はこの式の**実部** $\Re[\exp(i(A-B))]$ に等しいことが分かります。

一方で、指数関数の性質を用いると、
$$ \exp(i(A - B)) = \exp(iA)\exp(-iB) = (\cos A + i\sin A)(\cos B - i\sin B) $$
と展開できます。この右辺を展開すると、
$$ (\cos A \cos B + \sin A \sin B) + i(\sin A \cos B - \cos A \sin B) $$
となります。
実部同士を比較することで、
$$ \cos(A - B) = \cos A \cos B + \sin A \sin B $$
が証明されます。

### 3. $\sin(A - B) = \sin A \cos B - \cos A \sin B$ の証明
2. の展開式において、今度は虚部を比較します。
オイラーの公式から
$$ \exp(i(A - B)) = \cos(A - B) + i\sin(A - B) $$
なので、$\sin(A - B)$ は**虚部** $\Im[\exp(i(A-B))]$ となります。

先ほどの展開式
$$ (\cos A \cos B + \sin A \sin B) + i(\sin A \cos B - \cos A \sin B) $$
の虚部を取り出すと、
$$ \sin(A - B) = \sin A \cos B - \cos A \sin B $$
となり、加法定理が証明されます。
