# Exercise 2.58

## 問題
カーネル密度推定法によって得られる密度推定値の期待値が、真の確率密度とカーネル関数の畳み込み（convolution）として表されることを示せ。また、$h \to 0$ の極限でこの推定量が不偏（unbiased）になることを示せ。

## 解答と解説

データ点 $\mathbf{x}_n \sim p(\mathbf{x})$ ($n=1, \dots, N$) が未知の真の確率密度 $p(\mathbf{x})$ から独立に抽出されているとします。
カーネル密度推定量は次のように定義されます。
$$ \hat{p}(\mathbf{x}) = \frac{1}{N} \sum_{n=1}^N \frac{1}{h^D} k\left( \frac{\mathbf{x} - \mathbf{x}_n}{h} \right) $$

まず、この推定量 $\hat{p}(\mathbf{x})$ の期待値 $\mathbb{E}[\hat{p}(\mathbf{x})]$ を計算します。期待値は真の分布 $p$ の下で取られます。各データ点 $\mathbf{x}_n$ は同一の分布 $p$ に従うため、どの $n$ に対しても期待値は同じになります。
$$ \mathbb{E}[\hat{p}(\mathbf{x})] = \mathbb{E} \left[ \frac{1}{N} \sum_{n=1}^N \frac{1}{h^D} k\left( \frac{\mathbf{x} - \mathbf{x}_n}{h} \right) \right] = \frac{1}{h^D} \mathbb{E}_{\mathbf{x}' \sim p} \left[ k\left( \frac{\mathbf{x} - \mathbf{x}'}{h} \right) \right] $$

この期待値を積分で書き下すと、次のようになります。
$$ \mathbb{E}[\hat{p}(\mathbf{x})] = \int \frac{1}{h^D} k\left( \frac{\mathbf{x} - \mathbf{x}'}{h} \right) p(\mathbf{x}') d\mathbf{x}' $$
この式は、平滑化されたカーネル関数と真の密度 $p(\mathbf{x}')$ との畳み込み（convolution）積分そのものです。つまり、推定量の期待値は真の密度をカーネル関数によって平滑化（ぼかし）したものになります。

次に、この式で変数変換 $\mathbf{u} = \frac{\mathbf{x} - \mathbf{x}'}{h}$ を行います。これにより $\mathbf{x}' = \mathbf{x} - h\mathbf{u}$、微小体積要素は $d\mathbf{x}' = h^D d\mathbf{u}$ となります（積分の極限は全空間であるため変わりません）。
$$ \mathbb{E}[\hat{p}(\mathbf{x})] = \int \frac{1}{h^D} k(\mathbf{u}) p(\mathbf{x} - h\mathbf{u}) (h^D d\mathbf{u}) = \int k(\mathbf{u}) p(\mathbf{x} - h\mathbf{u}) d\mathbf{u} $$

ここで、平滑化パラメータ（バンド幅）を極限 $h \to 0$ に近づけることを考えます。真の密度 $p$ が連続であると仮定すると、$h \to 0$ のとき $p(\mathbf{x} - h\mathbf{u}) \to p(\mathbf{x})$ となります。これを積分に適用すると、
$$ \lim_{h \to 0} \mathbb{E}[\hat{p}(\mathbf{x})] = \int k(\mathbf{u}) p(\mathbf{x}) d\mathbf{u} = p(\mathbf{x}) \int k(\mathbf{u}) d\mathbf{u} $$

カーネル関数は $\int k(\mathbf{u}) d\mathbf{u} = 1$ に正規化されているため、
$$ \lim_{h \to 0} \mathbb{E}[\hat{p}(\mathbf{x})] = p(\mathbf{x}) $$
となります。

以上より、$h \to 0$ の極限においてカーネル密度推定量は真の確率密度 $p(\mathbf{x})$ に一致し、漸近的に不偏推定量（asymptotically unbiased estimator）になることが示されました。
