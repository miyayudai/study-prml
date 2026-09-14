# Exercise 2.57

## 問題
Parzen 窓（Parzen window）法を用いたカーネル密度推定において、カーネル関数 $k(\mathbf{u})$ の積分が 1 であること（$\int k(\mathbf{u}) d\mathbf{u} = 1$）を仮定したとき、得られる推定密度関数 $p(\mathbf{x})$ 全体の積分も 1 になることを示せ。

## 解答と解説

Parzen 窓密度推定（カーネル密度推定）により得られる確率密度関数 $p(\mathbf{x})$ は、与えられた $N$ 個のデータ点 $\mathbf{x}_1, \dots, \mathbf{x}_N$ と、平滑化パラメータ（バンド幅） $h$ を用いて次のように表されます。ここでデータの次元数を $D$ とします。
$$ p(\mathbf{x}) = \frac{1}{N} \sum_{n=1}^N \frac{1}{h^D} k\left( \frac{\mathbf{x} - \mathbf{x}_n}{h} \right) $$

この推定密度関数が正当な確率密度関数であるためには、全空間での積分が 1 になる必要があります。これを証明します。

空間全体にわたって $p(\mathbf{x})$ を積分します。和と積分の順序を交換すると以下のようになります。
$$ \int p(\mathbf{x}) d\mathbf{x} = \int \left( \frac{1}{N} \sum_{n=1}^N \frac{1}{h^D} k\left( \frac{\mathbf{x} - \mathbf{x}_n}{h} \right) \right) d\mathbf{x} $$
$$ \int p(\mathbf{x}) d\mathbf{x} = \frac{1}{N} \sum_{n=1}^N \frac{1}{h^D} \int k\left( \frac{\mathbf{x} - \mathbf{x}_n}{h} \right) d\mathbf{x} $$

ここで、積分内の各項について変数変換を行います。新しい変数 $\mathbf{u}$ を次のように定義します。
$$ \mathbf{u} = \frac{\mathbf{x} - \mathbf{x}_n}{h} $$
これにより、微小体積要素は $d\mathbf{x} = h^D d\mathbf{u}$ と変換されます。これを積分に代入します。
$$ \int k\left( \frac{\mathbf{x} - \mathbf{x}_n}{h} \right) d\mathbf{x} = \int k(\mathbf{u}) (h^D d\mathbf{u}) = h^D \int k(\mathbf{u}) d\mathbf{u} $$

問題文の仮定より、カーネル関数 $k(\mathbf{u})$ は全空間での積分が 1 に正規化されています（$\int k(\mathbf{u}) d\mathbf{u} = 1$）。したがって、
$$ h^D \int k(\mathbf{u}) d\mathbf{u} = h^D \times 1 = h^D $$
となります。

この結果を元の式に代入すると、
$$ \int p(\mathbf{x}) d\mathbf{x} = \frac{1}{N} \sum_{n=1}^N \frac{1}{h^D} (h^D) = \frac{1}{N} \sum_{n=1}^N 1 $$
となります。和の中身は定数 1 であり、それを $N$ 回足し合わせるため、結果は $N$ になります。
$$ \frac{1}{N} \sum_{n=1}^N 1 = \frac{1}{N} \times N = 1 $$

以上より、カーネル関数が正規化されていれば、Parzen 窓法による推定密度関数 $p(\mathbf{x})$ も自動的に正規化され、全空間での積分が 1 となることが示されました。
