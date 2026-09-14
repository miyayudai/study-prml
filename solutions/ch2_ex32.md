# 演習問題 2.32

## 問題
式 (2.99) と (2.100) で与えられる周辺分布と条件付き分布によって定義される同時分布 $p(\mathbf{x}, \mathbf{y})$ を考える。同時分布の指数部にある二次形式を調べ、2.3節で議論された「平方完成」の手法を用いることで、変数 $\mathbf{x}$ を積分消去した周辺分布 $p(\mathbf{y})$ の平均と共分散の表式を求めよ。これを行うために、Woodburyの行列反転公式 (2.289) を用いよ。また、これらの結果が2章の結果を用いて得られた (2.109) および (2.110) と一致することを確認せよ。

## 解答

同時分布 $p(\mathbf{x}, \mathbf{y}) = p(\mathbf{x})p(\mathbf{y}|\mathbf{x})$ の対数をとると、その指数部の二次形式は以下のように書けます。

$$
-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\mathbf{\Lambda}(\mathbf{x} - \boldsymbol{\mu}) - \frac{1}{2}(\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b})^T\mathbf{L}(\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b})
$$

$\mathbf{x}$ について積分消去するために、まず $\mathbf{x}$ に依存する項をまとめ、$\mathbf{x}$ についての平方完成を行います。展開して $\mathbf{x}$ についてまとめると以下のようになります。

$$
-\frac{1}{2}\mathbf{x}^T(\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})\mathbf{x} + \mathbf{x}^T(\mathbf{\Lambda}\boldsymbol{\mu} + \mathbf{A}^T\mathbf{L}(\mathbf{y} - \mathbf{b})) + \text{const}
$$

ここで、$\mathbf{x}$ の二次形式の精度行列（共分散の逆行列）と平均のシフトを表すパラメータを次のように定義します。
$$
\mathbf{\Sigma}_x = (\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})^{-1}
$$
$$
\mathbf{m}_x = \mathbf{\Sigma}_x(\mathbf{\Lambda}\boldsymbol{\mu} + \mathbf{A}^T\mathbf{L}(\mathbf{y} - \mathbf{b}))
$$

これらを用いると、$\mathbf{x}$ についての平方完成は次のように書けます。
$$
-\frac{1}{2}(\mathbf{x} - \mathbf{m}_x)^T\mathbf{\Sigma}_x^{-1}(\mathbf{x} - \mathbf{m}_x) + \frac{1}{2}\mathbf{m}_x^T\mathbf{\Sigma}_x^{-1}\mathbf{m}_x
$$

第一項は $\mathbf{x}$ に関するガウス分布の指数部であり、$\mathbf{x}$ について積分すると正規化定数の逆数となって消えます。したがって、$\mathbf{y}$ に依存する残りの項は、平方完成で生じた第二項と、最初からあった $\mathbf{x}$ に依存しない項の和になります。$\mathbf{y}$ の分布を決定する指数部は以下のようになります。

$$
\frac{1}{2}\mathbf{m}_x^T\mathbf{\Sigma}_x^{-1}\mathbf{m}_x - \frac{1}{2}\boldsymbol{\mu}^T\mathbf{\Lambda}\boldsymbol{\mu} - \frac{1}{2}(\mathbf{y} - \mathbf{b})^T\mathbf{L}(\mathbf{y} - \mathbf{b})
$$

$\mathbf{m}_x$ を代入して、$\mathbf{y}$ に関する二次形式を抽出します。
$$
\frac{1}{2} (\mathbf{\Lambda}\boldsymbol{\mu} + \mathbf{A}^T\mathbf{L}(\mathbf{y} - \mathbf{b}))^T (\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})^{-1} (\mathbf{\Lambda}\boldsymbol{\mu} + \mathbf{A}^T\mathbf{L}(\mathbf{y} - \mathbf{b})) - \frac{1}{2}(\mathbf{y} - \mathbf{b})^T\mathbf{L}(\mathbf{y} - \mathbf{b})
$$

この式から $\mathbf{y}$ に関する二次項（$\mathbf{y}$ の共分散の逆行列に対応する部分）を取り出すと次のようになります。
$$
-\frac{1}{2}(\mathbf{y} - \mathbf{b})^T \left[ \mathbf{L} - \mathbf{L}\mathbf{A}(\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})^{-1}\mathbf{A}^T\mathbf{L} \right] (\mathbf{y} - \mathbf{b})
$$

ここで、Woodburyの公式 (2.289) を用います。
$$
(\mathbf{Z} + \mathbf{U}\mathbf{W}\mathbf{V}^T)^{-1} = \mathbf{Z}^{-1} - \mathbf{Z}^{-1}\mathbf{U}(\mathbf{W}^{-1} + \mathbf{V}^T\mathbf{Z}^{-1}\mathbf{U})^{-1}\mathbf{V}^T\mathbf{Z}^{-1}
$$
これにおいて、$\mathbf{Z} = \mathbf{L}^{-1}, \mathbf{U} = \mathbf{A}, \mathbf{W} = \mathbf{\Lambda}^{-1}, \mathbf{V}^T = \mathbf{A}^T$ とおくと、
$$
(\mathbf{L}^{-1} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T)^{-1} = \mathbf{L} - \mathbf{L}\mathbf{A}(\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})^{-1}\mathbf{A}^T\mathbf{L}
$$
となります。したがって、周辺分布 $p(\mathbf{y})$ の共分散行列 $\mathbf{\Sigma}_y$ は次のようになります。
$$
\mathbf{\Sigma}_y = \mathbf{L}^{-1} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T
$$

次に、$\mathbf{y}$ に関する一次項（交差項）を取り出します。
$$
(\mathbf{y} - \mathbf{b})^T \mathbf{L}\mathbf{A}(\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})^{-1}\mathbf{\Lambda}\boldsymbol{\mu}
$$
ここで、行列の変形を行います。Woodbury公式の逆の操作から、以下の関係が成り立ちます。
$$
\mathbf{L}\mathbf{A}(\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A})^{-1}\mathbf{\Lambda} = (\mathbf{L}^{-1} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T)^{-1} \mathbf{A} \mathbf{\Lambda}^{-1} \mathbf{\Lambda} = \mathbf{\Sigma}_y^{-1}\mathbf{A}
$$
よって、交差項は次のように書けます。
$$
(\mathbf{y} - \mathbf{b})^T \mathbf{\Sigma}_y^{-1}\mathbf{A}\boldsymbol{\mu}
$$

以上の二次項と一次項を合わせると、$\mathbf{y}$ に依存する指数部は次のようにまとめられます。
$$
-\frac{1}{2}(\mathbf{y} - \mathbf{b})^T\mathbf{\Sigma}_y^{-1}(\mathbf{y} - \mathbf{b}) + (\mathbf{y} - \mathbf{b})^T\mathbf{\Sigma}_y^{-1}\mathbf{A}\boldsymbol{\mu} + \text{const}
$$
これをさらに平方完成すると、
$$
-\frac{1}{2}(\mathbf{y} - \mathbf{b} - \mathbf{A}\boldsymbol{\mu})^T\mathbf{\Sigma}_y^{-1}(\mathbf{y} - \mathbf{b} - \mathbf{A}\boldsymbol{\mu}) + \text{const}
$$
$$
= -\frac{1}{2}(\mathbf{y} - (\mathbf{A}\boldsymbol{\mu} + \mathbf{b}))^T\mathbf{\Sigma}_y^{-1}(\mathbf{y} - (\mathbf{A}\boldsymbol{\mu} + \mathbf{b})) + \text{const}
$$
となります。この形から、周辺分布 $p(\mathbf{y})$ は平均が $\mathbf{A}\boldsymbol{\mu} + \mathbf{b}$、共分散が $\mathbf{\Sigma}_y$ のガウス分布であることが分かります。

$$
\mathbb{E}[\mathbf{y}] = \mathbf{A}\boldsymbol{\mu} + \mathbf{b}
$$
$$
\text{cov}[\mathbf{y}] = \mathbf{L}^{-1} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T
$$

これは式 (2.109) および (2.110) と完全に一致しており、結果が確かめられました。
