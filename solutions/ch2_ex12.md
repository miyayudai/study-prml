# 演習問題 2.12 (Exercise 2.12)

## 問題の目的
演習問題2.11で導出した同時分布 $p(\mathbf{x}, \mathbf{y})$ を用いて、ベイズの定理に基づく逆方向の条件付き分布 $p(\mathbf{x}|\mathbf{y})$ および $\mathbf{y}$ の周辺分布 $p(\mathbf{y})$ を導出します。

## 周辺分布の導出

同時分布の共分散行列 $\mathbf{C}$ と平均から、$\mathbf{y}$ の周辺分布 $p(\mathbf{y})$ は、単純に $\mathbf{y}$ に対応するブロックを取り出すことで得られます（演習問題2.8の結果を利用）。
$$ p(\mathbf{y}) = \mathcal{N}(\mathbf{y} | \mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{L}^{-1} + \mathbf{A} \boldsymbol{\Lambda}^{-1} \mathbf{A}^T) $$

これは、入力 $\mathbf{x}$ の不確実性（分散 $\boldsymbol{\Lambda}^{-1}$）が、線形変換 $\mathbf{A}$ を通じて出力 $\mathbf{y}$ の分散に加算的に寄与することを示しています。

## 条件付き分布（事後分布）の導出

$p(\mathbf{x} | \mathbf{y})$ の導出には、演習問題2.9の条件付き分布の公式を適用します。
精度行列を利用した公式を用いると、計算がスムーズです。同時分布の精度行列 $\mathbf{R}$ のブロック要素は以下の通りでした。
$$ \mathbf{R}_{xx} = \boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A}, \quad \mathbf{R}_{xy} = -\mathbf{A}^T \mathbf{L} $$

条件付き分散 $\boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}}$ は、$\mathbf{R}_{xx}$ の逆行列です。
$$ \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}} = (\boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A})^{-1} $$

条件付き平均 $\boldsymbol{\mu}_{\mathbf{x}|\mathbf{y}}$ は、以下の式で与えられます。
$$ \boldsymbol{\mu}_{\mathbf{x}|\mathbf{y}} = \boldsymbol{\mu} - \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}} \mathbf{R}_{xy} (\mathbf{y} - (\mathbf{A}\boldsymbol{\mu} + \mathbf{b})) $$
$$ = \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}} \left( \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}}^{-1} \boldsymbol{\mu} + \mathbf{A}^T \mathbf{L} (\mathbf{y} - \mathbf{A}\boldsymbol{\mu} - \mathbf{b}) \right) $$
$$ = \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}} \left( (\boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A})\boldsymbol{\mu} + \mathbf{A}^T \mathbf{L} \mathbf{y} - \mathbf{A}^T \mathbf{L} \mathbf{A} \boldsymbol{\mu} - \mathbf{A}^T \mathbf{L} \mathbf{b} \right) $$
$$ = \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}} \left( \boldsymbol{\Lambda} \boldsymbol{\mu} + \mathbf{A}^T \mathbf{L} (\mathbf{y} - \mathbf{b}) \right) $$

この結果は、事前知識（平均 $\boldsymbol{\mu}$）と観測データ $\mathbf{y}$ の情報を、それぞれの精度行列を重みとして最適にブレンドした形になっています。ガウス分布のベイズ推論（線形回帰やカルマンフィルタなど）の基礎となる極めて重要な公式です。
