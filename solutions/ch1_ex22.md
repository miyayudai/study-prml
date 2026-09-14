# 演習問題 1.22

## 問題文
0-1損失行列（正分類の場合の損失が0、誤分類の場合の損失が1）を用いた場合、期待損失を最小化する決定則が、各 $\mathbf{x}$ に対して事後確率 $p(C_k|\mathbf{x})$ を最大化するクラス $C_k$ を選ぶことと同値になることを示せ。

## 解答と解説
### 1. 期待損失の定式化
一般的な損失行列 $L_{kj}$ における期待損失（リスク）は次のように定義されます。
$$ \mathbb{E}[L] = \sum_{k} \sum_{j} \int_{\mathcal{R}_j} L_{kj} p(\mathbf{x}, C_k) \mathrm{d}\mathbf{x} $$
ここで、$\mathcal{R}_j$ は入力 $\mathbf{x}$ をクラス $C_j$ に割り当てる決定領域を表します。

ベイズの定理より $p(\mathbf{x}, C_k) = p(C_k | \mathbf{x}) p(\mathbf{x})$ であるため、上式は以下のように書き換えられます。
$$ \mathbb{E}[L] = \int \left( \sum_{j} \sum_{k} L_{kj} p(C_k | \mathbf{x}) I(\mathbf{x} \in \mathcal{R}_j) \right) p(\mathbf{x}) \mathrm{d}\mathbf{x} $$
期待損失全体を最小化するためには、被積分関数がすべての $\mathbf{x}$ について最小になればよいことになります。したがって、各 $\mathbf{x}$ について以下の値を最小にするようなクラス $C_j$ を選択すべきです。
$$ \sum_{k} L_{kj} p(C_k | \mathbf{x}) $$

### 2. 0-1 損失行列の適用
問題設定より、0-1損失行列は次のように定義されます。
$$ L_{kj} = 1 - I_{kj} = \begin{cases} 0 & (k = j) \\ 1 & (k \neq j) \end{cases} $$
これを先ほどの最小化すべき式に代入します。
$$ \sum_{k} L_{kj} p(C_k | \mathbf{x}) = \sum_{k} (1 - I_{kj}) p(C_k | \mathbf{x}) $$
$$ = \sum_{k} p(C_k | \mathbf{x}) - \sum_{k} I_{kj} p(C_k | \mathbf{x}) $$
確率の和の性質から $\sum_{k} p(C_k | \mathbf{x}) = 1$ であり、クロネッカーのデルタの性質から $\sum_{k} I_{kj} p(C_k | \mathbf{x}) = p(C_j | \mathbf{x})$ となります。
したがって、最小化すべき式は以下のように簡略化されます。
$$ \sum_{k} L_{kj} p(C_k | \mathbf{x}) = 1 - p(C_j | \mathbf{x}) $$

### 3. 結論
上記の結果より、期待損失を最小化するためには各 $\mathbf{x}$ において $1 - p(C_j | \mathbf{x})$ を最小化するクラス $C_j$ を選べばよいことがわかります。
これはすなわち、**事後確率 $p(C_j | \mathbf{x})$ を最大化するクラス $C_j$ を選ぶこと**と完全に同値です。
したがって、0-1損失関数の下での最適な決定則は、事後確率を最大化するクラスに割り当てることであることが示されました。
