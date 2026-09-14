# 演習問題 1.23

## 問題文
期待損失の式
$$ \mathbb{E}[L] = \sum_k \sum_j \int_{\mathcal{R}_j} L_{kj} p(\mathbf{x}, C_k) \mathrm{d}\mathbf{x} $$
において、結合確率 $p(\mathbf{x}, C_k)$ ではなく、クラス条件付き確率 $p(\mathbf{x} | C_k)$ に基づいて決定則を構築する場合を考える。このとき、損失行列 $L_{kj}$ をクラスの事前確率 $p(C_k)$ を用いてどのように修正すれば、期待損失を最小化する最適決定則が等価になるかを示せ。

## 解答と解説
### 1. 期待損失の分解
期待損失は、同時確率密度 $p(\mathbf{x}, C_k)$ を用いて次のように表されます。
$$ \mathbb{E}[L] = \sum_k \sum_j \int_{\mathcal{R}_j} L_{kj} p(\mathbf{x}, C_k) \mathrm{d}\mathbf{x} $$
確率の乗法定理 $p(\mathbf{x}, C_k) = p(\mathbf{x} | C_k) p(C_k)$ を用いると、期待損失は以下のように書き換えられます。
$$ \mathbb{E}[L] = \sum_k \sum_j \int_{\mathcal{R}_j} L_{kj} p(C_k) p(\mathbf{x} | C_k) \mathrm{d}\mathbf{x} $$

### 2. 損失行列の再定義
ここで、修正された新しい損失行列 $\tilde{L}_{kj}$ を次のように定義します。
$$ \tilde{L}_{kj} = L_{kj} p(C_k) $$
これを期待損失の式に代入すると、式は次のように簡潔になります。
$$ \mathbb{E}[L] = \sum_k \sum_j \int_{\mathcal{R}_j} \tilde{L}_{kj} p(\mathbf{x} | C_k) \mathrm{d}\mathbf{x} $$
積分の順序と和の順序を入れ替えることで、次のように整理できます。
$$ \mathbb{E}[L] = \sum_j \int_{\mathcal{R}_j} \left( \sum_k \tilde{L}_{kj} p(\mathbf{x} | C_k) \right) \mathrm{d}\mathbf{x} $$

### 3. 最適な決定則
期待損失全体を最小化するためには、被積分関数である $\sum_k \tilde{L}_{kj} p(\mathbf{x} | C_k)$ を各入力 $\mathbf{x}$ について最小にするようなクラス $C_j$（すなわち領域 $\mathcal{R}_j$）を選択すればよいことになります。

したがって、クラス条件付き確率（尤度）$p(\mathbf{x} | C_k)$ に基づいて決定を行う場合、元の損失行列 $L_{kj}$ に事前確率 $p(C_k)$ を乗じた **$\tilde{L}_{kj} = L_{kj} p(C_k)$ を新たな損失行列として用いることで**、全く等価な期待損失最小化の基準を得ることができます。
