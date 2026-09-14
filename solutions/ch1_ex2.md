# 演習問題 1.2

## 問題
正則化項付きの二乗和誤差関数 (1.4)
$$ \tilde{E}(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N \{y(x_n, \mathbf{w}) - t_n\}^2 + \frac{\lambda}{2} \|\mathbf{w}\|^2 $$
を最小化する多項式 (1.1) の係数 $w_i$ を決定する連立一次方程式を書き下せ。

## 解答・解説

正則化項付き誤差関数 $\tilde{E}(\mathbf{w})$ は、通常の二乗和誤差関数 $E(\mathbf{w})$ に L2ノルム正則化項 $\frac{\lambda}{2} \|\mathbf{w}\|^2$ を加えたものです。

$\|\mathbf{w}\|^2$ は重みベクトルのL2ノルムの二乗であり、成分で表すと
$$ \|\mathbf{w}\|^2 = \mathbf{w}^T \mathbf{w} = \sum_{j=0}^M w_j^2 $$
となります。

したがって、誤差関数は成分を用いて次のように表されます。
$$ \tilde{E}(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N \left( \sum_{j=0}^M w_j x_n^j - t_n \right)^2 + \frac{\lambda}{2} \sum_{j=0}^M w_j^2 $$

この誤差関数を最小化するために、$\tilde{E}(\mathbf{w})$ を各パラメータ $w_i$ で偏微分して 0 とおきます。
第1項の微分は演習問題1.1と同じであり、第2項の微分が加わります。

第2項の偏微分は以下の通りです。
$$ \frac{\partial}{\partial w_i} \left( \frac{\lambda}{2} \sum_{j=0}^M w_j^2 \right) = \frac{\lambda}{2} \cdot 2 w_i = \lambda w_i $$

したがって、全体の偏微分は次のようになります。
$$ \frac{\partial \tilde{E}}{\partial w_i} = \sum_{n=1}^N \left( \sum_{j=0}^M w_j x_n^j - t_n \right) x_n^i + \lambda w_i $$

最適解ではこれが 0 になるため、
$$ \sum_{n=1}^N \left( \sum_{j=0}^M w_j x_n^{i+j} - t_n x_n^i \right) + \lambda w_i = 0 $$
$$ \sum_{j=0}^M \left( \sum_{n=1}^N x_n^{i+j} \right) w_j + \lambda w_i = \sum_{n=1}^N t_n x_n^i $$

ここで、クロネッカーのデルタ $\delta_{ij}$ を用いて $\lambda w_i = \sum_{j=0}^M \lambda \delta_{ij} w_j$ と表現します。これにより、$w_j$ の和としてまとめることができます。
$$ \sum_{j=0}^M \left( \sum_{n=1}^N x_n^{i+j} + \lambda \delta_{ij} \right) w_j = \sum_{n=1}^N t_n x_n^i $$

演習問題1.1と同様に $A_{ij} = \sum_{n=1}^N x_n^{i+j}$ および $T_i = \sum_{n=1}^N t_n x_n^i$ と定義すれば、求めるべき連立一次方程式は次のように書き下せます。
$$ \sum_{j=0}^M (A_{ij} + \lambda \delta_{ij}) w_j = T_i $$

行列表記を用いると、$\mathbf{A}$ を要素が $A_{ij}$ の行列、$\mathbf{T}$ を要素が $T_i$ のベクトルとして
$$ (\mathbf{A} + \lambda \mathbf{I}) \mathbf{w} = \mathbf{T} $$
となります。これが正則化項を考慮したリッジ回帰の正規方程式です。
