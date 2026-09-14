# 演習問題 1.1

## 問題
多項式 (1.1) $y(x, \mathbf{w}) = \sum_{j=0}^M w_j x^j$ を用いた二乗和誤差関数 (1.2) 
$$ E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N \{y(x_n, \mathbf{w}) - t_n\}^2 $$
を最小にする係数 $\mathbf{w} = \{w_i\}$ が、次の連立一次方程式の解として与えられることを示せ。
$$ \sum_{j=0}^M A_{ij} w_j = T_i $$
ここで、$A_{ij} = \sum_{n=1}^N x_n^{i+j}, \quad T_i = \sum_{n=1}^N t_n x_n^i$ である。

## 解答・解説

二乗和誤差関数 $E(\mathbf{w})$ を最小化するパラメータ $\mathbf{w}$ を求めるため、誤差関数を各パラメータ $w_i \ (i=0, \dots, M)$ で偏微分し、それを 0 とおきます。

まず、誤差関数に多項式のモデルを代入します。
$$ E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N \left( \sum_{j=0}^M w_j x_n^j - t_n \right)^2 $$

この式を特定のパラメータ $w_i$ について偏微分します。連鎖律（合成関数の微分法則）を用います。
$$ \frac{\partial E}{\partial w_i} = \frac{1}{2} \sum_{n=1}^N 2 \left( \sum_{j=0}^M w_j x_n^j - t_n \right) \frac{\partial}{\partial w_i} \left( \sum_{j=0}^M w_j x_n^j - t_n \right) $$

ここで、$\sum_{j=0}^M w_j x_n^j$ を $w_i$ で偏微分すると、$j=i$ の項のみが残り、$x_n^i$ となります。
$$ \frac{\partial}{\partial w_i} \left( \sum_{j=0}^M w_j x_n^j - t_n \right) = x_n^i $$

したがって、偏微分は次のように整理されます。
$$ \frac{\partial E}{\partial w_i} = \sum_{n=1}^N \left( \sum_{j=0}^M w_j x_n^j - t_n \right) x_n^i $$

誤差関数を最小化するための必要条件は、すべての $i$ について $\frac{\partial E}{\partial w_i} = 0$ となることですから、
$$ \sum_{n=1}^N \left( \sum_{j=0}^M w_j x_n^j - t_n \right) x_n^i = 0 $$
$$ \sum_{n=1}^N \left( \sum_{j=0}^M w_j x_n^{i+j} - t_n x_n^i \right) = 0 $$

項を分離して移項します。
$$ \sum_{n=1}^N \sum_{j=0}^M w_j x_n^{i+j} = \sum_{n=1}^N t_n x_n^i $$

左辺の和の順序を入れ替えます（$n$ についての和と $j$ についての和を交換）。
$$ \sum_{j=0}^M \left( \sum_{n=1}^N x_n^{i+j} \right) w_j = \sum_{n=1}^N t_n x_n^i $$

ここで、問題で定義された
$$ A_{ij} = \sum_{n=1}^N x_n^{i+j}, \quad T_i = \sum_{n=1}^N t_n x_n^i $$
を用いると、求めるべき連立一次方程式が得られます。
$$ \sum_{j=0}^M A_{ij} w_j = T_i $$
この式が $i = 0, 1, \dots, M$ について成り立つため、$M+1$ 個の未知数 $\{w_0, \dots, w_M\}$ に対する $M+1$ 本の正規方程式となります。
