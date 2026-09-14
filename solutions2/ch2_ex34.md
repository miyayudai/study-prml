## Exercise 2.34 ⭐ (ウィシャート分布の定義と性質)

多変量ガウス分布の精度行列の共役事前分布であるウィシャート分布の期待値が $\nu \mathbf{W}$ となることを確認せよ。

**【解答】**

ウィシャート分布（Wishart distribution）は、1次元のガンマ分布を多変量の正定値対称行列へ拡張したものであり、多変量ガウス分布の精度行列（共分散行列の逆行列）に対する共役事前分布として用いられる。
$D \times D$ の正定値対称行列 $\boldsymbol{\Lambda}$ に対するウィシャート分布は次のように定義される（教科書の式 (2.155)）。
$$ \mathcal{W}(\boldsymbol{\Lambda} | \mathbf{W}, \nu) = B(\mathbf{W}, \nu) |\boldsymbol{\Lambda}|^{(\nu - D - 1)/2} \exp\left( -\frac{1}{2} \mathrm{Tr}(\mathbf{W}^{-1} \boldsymbol{\Lambda}) \right) $$
ここで、$\nu$ は自由度パラメータ、$D$ は次元数、$\mathbf{W}$ は $D \times D$ の正定値対称なスケール行列であり、$B(\mathbf{W}, \nu)$ は規格化定数である：
$$ B(\mathbf{W}, \nu) = |\mathbf{W}|^{-\nu/2} \left( 2^{\nu D/2} \pi^{D(D-1)/4} \prod_{i=1}^D \Gamma\left(\frac{\nu + 1 - i}{2}\right) \right)^{-1} $$

ウィシャート分布の期待値 $\mathbb{E}[\boldsymbol{\Lambda}]$ を求めるには、分布の定義と規格化条件を利用する。
任意の正定値対称行列 $\mathbf{W}$ と $\nu > D - 1$ について、ウィシャート分布の確率密度関数の全空間での積分は $1$ になるので、以下の関係式が成立する：
$$ \int |\boldsymbol{\Lambda}|^{(\nu - D - 1)/2} \exp\left( -\frac{1}{2} \mathrm{Tr}(\mathbf{W}^{-1} \boldsymbol{\Lambda}) \right) d\boldsymbol{\Lambda} = B(\mathbf{W}, \nu)^{-1} $$
ここで計算の便宜上 $\mathbf{A} = \mathbf{W}^{-1}$ と置くと、
$$ \int |\boldsymbol{\Lambda}|^{(\nu - D - 1)/2} \exp\left( -\frac{1}{2} \mathrm{Tr}(\mathbf{A} \boldsymbol{\Lambda}) \right) d\boldsymbol{\Lambda} = B(\mathbf{A}^{-1}, \nu)^{-1} $$
となる。
$B(\mathbf{A}^{-1}, \nu)^{-1}$ の $\mathbf{A}$ への依存性は、定義から $|\mathbf{A}|^{-\nu/2}$ の部分だけである。それ以外の定数部分を $C(\nu, D)$ と置くと、右辺は $C(\nu, D) |\mathbf{A}|^{-\nu/2}$ と書ける。
$$ \int |\boldsymbol{\Lambda}|^{(\nu - D - 1)/2} \exp\left( -\frac{1}{2} \mathrm{Tr}(\mathbf{A} \boldsymbol{\Lambda}) \right) d\boldsymbol{\Lambda} = C(\nu, D) |\mathbf{A}|^{-\nu/2} $$

この式の両辺を、行列 $\mathbf{A}$ の $(i, j)$ 成分 $A_{ij}$ について偏微分する。
左辺の被積分関数における $\mathbf{A}$ の微分は、$\mathrm{Tr}(\mathbf{A} \boldsymbol{\Lambda}) = \sum_{k,l} A_{kl} \Lambda_{lk}$ であることから、
$$ \frac{\partial}{\partial A_{ij}} \exp\left( -\frac{1}{2} \mathrm{Tr}(\mathbf{A} \boldsymbol{\Lambda}) \right) = -\frac{1}{2} \Lambda_{ji} \exp\left( -\frac{1}{2} \mathrm{Tr}(\mathbf{A} \boldsymbol{\Lambda}) \right) = -\frac{1}{2} \Lambda_{ij} \exp\left( -\frac{1}{2} \mathrm{Tr}(\mathbf{A} \boldsymbol{\Lambda}) \right) $$
（$\boldsymbol{\Lambda}$ は対称行列であるため $\Lambda_{ji} = \Lambda_{ij}$）となる。

右辺の行列式 $|\mathbf{A}|^{-\nu/2}$ の $\mathbf{A}$ に関する偏微分は、ヤコビの公式 $\frac{\partial |\mathbf{A}|}{\partial A_{ij}} = |\mathbf{A}| (\mathbf{A}^{-1})_{ji}$ より、
$$ \frac{\partial}{\partial A_{ij}} C(\nu, D) |\mathbf{A}|^{-\nu/2} = C(\nu, D) \left( -\frac{\nu}{2} \right) |\mathbf{A}|^{-\nu/2 - 1} \frac{\partial |\mathbf{A}|}{\partial A_{ij}} = C(\nu, D) \left( -\frac{\nu}{2} \right) |\mathbf{A}|^{-\nu/2} (\mathbf{A}^{-1})_{ji} $$
（$\mathbf{A}$ も対称行列であるため、$(\mathbf{A}^{-1})_{ji} = (\mathbf{A}^{-1})_{ij}$ となる。）

両辺を等置して整理すると、
$$ \int \left( -\frac{1}{2} \Lambda_{ij} \right) |\boldsymbol{\Lambda}|^{(\nu - D - 1)/2} \exp\left( -\frac{1}{2} \mathrm{Tr}(\mathbf{A} \boldsymbol{\Lambda}) \right) d\boldsymbol{\Lambda} = -\frac{\nu}{2} (\mathbf{A}^{-1})_{ij} C(\nu, D) |\mathbf{A}|^{-\nu/2} $$
両辺に $-2$ を掛け、さらに右辺にある定数 $C(\nu, D) |\mathbf{A}|^{-\nu/2} = B(\mathbf{W}, \nu)^{-1}$ で両辺を割ると、
$$ \int \Lambda_{ij} \cdot B(\mathbf{W}, \nu) |\boldsymbol{\Lambda}|^{(\nu - D - 1)/2} \exp\left( -\frac{1}{2} \mathrm{Tr}(\mathbf{A} \boldsymbol{\Lambda}) \right) d\boldsymbol{\Lambda} = \nu (\mathbf{A}^{-1})_{ij} $$
左辺はまさにウィシャート分布に対する $\Lambda_{ij}$ の期待値 $\mathbb{E}[\Lambda_{ij}]$ そのものである。
$$ \mathbb{E}[\Lambda_{ij}] = \nu (\mathbf{A}^{-1})_{ij} $$
最後に $\mathbf{A} = \mathbf{W}^{-1}$ であったことから $\mathbf{A}^{-1} = \mathbf{W}$ を代入すると、
$$ \mathbb{E}[\Lambda_{ij}] = \nu W_{ij} $$
したがって、行列全体として
$$ \mathbb{E}[\boldsymbol{\Lambda}] = \nu \mathbf{W} $$
となり、ウィシャート分布の期待値が $\nu \mathbf{W}$ であることが示された。
