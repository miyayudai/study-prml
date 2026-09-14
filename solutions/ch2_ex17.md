# 演習問題 2.17: ガウス分布における無相関性と独立性

前問(2.16)では一般の確率変数において「無相関 $\nRightarrow$ 独立」であることを示しましたが、本問題では**多変量ガウス分布**に従う変数間においては「無相関ならば独立である」という特別な性質が成り立つことを証明します。

## 1. 問題の設定

$D$ 次元の確率変数ベクトル $\mathbf{x}$ が多変量ガウス分布 $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma})$ に従うとします。ベクトル $\mathbf{x}$ を2つの部分ベクトル $\mathbf{x}_a$ (次元 $M$) と $\mathbf{x}_b$ (次元 $D-M$) に分割します。

$$
\mathbf{x} = \begin{pmatrix} \mathbf{x}_a \\ \mathbf{x}_b \end{pmatrix}, \quad
\boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b \end{pmatrix}, \quad
\boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ \boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb} \end{pmatrix}
$$

ここで、$\mathbf{x}_a$ と $\mathbf{x}_b$ が**無相関**であると仮定します。すなわち、交差共分散行列がゼロ行列となります。
$$
\boldsymbol{\Sigma}_{ab} = \mathbf{0}, \quad \boldsymbol{\Sigma}_{ba} = \mathbf{0}^T = \mathbf{0}
$$

## 2. 精度行列と行列式の構造

交差成分がゼロであるため、共分散行列 $\boldsymbol{\Sigma}$ はブロック対角行列となります。
$$
\boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{aa} & \mathbf{0} \\ \mathbf{0} & \boldsymbol{\Sigma}_{bb} \end{pmatrix}
$$

ブロック対角行列の逆行列（精度行列 $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$）は、各ブロックの逆行列を対角に並べたものになります。
$$
\boldsymbol{\Sigma}^{-1} = \begin{pmatrix} \boldsymbol{\Sigma}_{aa}^{-1} & \mathbf{0} \\ \mathbf{0} & \boldsymbol{\Sigma}_{bb}^{-1} \end{pmatrix}
$$

また、ブロック対角行列の行列式は各ブロックの行列式の積になります。
$$
|\boldsymbol{\Sigma}| = |\boldsymbol{\Sigma}_{aa}| |\boldsymbol{\Sigma}_{bb}|
$$

## 3. 同時確率密度関数の分解

元の同時ガウス分布の確率密度関数は次のように書けます。
$$
p(\mathbf{x}_a, \mathbf{x}_b) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

指数部のマハラノビス距離を展開します。$\boldsymbol{\Sigma}^{-1}$ がブロック対角であるため、交差項は現れず、2つの二次形式の和に分解されます。
$$
(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Sigma}_{aa}^{-1} (\mathbf{x}_a - \boldsymbol{\mu}_a) + (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Sigma}_{bb}^{-1} (\mathbf{x}_b - \boldsymbol{\mu}_b)
$$

正規化係数についても、$D = M + (D-M)$ と行列式の積を用いて分解します。
$$
\frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} = \left( \frac{1}{(2\pi)^{M/2} |\boldsymbol{\Sigma}_{aa}|^{1/2}} \right) \left( \frac{1}{(2\pi)^{(D-M)/2} |\boldsymbol{\Sigma}_{bb}|^{1/2}} \right)
$$

これらをまとめると、同時分布は次のように積の形に完全に分離されます。
$$
p(\mathbf{x}_a, \mathbf{x}_b) = \mathcal{N}(\mathbf{x}_a | \boldsymbol{\mu}_a, \boldsymbol{\Sigma}_{aa}) \times \mathcal{N}(\mathbf{x}_b | \boldsymbol{\mu}_b, \boldsymbol{\Sigma}_{bb})
$$

## 4. 結論

$$
p(\mathbf{x}_a, \mathbf{x}_b) = p(\mathbf{x}_a) p(\mathbf{x}_b)
$$
となるため、$\mathbf{x}_a$ と $\mathbf{x}_b$ は**統計的に独立**であることが証明されました。

ガウス分布においては、共分散行列が分布の全ての依存関係を記述するため、2次のモーメント（共分散）がゼロであれば、それ以上の高次の依存関係も存在しないことがわかります。
