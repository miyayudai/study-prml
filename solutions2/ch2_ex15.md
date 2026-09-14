# Exercise 2.15

## 2.15 共分散行列の二次形式と固有値分解 (Quadratic Form and Eigenvalue Decomposition of Covariance Matrix)

### 問題設定 (Problem Setup)
多変量ガウス分布の指数部分には、次のような二次形式 (Quadratic form) が現れます。
$$ \Delta^2 = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) $$
ここで、$\boldsymbol{\Sigma}$ は $D \times D$ の共分散行列（対称正定値行列）です。$\Delta$ はマハラノビス距離 (Mahalanobis distance) と呼ばれます。
この二次形式が、共分散行列の固有値分解を用いることで、主軸座標系 (Principal axis coordinate system) における独立な変数群の平方和として対角化（分解）できることを示します。

### 固有値分解 (Eigenvalue Decomposition)
$\boldsymbol{\Sigma}$ は実対称行列であるため、直交行列 $\mathbf{U}$ と対角行列 $\boldsymbol{\Lambda}$ を用いて対角化可能です。
$$ \boldsymbol{\Sigma} \mathbf{u}_i = \lambda_i \mathbf{u}_i $$
$$ \boldsymbol{\Sigma} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T $$
ここで、$\mathbf{U} = (\mathbf{u}_1, \dots, \mathbf{u}_D)$ は固有ベクトル $\mathbf{u}_i$ を列ベクトルとして並べた直交行列であり、$\mathbf{U}^T \mathbf{U} = \mathbf{I}$ を満たします。$\boldsymbol{\Lambda}$ は対角成分に固有値 $\lambda_i$ が並ぶ対角行列です。

このとき、逆行列 $\boldsymbol{\Sigma}^{-1}$ も同様に分解できます。
$$ \boldsymbol{\Sigma}^{-1} = (\mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T)^{-1} = (\mathbf{U}^T)^{-1} \boldsymbol{\Lambda}^{-1} \mathbf{U}^{-1} $$
$\mathbf{U}$ が直交行列であることから $\mathbf{U}^{-1} = \mathbf{U}^T$ となるため、次のように書けます。
$$ \boldsymbol{\Sigma}^{-1} = \mathbf{U} \boldsymbol{\Lambda}^{-1} \mathbf{U}^T $$

### 二次形式の変換と対角化
上で得られた $\boldsymbol{\Sigma}^{-1}$ の分解を、二次形式の式に代入します。
$$ \Delta^2 = (\mathbf{x} - \boldsymbol{\mu})^T (\mathbf{U} \boldsymbol{\Lambda}^{-1} \mathbf{U}^T) (\mathbf{x} - \boldsymbol{\mu}) $$
ここで、変数変換 $\mathbf{y} = \mathbf{U}^T (\mathbf{x} - \boldsymbol{\mu})$ を導入します。この変換は、中心を $\boldsymbol{\mu}$ に平行移動させた後、固有ベクトル $\mathbf{U}$ によって座標系を回転（直交変換）させることを意味します。この新たな座標系 $\mathbf{y}$ を**主軸座標系**と呼びます。

この変換に伴い、転置は $\mathbf{y}^T = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{U}$ となります。これを二次形式に当てはめると、
$$ \Delta^2 = \mathbf{y}^T \boldsymbol{\Lambda}^{-1} \mathbf{y} $$
となります。

$\boldsymbol{\Lambda}^{-1}$ は対角行列であり、その成分は $1/\lambda_1, \dots, 1/\lambda_D$ です。ベクトル $\mathbf{y} = (y_1, \dots, y_D)^T$ を用いて行列表記を展開すると、次のように独立した和として表現できます。
$$ \Delta^2 = \sum_{i=1}^D y_i \left( \frac{1}{\lambda_i} \right) y_i = \sum_{i=1}^D \frac{y_i^2}{\lambda_i} $$

### 結論と幾何学的解釈 (Geometric Interpretation)
$$ \Delta^2 = \sum_{i=1}^D \frac{y_i^2}{\lambda_i} $$
この結果は、マハラノビス距離が新しい座標系 $\mathbf{y}$ において、各成分 $y_i$ が独立に分散 $\lambda_i$ を持つ標準的なユークリッド距離の拡張として表されることを示しています。

等高線（$\Delta^2 = \mathrm{const}$ となる点の集合）を考えると、各成分の平方和の形になっているため、これは $D$ 次元空間上の**楕円体 (Ellipsoid)** の方程式に他なりません。
- 楕円体の各主軸の方向は、共分散行列の固有ベクトル $\mathbf{u}_i$ によって与えられます。
- 楕円体の各主軸方向への広がり（半径）は、対応する固有値の平方根 $\sqrt{\lambda_i}$ に比例します。

このように、共分散行列の固有値分解を用いることで、変数間に相関がある多変量ガウス分布も、互いに無相関（独立）な一次元ガウス分布の積として幾何学的に分解・解釈できることが確認できます。
