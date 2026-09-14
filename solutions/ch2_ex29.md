# 演習問題 2.29

## 問題
分割行列の逆行列の公式 (2.76) を用いて、精度行列 (2.104) の逆行列が共分散行列 (2.105) で与えられることを示せ。

## 解答

分割行列の逆行列の公式 (2.76) は以下のように与えられます。

$$
\begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{pmatrix}^{-1} = \begin{pmatrix} \mathbf{M} & -\mathbf{M}\mathbf{B}\mathbf{D}^{-1} \\ -\mathbf{D}^{-1}\mathbf{C}\mathbf{M} & \mathbf{D}^{-1} + \mathbf{D}^{-1}\mathbf{C}\mathbf{M}\mathbf{B}\mathbf{D}^{-1} \end{pmatrix}
$$

ここで、$\mathbf{M}$ はシューア補行列を用いて次のように定義されます。
$$
\mathbf{M} = (\mathbf{A} - \mathbf{B}\mathbf{D}^{-1}\mathbf{C})^{-1}
$$

式 (2.104) で与えられる同時分布の精度行列 $\mathbf{R}$ は次の形をしています。

$$
\mathbf{R} = \begin{pmatrix} \mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A} & -\mathbf{A}^T\mathbf{L} \\ -\mathbf{L}\mathbf{A} & \mathbf{L} \end{pmatrix}
$$

これを公式に当てはめると、各ブロック行列は以下のように対応します。
- $\mathbf{A}_{mat} = \mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A}$
- $\mathbf{B}_{mat} = -\mathbf{A}^T\mathbf{L}$
- $\mathbf{C}_{mat} = -\mathbf{L}\mathbf{A}$
- $\mathbf{D}_{mat} = \mathbf{L}$

まず、$\mathbf{M}$ を計算します。
$$
\mathbf{B}_{mat}\mathbf{D}_{mat}^{-1}\mathbf{C}_{mat} = (-\mathbf{A}^T\mathbf{L})\mathbf{L}^{-1}(-\mathbf{L}\mathbf{A}) = \mathbf{A}^T\mathbf{L}\mathbf{L}^{-1}\mathbf{L}\mathbf{A} = \mathbf{A}^T\mathbf{L}\mathbf{A}
$$
したがって、
$$
\mathbf{A}_{mat} - \mathbf{B}_{mat}\mathbf{D}_{mat}^{-1}\mathbf{C}_{mat} = (\mathbf{\Lambda} + \mathbf{A}^T\mathbf{L}\mathbf{A}) - \mathbf{A}^T\mathbf{L}\mathbf{A} = \mathbf{\Lambda}
$$
よって、
$$
\mathbf{M} = \mathbf{\Lambda}^{-1}
$$
となります。

次に、逆行列の各ブロック成分を計算します。

**右上ブロック:**
$$
-\mathbf{M}\mathbf{B}_{mat}\mathbf{D}_{mat}^{-1} = -\mathbf{\Lambda}^{-1}(-\mathbf{A}^T\mathbf{L})\mathbf{L}^{-1} = \mathbf{\Lambda}^{-1}\mathbf{A}^T
$$

**左下ブロック:**
$$
-\mathbf{D}_{mat}^{-1}\mathbf{C}_{mat}\mathbf{M} = -\mathbf{L}^{-1}(-\mathbf{L}\mathbf{A})\mathbf{\Lambda}^{-1} = \mathbf{A}\mathbf{\Lambda}^{-1}
$$

**右下ブロック:**
$$
\mathbf{D}_{mat}^{-1} + \mathbf{D}_{mat}^{-1}\mathbf{C}_{mat}\mathbf{M}\mathbf{B}_{mat}\mathbf{D}_{mat}^{-1} = \mathbf{L}^{-1} + (-\mathbf{D}_{mat}^{-1}\mathbf{C}_{mat}\mathbf{M})(-\mathbf{B}_{mat}\mathbf{D}_{mat}^{-1})
$$
上で求めた左下ブロック $\mathbf{A}\mathbf{\Lambda}^{-1}$ と $\mathbf{B}_{mat}\mathbf{D}_{mat}^{-1} = -\mathbf{A}^T$ を代入すると、
$$
\mathbf{L}^{-1} + (\mathbf{A}\mathbf{\Lambda}^{-1})(\mathbf{A}^T) = \mathbf{L}^{-1} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T
$$

以上の結果をまとめると、精度行列 $\mathbf{R}$ の逆行列である共分散行列 $\text{cov}[\mathbf{z}]$ は次のようになります。

$$
\text{cov}[\mathbf{z}] = \mathbf{R}^{-1} = \begin{pmatrix} \mathbf{\Lambda}^{-1} & \mathbf{\Lambda}^{-1}\mathbf{A}^T \\ \mathbf{A}\mathbf{\Lambda}^{-1} & \mathbf{L}^{-1} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T \end{pmatrix}
$$

これは式 (2.105) と完全に一致し、題意が示されました。
