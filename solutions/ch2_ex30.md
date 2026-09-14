# 演習問題 2.30

## 問題
式 (2.107) から出発し、結果 (2.105) を用いて、式 (2.108) を確かめよ。

## 解答

式 (2.107) は、同時分布 $p(\mathbf{z})$ における $\mathbf{z}$ の平均 $\mathbb{E}[\mathbf{z}]$ を次のように与えています。

$$
\mathbb{E}[\mathbf{z}] = \mathbf{R}^{-1} \begin{pmatrix} \mathbf{\Lambda}\boldsymbol{\mu} - \mathbf{A}^T\mathbf{L}\mathbf{b} \\ \mathbf{L}\mathbf{b} \end{pmatrix}
$$

ここで、式 (2.105) によって $\mathbf{R}^{-1}$ （共分散行列 $\text{cov}[\mathbf{z}]$）は次のように求められています。

$$
\mathbf{R}^{-1} = \begin{pmatrix} \mathbf{\Lambda}^{-1} & \mathbf{\Lambda}^{-1}\mathbf{A}^T \\ \mathbf{A}\mathbf{\Lambda}^{-1} & \mathbf{L}^{-1} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T \end{pmatrix}
$$

これを式 (2.107) に代入して、行列とベクトルの積を計算します。$\mathbf{z} = (\mathbf{x}^T, \mathbf{y}^T)^T$ であるため、期待値は上側ブロック $\mathbb{E}[\mathbf{x}]$ と下側ブロック $\mathbb{E}[\mathbf{y}]$ に分けて計算できます。

**上側ブロック $\mathbb{E}[\mathbf{x}]$ の計算:**
行列の1行目とベクトルを掛け合わせます。
$$
\mathbb{E}[\mathbf{x}] = \mathbf{\Lambda}^{-1}(\mathbf{\Lambda}\boldsymbol{\mu} - \mathbf{A}^T\mathbf{L}\mathbf{b}) + (\mathbf{\Lambda}^{-1}\mathbf{A}^T)(\mathbf{L}\mathbf{b})
$$
$$
= \mathbf{\Lambda}^{-1}\mathbf{\Lambda}\boldsymbol{\mu} - \mathbf{\Lambda}^{-1}\mathbf{A}^T\mathbf{L}\mathbf{b} + \mathbf{\Lambda}^{-1}\mathbf{A}^T\mathbf{L}\mathbf{b}
$$
第2項と第3項が相殺し合い、$\mathbf{\Lambda}^{-1}\mathbf{\Lambda} = \mathbf{I}$ より、
$$
\mathbb{E}[\mathbf{x}] = \boldsymbol{\mu}
$$
となります。

**下側ブロック $\mathbb{E}[\mathbf{y}]$ の計算:**
行列の2行目とベクトルを掛け合わせます。
$$
\mathbb{E}[\mathbf{y}] = (\mathbf{A}\mathbf{\Lambda}^{-1})(\mathbf{\Lambda}\boldsymbol{\mu} - \mathbf{A}^T\mathbf{L}\mathbf{b}) + (\mathbf{L}^{-1} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T)(\mathbf{L}\mathbf{b})
$$
式を展開すると、
$$
= \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{\Lambda}\boldsymbol{\mu} - \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T\mathbf{L}\mathbf{b} + \mathbf{L}^{-1}\mathbf{L}\mathbf{b} + \mathbf{A}\mathbf{\Lambda}^{-1}\mathbf{A}^T\mathbf{L}\mathbf{b}
$$
ここでも $\mathbf{\Lambda}^{-1}\mathbf{\Lambda} = \mathbf{I}$ および $\mathbf{L}^{-1}\mathbf{L} = \mathbf{I}$ を適用し、項を整理します。第2項と第4項が相殺し合うため、
$$
= \mathbf{A}\boldsymbol{\mu} + \mathbf{b}
$$
となります。

以上の結果をベクトルとしてまとめると、以下のようになります。
$$
\mathbb{E}[\mathbf{z}] = \begin{pmatrix} \mathbb{E}[\mathbf{x}] \\ \mathbb{E}[\mathbf{y}] \end{pmatrix} = \begin{pmatrix} \boldsymbol{\mu} \\ \mathbf{A}\boldsymbol{\mu} + \mathbf{b} \end{pmatrix}
$$

これはまさに式 (2.108) の結果と一致しており、題意が示されました。
