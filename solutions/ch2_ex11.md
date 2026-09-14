# 演習問題 2.11 (Exercise 2.11)

## 問題の目的
線形ガウスモデルにおいて、周辺分布 $p(\mathbf{x})$ と条件付き分布 $p(\mathbf{y}|\mathbf{x})$ が与えられたとき、同時分布 $p(\mathbf{x}, \mathbf{y})$ がガウス分布になることを示し、その平均と共分散を求めます。

## モデルの定義

以下のような線形ガウスモデルを考えます。
$$ p(\mathbf{x}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1}) $$
$$ p(\mathbf{y} | \mathbf{x}) = \mathcal{N}(\mathbf{y} | \mathbf{A}\mathbf{x} + \mathbf{b}, \mathbf{L}^{-1}) $$

ここで $\mathbf{x}$ と $\mathbf{y}$ は確率変数ベクトル、$\mathbf{A}$ と $\mathbf{b}$ は線形変換のパラメータ、$\boldsymbol{\Lambda}$ と $\mathbf{L}$ はそれぞれの精度行列です。

## 同時分布の導出

同時分布は積の法則から $p(\mathbf{x}, \mathbf{y}) = p(\mathbf{y} | \mathbf{x}) p(\mathbf{x})$ で与えられます。両者がガウス分布であるため、対数同時確率は二次形式の和となります。
$$ \ln p(\mathbf{x}, \mathbf{y}) = -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Lambda} (\mathbf{x} - \boldsymbol{\mu}) - \frac{1}{2} (\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b})^T \mathbf{L} (\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b}) + \text{const} $$

変数を連結したベクトル $\mathbf{z} = \begin{pmatrix} \mathbf{x} \\ \mathbf{y} \end{pmatrix}$ について、この二次形式を整理し平方完成します。
まず2次の項を展開して抽出します。
$$ -\frac{1}{2} \left[ \mathbf{x}^T (\boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A}) \mathbf{x} - 2 \mathbf{x}^T \mathbf{A}^T \mathbf{L} \mathbf{y} + \mathbf{y}^T \mathbf{L} \mathbf{y} \right] $$

これを行列表記でまとめると、同時分布の精度行列 $\mathbf{R}$ が得られます。
$$ \mathbf{R} = \begin{pmatrix} \boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A} & -\mathbf{A}^T \mathbf{L} \\ -\mathbf{L} \mathbf{A} & \mathbf{L} \end{pmatrix} $$

同時分布の共分散行列 $\mathbf{C}$ は $\mathbf{R}$ の逆行列として求まります。ブロック行列の反転補題を用いると、
$$ \mathbf{C} = \begin{pmatrix} \boldsymbol{\Lambda}^{-1} & \boldsymbol{\Lambda}^{-1} \mathbf{A}^T \\ \mathbf{A} \boldsymbol{\Lambda}^{-1} & \mathbf{L}^{-1} + \mathbf{A} \boldsymbol{\Lambda}^{-1} \mathbf{A}^T \end{pmatrix} $$

次に1次の項の係数を比較することで、同時分布の平均 $\mathbb{E}[\mathbf{z}]$ が求まります。計算を進めると、
$$ \mathbb{E}[\mathbf{z}] = \begin{pmatrix} \boldsymbol{\mu} \\ \mathbf{A}\boldsymbol{\mu} + \mathbf{b} \end{pmatrix} $$

よって、同時分布は以下のガウス分布となることが証明されました。
$$ p(\mathbf{x}, \mathbf{y}) = \mathcal{N} \left( \begin{pmatrix} \mathbf{x} \\ \mathbf{y} \end{pmatrix} \middle| \begin{pmatrix} \boldsymbol{\mu} \\ \mathbf{A}\boldsymbol{\mu} + \mathbf{b} \end{pmatrix}, \begin{pmatrix} \boldsymbol{\Lambda}^{-1} & \boldsymbol{\Lambda}^{-1} \mathbf{A}^T \\ \mathbf{A} \boldsymbol{\Lambda}^{-1} & \mathbf{L}^{-1} + \mathbf{A} \boldsymbol{\Lambda}^{-1} \mathbf{A}^T \end{pmatrix} \right) $$
