# 演習問題 2.23

## 問題設定
同時ガウス分布 $p(\mathbf{x}_a, \mathbf{x}_b)$ において、周辺分布 $p(\mathbf{x}_a)$ がガウス分布になることを示し、その平均と共分散行列を導出せよ。

## 解答と証明
同時分布を以下のように分割する。
$$ \mathbf{x} = \begin{pmatrix} \mathbf{x}_a \\ \mathbf{x}_b \end{pmatrix}, \quad \boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ \boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb} \end{pmatrix} $$

周辺分布 $p(\mathbf{x}_a)$ は、同時分布を $\mathbf{x}_b$ について積分消去することで得られる。
$$ p(\mathbf{x}_a) = \int p(\mathbf{x}_a, \mathbf{x}_b) d\mathbf{x}_b $$

同時分布の指数部分は二次形式の形をとるため、$\mathbf{x}_b$ についての平方完成を行う。
精度行列を $\mathbf{\Lambda} = \boldsymbol{\Sigma}^{-1}$ とし、同様にブロック分割する。
指数部分に現れる二次形式は以下のようになる。
$$ -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Lambda} (\mathbf{x} - \boldsymbol{\mu}) $$

これを $\mathbf{x}_b$ についてまとめると、
$$ -\frac{1}{2} \left( \mathbf{x}_b^T \mathbf{\Lambda}_{bb} \mathbf{x}_b - 2\mathbf{x}_b^T \mathbf{m} + \text{const} \right) $$
の形になる。積分消去により、$\mathbf{x}_a$ に依存する部分は元の精度行列 $\mathbf{\Lambda}$ のシュール補行列を用いて表される。

結果として、ガウス分布の積分の性質から、周辺分布 $p(\mathbf{x}_a)$ もガウス分布となり、そのパラメータは直接分割された行列から得られる。
$$ \mathbb{E}[\mathbf{x}_a] = \boldsymbol{\mu}_a $$
$$ \text{cov}[\mathbf{x}_a] = \boldsymbol{\Sigma}_{aa} $$
(証明終)
