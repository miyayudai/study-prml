# 演習問題 2.10 (Exercise 2.10)

## 問題の目的
分割された行列の逆行列（ブロック行列の反転補題）を利用して、精度行列 $\boldsymbol{\Lambda}$ の要素を共分散行列 $\boldsymbol{\Sigma}$ の要素を用いて表現し、周辺分布と条件付き分布のパラメータの代替表現を導きます。

## 導出のステップ

ブロック行列 $\boldsymbol{\Sigma}$ とその逆行列 $\boldsymbol{\Lambda}$ の関係 $\boldsymbol{\Sigma} \boldsymbol{\Lambda} = \mathbf{I}$ より、以下の連立方程式が成り立ちます。
1. $\boldsymbol{\Sigma}_{aa}\boldsymbol{\Lambda}_{aa} + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Lambda}_{ba} = \mathbf{I}$
2. $\boldsymbol{\Sigma}_{aa}\boldsymbol{\Lambda}_{ab} + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Lambda}_{bb} = \mathbf{0}$
3. $\boldsymbol{\Sigma}_{ba}\boldsymbol{\Lambda}_{aa} + \boldsymbol{\Sigma}_{bb}\boldsymbol{\Lambda}_{ba} = \mathbf{0}$
4. $\boldsymbol{\Sigma}_{ba}\boldsymbol{\Lambda}_{ab} + \boldsymbol{\Sigma}_{bb}\boldsymbol{\Lambda}_{bb} = \mathbf{I}$

第2式から $\boldsymbol{\Lambda}_{ab}$ を解くと、
$$ \boldsymbol{\Lambda}_{ab} = -\boldsymbol{\Sigma}_{aa}^{-1} \boldsymbol{\Sigma}_{ab} \boldsymbol{\Lambda}_{bb} $$

これを第4式に代入します。
$$ -\boldsymbol{\Sigma}_{ba} \boldsymbol{\Sigma}_{aa}^{-1} \boldsymbol{\Sigma}_{ab} \boldsymbol{\Lambda}_{bb} + \boldsymbol{\Sigma}_{bb} \boldsymbol{\Lambda}_{bb} = \mathbf{I} $$
$$ (\boldsymbol{\Sigma}_{bb} - \boldsymbol{\Sigma}_{ba} \boldsymbol{\Sigma}_{aa}^{-1} \boldsymbol{\Sigma}_{ab}) \boldsymbol{\Lambda}_{bb} = \mathbf{I} $$
$$ \boldsymbol{\Lambda}_{bb} = (\boldsymbol{\Sigma}_{bb} - \boldsymbol{\Sigma}_{ba} \boldsymbol{\Sigma}_{aa}^{-1} \boldsymbol{\Sigma}_{ab})^{-1} $$
ここで得られた逆行列の括弧内の式は、シューア補行列 (Schur complement) と呼ばれます。

対称性から $\boldsymbol{\Lambda}_{aa}$ も同様に求められます。
$$ \boldsymbol{\Lambda}_{aa} = (\boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} \boldsymbol{\Sigma}_{ba})^{-1} $$

これを利用することで、演習問題2.9で求めた条件付き分布のパラメータを、精度行列 $\boldsymbol{\Lambda}$ ではなく共分散行列 $\boldsymbol{\Sigma}$ で書き換えることができます。
$$ \boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$
$$ \boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} \boldsymbol{\Sigma}_{ba} $$

この表現は、共分散の形で情報が与えられている実用的なケース（カルマンフィルタなど）で非常に有用です。
