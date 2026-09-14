# 演習問題 2.9 (Exercise 2.9)

## 問題の目的
多変量ガウス分布において、一部の変数が観測された場合の残りの変数の条件付き分布 $p(\mathbf{x}_a | \mathbf{x}_b)$ を導出します。周辺分布と同様に、条件付き分布もまたガウス分布になることを示します。

## 導出のステップ

同時分布 $p(\mathbf{x}_a, \mathbf{x}_b)$ の指数部分の二次形式 $\Delta$ は、演習問題2.8の式から出発します。
$$ \Delta = (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{aa} (\mathbf{x}_a - \boldsymbol{\mu}_a) + 2 (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) + (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Lambda}_{bb} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$

条件付き分布 $p(\mathbf{x}_a | \mathbf{x}_b)$ を考えるため、$\mathbf{x}_b$ は定数として扱い、$\mathbf{x}_a$ について平方完成を行います。$\mathbf{x}_a$ に依存する項をまとめると以下のようになります。
$$ \Delta_a = \mathbf{x}_a^T \boldsymbol{\Lambda}_{aa} \mathbf{x}_a - 2 \mathbf{x}_a^T (\boldsymbol{\Lambda}_{aa} \boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b)) + \text{const} $$

ここで、ガウス分布の一般的な二次形式 $(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} - 2 \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu} + \text{const}$ と係数比較を行います。

まず、2次の項 $\mathbf{x}_a^T \boldsymbol{\Lambda}_{aa} \mathbf{x}_a$ の係数から、条件付き共分散行列 $\boldsymbol{\Sigma}_{a|b}$ は精度行列の対応するブロックの逆行列であることがわかります。
$$ \boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Lambda}_{aa}^{-1} $$

次に、1次の項の係数を比較すると、
$$ \boldsymbol{\Sigma}_{a|b}^{-1} \boldsymbol{\mu}_{a|b} = \boldsymbol{\Lambda}_{aa} \boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$
$$ \boldsymbol{\Lambda}_{aa} \boldsymbol{\mu}_{a|b} = \boldsymbol{\Lambda}_{aa} \boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$

両辺に左から $\boldsymbol{\Lambda}_{aa}^{-1}$ を掛けることで、条件付き平均 $\boldsymbol{\mu}_{a|b}$ が得られます。
$$ \boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{aa}^{-1} \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$

以上の結果から、条件付き分布 $p(\mathbf{x}_a | \mathbf{x}_b)$ は以下のパラメータを持つガウス分布となることが証明されました。
$$ p(\mathbf{x}_a | \mathbf{x}_b) = \mathcal{N}(\mathbf{x}_a | \boldsymbol{\mu}_{a|b}, \boldsymbol{\Sigma}_{a|b}) $$
