# 演習問題 2.22

## 問題設定
ベクトルの二次形式 $\mathbf{x}^T \mathbf{A} \mathbf{x}$ の期待値に関する公式を証明する。確率変数ベクトル $\mathbf{x}$ の平均を $\boldsymbol{\mu}$、共分散行列を $\boldsymbol{\Sigma}$ としたとき、定数行列 $\mathbf{A}$ に対して次式が成り立つことを示せ。
$$ \mathbb{E}[\mathbf{x}^T \mathbf{A} \mathbf{x}] = \boldsymbol{\mu}^T \mathbf{A} \boldsymbol{\mu} + \text{Tr}(\mathbf{A} \boldsymbol{\Sigma}) $$

## 解答と証明

スカラー値である二次形式に対してトレースを取っても値は変わらない。トレースの巡回不変性を用いると、
$$ \mathbf{x}^T \mathbf{A} \mathbf{x} = \text{Tr}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = \text{Tr}(\mathbf{A} \mathbf{x} \mathbf{x}^T) $$
となる。両辺の期待値をとる。期待値演算とトレース演算はともに線形であるため交換可能である。
$$ \mathbb{E}[\mathbf{x}^T \mathbf{A} \mathbf{x}] = \mathbb{E}[\text{Tr}(\mathbf{A} \mathbf{x} \mathbf{x}^T)] = \text{Tr}(\mathbf{A} \mathbb{E}[\mathbf{x} \mathbf{x}^T]) $$

共分散行列の定義より、
$$ \boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T] = \mathbb{E}[\mathbf{x}\mathbf{x}^T] - \boldsymbol{\mu}\boldsymbol{\mu}^T $$
これを整理すると、$\mathbb{E}[\mathbf{x}\mathbf{x}^T] = \boldsymbol{\Sigma} + \boldsymbol{\mu}\boldsymbol{\mu}^T$ となる。これを代入すると、
$$ \mathbb{E}[\mathbf{x}^T \mathbf{A} \mathbf{x}] = \text{Tr}(\mathbf{A} (\boldsymbol{\Sigma} + \boldsymbol{\mu}\boldsymbol{\mu}^T)) = \text{Tr}(\mathbf{A} \boldsymbol{\Sigma}) + \text{Tr}(\mathbf{A} \boldsymbol{\mu}\boldsymbol{\mu}^T) $$

再びトレースの性質を用いると、$\text{Tr}(\mathbf{A} \boldsymbol{\mu}\boldsymbol{\mu}^T) = \text{Tr}(\boldsymbol{\mu}^T \mathbf{A} \boldsymbol{\mu})$ である。$\boldsymbol{\mu}^T \mathbf{A} \boldsymbol{\mu}$ はスカラーであるため、トレースを外すことができる。
したがって、
$$ \mathbb{E}[\mathbf{x}^T \mathbf{A} \mathbf{x}] = \boldsymbol{\mu}^T \mathbf{A} \boldsymbol{\mu} + \text{Tr}(\mathbf{A} \boldsymbol{\Sigma}) $$
が示された。(証明終)
