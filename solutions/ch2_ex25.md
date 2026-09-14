# 演習問題 2.25

## 問題設定
ガウス分布の尤度関数の対数微分について調べる。共分散行列 $\boldsymbol{\Sigma}$ に対する最尤推定量を導出する前段階として、トレースの微分の性質を証明せよ。

## 解答と証明
共分散行列 $\boldsymbol{\Sigma}$ の最尤推定を行う際、対数尤度関数は次のように表される。
$$ \ln p(\mathbf{X} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{N}{2} \ln |\boldsymbol{\Sigma}| - \frac{1}{2} \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}) + \text{const} $$

精度行列 $\mathbf{\Lambda} = \boldsymbol{\Sigma}^{-1}$ で微分することを考える。対数行列式の微分公式 $\frac{\partial}{\partial \mathbf{\Lambda}} \ln |\mathbf{\Lambda}| = \mathbf{\Lambda}^{-1} = \boldsymbol{\Sigma}$ を用いる。
$$ \frac{\partial}{\partial \mathbf{\Lambda}} \ln |\boldsymbol{\Sigma}|^{-1} = \frac{\partial}{\partial \mathbf{\Lambda}} \ln |\mathbf{\Lambda}| = \mathbf{\Lambda}^{-T} = \boldsymbol{\Sigma}^T $$
$\boldsymbol{\Sigma}$ は対称行列であるため $\boldsymbol{\Sigma}$ に等しい。

次に、トレース公式を用いて二次形式の和を書き換える。
$$ \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})^T \mathbf{\Lambda} (\mathbf{x}_n - \boldsymbol{\mu}) = \text{Tr}\left( \mathbf{\Lambda} \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T \right) $$

この項の $\mathbf{\Lambda}$ に対する微分は、$\frac{\partial}{\partial \mathbf{\Lambda}} \text{Tr}(\mathbf{\Lambda}\mathbf{A}) = \mathbf{A}^T$ を用いると、
$$ \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T $$
となる。これらを組み合わせることで最尤推定の導出が可能になる。
(証明終)
