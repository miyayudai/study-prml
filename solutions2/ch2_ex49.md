# Exercise 2.49 (対数分配関数の2階微分と共分散)

## 課題
指数型分布族における対数分配関数 $A(\boldsymbol{\eta})$ の2階微分（ヘッセ行列）が、十分統計量 $\mathbf{u}(\mathbf{x})$ の共分散行列に等しいこと、すなわち $\nabla^2 A(\boldsymbol{\eta}) = \mathrm{cov}[\mathbf{u}(\mathbf{x})]$ を示せ。さらに、この結果から $A(\boldsymbol{\eta})$ が凸関数であることを説明せよ。

## 解答と解説
指数型分布族の一般的な確率密度（または質量）関数は以下のように定義されます。
$$ p(\mathbf{x} | \boldsymbol{\eta}) = h(\mathbf{x}) \exp\left( \boldsymbol{\eta}^T \mathbf{u}(\mathbf{x}) - A(\boldsymbol{\eta}) \right) $$

確率密度関数の全体に対する積分は 1 に等しいという規格化条件を考えます。
$$ \int p(\mathbf{x} | \boldsymbol{\eta}) d\mathbf{x} = 1 $$

両辺に $\exp(A(\boldsymbol{\eta}))$ を掛けると、
$$ \exp(A(\boldsymbol{\eta})) = \int h(\mathbf{x}) \exp\left( \boldsymbol{\eta}^T \mathbf{u}(\mathbf{x}) \right) d\mathbf{x} $$
となります。

### 1階微分の計算（期待値）
両辺を自然パラメータ $\boldsymbol{\eta}$ に関して微分（勾配ベクトル）します。
左辺の微分はチェーンルールにより、
$$ \nabla_{\boldsymbol{\eta}} \exp(A(\boldsymbol{\eta})) = \exp(A(\boldsymbol{\eta})) \nabla A(\boldsymbol{\eta}) $$

右辺の微分（積分と微分の順序交換を仮定）は、
$$ \nabla_{\boldsymbol{\eta}} \int h(\mathbf{x}) \exp\left( \boldsymbol{\eta}^T \mathbf{u}(\mathbf{x}) \right) d\mathbf{x} = \int h(\mathbf{x}) \mathbf{u}(\mathbf{x}) \exp\left( \boldsymbol{\eta}^T \mathbf{u}(\mathbf{x}) \right) d\mathbf{x} $$

両者を等置し、全体を $\exp(A(\boldsymbol{\eta}))$ で割ります（または右辺の積分の中に $\exp(-A(\boldsymbol{\eta}))$ として戻します）。
$$ \nabla A(\boldsymbol{\eta}) = \int \mathbf{u}(\mathbf{x}) h(\mathbf{x}) \exp\left( \boldsymbol{\eta}^T \mathbf{u}(\mathbf{x}) - A(\boldsymbol{\eta}) \right) d\mathbf{x} $$
$$ \nabla A(\boldsymbol{\eta}) = \int \mathbf{u}(\mathbf{x}) p(\mathbf{x} | \boldsymbol{\eta}) d\mathbf{x} = \mathbb{E}[\mathbf{u}(\mathbf{x})] $$
これが、対数分配関数の1階微分が十分統計量の期待値となる証明です。

### 2階微分の計算（共分散行列）
次に、上記の式 $\nabla A(\boldsymbol{\eta}) = \int \mathbf{u}(\mathbf{x}) p(\mathbf{x} | \boldsymbol{\eta}) d\mathbf{x}$ の両辺をもう一度 $\boldsymbol{\eta}$ で微分します。
左辺は2階微分（ヘッセ行列）になります。
$$ \nabla^2 A(\boldsymbol{\eta}) $$

右辺の積分内を $\boldsymbol{\eta}$ について微分します。ここで、$p(\mathbf{x} | \boldsymbol{\eta}) = h(\mathbf{x}) \exp(\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x}) - A(\boldsymbol{\eta}))$ であるため、その微分は、
$$ \nabla_{\boldsymbol{\eta}} p(\mathbf{x} | \boldsymbol{\eta}) = p(\mathbf{x} | \boldsymbol{\eta}) \left( \mathbf{u}(\mathbf{x}) - \nabla A(\boldsymbol{\eta}) \right)^T $$
となります（ここでは行列の次元を合わせるために転置を考慮します）。

これを右辺の微分に適用すると、
$$ \nabla_{\boldsymbol{\eta}} \int \mathbf{u}(\mathbf{x}) p(\mathbf{x} | \boldsymbol{\eta}) d\mathbf{x} = \int \mathbf{u}(\mathbf{x}) \left( \mathbf{u}(\mathbf{x}) - \nabla A(\boldsymbol{\eta}) \right)^T p(\mathbf{x} | \boldsymbol{\eta}) d\mathbf{x} $$

これを展開します。
$$ = \int \mathbf{u}(\mathbf{x}) \mathbf{u}(\mathbf{x})^T p(\mathbf{x} | \boldsymbol{\eta}) d\mathbf{x} - \left( \int \mathbf{u}(\mathbf{x}) p(\mathbf{x} | \boldsymbol{\eta}) d\mathbf{x} \right) \nabla A(\boldsymbol{\eta})^T $$

期待値の定義と $\nabla A(\boldsymbol{\eta}) = \mathbb{E}[\mathbf{u}(\mathbf{x})]$ より、
$$ = \mathbb{E}[\mathbf{u}(\mathbf{x})\mathbf{u}(\mathbf{x})^T] - \mathbb{E}[\mathbf{u}(\mathbf{x})] \mathbb{E}[\mathbf{u}(\mathbf{x})]^T $$

この結果はまさに十分統計量 $\mathbf{u}(\mathbf{x})$ の共分散行列の定義そのものです。したがって、
$$ \nabla^2 A(\boldsymbol{\eta}) = \mathrm{cov}[\mathbf{u}(\mathbf{x})] $$
が証明されました。

### $A(\boldsymbol{\eta})$ の凸性について
任意の確率変数ベクトル $\mathbf{u}$ の共分散行列は、半正定値行列 (positive semi-definite) であることが知られています。
すなわち、任意の非ゼロベクトル $\mathbf{v}$ に対して $\mathbf{v}^T \mathrm{cov}[\mathbf{u}(\mathbf{x})] \mathbf{v} \geq 0$ が成り立ちます。
ヘッセ行列 $\nabla^2 A(\boldsymbol{\eta})$ が至る所で半正定値であるということは、関数 $A(\boldsymbol{\eta})$ が自然パラメータ $\boldsymbol{\eta}$ について**凸関数 (convex function)** であることを意味します。この性質は、指数型分布族における対数尤度の最適化などにおいて非常に重要な役割を果たします。
