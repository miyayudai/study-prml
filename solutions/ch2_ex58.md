# 演習問題 2.58

## 問題文（要約）
指数型分布族において、対数正規化項 $\ln g(\boldsymbol{\eta})$ の二階微分が十分統計量 $\mathbf{u}(\mathbf{x})$ の共分散行列と一致すること、すなわち $-\nabla \nabla \ln g(\boldsymbol{\eta}) = \text{cov}[\mathbf{u}(\mathbf{x})]$ を示せ。

## 解答と解説

指数型分布族の確率分布は以下のように表されます。
$$ p(\mathbf{x}|\boldsymbol{\eta}) = h(\mathbf{x}) g(\boldsymbol{\eta}) \exp\{\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} $$

確率分布の積分は1になるという条件より：
$$ \int p(\mathbf{x}|\boldsymbol{\eta}) d\mathbf{x} = 1 \implies g(\boldsymbol{\eta}) \int h(\mathbf{x}) \exp\{\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} d\mathbf{x} = 1 $$

この両辺の自然パラメータ $\boldsymbol{\eta}$ に対する勾配（一階微分）をとると、以下が得られます。
$$ \nabla g(\boldsymbol{\eta}) \int h(\mathbf{x}) \exp\{\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} d\mathbf{x} + g(\boldsymbol{\eta}) \int h(\mathbf{x}) \exp\{\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} \mathbf{u}(\mathbf{x}) d\mathbf{x} = 0 $$

両辺に $g(\boldsymbol{\eta})$ を掛けることで整理すると（あるいは前式の積分に $1/g(\boldsymbol{\eta})$ を代入すると）：
$$ -\frac{\nabla g(\boldsymbol{\eta})}{g(\boldsymbol{\eta})} = \int h(\mathbf{x}) g(\boldsymbol{\eta}) \exp\{\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} \mathbf{u}(\mathbf{x}) d\mathbf{x} $$

左辺は合成関数の微分により $-\nabla \ln g(\boldsymbol{\eta})$ となり、右辺は $\mathbf{u}(\mathbf{x})$ の期待値 $\mathbb{E}[\mathbf{u}(\mathbf{x})]$ に他なりません。
$$ -\nabla \ln g(\boldsymbol{\eta}) = \mathbb{E}[\mathbf{u}(\mathbf{x})] $$

次に、この式をさらに $\boldsymbol{\eta}$ について微分（二階微分）します。
元の積分の式：
$$ -\nabla \ln g(\boldsymbol{\eta}) = \int g(\boldsymbol{\eta}) h(\mathbf{x}) \exp\{\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} \mathbf{u}(\mathbf{x}) d\mathbf{x} $$
この両辺を $\boldsymbol{\eta}$ で微分します。（※ 左辺は行列になります）

$$ -\nabla \nabla \ln g(\boldsymbol{\eta}) = \int \left[ \nabla g(\boldsymbol{\eta}) \right] h(\mathbf{x}) \exp\{\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} \mathbf{u}(\mathbf{x})^T d\mathbf{x} + \int g(\boldsymbol{\eta}) h(\mathbf{x}) \exp\{\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} \mathbf{u}(\mathbf{x}) \mathbf{u}(\mathbf{x})^T d\mathbf{x} $$

ここで、$\nabla g(\boldsymbol{\eta}) = g(\boldsymbol{\eta}) \nabla \ln g(\boldsymbol{\eta})$ の関係を用いると、右辺の第一項は次のように変形できます：
$$ \text{第一項} = \nabla \ln g(\boldsymbol{\eta}) \int g(\boldsymbol{\eta}) h(\mathbf{x}) \exp\{\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} \mathbf{u}(\mathbf{x})^T d\mathbf{x} $$
$$ = \nabla \ln g(\boldsymbol{\eta}) \mathbb{E}[\mathbf{u}(\mathbf{x})^T] $$
先ほど導出した $-\nabla \ln g(\boldsymbol{\eta}) = \mathbb{E}[\mathbf{u}(\mathbf{x})]$ を代入すると：
$$ \text{第一項} = -\mathbb{E}[\mathbf{u}(\mathbf{x})] \mathbb{E}[\mathbf{u}(\mathbf{x})^T] $$

また、右辺の第二項は $\mathbf{u}(\mathbf{x})\mathbf{u}(\mathbf{x})^T$ の期待値の定義そのものです。
$$ \text{第二項} = \mathbb{E}[\mathbf{u}(\mathbf{x})\mathbf{u}(\mathbf{x})^T] $$

これらをまとめると：
$$ -\nabla \nabla \ln g(\boldsymbol{\eta}) = \mathbb{E}[\mathbf{u}(\mathbf{x})\mathbf{u}(\mathbf{x})^T] - \mathbb{E}[\mathbf{u}(\mathbf{x})] \mathbb{E}[\mathbf{u}(\mathbf{x})^T] $$

この式の右辺は、まさに $\mathbf{u}(\mathbf{x})$ の共分散行列 $\text{cov}[\mathbf{u}(\mathbf{x})]$ の定義に他なりません。したがって、
$$ -\nabla \nabla \ln g(\boldsymbol{\eta}) = \text{cov}[\mathbf{u}(\mathbf{x})] $$
が示されました。これは、対数正規化項がキュムラント母関数として機能することを示す重要な性質です。
