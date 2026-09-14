# 演習問題 2.38

## 問題設定
指数分布族の2階のモーメントに関する性質を導出せよ。

## 解答と証明

前問（演習問題 2.37）で導出した結果
$$ \int h(\mathbf{x})g(oldsymbol{\eta})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} d\mathbf{x} = 1 $$
をさらに $oldsymbol{\eta}$ に関して2階微分します。

まず1階微分の結果から出発します。
$$ 
abla_{oldsymbol{\eta}} g(oldsymbol{\eta}) \int h(\mathbf{x})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} d\mathbf{x} + g(oldsymbol{\eta}) \int h(\mathbf{x})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} \mathbf{u}(\mathbf{x}) d\mathbf{x} = 0 $$

もう一度 $oldsymbol{\eta}$ で微分（勾配ベクトルに対するヤコビアンの計算）を行います。
$$

abla_{oldsymbol{\eta}}^2 g(oldsymbol{\eta}) \int h(\mathbf{x})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} d\mathbf{x} \
+ 
abla_{oldsymbol{\eta}} g(oldsymbol{\eta}) \int h(\mathbf{x})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} \mathbf{u}(\mathbf{x})^T d\mathbf{x} \
+ 
abla_{oldsymbol{\eta}} g(oldsymbol{\eta}) \int h(\mathbf{x})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} \mathbf{u}(\mathbf{x})^T d\mathbf{x} \
+ g(oldsymbol{\eta}) \int h(\mathbf{x})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} \mathbf{u}(\mathbf{x})\mathbf{u}(\mathbf{x})^T d\mathbf{x} = 0
$$

ここで、$g(oldsymbol{\eta}) \int h(\mathbf{x})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} d\mathbf{x} = 1$ の関係式を用いて各項を $g(oldsymbol{\eta})$ で割ると、
$$ rac{
abla_{oldsymbol{\eta}}^2 g(oldsymbol{\eta})}{g(oldsymbol{\eta})} + 2 rac{
abla_{oldsymbol{\eta}} g(oldsymbol{\eta})}{g(oldsymbol{\eta})} \mathbb{E}[\mathbf{u}(\mathbf{x})^T] + \mathbb{E}[\mathbf{u}(\mathbf{x})\mathbf{u}(\mathbf{x})^T] = 0 $$

一方、$-
abla_{oldsymbol{\eta}} \ln g(oldsymbol{\eta}) = \mathbb{E}[\mathbf{u}(\mathbf{x})]$ であることを考慮し、共分散行列 $	ext{cov}[\mathbf{u}(\mathbf{x})] = \mathbb{E}[\mathbf{u}(\mathbf{x})\mathbf{u}(\mathbf{x})^T] - \mathbb{E}[\mathbf{u}(\mathbf{x})]\mathbb{E}[\mathbf{u}(\mathbf{x})]^T$ を計算すると、
$$ -
abla_{oldsymbol{\eta}}^2 \ln g(oldsymbol{\eta}) = 	ext{cov}[\mathbf{u}(\mathbf{x})] $$
となることが示されます。これにより、対数正規化関数の2階微分が十分統計量の共分散行列となることがわかります。
