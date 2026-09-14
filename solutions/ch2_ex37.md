# 演習問題 2.37

## 問題設定
指数分布族の一般形と、その標準的なパラメータ表現に関する性質を導出せよ。

## 解答と証明

確率分布が指数分布族（Exponential Family）に属するとは、その確率密度関数（または確率質量関数）が次のように表されることを指します。
$$ p(\mathbf{x}|oldsymbol{\eta}) = h(\mathbf{x})g(oldsymbol{\eta})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} $$
ここで、$oldsymbol{\eta}$ は自然パラメータ（natural parameters）、$\mathbf{u}(\mathbf{x})$ は十分統計量（sufficient statistics）です。

確率分布の正規化条件より、全空間にわたる積分は1にならなければなりません。
$$ \int p(\mathbf{x}|oldsymbol{\eta}) d\mathbf{x} = 1 $$

この式に指数分布族の定義を代入すると、
$$ g(oldsymbol{\eta}) \int h(\mathbf{x})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} d\mathbf{x} = 1 $$
したがって、正規化係数 $g(oldsymbol{\eta})$ は次のように求められます。
$$ g(oldsymbol{\eta}) = \left[ \int h(\mathbf{x})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} d\mathbf{x} ight]^{-1} $$

次に、この正規化条件の両辺を $oldsymbol{\eta}$ に関して微分し、期待値 $\mathbb{E}[\mathbf{u}(\mathbf{x})]$ との関係を導出します。
$$ 
abla_{oldsymbol{\eta}} \int h(\mathbf{x})g(oldsymbol{\eta})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} d\mathbf{x} = 0 $$

積の微分法則を適用すると、
$$ \int h(\mathbf{x}) \left\{ (
abla_{oldsymbol{\eta}}g(oldsymbol{\eta}))\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} + g(oldsymbol{\eta})\exp\{oldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\}\mathbf{u}(\mathbf{x}) ight\} d\mathbf{x} = 0 $$

これを整理すると、
$$ rac{
abla_{oldsymbol{\eta}}g(oldsymbol{\eta})}{g(oldsymbol{\eta})} \int p(\mathbf{x}|oldsymbol{\eta}) d\mathbf{x} + \int p(\mathbf{x}|oldsymbol{\eta}) \mathbf{u}(\mathbf{x}) d\mathbf{x} = 0 $$
$$ 
abla_{oldsymbol{\eta}} \ln g(oldsymbol{\eta}) + \mathbb{E}[\mathbf{u}(\mathbf{x})] = 0 $$

したがって、十分統計量の期待値は次のように表されます。
$$ \mathbb{E}[\mathbf{u}(\mathbf{x})] = -
abla_{oldsymbol{\eta}} \ln g(oldsymbol{\eta}) $$
これにより、対数正規化関数の1階微分が十分統計量の期待値に一致することが示されました。
