# PRML Exercise 2.56

## 問題
ベータ分布（Beta distribution）が指数型分布族（Exponential Family）のクラスに属することを示せ。また、その際の自然パラメータ $\boldsymbol{\eta}$、十分統計量 $\mathbf{u}(x)$、および各関数 $h(x), g(\boldsymbol{\eta})$ を明示せよ。

## 解答と解説
指数型分布族の標準形は次のように定義されます。
$$ p(x|\boldsymbol{\eta}) = h(x) g(\boldsymbol{\eta}) \exp\left( \boldsymbol{\eta}^T \mathbf{u}(x) \right) $$

一方、パラメータ $a, b > 0$ を持つベータ分布の定義式は以下の通りです。
$$ \text{Beta}(x|a, b) = \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} x^{a-1} (1 - x)^{b-1} $$
（ただし、$0 \le x \le 1$ であり、それ以外の範囲では $0$ です。）

この式を指数型分布族の形に変形します。$x^{a-1} (1 - x)^{b-1}$ の部分を指数関数を用いて書き直すと、
$$ x^{a-1} (1 - x)^{b-1} = \exp\left( \ln\left( x^{a-1} (1 - x)^{b-1} \right) \right) = \exp\left( (a - 1)\ln x + (b - 1)\ln(1 - x) \right) $$
となります。これをベータ分布の式に代入すると、
$$ \text{Beta}(x|a, b) = \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} \exp\left( (a - 1)\ln x + (b - 1)\ln(1 - x) \right) $$
となります。

### パラメータの対応づけ
この式を指数型分布族の標準形 $h(x) g(\boldsymbol{\eta}) \exp\left( \boldsymbol{\eta}^T \mathbf{u}(x) \right)$ と比較して、各要素を次のように割り当てます。

1. **自然パラメータ $\boldsymbol{\eta}$**:
   指数部にある係数をベクトルとしてまとめます。
   $$ \boldsymbol{\eta} = \begin{pmatrix} a - 1 \\ b - 1 \end{pmatrix} $$
   ※文献によっては $\boldsymbol{\eta} = (a, b)^T$ と定義し、$h(x)$ 側に $- \ln x - \ln(1-x)$ を押し付けることもありますが、どちらも等価な表現です。ここでは係数をそのまま $\boldsymbol{\eta}$ とします。

2. **十分統計量 $\mathbf{u}(x)$**:
   $\boldsymbol{\eta}$ の内積の相手となる、変数 $x$ に依存する部分です。
   $$ \mathbf{u}(x) = \begin{pmatrix} \ln x \\ \ln(1 - x) \end{pmatrix} $$
   これにより、指数部は $\boldsymbol{\eta}^T \mathbf{u}(x) = (a - 1)\ln x + (b - 1)\ln(1 - x)$ となります。

3. **正規化係数 $g(\boldsymbol{\eta})$**:
   $x$ に依存せず、パラメータにのみ依存する係数部分です。
   $\eta_1 = a - 1, \eta_2 = b - 1$ より $a = \eta_1 + 1, b = \eta_2 + 1$ なので、
   $$ g(\boldsymbol{\eta}) = \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} = \frac{\Gamma(\eta_1 + \eta_2 + 2)}{\Gamma(\eta_1 + 1)\Gamma(\eta_2 + 1)} $$

4. **ベース測度 $h(x)$**:
   $x$ のみに依存し、パラメータに依存しない項です。今回は残った項が定数 $1$ のみですので、
   $$ h(x) = 1 $$
   となります。

### 結論
ベータ分布は $h(x) g(\boldsymbol{\eta}) \exp\left( \boldsymbol{\eta}^T \mathbf{u}(x) \right)$ という形に正確に変形できるため、指数型分布族の一員であることが証明されました。
