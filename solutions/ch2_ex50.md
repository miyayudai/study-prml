# PRML Exercise 2.50

## 問題
自由度パラメータ $\nu \to \infty$ の極限において、Studentのt分布（2.159）が平均 $\boldsymbol{\mu}$、精度行列 $\mathbf{\Lambda}$ のガウス分布に帰着することを示せ。

## 解答と解説
多変量Studentのt分布は次のように定義されます。
$$ \text{St}(\mathbf{x}|\boldsymbol{\mu}, \mathbf{\Lambda}, \nu) = \frac{\Gamma(D/2 + \nu/2)}{\Gamma(\nu/2)} \frac{|\mathbf{\Lambda}|^{1/2}}{(\pi \nu)^{D/2}} \left[ 1 + \frac{\Delta^2}{\nu} \right]^{-D/2 - \nu/2} $$
ここで、$\Delta^2 = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Lambda} (\mathbf{x} - \boldsymbol{\mu})$ はマハラノビス距離の2乗を表します。

$\nu \to \infty$ の極限を考えるため、上記の式を指数部（カーネル）と正規化定数に分けて評価します。

### 1. 指数部（カーネル）の極限
$\mathbf{x}$ に依存する部分は以下のようになります。
$$ \left( 1 + \frac{\Delta^2}{\nu} \right)^{-\nu/2 - D/2} = \left( 1 + \frac{\Delta^2}{\nu} \right)^{-\nu/2} \left( 1 + \frac{\Delta^2}{\nu} \right)^{-D/2} $$
ここで、指数関数の自然な定義 $\lim_{n \to \infty} (1 + x/n)^n = e^x$ を用いると、第一項は
$$ \lim_{\nu \to \infty} \left( 1 + \frac{\Delta^2}{\nu} \right)^{-\nu/2} = \exp\left(-\frac{\Delta^2}{2}\right) $$
となります。第二項は $\nu \to \infty$ において
$$ \lim_{\nu \to \infty} \left( 1 + \frac{\Delta^2}{\nu} \right)^{-D/2} = 1^{-D/2} = 1 $$
に収束します。したがって、指数部は全体として $\exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Lambda} (\mathbf{x} - \boldsymbol{\mu})\right)$ に帰着し、これはガウス分布のカーネルと一致します。

### 2. 正規化定数の極限
次に、$\mathbf{x}$ に依存しない係数部分の極限を考えます。
$$ C(\nu) = \frac{\Gamma(D/2 + \nu/2)}{\Gamma(\nu/2)} \frac{|\mathbf{\Lambda}|^{1/2}}{(\pi \nu)^{D/2}} $$
ガンマ関数の漸近展開（スターリングの近似から導かれる性質）として、$x \to \infty$ のとき
$$ \frac{\Gamma(x + a)}{\Gamma(x)} \approx x^a $$
が成り立ちます。ここで $x = \nu/2$、$a = D/2$ とおくと、
$$ \frac{\Gamma(D/2 + \nu/2)}{\Gamma(\nu/2)} \approx \left(\frac{\nu}{2}\right)^{D/2} $$
となります。これを係数部分に代入すると、
$$ \lim_{\nu \to \infty} C(\nu) = \left(\frac{\nu}{2}\right)^{D/2} \frac{|\mathbf{\Lambda}|^{1/2}}{(\pi \nu)^{D/2}} = \frac{1}{2^{D/2}} \frac{|\mathbf{\Lambda}|^{1/2}}{\pi^{D/2}} = \frac{|\mathbf{\Lambda}|^{1/2}}{(2\pi)^{D/2}} $$
となり、これは多変量ガウス分布の正規化定数と完全に一致します。

### 結論
以上の2点から、$\nu \to \infty$ の極限においてStudentのt分布は、
$$ \lim_{\nu \to \infty} \text{St}(\mathbf{x}|\boldsymbol{\mu}, \mathbf{\Lambda}, \nu) = \frac{|\mathbf{\Lambda}|^{1/2}}{(2\pi)^{D/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Lambda} (\mathbf{x} - \boldsymbol{\mu})\right) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \mathbf{\Lambda}^{-1}) $$
となることが証明されました。
自由度 $\nu$ が無限大に近づくと、外れ値への頑健性が失われ、通常のガウス分布と同じ振る舞いを示すことが数学的に確認できます。
