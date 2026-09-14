# Exercise 2.11

## 2.11 ディリクレ分布のモーメント (Moments of the Dirichlet Distribution)

### 問題設定 (Problem Setup)
ディリクレ分布 $\mathrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha})$ の周辺分布についての平均と分散、および共分散を導出する。

ディリクレ分布は多項分布の共役事前分布であり、次のように定義されます：
$$ \mathrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha}) = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_K)} \prod_{k=1}^{K} \mu_k^{\alpha_k - 1} = C(\boldsymbol{\alpha}) \prod_{k=1}^{K} \mu_k^{\alpha_k - 1} $$
ここで、$\alpha_0 = \sum_{k=1}^K \alpha_k$ であり、$C(\boldsymbol{\alpha})$ は正規化定数です。
$\boldsymbol{\mu} = (\mu_1, \dots, \mu_K)^{\mathrm{T}}$ は総和が1になる確率変数のベクトルであり、制約 $0 \leq \mu_k \leq 1$ および $\sum_{k=1}^K \mu_k = 1$ を満たします。

この分布に対して、特定の成分 $\mu_k$ の平均（期待値） $\mathbb{E}[\mu_k]$ と、分散 $\mathrm{var}[\mu_k]$ が以下のようになることを示します：
$$ \mathbb{E}[\mu_k] = \frac{\alpha_k}{\alpha_0} $$
$$ \mathrm{var}[\mu_k] = \frac{\alpha_k(\alpha_0-\alpha_k)}{\alpha_0^2(\alpha_0+1)} $$

### 平均 $\mathbb{E}[\mu_k]$ の導出
確率変数 $\mu_k$ の期待値は、確率密度関数に $\mu_k$ をかけて全区間で積分することで求められます。
$$ \mathbb{E}[\mu_k] = \int \mu_k \, \mathrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha}) \, d\boldsymbol{\mu} $$
$$ \mathbb{E}[\mu_k] = \int \mu_k \, C(\boldsymbol{\alpha}) \prod_{j=1}^{K} \mu_j^{\alpha_j - 1} \, d\boldsymbol{\mu} $$
$$ \mathbb{E}[\mu_k] = C(\boldsymbol{\alpha}) \int \mu_k^{(\alpha_k + 1) - 1} \prod_{j \neq k} \mu_j^{\alpha_j - 1} \, d\boldsymbol{\mu} $$

この積分は、新しいパラメータ $\boldsymbol{\alpha}'$ を持つディリクレ分布の正規化定数の逆数に等しくなります。ここで、新しいパラメータは $\alpha'_k = \alpha_k + 1$ であり、その他の $j \neq k$ については $\alpha'_j = \alpha_j$ となります。
ディリクレ分布が積分して1になる性質 $\int \prod_{j=1}^K \mu_j^{\alpha'_j - 1} d\boldsymbol{\mu} = \frac{1}{C(\boldsymbol{\alpha}')}$ を用いると、
$$ \mathbb{E}[\mu_k] = \frac{C(\boldsymbol{\alpha})}{C(\boldsymbol{\alpha}')} = \frac{\frac{\Gamma(\alpha_0)}{\prod_{j=1}^K \Gamma(\alpha_j)}}{\frac{\Gamma(\alpha_0 + 1)}{\Gamma(\alpha_k + 1) \prod_{j \neq k} \Gamma(\alpha_j)}} $$

分母と分子を整理すると以下のようになります：
$$ \mathbb{E}[\mu_k] = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_0 + 1)} \cdot \frac{\Gamma(\alpha_k + 1) \prod_{j \neq k} \Gamma(\alpha_j)}{\prod_{j=1}^K \Gamma(\alpha_j)} $$
ここで、ガンマ関数の性質 $\Gamma(x+1) = x\Gamma(x)$ を利用します。
- $\Gamma(\alpha_0 + 1) = \alpha_0 \Gamma(\alpha_0)$
- $\Gamma(\alpha_k + 1) = \alpha_k \Gamma(\alpha_k)$

これらを代入すると、
$$ \mathbb{E}[\mu_k] = \frac{\Gamma(\alpha_0)}{\alpha_0 \Gamma(\alpha_0)} \cdot \frac{\alpha_k \Gamma(\alpha_k)}{\Gamma(\alpha_k)} = \frac{\alpha_k}{\alpha_0} $$
これがディリクレ分布の平均（期待値）です。

### 二次のモーメント $\mathbb{E}[\mu_k^2]$ の導出
分散を求めるために、まずは $\mu_k^2$ の期待値を計算します。先ほどと同様にディリクレ分布の性質を利用します。
$$ \mathbb{E}[\mu_k^2] = \int \mu_k^2 \, \mathrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha}) \, d\boldsymbol{\mu} $$
$$ \mathbb{E}[\mu_k^2] = C(\boldsymbol{\alpha}) \int \mu_k^{(\alpha_k + 2) - 1} \prod_{j \neq k} \mu_j^{\alpha_j - 1} \, d\boldsymbol{\mu} $$

この積分の結果は、パラメータ $\alpha_k$ に $2$ を足した新しいディリクレ分布の正規化項の逆数になります。
$$ \mathbb{E}[\mu_k^2] = \frac{C(\boldsymbol{\alpha})}{C(\boldsymbol{\alpha}'')} = \frac{\frac{\Gamma(\alpha_0)}{\prod_{j=1}^K \Gamma(\alpha_j)}}{\frac{\Gamma(\alpha_0 + 2)}{\Gamma(\alpha_k + 2) \prod_{j \neq k} \Gamma(\alpha_j)}} $$
$$ \mathbb{E}[\mu_k^2] = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_0 + 2)} \cdot \frac{\Gamma(\alpha_k + 2)}{\Gamma(\alpha_k)} $$

再びガンマ関数の性質を繰り返し用います：
- $\Gamma(\alpha_0 + 2) = (\alpha_0 + 1)\alpha_0\Gamma(\alpha_0)$
- $\Gamma(\alpha_k + 2) = (\alpha_k + 1)\alpha_k\Gamma(\alpha_k)$

したがって、
$$ \mathbb{E}[\mu_k^2] = \frac{\alpha_k(\alpha_k + 1)}{\alpha_0(\alpha_0 + 1)} $$

### 分散 $\mathrm{var}[\mu_k]$ の導出
分散の一般的な定義 $\mathrm{var}[x] = \mathbb{E}[x^2] - (\mathbb{E}[x])^2$ を用います。
$$ \mathrm{var}[\mu_k] = \mathbb{E}[\mu_k^2] - (\mathbb{E}[\mu_k])^2 $$
$$ \mathrm{var}[\mu_k] = \frac{\alpha_k(\alpha_k + 1)}{\alpha_0(\alpha_0 + 1)} - \left( \frac{\alpha_k}{\alpha_0} \right)^2 $$
共通の分母でまとめると、
$$ \mathrm{var}[\mu_k] = \frac{\alpha_k(\alpha_k + 1)\alpha_0 - \alpha_k^2(\alpha_0 + 1)}{\alpha_0^2(\alpha_0 + 1)} $$
分子を展開して整理します：
$$ \alpha_k (\alpha_k \alpha_0 + \alpha_0) - (\alpha_k^2 \alpha_0 + \alpha_k^2) = \alpha_k^2 \alpha_0 + \alpha_k \alpha_0 - \alpha_k^2 \alpha_0 - \alpha_k^2 $$
$$ = \alpha_k \alpha_0 - \alpha_k^2 = \alpha_k (\alpha_0 - \alpha_k) $$

これを元の式に戻すと、分散が得られます。
$$ \mathrm{var}[\mu_k] = \frac{\alpha_k(\alpha_0 - \alpha_k)}{\alpha_0^2(\alpha_0 + 1)} $$

### 考察と解釈
ディリクレ分布の平均 $\mathbb{E}[\mu_k] = \frac{\alpha_k}{\alpha_0}$ は、事前分布の各クラスのパラメータ $\alpha_k$ が、全体のパラメータの和 $\alpha_0$ に対して占める割合に等しいことを示しています。これは多項分布の各事象の生起確率の事前推定値として非常に直感的な結果です。
また、分散について見ると、分母に $\alpha_0^2(\alpha_0 + 1)$ があるため、全体の観測データ（または事前データの強度） $\alpha_0$ が大きくなると分散は $O(1/\alpha_0)$ のオーダーで減少します。つまり、データが増えるほど推定の不確実性が小さくなるというベイズ推論の一般的な性質を数学的に裏付ける結果となっています。
