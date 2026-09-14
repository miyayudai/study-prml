# 演習問題 2.49

## 問題の概要
混合ガウス分布のEMアルゴリズムにおける周辺対数尤度の下界（lower bound）が、期待完全対数尤度とエントロピーの和の形になることを示し、EMアルゴリズムのEステップがこの下界を最大化することを確認する。

## 解答

観測データ $\mathbf{X}$ と潜在変数 $\mathbf{Z}$ を持つモデルにおいて、不完全対数尤度（周辺対数尤度）$\ln p(\mathbf{X}|\boldsymbol{\theta})$ を考えます。任意の分布 $q(\mathbf{Z})$ を導入すると、対数尤度は次のように分解できます。

$$
\ln p(\mathbf{X}|\boldsymbol{\theta}) = \mathcal{L}(q, \boldsymbol{\theta}) + \text{KL}(q || p)
$$

ここで、$\mathcal{L}(q, \boldsymbol{\theta})$ はエビデンスの下界（ELBO: Evidence Lower BOund）、$\text{KL}$ はカルバック・ライブラー情報量です。

$$
\mathcal{L}(q, \boldsymbol{\theta}) = \sum_{\mathbf{Z}} q(\mathbf{Z}) \ln \frac{p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta})}{q(\mathbf{Z})}
$$
$$
\text{KL}(q || p) = -\sum_{\mathbf{Z}} q(\mathbf{Z}) \ln \frac{p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta})}{q(\mathbf{Z})}
$$

KL情報量は常に非負（$\text{KL} \ge 0$）であるため、$\mathcal{L}(q, \boldsymbol{\theta})$ は対数尤度の下界を与えます。

$$
\ln p(\mathbf{X}|\boldsymbol{\theta}) \ge \mathcal{L}(q, \boldsymbol{\theta})
$$

### 下界の分解
下界 $\mathcal{L}(q, \boldsymbol{\theta})$ の式を展開します。

$$
\mathcal{L}(q, \boldsymbol{\theta}) = \sum_{\mathbf{Z}} q(\mathbf{Z}) \ln p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta}) - \sum_{\mathbf{Z}} q(\mathbf{Z}) \ln q(\mathbf{Z})
$$

第1項は、分布 $q(\mathbf{Z})$ に対する完全データ対数尤度 $\ln p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta})$ の期待値です。これを期待完全対数尤度 $\mathcal{Q}(\boldsymbol{\theta}, \boldsymbol{\theta}^{\text{old}})$ と関連付けることができます。
第2項は、分布 $q(\mathbf{Z})$ のエントロピー $\text{H}[q]$ です。

$$
\mathcal{L}(q, \boldsymbol{\theta}) = \mathbb{E}_{q(\mathbf{Z})}[\ln p(\mathbf{X}, \mathbf{Z}|\boldsymbol{\theta})] + \text{H}[q]
$$

### Eステップの解釈
EMアルゴリズムのEステップでは、現在のパラメータ $\boldsymbol{\theta}^{\text{old}}$ を固定した状態で、下界 $\mathcal{L}(q, \boldsymbol{\theta}^{\text{old}})$ を最大化するような $q(\mathbf{Z})$ を見つけます。

$\ln p(\mathbf{X}|\boldsymbol{\theta}^{\text{old}})$ は $q(\mathbf{Z})$ に依存しないため定数です。したがって、下界 $\mathcal{L}$ を最大化することは、$\text{KL}(q || p)$ を最小化することと等価です。
KL情報量が最小値0をとるのは、2つの分布が完全に一致するときです。つまり、

$$
q(\mathbf{Z}) = p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta}^{\text{old}})
$$

と選ぶときです。このとき下界は真の対数尤度に一致し、最大化されます。
混合ガウス分布の場合、事後分布 $p(\mathbf{Z}|\mathbf{X}, \boldsymbol{\theta}^{\text{old}})$ はまさに負担率 $\gamma(z_{nk})$ で与えられるため、Eステップでの負担率の計算は、下界を $q$ について最大化するステップとしてエレガントに解釈できます。
