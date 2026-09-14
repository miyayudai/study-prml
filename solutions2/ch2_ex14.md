# Exercise 2.14

## 2.14 ディリクレ分布の事後予測分布 (Posterior Predictive Distribution of Dirichlet)

### 問題設定 (Problem Setup)
$N$ 個のデータ $\mathcal{D}$ を観測したあとに、新たに得られる次の観測値 $\mathbf{x}$ が特定のクラス $k$ （すなわち $x_k = 1$ かつ他の $j \neq k$ で $x_j = 0$）となる**事後予測確率 (Posterior Predictive Probability)** を求めます。

前回の問題 (Exercise 2.13) より、事後分布はパラメータ $\boldsymbol{\alpha}^* = \boldsymbol{\alpha} + \mathbf{m}$ を持つディリクレ分布となることが分かっています。
$$ p(\boldsymbol{\mu}|\mathcal{D}, \boldsymbol{\alpha}) = \mathrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha} + \mathbf{m}) $$

このとき、事後予測確率 $p(x_k = 1 | \mathcal{D})$ が次のように表されることを示します。
$$ p(x_k = 1 | \mathcal{D}) = \frac{\alpha_k + m_k}{\alpha_0 + N} $$
ここで $\alpha_0 = \sum_{j=1}^K \alpha_j$、および $N = \sum_{j=1}^K m_j$ です。

### 事後予測分布の導出
次に観測されるデータ $\mathbf{x}$ の確率は、真のパラメータ $\boldsymbol{\mu}$ に依存しますが、$\boldsymbol{\mu}$ は未知の変数です。ベイズ推論においては、事後分布 $p(\boldsymbol{\mu}|\mathcal{D})$ に従って $\boldsymbol{\mu}$ を周辺化（積分消去）することで予測分布を得ます。
$$ p(x_k = 1 | \mathcal{D}) = \int p(x_k = 1 | \boldsymbol{\mu}) p(\boldsymbol{\mu}|\mathcal{D}) \, d\boldsymbol{\mu} $$

観測モデルから、あるパラメータ $\boldsymbol{\mu}$ が与えられたときのクラス $k$ が観測される確率は $p(x_k = 1 | \boldsymbol{\mu}) = \mu_k$ です。これを代入します。
$$ p(x_k = 1 | \mathcal{D}) = \int \mu_k \, p(\boldsymbol{\mu}|\mathcal{D}) \, d\boldsymbol{\mu} $$

この積分の形は、事後分布 $p(\boldsymbol{\mu}|\mathcal{D})$ のもとでの $\mu_k$ の期待値 $\mathbb{E}[\mu_k | \mathcal{D}]$ そのものです。
事後分布は $\mathrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha} + \mathbf{m})$ であり、ディリクレ分布の平均は Exercise 2.11 で導出した通り、パラメータの比となります。
事後分布におけるクラス $k$ のパラメータは $\alpha_k + m_k$ です。
事後分布におけるすべてのパラメータの総和は以下のようになります。
$$ \sum_{j=1}^K (\alpha_j + m_j) = \sum_{j=1}^K \alpha_j + \sum_{j=1}^K m_j = \alpha_0 + N $$

したがって、事後分布の平均の公式を用いると、次の結果が直ちに得られます。
$$ p(x_k = 1 | \mathcal{D}) = \mathbb{E}[\mu_k | \mathcal{D}] = \frac{\alpha_k + m_k}{\alpha_0 + N} $$

### 考察と解釈 (Discussion)
得られた予測確率の式 $\frac{\alpha_k + m_k}{\alpha_0 + N}$ は、非常に直感的な意味を持っています。
- 分母 $\alpha_0 + N$ は、「事前知識の強さ $\alpha_0$」と「実際の観測データ数 $N$」の合計、すなわち**有効総データ数**を表します。
- 分子 $\alpha_k + m_k$ は、クラス $k$ に対する「事前の仮想観測数 $\alpha_k$」と「実際の観測回数 $m_k$」の合計を表します。

データ数 $N \to \infty$ の極限では、事前分布の影響 ($\alpha_k$, $\alpha_0$) は無視できるほど小さくなり、予測確率は $m_k / N$、すなわち最尤推定（経験確率）に一致します。
一方でデータが少ない（$N$ が小さい）ときは、事前知識（スムージングの効果）が強く働き、未観測のクラス（$m_k = 0$）に対しても確率が $0$ になることを防ぐ役割を果たします（例えば、自然言語処理におけるラプラススムージングと等価になります）。
