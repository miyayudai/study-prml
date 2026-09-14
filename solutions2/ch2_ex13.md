# Exercise 2.13

## 2.13 多項尤度とディリクレ事前分布の共役性 (Conjugacy of Dirichlet Prior and Multinomial Likelihood)

### 問題設定 (Problem Setup)
ベイズ推論において、ある尤度関数に対して事後分布が事前分布と同じ関数形になる性質を**共役性 (Conjugacy)** と呼びます。
ここでは、多項分布の尤度関数とディリクレ事前分布の組み合わせが共役性を持つこと、すなわち事後分布が新たなパラメータを持つディリクレ分布となることを示します。

### 尤度関数と事前分布
**多項分布の尤度関数 (Multinomial Likelihood):**
$N$ 回の独立な試行からなるデータセット $\mathcal{D}$ において、各クラス $k \in \{1, \dots, K\}$ が観測された回数を $m_k$ とします（$\sum_{k=1}^K m_k = N$）。
パラメータ $\boldsymbol{\mu} = (\mu_1, \dots, \mu_K)^\mathrm{T}$ が与えられたときの尤度関数は以下のようになります。
$$ p(\mathcal{D}|\boldsymbol{\mu}) \propto \prod_{k=1}^K \mu_k^{m_k} $$
（多項係数は $\boldsymbol{\mu}$ に依存しない定数であるため、比例定数に含めて省略します。）

**ディリクレ事前分布 (Dirichlet Prior):**
パラメータ $\boldsymbol{\mu}$ に対する事前分布として、パラメータ $\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_K)^\mathrm{T}$ を持つディリクレ分布を仮定します。
$$ p(\boldsymbol{\mu}|\boldsymbol{\alpha}) = \mathrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha}) \propto \prod_{k=1}^K \mu_k^{\alpha_k - 1} $$

### 事後分布の導出 (Derivation of the Posterior)
ベイズの定理に従い、事後分布は尤度関数と事前分布の積に比例します。
$$ p(\boldsymbol{\mu}|\mathcal{D}, \boldsymbol{\alpha}) = \frac{p(\mathcal{D}|\boldsymbol{\mu}) p(\boldsymbol{\mu}|\boldsymbol{\alpha})}{p(\mathcal{D})} $$
周辺尤度 $p(\mathcal{D})$ は $\boldsymbol{\mu}$ に依存しない正規化定数であるため、事後分布の形状（$\boldsymbol{\mu}$ に対する依存性）は以下の比例関係から定まります。
$$ p(\boldsymbol{\mu}|\mathcal{D}, \boldsymbol{\alpha}) \propto p(\mathcal{D}|\boldsymbol{\mu}) p(\boldsymbol{\mu}|\boldsymbol{\alpha}) $$

それぞれの式を代入します。
$$ p(\boldsymbol{\mu}|\mathcal{D}, \boldsymbol{\alpha}) \propto \left( \prod_{k=1}^K \mu_k^{m_k} \right) \left( \prod_{k=1}^K \mu_k^{\alpha_k - 1} \right) $$
同じ底の累乗をまとめると、以下のようになります。
$$ p(\boldsymbol{\mu}|\mathcal{D}, \boldsymbol{\alpha}) \propto \prod_{k=1}^K \mu_k^{m_k + \alpha_k - 1} $$

### 結論と解釈
この結果は、事後分布がパラメータ $\alpha'_k = \alpha_k + m_k$ を持つディリクレ分布の関数形（カーネル）と全く同じ形をしていることを示しています。
ディリクレ分布は正規化されるため、比例定数を補うことで、最終的な事後分布は以下のように書けます。
$$ p(\boldsymbol{\mu}|\mathcal{D}, \boldsymbol{\alpha}) = \mathrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha} + \mathbf{m}) $$
ここで $\mathbf{m} = (m_1, \dots, m_K)^\mathrm{T}$ は観測された各クラスのカウントベクトルです。

この結果から、ディリクレ分布のパラメータ $\alpha_k$ は「観測データが得られる前に、クラス $k$ が $\alpha_k$ 回観測されたという**疑似カウント (pseudo-counts)** 」として直感的に解釈できることが分かります。データが観測されるたびに、事後分布のパラメータはそのクラスのカウント分だけ増加（$\alpha_k \to \alpha_k + m_k$）更新されていくという、オンライン学習における非常に自然な性質を持っています。
