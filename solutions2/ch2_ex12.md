# Exercise 2.12

## 2.12 ディリクレ分布の最頻値 (Mode of the Dirichlet Distribution)

### 問題設定 (Problem Setup)
ディリクレ分布 $\mathrm{Dir}(\boldsymbol{\mu}|\boldsymbol{\alpha})$ が与えられたとき、確率密度関数が最大となる $\boldsymbol{\mu}$ の値（最頻値）を求めます。
ディリクレ分布は次のように定義されます：
$$ p(\boldsymbol{\mu}|\boldsymbol{\alpha}) \propto \prod_{k=1}^K \mu_k^{\alpha_k - 1} $$
変数 $\boldsymbol{\mu}$ は、制約条件 $\sum_{k=1}^K \mu_k = 1$ を満たすシンプレックス上に存在します。
すべての $k$ に対して $\alpha_k > 1$ であると仮定します。このとき、最頻値が以下のようになることを示します：
$$ \mu_k^* = \frac{\alpha_k - 1}{\alpha_0 - K} $$
ここで $\alpha_0 = \sum_{k=1}^K \alpha_k$ です。

### 対数確率とラグランジュ未定乗数法
関数を最大化する際、単調増加関数である対数をとることで計算が大幅に容易になります。ディリクレ分布の対数をとると、
$$ \ln p(\boldsymbol{\mu}|\boldsymbol{\alpha}) = \sum_{k=1}^K (\alpha_k - 1) \ln \mu_k + \mathrm{const} $$
この関数を、制約条件 $\sum_{k=1}^K \mu_k = 1$ のもとで最大化します。制約付き最適化問題であるため、**ラグランジュの未定乗数法 (Lagrange Multipliers)** を導入します。

ラグランジュ乗数 $\lambda$ を用いて、次のようなラグランジュ関数 $L(\boldsymbol{\mu}, \lambda)$ を定義します：
$$ L(\boldsymbol{\mu}, \lambda) = \sum_{k=1}^K (\alpha_k - 1) \ln \mu_k + \lambda \left( \sum_{k=1}^K \mu_k - 1 \right) $$

### 微分と最適性の条件
最頻値を求めるため、ラグランジュ関数を各 $\mu_k$ で偏微分し、それを 0 と置きます。
$$ \frac{\partial L}{\partial \mu_k} = \frac{\alpha_k - 1}{\mu_k} + \lambda = 0 $$
この方程式を $\mu_k$ について解くと、
$$ \mu_k = -\frac{\alpha_k - 1}{\lambda} $$
となります。これが各カテゴリにおける $\mu_k$ の極値を与える条件です。

### ラグランジュ乗数 $\lambda$ の決定
得られた $\mu_k$ の表式を、元の制約条件 $\sum_{k=1}^K \mu_k = 1$ に代入して未知のパラメータ $\lambda$ を消去します。
$$ \sum_{k=1}^K \left( -\frac{\alpha_k - 1}{\lambda} \right) = 1 $$
$$ -\frac{1}{\lambda} \sum_{k=1}^K (\alpha_k - 1) = 1 $$
ここで、$\sum_{k=1}^K \alpha_k = \alpha_0$ であることを用いると、
$$ \sum_{k=1}^K (\alpha_k - 1) = \sum_{k=1}^K \alpha_k - \sum_{k=1}^K 1 = \alpha_0 - K $$
となります。これを代入すると、
$$ -\frac{1}{\lambda} (\alpha_0 - K) = 1 \implies \lambda = -(\alpha_0 - K) $$

### 最頻値 $\mu_k^*$ の導出
決定された $\lambda$ の値を、先ほど求めた $\mu_k$ の式に代入します。
$$ \mu_k^* = -\frac{\alpha_k - 1}{-(\alpha_0 - K)} = \frac{\alpha_k - 1}{\alpha_0 - K} $$
これにより、ディリクレ分布の最頻値が導出されました。

### 考察 (Discussion)
この結果 $\mu_k^* = \frac{\alpha_k - 1}{\alpha_0 - K}$ は、平均 $\mathbb{E}[\mu_k] = \frac{\alpha_k}{\alpha_0}$ と似ていますが、分母と分子からそれぞれ定数が引かれています。
これは $\alpha_k > 1$ の場合において分布のピーク（最も確率密度が高い点）を表しています。もし $\alpha_k = 1$ の場合、最頻値は一様分布に帰着し、特定のピークを持たなくなります。また $\alpha_k < 1$ の場合は、確率密度関数はシンプレックスの境界（$\mu_k \to 0$）で無限大に発散するため、内部に最頻値を持ちません。
