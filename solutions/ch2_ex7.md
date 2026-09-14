# 演習問題 2.7

## 問題設定
ディリクレ分布 (Dirichlet distribution) は多項分布の共役事前分布であり、次のように定義される。
$$ \text{Dir}(\boldsymbol{\mu} | \boldsymbol{\alpha}) = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_K)} \prod_{k=1}^K \mu_k^{\alpha_k - 1} $$
ここで、$\alpha_0 = \sum_{k=1}^K \alpha_k$ であり、$\boldsymbol{\mu}$ は $\sum_{k=1}^K \mu_k = 1$ かつ $\mu_k \ge 0$ を満たす。

このディリクレ分布について、$\mu_k$ の期待値、分散、および共分散がそれぞれ以下で与えられることを示せ。
$$ \mathbb{E}[\mu_k] = \frac{\alpha_k}{\alpha_0} $$
$$ \text{var}[\mu_k] = \frac{\alpha_k(\alpha_0 - \alpha_k)}{\alpha_0^2(\alpha_0 + 1)} $$
$$ \text{cov}[\mu_j, \mu_k] = -\frac{\alpha_j \alpha_k}{\alpha_0^2(\alpha_0 + 1)} \quad (j \neq k) $$

## 解答と導出

ディリクレ分布が正しく正規化されていること、すなわち
$$ \int \prod_{i=1}^K \mu_i^{\alpha_i - 1} d\boldsymbol{\mu} = \frac{\Gamma(\alpha_1) \cdots \Gamma(\alpha_K)}{\Gamma(\alpha_0)} $$
であることを用いて、各種のモーメントを計算する。

### 1. 期待値の導出
$\mu_k$ の期待値は定義より以下のように書ける。
$$
\begin{aligned}
\mathbb{E}[\mu_k] &= \int \mu_k \text{Dir}(\boldsymbol{\mu} | \boldsymbol{\alpha}) d\boldsymbol{\mu} \\
&= \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_K)} \int \mu_k \prod_{i=1}^K \mu_i^{\alpha_i - 1} d\boldsymbol{\mu} \\
&= \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_K)} \int \mu_k^{\alpha_k} \prod_{i \neq k} \mu_i^{\alpha_i - 1} d\boldsymbol{\mu}
\end{aligned}
$$

積分の部分は、パラメータが $\alpha_k + 1$ （その他のパラメータは変化なし）であるディリクレ分布の正規化定数の逆数に等しい。また、パラメータの和は $\alpha_0 + 1$ となる。したがって、
$$ \int \mu_k^{\alpha_k} \prod_{i \neq k} \mu_i^{\alpha_i - 1} d\boldsymbol{\mu} = \frac{\Gamma(\alpha_1) \cdots \Gamma(\alpha_k + 1) \cdots \Gamma(\alpha_K)}{\Gamma(\alpha_0 + 1)} $$

これを元の式に代入し、ガンマ関数の性質 $\Gamma(x+1) = x\Gamma(x)$ を用いる。
$$
\begin{aligned}
\mathbb{E}[\mu_k] &= \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_K)} \cdot \frac{\Gamma(\alpha_1) \cdots \Gamma(\alpha_k + 1) \cdots \Gamma(\alpha_K)}{\Gamma(\alpha_0 + 1)} \\
&= \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_0 + 1)} \cdot \frac{\Gamma(\alpha_k + 1)}{\Gamma(\alpha_k)} \\
&= \frac{1}{\alpha_0} \cdot \alpha_k \\
&= \frac{\alpha_k}{\alpha_0}
\end{aligned}
$$
よって期待値が示された。

### 2. 分散の導出
分散を求めるために、まず $\mathbb{E}[\mu_k^2]$ を計算する。期待値の導出と同様の手順を用いると、積分の結果はパラメータが $\alpha_k + 2$ （和は $\alpha_0 + 2$）のディリクレ正規化定数となる。

$$
\begin{aligned}
\mathbb{E}[\mu_k^2] &= \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_k)} \cdot \frac{\Gamma(\alpha_k + 2)}{\Gamma(\alpha_0 + 2)} \\
&= \frac{(\alpha_k + 1)\alpha_k}{(\alpha_0 + 1)\alpha_0}
\end{aligned}
$$

分散の公式に代入する。
$$
\begin{aligned}
\text{var}[\mu_k] &= \mathbb{E}[\mu_k^2] - (\mathbb{E}[\mu_k])^2 \\
&= \frac{\alpha_k(\alpha_k + 1)}{\alpha_0(\alpha_0 + 1)} - \left( \frac{\alpha_k}{\alpha_0} \right)^2 \\
&= \frac{\alpha_k(\alpha_k + 1)\alpha_0 - \alpha_k^2(\alpha_0 + 1)}{\alpha_0^2(\alpha_0 + 1)} \\
&= \frac{\alpha_k (\alpha_k\alpha_0 + \alpha_0 - \alpha_k\alpha_0 - \alpha_k)}{\alpha_0^2(\alpha_0 + 1)} \\
&= \frac{\alpha_k(\alpha_0 - \alpha_k)}{\alpha_0^2(\alpha_0 + 1)}
\end{aligned}
$$
これにより分散が示された。

### 3. 共分散の導出
異なる成分 $j, k$ についての共分散を求めるため、$\mathbb{E}[\mu_j \mu_k]$ を計算する。
この場合、パラメータが $\alpha_j + 1$ および $\alpha_k + 1$ となり、パラメータの和は $\alpha_0 + 2$ になる。

$$
\begin{aligned}
\mathbb{E}[\mu_j \mu_k] &= \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_j)\Gamma(\alpha_k)} \cdot \frac{\Gamma(\alpha_j + 1)\Gamma(\alpha_k + 1)}{\Gamma(\alpha_0 + 2)} \\
&= \frac{\alpha_j \alpha_k}{\alpha_0(\alpha_0 + 1)}
\end{aligned}
$$

共分散の公式 $\text{cov}[\mu_j, \mu_k] = \mathbb{E}[\mu_j \mu_k] - \mathbb{E}[\mu_j]\mathbb{E}[\mu_k]$ に代入する。
$$
\begin{aligned}
\text{cov}[\mu_j, \mu_k] &= \frac{\alpha_j \alpha_k}{\alpha_0(\alpha_0 + 1)} - \frac{\alpha_j}{\alpha_0} \frac{\alpha_k}{\alpha_0} \\
&= \frac{\alpha_j \alpha_k \alpha_0 - \alpha_j \alpha_k (\alpha_0 + 1)}{\alpha_0^2(\alpha_0 + 1)} \\
&= \frac{\alpha_j \alpha_k (\alpha_0 - \alpha_0 - 1)}{\alpha_0^2(\alpha_0 + 1)} \\
&= -\frac{\alpha_j \alpha_k}{\alpha_0^2(\alpha_0 + 1)}
\end{aligned}
$$

以上より、共分散が $-\frac{\alpha_j \alpha_k}{\alpha_0^2(\alpha_0 + 1)}$ となることが証明された。多項分布の場合と同様に、合計が1に制約されているため、変数間に負の相関が存在する。
