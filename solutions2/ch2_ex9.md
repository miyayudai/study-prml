## Exercise 2.9 (多項分布の規格化)

$K$ 個の互いに排他的な状態をとる離散変数を考え、それぞれの状態をとる確率を $\boldsymbol{\mu} = (\mu_1, \dots, \mu_K)^T$ とします。$\sum_{k=1}^K \mu_k = 1$ が成り立ちます。
この試行を独立に $N$ 回繰り返したとき、各状態 $k$ が観測された回数 $m_k$ を要素とするベクトル $\mathbf{m} = (m_1, \dots, m_K)^T$ が従う確率分布が**多項分布 (multinomial distribution)**です。
$$ \mathrm{Mult}(\mathbf{m} | \boldsymbol{\mu}, N) = \binom{N}{m_1 m_2 \dots m_K} \prod_{k=1}^K \mu_k^{m_k} = \frac{N!}{m_1! m_2! \dots m_K!} \prod_{k=1}^K \mu_k^{m_k} $$
ここで $\sum_{k=1}^K m_k = N$ です。本演習では、この確率分布の総和が 1 になる（規格化されている）ことを示します。

### 1. 多項定理の導入

証明には、二項定理を一般化した**多項定理 (multinomial theorem)**を用います。多項定理は次のように表されます：
$$ (x_1 + x_2 + \dots + x_K)^N = \sum_{\substack{m_1, \dots, m_K \ge 0 \\ \sum m_k = N}} \frac{N!}{m_1! m_2! \dots m_K!} x_1^{m_1} x_2^{m_2} \dots x_K^{m_K} $$
ここで、総和は $m_1 + m_2 + \dots + m_K = N$ を満たす全ての非負整数の組 $(m_1, \dots, m_K)$ について行われます。

### 2. 規格化の証明

多項分布の確率のすべての可能な観測パターン $\mathbf{m}$ にわたる総和をとります：
$$ \sum_{\mathbf{m}} \mathrm{Mult}(\mathbf{m} | \boldsymbol{\mu}, N) = \sum_{\substack{m_1, \dots, m_K \ge 0 \\ \sum m_k = N}} \frac{N!}{m_1! m_2! \dots m_K!} \mu_1^{m_1} \mu_2^{m_2} \dots \mu_K^{m_K} $$

多項定理において $x_k = \mu_k$ ($k=1, \dots, K$) と代入すると、上記の式はまさに多項定理の右辺の形をしています。
したがって、この総和は次のようにまとめることができます：
$$ \sum_{\mathbf{m}} \mathrm{Mult}(\mathbf{m} | \boldsymbol{\mu}, N) = (\mu_1 + \mu_2 + \dots + \mu_K)^N = \left( \sum_{k=1}^K \mu_k \right)^N $$

ここで、確率の基本的な公理（全ての状態の確率の和は 1）により、$\sum_{k=1}^K \mu_k = 1$ が成り立っています。
これを代入すると：
$$ \sum_{\mathbf{m}} \mathrm{Mult}(\mathbf{m} | \boldsymbol{\mu}, N) = (1)^N = 1 $$

以上により、多項分布が確率分布として正しく規格化されていることが示されました。
