# 演習問題 2.6

## 問題設定
多項分布 (Multinomial distribution) は、パラメータ $N$ と $\boldsymbol{\mu} = (\mu_1, \dots, \mu_K)^T$ を用いて次のように定義される。
$$ \text{Mult}(m_1, \dots, m_K | \boldsymbol{\mu}, N) = \binom{N}{m_1 m_2 \dots m_K} \prod_{k=1}^K \mu_k^{m_k} $$
ここで $\sum_{k=1}^K \mu_k = 1$ および $\sum_{k=1}^K m_k = N$ である。
このとき、各 $m_k$ の期待値、分散、および異なる要素間の共分散がそれぞれ以下になることを示せ。
$$ \mathbb{E}[m_k] = N\mu_k $$
$$ \text{var}[m_k] = N\mu_k(1-\mu_k) $$
$$ \text{cov}[m_j, m_k] = -N\mu_j\mu_k \quad (j \neq k) $$

## 解答と導出

多項分布は $N$ 回の独立な試行を行い、それぞれが確率 $\mu_k$ でクラス $k$ に属する場合の、各クラスの出現回数 $m_k$ の分布である。これを独立な変数の和として定式化することで、容易に導出できる。

### 1. 独立変数による表現
$i$ 回目の試行の結果を 1-of-K 符号化（one-hot エンコーディング）されたベクトル $\mathbf{x}^{(i)}$ で表す。すなわち、$\mathbf{x}^{(i)} = (x_1^{(i)}, \dots, x_K^{(i)})^T$ であり、試行がクラス $k$ だった場合は $x_k^{(i)} = 1$、それ以外は $0$ となる。
すると、各成分は次のような性質を持つベルヌーイ変数に近い振る舞いをする。
$$ p(x_k^{(i)} = 1) = \mu_k, \quad \mathbb{E}[x_k^{(i)}] = \mu_k $$
ある試行 $i$ において、ただ1つの成分のみが1になるため、同じ試行内の異なる成分 $j, k$ ($j \neq k$) については排反である。したがって、
$$ x_j^{(i)} x_k^{(i)} = 0 \quad (j \neq k) $$
このため、
$$ \mathbb{E}[x_j^{(i)} x_k^{(i)}] = 0 $$
が成り立つ。

各クラスの総出現回数 $m_k$ は、これら $N$ 回の試行の和として表される。
$$ m_k = \sum_{i=1}^N x_k^{(i)} $$

### 2. 期待値の導出
期待値の線形性を用いて計算する。
$$
\begin{aligned}
\mathbb{E}[m_k] &= \mathbb{E} \left[ \sum_{i=1}^N x_k^{(i)} \right] \\
&= \sum_{i=1}^N \mathbb{E}[x_k^{(i)}] \\
&= \sum_{i=1}^N \mu_k \\
&= N\mu_k
\end{aligned}
$$

### 3. 分散の導出
各試行 $\mathbf{x}^{(i)}$ は互いに独立であるため、$m_k$ の分散は各 $x_k^{(i)}$ の分散の和になる。
$x_k^{(i)} \in \{0, 1\}$ より $(x_k^{(i)})^2 = x_k^{(i)}$ であるから、
$$ \mathbb{E}[(x_k^{(i)})^2] = \mathbb{E}[x_k^{(i)}] = \mu_k $$
したがって、$x_k^{(i)}$ の分散は、
$$ \text{var}[x_k^{(i)}] = \mathbb{E}[(x_k^{(i)})^2] - (\mathbb{E}[x_k^{(i)}])^2 = \mu_k - \mu_k^2 = \mu_k(1-\mu_k) $$

これを合計して $m_k$ の分散を得る。
$$
\begin{aligned}
\text{var}[m_k] &= \sum_{i=1}^N \text{var}[x_k^{(i)}] \\
&= \sum_{i=1}^N \mu_k(1-\mu_k) \\
&= N\mu_k(1-\mu_k)
\end{aligned}
$$

### 4. 共分散の導出
異なるクラス $j, k$ についての共分散 $\text{cov}[m_j, m_k]$ を求める。
$$ \text{cov}[m_j, m_k] = \mathbb{E}[m_j m_k] - \mathbb{E}[m_j]\mathbb{E}[m_k] $$

まず $\mathbb{E}[m_j m_k]$ を計算する。
$$
\begin{aligned}
\mathbb{E}[m_j m_k] &= \mathbb{E} \left[ \left( \sum_{i=1}^N x_j^{(i)} \right) \left( \sum_{l=1}^N x_k^{(l)} \right) \right] \\
&= \mathbb{E} \left[ \sum_{i=1}^N \sum_{l=1}^N x_j^{(i)} x_k^{(l)} \right]
\end{aligned}
$$

この和を $i = l$ の場合と $i \neq l$ の場合に分ける。
- $i = l$ のとき：同じ試行内で異なるクラスが同時に1になることはないため、$x_j^{(i)} x_k^{(i)} = 0$。したがって期待値も0。
- $i \neq l$ のとき：異なる試行は独立であるため、$\mathbb{E}[x_j^{(i)} x_k^{(l)}] = \mathbb{E}[x_j^{(i)}]\mathbb{E}[x_k^{(l)}] = \mu_j \mu_k$。このような項は $N(N-1)$ 個存在する。

よって、
$$ \mathbb{E}[m_j m_k] = N(N-1)\mu_j \mu_k $$

これを共分散の式に代入する。
$$
\begin{aligned}
\text{cov}[m_j, m_k] &= N(N-1)\mu_j \mu_k - (N\mu_j)(N\mu_k) \\
&= (N^2 - N)\mu_j \mu_k - N^2\mu_j \mu_k \\
&= -N\mu_j \mu_k
\end{aligned}
$$

以上より、共分散は $-N\mu_j \mu_k$ となり、2つのカウント間に負の相関があることが示された。これは合計 $N$ が固定されているため、一方が増えれば他方が減らざるを得ないという性質を反映している。
