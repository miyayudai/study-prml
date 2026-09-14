## Exercise 2.10 (多項分布のモーメント)

多項分布 $\mathrm{Mult}(\mathbf{m} | \boldsymbol{\mu}, N)$ に従う確率変数 $\mathbf{m} = (m_1, \dots, m_K)^T$ について、各成分の期待値、分散、および共分散を導出します。
$m_k$ は $N$ 回の独立な試行のうち、状態 $k$ が観測された回数です。

### 1. 指示変数の導入
解析を簡単にするために、個々の試行の結果を表す**指示変数 (indicator variable)** $x_{nk}$ を導入します。$n$ 回目の試行で状態が $k$ だった場合は $x_{nk}=1$、それ以外の場合は $x_{nk}=0$ とします。
このとき、観測回数 $m_k$ は $N$ 回の試行の和として表せます：
$$ m_k = \sum_{n=1}^N x_{nk} $$

各試行は独立であり、特定の状態 $k$ が出る確率は $\mu_k$ です。したがって $x_{nk}$ について以下の性質が成り立ちます：
1. 期待値: $\mathbb{E}[x_{nk}] = 1 \cdot p(x_{nk}=1) + 0 \cdot p(x_{nk}=0) = \mu_k$
2. 2乗の期待値: $\mathbb{E}[x_{nk}^2] = 1^2 \cdot \mu_k + 0^2 \cdot (1-\mu_k) = \mu_k$
3. 分散: $\mathrm{var}[x_{nk}] = \mathbb{E}[x_{nk}^2] - (\mathbb{E}[x_{nk}])^2 = \mu_k - \mu_k^2 = \mu_k(1-\mu_k)$

### 2. 平均（期待値）の導出
期待値の線形性を用いて $\mathbb{E}[m_k]$ を求めます：
$$ \mathbb{E}[m_k] = \mathbb{E} \left[ \sum_{n=1}^N x_{nk} \right] = \sum_{n=1}^N \mathbb{E}[x_{nk}] = \sum_{n=1}^N \mu_k = N\mu_k $$

### 3. 分散の導出
各試行 $n$ は互いに独立であるため、和の分散は分散の和になります：
$$ \mathrm{var}[m_k] = \mathrm{var} \left[ \sum_{n=1}^N x_{nk} \right] = \sum_{n=1}^N \mathrm{var}[x_{nk}] $$
先に求めた $\mathrm{var}[x_{nk}] = \mu_k(1-\mu_k)$ を代入して：
$$ \mathrm{var}[m_k] = \sum_{n=1}^N \mu_k(1-\mu_k) = N\mu_k(1-\mu_k) $$

### 4. 共分散の導出
異なる状態 $j$ と $k$ ($j \ne k$) について、共分散 $\mathrm{cov}[m_j, m_k]$ を求めます。双線形性より：
$$ \mathrm{cov}[m_j, m_k] = \mathrm{cov} \left[ \sum_{n=1}^N x_{nj}, \sum_{l=1}^N x_{lk} \right] = \sum_{n=1}^N \sum_{l=1}^N \mathrm{cov}[x_{nj}, x_{lk}] $$

異なる試行 ($n \ne l$) の結果は独立なので、$\mathrm{cov}[x_{nj}, x_{lk}] = 0$ です。したがって、同じ試行 ($n = l$) の項のみが残ります：
$$ \mathrm{cov}[m_j, m_k] = \sum_{n=1}^N \mathrm{cov}[x_{nj}, x_{nk}] $$

1回の試行 $n$ における状態 $j$ と $k$ の共分散を求めます：
$$ \mathrm{cov}[x_{nj}, x_{nk}] = \mathbb{E}[x_{nj}x_{nk}] - \mathbb{E}[x_{nj}]\mathbb{E}[x_{nk}] $$
ここで、$j \ne k$ のとき、1回の試行で状態 $j$ と状態 $k$ が同時に起こることはありません（排反事象）。よって、常に $x_{nj}x_{nk} = 0$ となり、$\mathbb{E}[x_{nj}x_{nk}] = 0$ です。
$$ \mathrm{cov}[x_{nj}, x_{nk}] = 0 - \mu_j\mu_k = -\mu_j\mu_k $$

これを和の式に代入します：
$$ \mathrm{cov}[m_j, m_k] = \sum_{n=1}^N (-\mu_j\mu_k) = -N\mu_j\mu_k $$

**解釈:** 共分散が負になるのは、$N$ 回という限られた試行回数の中で、ある状態 $j$ が多く観測されればされるほど、別の状態 $k$ が観測される回数は必然的に減るというトレードオフの関係を直感的に表しています。
