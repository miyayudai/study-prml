# Exercise 2.47 (混合ガウス分布のモーメント)

## 課題
全分散の法則（または期待値と分散の定義）を用いて、混合ガウス分布の期待値と共分散を導出せよ。

## 解答と解説
混合ガウス分布の確率密度関数は以下で与えられます。
$$ p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) $$
ここで、$\sum_{k=1}^K \pi_k = 1$ です。

### 1. 期待値（平均）の導出
期待値の定義より、$\mathbb{E}[\mathbf{x}]$ は次のように計算されます。
$$ \mathbb{E}[\mathbf{x}] = \int \mathbf{x} p(\mathbf{x}) d\mathbf{x} = \int \mathbf{x} \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) d\mathbf{x} $$

積分と和の順序を交換すると、
$$ \mathbb{E}[\mathbf{x}] = \sum_{k=1}^K \pi_k \int \mathbf{x} \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) d\mathbf{x} $$

ここで、積分部分は第 $k$ 成分のガウス分布の平均値 $\boldsymbol{\mu}_k$ を表しています。
$$ \int \mathbf{x} \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) d\mathbf{x} = \boldsymbol{\mu}_k $$

したがって、混合ガウス分布全体の平均 $\mathbb{E}[\mathbf{x}]$ は、各成分の平均の重み付き和となります。
$$ \mathbb{E}[\mathbf{x}] = \sum_{k=1}^K \pi_k \boldsymbol{\mu}_k $$
便宜上、この全体の平均を $\boldsymbol{\mu}$ とおきます。

### 2. 共分散の導出
共分散行列 $\mathrm{cov}[\mathbf{x}]$ の定義は以下の通りです。
$$ \mathrm{cov}[\mathbf{x}] = \mathbb{E}[\mathbf{x}\mathbf{x}^T] - \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^T $$

まず、2次のモーメント $\mathbb{E}[\mathbf{x}\mathbf{x}^T]$ を計算します。
$$ \mathbb{E}[\mathbf{x}\mathbf{x}^T] = \int \mathbf{x}\mathbf{x}^T p(\mathbf{x}) d\mathbf{x} = \sum_{k=1}^K \pi_k \int \mathbf{x}\mathbf{x}^T \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) d\mathbf{x} $$

第 $k$ 成分のガウス分布において、共分散は $\boldsymbol{\Sigma}_k = \mathbb{E}_k[\mathbf{x}\mathbf{x}^T] - \boldsymbol{\mu}_k\boldsymbol{\mu}_k^T$ ですから、次のように書き換えられます。
$$ \int \mathbf{x}\mathbf{x}^T \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) d\mathbf{x} = \boldsymbol{\Sigma}_k + \boldsymbol{\mu}_k\boldsymbol{\mu}_k^T $$

これを代入すると、2次のモーメントは次のようになります。
$$ \mathbb{E}[\mathbf{x}\mathbf{x}^T] = \sum_{k=1}^K \pi_k (\boldsymbol{\Sigma}_k + \boldsymbol{\mu}_k\boldsymbol{\mu}_k^T) $$

全体の共分散行列は、これに $\mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^T$ を引くことで得られます。
$$ \mathrm{cov}[\mathbf{x}] = \sum_{k=1}^K \pi_k (\boldsymbol{\Sigma}_k + \boldsymbol{\mu}_k\boldsymbol{\mu}_k^T) - \boldsymbol{\mu}\boldsymbol{\mu}^T $$

ここで、$\boldsymbol{\mu} = \sum_{k=1}^K \pi_k \boldsymbol{\mu}_k$ であり、また $\sum_{k=1}^K \pi_k = 1$ を利用して $\boldsymbol{\mu}\boldsymbol{\mu}^T = \sum_{k=1}^K \pi_k \boldsymbol{\mu}\boldsymbol{\mu}^T$ と表すことができます。式を整理すると、
$$ \mathrm{cov}[\mathbf{x}] = \sum_{k=1}^K \pi_k \boldsymbol{\Sigma}_k + \sum_{k=1}^K \pi_k \boldsymbol{\mu}_k\boldsymbol{\mu}_k^T - \sum_{k=1}^K \pi_k \boldsymbol{\mu}\boldsymbol{\mu}^T $$
$$ \mathrm{cov}[\mathbf{x}] = \sum_{k=1}^K \pi_k \boldsymbol{\Sigma}_k + \sum_{k=1}^K \pi_k (\boldsymbol{\mu}_k\boldsymbol{\mu}_k^T - \boldsymbol{\mu}\boldsymbol{\mu}^T) $$

ここで第2項のカッコ内に $\boldsymbol{\mu}_k\boldsymbol{\mu}^T - \boldsymbol{\mu}_k\boldsymbol{\mu}^T$ などを足し引きして平方完成の形に変形します。
$$ \sum_{k=1}^K \pi_k (\boldsymbol{\mu}_k\boldsymbol{\mu}_k^T - \boldsymbol{\mu}_k\boldsymbol{\mu}^T - \boldsymbol{\mu}\boldsymbol{\mu}_k^T + \boldsymbol{\mu}\boldsymbol{\mu}^T) $$
※ なぜなら、$\sum \pi_k \boldsymbol{\mu}_k \boldsymbol{\mu}^T = \boldsymbol{\mu}\boldsymbol{\mu}^T$ および $\sum \pi_k \boldsymbol{\mu} \boldsymbol{\mu}_k^T = \boldsymbol{\mu}\boldsymbol{\mu}^T$ が成り立つため、$\boldsymbol{\mu}_k\boldsymbol{\mu}_k^T - \boldsymbol{\mu}\boldsymbol{\mu}^T$ は上式と全く等価になります。

これは次のようにまとめられます。
$$ \sum_{k=1}^K \pi_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T $$

以上より、混合分布の全体の共分散は、各成分の共分散の期待値（成分内分散）と、成分の平均値間の分散（成分間分散）の和として表されます。
$$ \mathrm{cov}[\mathbf{x}] = \sum_{k=1}^K \pi_k \boldsymbol{\Sigma}_k + \sum_{k=1}^K \pi_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T $$

これにより、全分散の法則と同様の構造をもつことが示されました。
