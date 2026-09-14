## Exercise 2.24 (線形ガウスモデルの周辺分布)

**問題:**
線形ガウスモデルにおける周辺分布 $p(\mathbf{y})$ の平均と共分散を導出せよ。

**解答と解説:**

線形ガウスモデルでは、潜在変数 $\mathbf{x}$ と観測変数 $\mathbf{y}$ の間に以下の関係が仮定されます。
$$ p(\mathbf{x}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1}) $$
$$ p(\mathbf{y} | \mathbf{x}) = \mathcal{N}(\mathbf{y} | \mathbf{A}\mathbf{x} + \mathbf{b}, \mathbf{L}^{-1}) $$
ここで、$\boldsymbol{\Lambda}$ と $\mathbf{L}$ は精度行列を表します。

目的は、周辺分布 $p(\mathbf{y}) = \int p(\mathbf{y} | \mathbf{x}) p(\mathbf{x}) d\mathbf{x}$ の平均と共分散を求めることです。周辺分布もガウス分布になることが知られているため、その期待値と分散（共分散）を直接計算することで導出できます。

まず、$p(\mathbf{y} | \mathbf{x})$ は $\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{b} + \boldsymbol{\epsilon}$ と等価であり、$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{L}^{-1})$ は $\mathbf{x}$ と独立なノイズです。

**1. 平均の導出**
$\mathbf{y}$ の期待値 $\mathbb{E}[\mathbf{y}]$ は、期待値の線形性を用いて計算できます。
$$ \mathbb{E}[\mathbf{y}] = \mathbb{E}[\mathbf{A}\mathbf{x} + \mathbf{b} + \boldsymbol{\epsilon}] = \mathbf{A}\mathbb{E}[\mathbf{x}] + \mathbf{b} + \mathbb{E}[\boldsymbol{\epsilon}] $$
$\mathbb{E}[\mathbf{x}] = \boldsymbol{\mu}$ であり、$\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$ であるため、
$$ \mathbb{E}[\mathbf{y}] = \mathbf{A}\boldsymbol{\mu} + \mathbf{b} $$
となります。

**2. 共分散の導出**
$\mathbf{y}$ の共分散行列 $\mathrm{cov}[\mathbf{y}]$ を求めます。$\mathbf{x}$ と $\boldsymbol{\epsilon}$ は独立であるため、和の共分散は共分散の和になります。
$$ \mathrm{cov}[\mathbf{y}] = \mathrm{cov}[\mathbf{A}\mathbf{x} + \mathbf{b} + \boldsymbol{\epsilon}] = \mathrm{cov}[\mathbf{A}\mathbf{x}] + \mathrm{cov}[\boldsymbol{\epsilon}] $$
定数ベクトル $\mathbf{b}$ は分散を持たないため共分散に影響しません。また、$\mathrm{cov}[\mathbf{A}\mathbf{x}] = \mathbf{A} \mathrm{cov}[\mathbf{x}] \mathbf{A}^T$ となります。
$\mathrm{cov}[\mathbf{x}] = \boldsymbol{\Lambda}^{-1}$ であり、$\mathrm{cov}[\boldsymbol{\epsilon}] = \mathbf{L}^{-1}$ なので、代入すると、
$$ \mathrm{cov}[\mathbf{y}] = \mathbf{A} \boldsymbol{\Lambda}^{-1} \mathbf{A}^T + \mathbf{L}^{-1} $$
となります。

したがって、周辺分布 $p(\mathbf{y})$ は平均 $\mathbf{A}\boldsymbol{\mu} + \mathbf{b}$、共分散 $\mathbf{L}^{-1} + \mathbf{A}\boldsymbol{\Lambda}^{-1}\mathbf{A}^T$ のガウス分布になります。
$$ p(\mathbf{y}) = \mathcal{N}(\mathbf{y} | \mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{L}^{-1} + \mathbf{A}\boldsymbol{\Lambda}^{-1}\mathbf{A}^T) $$
