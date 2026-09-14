# Exercise 2.46 (混合ガウス分布の規格化)

## 課題
混合ガウス分布（Gaussian Mixture Model; GMM）の確率密度関数は次のように定義されます。
$$ p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) $$
ここで、各成分のガウス分布は規格化されており、混合係数に関する条件 $\sum_{k=1}^K \pi_k = 1$ が成り立ちます。
このとき、混合ガウス分布 $p(\mathbf{x})$ 全体が規格化されていること（すなわち、全空間にわたる積分が 1 となること）を証明してください。

## 解答と解説
確率密度関数 $p(\mathbf{x})$ を入力空間全体 $\mathbf{x} \in \mathbb{R}^D$ について積分し、その結果が 1 になることを示します。

定義より、積分は次のように記述できます。
$$ \int p(\mathbf{x}) d\mathbf{x} = \int \left( \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right) d\mathbf{x} $$

積分と有限和の順序を交換すると（積分と和の線形性より）、
$$ \int p(\mathbf{x}) d\mathbf{x} = \sum_{k=1}^K \pi_k \left( \int \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) d\mathbf{x} \right) $$
となります。

ここで、各コンポーネント $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ は正当な確率密度関数（多変量ガウス分布）であるため、その全空間での積分は 1 に等しくなります。すなわち、すべての $k \in \{1, \dots, K\}$ において、
$$ \int \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) d\mathbf{x} = 1 $$
が成り立ちます。

この性質を式に代入すると、
$$ \int p(\mathbf{x}) d\mathbf{x} = \sum_{k=1}^K \pi_k (1) = \sum_{k=1}^K \pi_k $$
が得られます。

問題の仮定により、混合係数の総和は 1 である（$\sum_{k=1}^K \pi_k = 1$）ため、最終的に以下が示されます。
$$ \int p(\mathbf{x}) d\mathbf{x} = 1 $$

以上により、混合係数の総和が 1 であることと各ガウス分布が規格化されていることから、混合ガウス分布全体も規格化された確率密度関数であることが証明されました。
