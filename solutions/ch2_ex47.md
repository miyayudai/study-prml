# 演習問題 2.47

## 問題の概要
ベルヌーイ分布の混合モデルについて、ラグランジュの未定乗数法を用いて混合係数 $\pi_k$ の最尤推定の更新式を導出する。

## 解答

ベルヌーイ分布の混合モデルの対数尤度は次のように与えられます。

$$
\ln p(\mathbf{X}|\boldsymbol{\mu}, \boldsymbol{\pi}) = \sum_{n=1}^N \ln \left\{ \sum_{k=1}^K \pi_k p(\mathbf{x}_n|\boldsymbol{\mu}_k) \right\}
$$

ここで、$p(\mathbf{x}_n|\boldsymbol{\mu}_k)$ はパラメータ $\boldsymbol{\mu}_k$ を持つベルヌーイ分布です。
混合係数 $\pi_k$ は確率の公理を満たす必要があるため、制約条件 $\sum_{k=1}^K \pi_k = 1$ が課されます。

### ラグランジュ関数の構築
この制約条件付き最適化問題を解くために、ラグランジュ乗数 $\lambda$ を用いてラグランジュ関数を構築します。

$$
L(\boldsymbol{\pi}, \lambda) = \sum_{n=1}^N \ln \left\{ \sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j) \right\} + \lambda \left( \sum_{k=1}^K \pi_k - 1 \right)
$$

### $\pi_k$ についての微分
ラグランジュ関数 $L$ を $\pi_k$ について偏微分し、0と置きます。

$$
\frac{\partial L}{\partial \pi_k} = \sum_{n=1}^N \frac{p(\mathbf{x}_n|\boldsymbol{\mu}_k)}{\sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j)} + \lambda = 0
$$

ここで、事後確率（負担率）$\gamma(z_{nk})$ を次のように定義します。

$$
\gamma(z_{nk}) = \frac{\pi_k p(\mathbf{x}_n|\boldsymbol{\mu}_k)}{\sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j)}
$$

この定義式を変形して微分の式に代入したいのですが、微分の式には分子に $\pi_k$ がありません。そこで、微分の式の両辺に $\pi_k$ を掛けます。

$$
\pi_k \frac{\partial L}{\partial \pi_k} = \sum_{n=1}^N \frac{\pi_k p(\mathbf{x}_n|\boldsymbol{\mu}_k)}{\sum_{j=1}^K \pi_j p(\mathbf{x}_n|\boldsymbol{\mu}_j)} + \lambda \pi_k = 0
$$

第1項はまさに負担率 $\gamma(z_{nk})$ の和になります。

$$
\sum_{n=1}^N \gamma(z_{nk}) + \lambda \pi_k = 0
$$

### ラグランジュ乗数 $\lambda$ の決定
コンポーネント $k$ に割り当てられた実効的なデータ数（負担率の和）を $N_k$ とおきます。

$$
N_k = \sum_{n=1}^N \gamma(z_{nk})
$$

すると式は $N_k + \lambda \pi_k = 0$ となります。
この式をすべての $k=1, \dots, K$ について足し合わせます。

$$
\sum_{k=1}^K N_k + \lambda \sum_{k=1}^K \pi_k = 0
$$

負担率の定義から、$\sum_{k=1}^K \gamma(z_{nk}) = 1$ であるため、
$$
\sum_{k=1}^K N_k = \sum_{k=1}^K \sum_{n=1}^N \gamma(z_{nk}) = \sum_{n=1}^N 1 = N
$$
となります。また制約条件より $\sum_{k=1}^K \pi_k = 1$ です。したがって、

$$
N + \lambda \cdot 1 = 0 \implies \lambda = -N
$$

### 最終的な更新式
$\lambda = -N$ を $N_k + \lambda \pi_k = 0$ に代入します。

$$
N_k - N \pi_k = 0 \implies \pi_k = \frac{N_k}{N}
$$

すなわち、

$$
\pi_k = \frac{1}{N} \sum_{n=1}^N \gamma(z_{nk})
$$

となり、混合係数 $\pi_k$ は、すべてのデータ点に対するコンポーネント $k$ の負担率の平均として更新されることが導出されました。これは混合ガウスモデルの場合と全く同じ形をしています。
