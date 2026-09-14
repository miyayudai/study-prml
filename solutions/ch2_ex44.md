# 演習問題 2.44

## 問題の概要
Studentのt分布のパラメータ（$\mu, \lambda, \nu$）に関する最尤推定方程式を導出する。

## 解答

データ集合 $X = \{x_1, \dots, x_N\}$ が独立に同一のStudentのt分布 $\text{St}(x|\mu, \lambda, \nu)$ から生成されたとします。対数尤度関数は次のように与えられます。

$$
\ln p(X|\mu, \lambda, \nu) = \sum_{n=1}^N \ln \text{St}(x_n|\mu, \lambda, \nu)
$$

t分布の確率密度関数は以下の形を持ちます。

$$
\text{St}(x|\mu, \lambda, \nu) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})} \left(\frac{\lambda}{\pi \nu}\right)^{1/2} \left[1 + \frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}
$$

### 1. $\mu$ についての微分
対数尤度を $\mu$ について偏微分し、0とおきます。

$$
\frac{\partial}{\partial \mu} \ln p(X|\mu, \lambda, \nu) = \sum_{n=1}^N \frac{\partial}{\partial \mu} \left( -\frac{\nu+1}{2} \ln \left[1 + \frac{\lambda(x_n-\mu)^2}{\nu}\right] \right)
$$

連鎖律を用いて微分すると、

$$
= \sum_{n=1}^N \left( -\frac{\nu+1}{2} \right) \frac{1}{1 + \frac{\lambda(x_n-\mu)^2}{\nu}} \left( -\frac{2\lambda(x_n-\mu)}{\nu} \right)
$$
$$
= \sum_{n=1}^N \frac{\lambda(\nu+1)}{\nu} \frac{x_n-\mu}{1 + \frac{\lambda(x_n-\mu)^2}{\nu}} = 0
$$

ここで、重み $E_n$ を次のように定義します。

$$
E_n = \frac{\nu+1}{\nu + \lambda(x_n-\mu)^2}
$$

すると方程式は、

$$
\sum_{n=1}^N E_n (x_n - \mu) = 0 \implies \mu = \frac{\sum_{n=1}^N E_n x_n}{\sum_{n=1}^N E_n}
$$

となります。

### 2. $\lambda$ についての微分
同様に $\lambda$ について偏微分します。

$$
\frac{\partial}{\partial \lambda} \ln p(X|\mu, \lambda, \nu) = \sum_{n=1}^N \left[ \frac{1}{2\lambda} - \frac{\nu+1}{2} \frac{\frac{(x_n-\mu)^2}{\nu}}{1 + \frac{\lambda(x_n-\mu)^2}{\nu}} \right] = 0
$$

両辺に $2$ を掛け整理すると、

$$
\frac{N}{\lambda} = \sum_{n=1}^N \frac{\nu+1}{\nu + \lambda(x_n-\mu)^2} (x_n-\mu)^2 = \sum_{n=1}^N E_n (x_n-\mu)^2
$$

したがって、

$$
\frac{1}{\lambda} = \frac{1}{N} \sum_{n=1}^N E_n (x_n-\mu)^2
$$

これらの結果は、重み $E_n$ がパラメータ $\mu, \lambda$ に依存しているため、閉じた形の解ではなく、EMアルゴリズムなどを用いた反復的な更新が必要であることを示しています。
