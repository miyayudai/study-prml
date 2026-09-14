# PRML Exercise 2.54

## 問題
von Mises分布 $p(\theta|\theta_0, m)$ について、$\theta$ に関する1階および2階の導関数を計算し、この分布の最頻値（モード）を求めよ。

## 解答と解説
von Mises分布の式は次のように与えられます。
$$ p(\theta|\theta_0, m) = \frac{1}{2\pi I_0(m)} \exp(m \cos(\theta - \theta_0)) $$

この分布の最頻値（分布が最大となる $\theta$）を求めるため、$\theta$ で微分して極値を持つ条件を調べます。

### 1. 1階導関数と極値の条件
$\theta$ についての1階導関数 $p'(\theta|\theta_0, m)$ を計算します。
合成関数の微分則を用いると、
$$ \frac{d}{d\theta} \exp(m \cos(\theta - \theta_0)) = \exp(m \cos(\theta - \theta_0)) \cdot (-m \sin(\theta - \theta_0)) $$
となるので、
$$ p'(\theta|\theta_0, m) = -m \sin(\theta - \theta_0) \cdot \frac{1}{2\pi I_0(m)} \exp(m \cos(\theta - \theta_0)) = -m \sin(\theta - \theta_0) p(\theta|\theta_0, m) $$
となります。

極値の条件は $p'(\theta|\theta_0, m) = 0$ です。
$p(\theta|\theta_0, m) > 0$ かつ $m > 0$ であるため、これが $0$ となる条件は
$$ \sin(\theta - \theta_0) = 0 $$
です。これを満たす $\theta$ は
$$ \theta = \theta_0 + k\pi \quad (k \in \mathbb{Z}) $$
となります。

### 2. 2階導関数と極大値の判定
得られた極値が極大値（最頻値）であるか極小値であるかを判定するため、2階導関数 $p''(\theta|\theta_0, m)$ を計算します。
積の微分則を適用して、
$$ p''(\theta|\theta_0, m) = \frac{d}{d\theta} \left[ -m \sin(\theta - \theta_0) p(\theta|\theta_0, m) \right] $$
$$ = -m \cos(\theta - \theta_0) p(\theta|\theta_0, m) - m \sin(\theta - \theta_0) p'(\theta|\theta_0, m) $$
ここで $p'(\theta|\theta_0, m) = -m \sin(\theta - \theta_0) p(\theta|\theta_0, m)$ を代入すると、
$$ p''(\theta|\theta_0, m) = \left[ -m \cos(\theta - \theta_0) + m^2 \sin^2(\theta - \theta_0) \right] p(\theta|\theta_0, m) $$
となります。

極値となる $\theta = \theta_0 + k\pi$ における2階導関数の符号を評価します。
* **$k$ が偶数の場合（例：$k=0$）**
  $\theta = \theta_0$（$2\pi$ を法として）のとき、$\cos(\theta - \theta_0) = 1$、$\sin(\theta - \theta_0) = 0$ となります。
  代入すると、
  $$ p''(\theta_0|\theta_0, m) = -m p(\theta_0|\theta_0, m) $$
  $m > 0$ かつ $p(\theta_0) > 0$ なので、2階導関数は負（$< 0$）となります。したがって、この点で分布は**極大値**（最大値）をとります。

* **$k$ が奇数の場合（例：$k=1$）**
  $\theta = \theta_0 + \pi$ のとき、$\cos(\theta - \theta_0) = -1$、$\sin(\theta - \theta_0) = 0$ となります。
  代入すると、
  $$ p''(\theta_0+\pi|\theta_0, m) = m p(\theta_0+\pi|\theta_0, m) $$
  となり、2階導関数は正（$> 0$）となります。したがって、この点で分布は**極小値**をとります。

### 結論
極大かつ分布の最大値を与える点は $\theta = \theta_0$（$2\pi$ の整数倍を除く）であることが分かりました。
よって、von Mises分布の最頻値（モード）は $\theta_0$ となります。
