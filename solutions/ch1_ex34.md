# 演習問題 1.34

## 問題の概要
連続確率変数 $x$ の微分エントロピーを最大化する分布が、平均 $\mu$ と分散 $\sigma^2$ が与えられた制約のもとではガウス分布になることを、ラグランジュの未定乗数法を用いて証明する。

## 証明
連続変数 $x$ の分布 $p(x)$ に対する微分エントロピー $H[x]$ は次のように定義される。
$$
H[x] = -\int_{-\infty}^{\infty} p(x) \ln p(x) dx
$$
このエントロピーを以下の3つの制約条件のもとで最大化する。
1. 確率分布の正規化条件：
$$ \int_{-\infty}^{\infty} p(x) dx = 1 $$
2. 平均が $\mu$ であること：
$$ \int_{-\infty}^{\infty} x p(x) dx = \mu $$
3. 分散が $\sigma^2$ であること：
$$ \int_{-\infty}^{\infty} (x-\mu)^2 p(x) dx = \sigma^2 $$

ラグランジュの未定乗数 $\lambda_1, \lambda_2, \lambda_3$ を導入し、ラグランジュ関数（汎関数）を次のように設定する。
$$
L(p) = -\int p(x) \ln p(x) dx + \lambda_1 \left( \int p(x) dx - 1 \right) + \lambda_2 \left( \int x p(x) dx - \mu \right) + \lambda_3 \left( \int (x-\mu)^2 p(x) dx - \sigma^2 \right)
$$
変分法を用いて、$p(x)$ に対する $L$ の変分（関数微分）を $0$ と置く。
$$
\frac{\delta L}{\delta p(x)} = -\ln p(x) - 1 + \lambda_1 + \lambda_2 x + \lambda_3 (x-\mu)^2 = 0
$$
これを $p(x)$ について解くと、
$$
p(x) = \exp \left( -1 + \lambda_1 + \lambda_2 x + \lambda_3 (x-\mu)^2 \right)
$$
となる。これを平方完成を用いて整理すると、ガウス関数の形になることが予想できる。指数部分を $x-\mu$ で整理するため、$\lambda_2 x = \lambda_2 (x-\mu) + \lambda_2 \mu$ と書き換えると、
$$
p(x) = \exp \left( \lambda_3 (x-\mu)^2 + \lambda_2 (x-\mu) + (-1 + \lambda_1 + \lambda_2 \mu) \right)
$$
ガウス分布の形 $p(x) \propto \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ と比較すると、対称性（制約 2 と 3）から $x-\mu$ の1次の項は不要であるため、$\lambda_2 = 0$ となることがわかる。（厳密には、$\int x p(x) dx = \mu$ の制約を満たすために $\lambda_2 = 0$ が要求される。）
したがって、
$$
p(x) = \exp \left( \lambda_3 (x-\mu)^2 + \lambda_1 - 1 \right)
$$
制約条件を満たすように定数 $\lambda_3$ と $\lambda_1$ を決定する。正規化条件と分散の条件より、$\lambda_3 = -\frac{1}{2\sigma^2}$ となり、正規化係数は $\exp(\lambda_1 - 1) = \frac{1}{\sqrt{2\pi\sigma^2}}$ となる。
これを代入すると、
$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(x-\mu)^2}{2\sigma^2} \right)
$$
が得られる。これは平均 $\mu$、分散 $\sigma^2$ のガウス分布 $\mathcal{N}(x|\mu, \sigma^2)$ そのものである。
以上より、与えられた平均と分散の制約のもとで微分エントロピーを最大化するのはガウス分布であることが示された。
