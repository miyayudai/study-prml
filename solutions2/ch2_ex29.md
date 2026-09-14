## Exercise 2.29 (逐次分散推定式の導出)

**【問題】**
分散の逐次更新式
$$ \sigma_{(N)}^2 = \sigma_{(N-1)}^2 + \frac{1}{N} \left( (x_N - \mu)^2 - \sigma_{(N-1)}^2 \right) $$
を導出せよ。（$\mu$ は既知とする）

**【解答】**
$N$ 個のデータに対する分散の最尤推定量（$\mu$ が既知の場合）は以下のように定義されます。
$$ \sigma_{(N)}^2 = \frac{1}{N} \sum_{n=1}^N (x_n - \mu)^2 $$
ここで、直前までの $N-1$ 個のデータにおける分散は以下になります。
$$ \sigma_{(N-1)}^2 = \frac{1}{N-1} \sum_{n=1}^{N-1} (x_n - \mu)^2 $$

和の定義を $N-1$ 個の部分と最後の 1 個（$N$ 番目）に分解して整理します。
$$
\begin{align*}
\sigma_{(N)}^2 
&= \frac{1}{N} \sum_{n=1}^N (x_n - \mu)^2 \\
&= \frac{1}{N} \left( \sum_{n=1}^{N-1} (x_n - \mu)^2 + (x_N - \mu)^2 \right)
\end{align*}
$$
ここで、第一項の和は $N-1$ 個のデータに対する分散を用いて表すことができます。
$$ \sum_{n=1}^{N-1} (x_n - \mu)^2 = (N-1)\sigma_{(N-1)}^2 $$
これを上の式に代入します。
$$
\begin{align*}
\sigma_{(N)}^2 
&= \frac{1}{N} \left( (N-1)\sigma_{(N-1)}^2 + (x_N - \mu)^2 \right) \\
&= \frac{N-1}{N}\sigma_{(N-1)}^2 + \frac{1}{N}(x_N - \mu)^2 \\
&= \left( 1 - \frac{1}{N} \right)\sigma_{(N-1)}^2 + \frac{1}{N}(x_N - \mu)^2 \\
&= \sigma_{(N-1)}^2 - \frac{1}{N}\sigma_{(N-1)}^2 + \frac{1}{N}(x_N - \mu)^2 \\
&= \sigma_{(N-1)}^2 + \frac{1}{N} \left( (x_N - \mu)^2 - \sigma_{(N-1)}^2 \right)
\end{align*}
$$

以上により、逐次分散推定の更新式が導出されました。この形は Robbins-Monro アルゴリズムの更新式
$$ \theta^{(N)} = \theta^{(N-1)} + a_N \left( z(\theta^{(N-1)}) \right) $$
と一致しており、新しい観測データ $(x_N - \mu)^2$ と現在の推定値 $\sigma_{(N-1)}^2$ との誤差によって推定値を補正していく形になっていることが分かります。
