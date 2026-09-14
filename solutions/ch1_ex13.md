# 演習問題 1.13

## 問題設定
演習問題 1.11 で求めた分散の最尤推定量 $\sigma_{ML}^2 = \frac{1}{N} \sum_{n=1}^N (x_n - \mu_{ML})^2$ について、その期待値を計算し、真の分散 $\sigma^2$ との間にあるバイアスを明らかにする。具体的には、以下を証明する：
$$ \mathbb{E}[\sigma_{ML}^2] = \frac{N-1}{N} \sigma^2 $$

## 解答
まず、分散の最尤推定量の式を展開する。ここで $\mu_{ML} = \frac{1}{N} \sum_{n=1}^N x_n$ である。
$$ \sigma_{ML}^2 = \frac{1}{N} \sum_{n=1}^N (x_n - \mu_{ML})^2 $$
$$ = \frac{1}{N} \sum_{n=1}^N \left( x_n^2 - 2x_n\mu_{ML} + \mu_{ML}^2 \right) $$
$$ = \frac{1}{N} \sum_{n=1}^N x_n^2 - \frac{2}{N}\mu_{ML} \sum_{n=1}^N x_n + \frac{1}{N} \sum_{n=1}^N \mu_{ML}^2 $$
ここで、$\sum_{n=1}^N x_n = N\mu_{ML}$ であることを用いると：
$$ = \frac{1}{N} \sum_{n=1}^N x_n^2 - 2\mu_{ML}^2 + \mu_{ML}^2 = \frac{1}{N} \sum_{n=1}^N x_n^2 - \mu_{ML}^2 $$

この式全体の期待値をとる：
$$ \mathbb{E}[\sigma_{ML}^2] = \mathbb{E} \left[ \frac{1}{N} \sum_{n=1}^N x_n^2 - \mu_{ML}^2 \right] = \frac{1}{N} \sum_{n=1}^N \mathbb{E}[x_n^2] - \mathbb{E}[\mu_{ML}^2] $$

### 1. 第1項の評価
演習問題 1.12 より、 $\mathbb{E}[x_n^2] = \mu^2 + \sigma^2$ であるから：
$$ \frac{1}{N} \sum_{n=1}^N \mathbb{E}[x_n^2] = \frac{1}{N} \sum_{n=1}^N (\mu^2 + \sigma^2) = \mu^2 + \sigma^2 $$

### 2. 第2項の評価
次に、標本平均の二乗の期待値 $\mathbb{E}[\mu_{ML}^2]$ を計算する。
$$ \mathbb{E}[\mu_{ML}^2] = \mathbb{E} \left[ \left( \frac{1}{N} \sum_{n=1}^N x_n \right) \left( \frac{1}{N} \sum_{m=1}^N x_m \right) \right] $$
$$ = \frac{1}{N^2} \sum_{n=1}^N \sum_{m=1}^N \mathbb{E}[x_n x_m] $$
ここで、演習問題 1.12 の結果 $\mathbb{E}[x_n x_m] = \mu^2 + I_{nm} \sigma^2$ を代入する：
$$ = \frac{1}{N^2} \sum_{n=1}^N \sum_{m=1}^N (\mu^2 + I_{nm} \sigma^2) $$
和を分けると、
$$ = \frac{1}{N^2} \left( \sum_{n=1}^N \sum_{m=1}^N \mu^2 \right) + \frac{1}{N^2} \left( \sum_{n=1}^N \sum_{m=1}^N I_{nm} \sigma^2 \right) $$
二重和の第1項は $N^2$ 個の $\mu^2$ の和である。第2項は $I_{nm} = 1$ となる $n=m$ の項のみが残るため、$N$ 個の $\sigma^2$ の和となる。
$$ = \frac{1}{N^2} (N^2 \mu^2) + \frac{1}{N^2} (N \sigma^2) = \mu^2 + \frac{1}{N} \sigma^2 $$

### 3. まとめる
評価した第1項と第2項を元の式に代入する：
$$ \mathbb{E}[\sigma_{ML}^2] = (\mu^2 + \sigma^2) - \left( \mu^2 + \frac{1}{N} \sigma^2 \right) $$
$$ = \sigma^2 - \frac{1}{N} \sigma^2 = \frac{N-1}{N} \sigma^2 $$

以上より、分散の最尤推定量 $\sigma_{ML}^2$ の期待値が $\frac{N-1}{N} \sigma^2$ となることが示された。この結果は、最尤推定量が真の分散を過小評価する（バイアスを持つ）ことを数学的に説明している。
