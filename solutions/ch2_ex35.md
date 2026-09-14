# 演習問題 2.35

## 問題
結果 (2.59) を用いて (2.62) を証明せよ。次に、結果 (2.59) と (2.62) を用いて、
$$
\mathbb{E}[\mathbf{x}_n \mathbf{x}_m^T] = \boldsymbol{\mu} \boldsymbol{\mu}^T + I_{nm} \mathbf{\Sigma} \quad (2.291)
$$
を示せ。ここで、$\mathbf{x}_n$ は平均 $\boldsymbol{\mu}$、共分散 $\mathbf{\Sigma}$ のガウス分布からサンプリングされたデータ点を示し、$I_{nm}$ は単位行列の $(n, m)$ 成分（クロネッカーのデルタ $\delta_{nm}$）を示す。これを用いて、結果 (2.124) を証明せよ。

## 解答

### 1. (2.62) の証明
式 (2.59) は確率変数の平均が $\mathbb{E}[\mathbf{x}] = \boldsymbol{\mu}$ であることを示しています。
共分散行列の定義より、
$$
\mathbf{\Sigma} = \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T]
$$
この期待値の中身を展開すると、
$$
(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T = \mathbf{x}\mathbf{x}^T - \mathbf{x}\boldsymbol{\mu}^T - \boldsymbol{\mu}\mathbf{x}^T + \boldsymbol{\mu}\boldsymbol{\mu}^T
$$
となります。期待値は線形演算であるため、各項の期待値を取ります。$\boldsymbol{\mu}$ は定数ベクトルであることに注意すると、
$$
\mathbb{E}[\mathbf{x}\mathbf{x}^T] - \mathbb{E}[\mathbf{x}]\boldsymbol{\mu}^T - \boldsymbol{\mu}\mathbb{E}[\mathbf{x}^T] + \boldsymbol{\mu}\boldsymbol{\mu}^T
$$
式 (2.59) より $\mathbb{E}[\mathbf{x}] = \boldsymbol{\mu}$ を代入すると、
$$
= \mathbb{E}[\mathbf{x}\mathbf{x}^T] - \boldsymbol{\mu}\boldsymbol{\mu}^T - \boldsymbol{\mu}\boldsymbol{\mu}^T + \boldsymbol{\mu}\boldsymbol{\mu}^T = \mathbb{E}[\mathbf{x}\mathbf{x}^T] - \boldsymbol{\mu}\boldsymbol{\mu}^T
$$
これが $\mathbf{\Sigma}$ に等しいため、移項して式 (2.62) が得られます。
$$
\mathbb{E}[\mathbf{x}\mathbf{x}^T] = \boldsymbol{\mu} \boldsymbol{\mu}^T + \mathbf{\Sigma}
$$

### 2. (2.291) の証明
異なるデータ点 $\mathbf{x}_n$ と $\mathbf{x}_m$ は独立に同一の分布（i.i.d.）から抽出されているとします。
- $n \neq m$ のとき（$I_{nm} = 0$）：$\mathbf{x}_n$ と $\mathbf{x}_m$ は独立なので、期待値は積に分解できます。
  $$
  \mathbb{E}[\mathbf{x}_n \mathbf{x}_m^T] = \mathbb{E}[\mathbf{x}_n]\mathbb{E}[\mathbf{x}_m^T] = \boldsymbol{\mu} \boldsymbol{\mu}^T
  $$
- $n = m$ のとき（$I_{nm} = 1$）：先ほど証明した (2.62) をそのまま用います。
  $$
  \mathbb{E}[\mathbf{x}_n \mathbf{x}_n^T] = \boldsymbol{\mu} \boldsymbol{\mu}^T + \mathbf{\Sigma}
  $$
これらをひとつの式にまとめると、単位行列の要素 $I_{nm}$（または $\delta_{nm}$）を用いて次のように書けます。
$$
\mathbb{E}[\mathbf{x}_n \mathbf{x}_m^T] = \boldsymbol{\mu} \boldsymbol{\mu}^T + I_{nm} \mathbf{\Sigma}
$$
これが式 (2.291) です。

### 3. (2.124) の証明
式 (2.124) は最尤推定量の期待値に関するものです。
まず、標本平均 $\boldsymbol{\mu}_{ML} = \frac{1}{N} \sum_{n=1}^N \mathbf{x}_n$ の期待値は、
$$
\mathbb{E}[\boldsymbol{\mu}_{ML}] = \mathbb{E}\left[ \frac{1}{N} \sum_{n=1}^N \mathbf{x}_n \right] = \frac{1}{N} \sum_{n=1}^N \mathbb{E}[\mathbf{x}_n] = \frac{1}{N} \sum_{n=1}^N \boldsymbol{\mu} = \boldsymbol{\mu}
$$
となり、不偏推定量であることが分かります。

次に、標本共分散 $\mathbf{\Sigma}_{ML}$ の期待値を計算します。$\mathbf{\Sigma}_{ML}$ の定義を展開します。
$$
\mathbf{\Sigma}_{ML} = \frac{1}{N} \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu}_{ML})(\mathbf{x}_n - \boldsymbol{\mu}_{ML})^T
$$
$$
= \frac{1}{N} \sum_{n=1}^N (\mathbf{x}_n \mathbf{x}_n^T - \mathbf{x}_n \boldsymbol{\mu}_{ML}^T - \boldsymbol{\mu}_{ML} \mathbf{x}_n^T + \boldsymbol{\mu}_{ML} \boldsymbol{\mu}_{ML}^T)
$$
ここで、$\sum_{n=1}^N \mathbf{x}_n = N \boldsymbol{\mu}_{ML}$ を用いると、
$$
= \left( \frac{1}{N} \sum_{n=1}^N \mathbf{x}_n \mathbf{x}_n^T \right) - \boldsymbol{\mu}_{ML} \boldsymbol{\mu}_{ML}^T - \boldsymbol{\mu}_{ML} \boldsymbol{\mu}_{ML}^T + \boldsymbol{\mu}_{ML} \boldsymbol{\mu}_{ML}^T
= \left( \frac{1}{N} \sum_{n=1}^N \mathbf{x}_n \mathbf{x}_n^T \right) - \boldsymbol{\mu}_{ML} \boldsymbol{\mu}_{ML}^T
$$

この式の期待値を取ります。
$$
\mathbb{E}[\mathbf{\Sigma}_{ML}] = \frac{1}{N} \sum_{n=1}^N \mathbb{E}[\mathbf{x}_n \mathbf{x}_n^T] - \mathbb{E}[\boldsymbol{\mu}_{ML} \boldsymbol{\mu}_{ML}^T]
$$
第一項は、(2.62) より $\mathbb{E}[\mathbf{x}_n \mathbf{x}_n^T] = \boldsymbol{\mu}\boldsymbol{\mu}^T + \mathbf{\Sigma}$ であるため、
$$
\frac{1}{N} \sum_{n=1}^N (\boldsymbol{\mu}\boldsymbol{\mu}^T + \mathbf{\Sigma}) = \boldsymbol{\mu}\boldsymbol{\mu}^T + \mathbf{\Sigma}
$$
第二項について、$\boldsymbol{\mu}_{ML}$ の定義と式 (2.291) を用いて計算します。
$$
\mathbb{E}[\boldsymbol{\mu}_{ML} \boldsymbol{\mu}_{ML}^T] = \mathbb{E} \left[ \left(\frac{1}{N} \sum_{n=1}^N \mathbf{x}_n\right) \left(\frac{1}{N} \sum_{m=1}^N \mathbf{x}_m^T\right) \right] = \frac{1}{N^2} \sum_{n=1}^N \sum_{m=1}^N \mathbb{E}[\mathbf{x}_n \mathbf{x}_m^T]
$$
$$
= \frac{1}{N^2} \sum_{n=1}^N \sum_{m=1}^N (\boldsymbol{\mu} \boldsymbol{\mu}^T + I_{nm} \mathbf{\Sigma})
$$
総和を計算すると、$\boldsymbol{\mu} \boldsymbol{\mu}^T$ の項は $N \times N = N^2$ 個あり、$I_{nm} \mathbf{\Sigma}$ の項は $n=m$ のときのみ $1$ となるため $N$ 個あります。
$$
= \frac{1}{N^2} (N^2 \boldsymbol{\mu} \boldsymbol{\mu}^T + N \mathbf{\Sigma}) = \boldsymbol{\mu} \boldsymbol{\mu}^T + \frac{1}{N} \mathbf{\Sigma}
$$

これらを $\mathbb{E}[\mathbf{\Sigma}_{ML}]$ の式に代入します。
$$
\mathbb{E}[\mathbf{\Sigma}_{ML}] = (\boldsymbol{\mu}\boldsymbol{\mu}^T + \mathbf{\Sigma}) - \left( \boldsymbol{\mu} \boldsymbol{\mu}^T + \frac{1}{N} \mathbf{\Sigma} \right)
= \mathbf{\Sigma} - \frac{1}{N} \mathbf{\Sigma} = \frac{N-1}{N} \mathbf{\Sigma}
$$

これにより、最尤推定量の期待値に関する式 (2.124) が証明されました。
