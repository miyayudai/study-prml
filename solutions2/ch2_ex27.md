## Exercise 2.27 (標本共分散のバイアス)

**【問題】**
最尤共分散の期待値が $\mathbb{E}[\boldsymbol{\Sigma}_{\mathrm{ML}}] = \frac{N-1}{N}\boldsymbol{\Sigma}$ となることを証明せよ。

**【解答】**
最尤推定量 $\boldsymbol{\Sigma}_{\mathrm{ML}}$ は次のように定義されます。
$$ \boldsymbol{\Sigma}_{\mathrm{ML}} = \frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})^T $$
ここで、データ点は互いに独立であり、真の平均を $\boldsymbol{\mu}$、真の共分散を $\boldsymbol{\Sigma}$ とします。すなわち、
$$ \mathbb{E}[\mathbf{x}_n] = \boldsymbol{\mu}, \quad \mathbb{E}[(\mathbf{x}_n-\boldsymbol{\mu})(\mathbf{x}_m-\boldsymbol{\mu})^T] = \delta_{nm}\boldsymbol{\Sigma} $$
ここで $\delta_{nm}$ はクロネッカーのデルタです。

まず、各データ点から標本平均 $\boldsymbol{\mu}_{\mathrm{ML}}$ を引いた項を、真の平均 $\boldsymbol{\mu}$ を基準にして変形します。
$$ \mathbf{x}_n - \boldsymbol{\mu}_{\mathrm{ML}} = (\mathbf{x}_n - \boldsymbol{\mu}) - (\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu}) $$
これを用いて $\boldsymbol{\Sigma}_{\mathrm{ML}}$ の中身を展開します。
$$
\begin{align*}
\sum_{n=1}^N (\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})^T 
&= \sum_{n=1}^N \left\{ (\mathbf{x}_n - \boldsymbol{\mu}) - (\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu}) \right\} \left\{ (\mathbf{x}_n - \boldsymbol{\mu}) - (\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu}) \right\}^T \\
&= \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T - 2 \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})^T + \sum_{n=1}^N (\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})(\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})^T
\end{align*}
$$
ここで、$\sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu}) = N(\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})$ であることを用いると、第二項は $-2N(\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})(\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})^T$ となり、第三項は $N(\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})(\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})^T$ となります。これらを整理すると、
$$ \sum_{n=1}^N (\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})^T = \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T - N(\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})(\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})^T $$

次に、この式の両辺の期待値をとります。
$$ \mathbb{E}\left[ (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T \right] = \boldsymbol{\Sigma} $$
であるため、第一項の期待値は $N\boldsymbol{\Sigma}$ となります。
続いて、第二項の期待値を評価します。標本平均と真の平均の差の共分散は、
$$
\begin{align*}
\mathbb{E}\left[ (\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})(\boldsymbol{\mu}_{\mathrm{ML}} - \boldsymbol{\mu})^T \right] 
&= \mathbb{E}\left[ \left( \frac{1}{N}\sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu}) \right) \left( \frac{1}{N}\sum_{m=1}^N (\mathbf{x}_m - \boldsymbol{\mu}) \right)^T \right] \\
&= \frac{1}{N^2} \sum_{n=1}^N \sum_{m=1}^N \mathbb{E}\left[ (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_m - \boldsymbol{\mu})^T \right] \\
&= \frac{1}{N^2} \sum_{n=1}^N \boldsymbol{\Sigma} = \frac{1}{N}\boldsymbol{\Sigma}
\end{align*}
$$
となります（異るインデックスの項は無相関なので $0$ になります）。

したがって、展開式の期待値は次のようになります。
$$ \mathbb{E}\left[ \sum_{n=1}^N (\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})(\mathbf{x}_n-\boldsymbol{\mu}_{\mathrm{ML}})^T \right] = N\boldsymbol{\Sigma} - N\left(\frac{1}{N}\boldsymbol{\Sigma}\right) = (N - 1)\boldsymbol{\Sigma} $$

最後に、これを $N$ で割ることで $\boldsymbol{\Sigma}_{\mathrm{ML}}$ の期待値が得られます。
$$ \mathbb{E}[\boldsymbol{\Sigma}_{\mathrm{ML}}] = \frac{1}{N} (N - 1)\boldsymbol{\Sigma} = \frac{N - 1}{N}\boldsymbol{\Sigma} $$
以上により、最尤推定量 $\boldsymbol{\Sigma}_{\mathrm{ML}}$ は真の共分散 $\boldsymbol{\Sigma}$ を過小評価するバイアスを持っており、不偏推定量とするためには $1/(N-1)$ で割る必要があることが示されました。
