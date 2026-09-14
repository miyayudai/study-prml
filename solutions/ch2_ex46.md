# 演習問題 2.46

## 問題の概要
混合ガウス分布の最尤推定において、あるコンポーネントの平均がデータ点の一つに一致し、その分散が0に近づくとき、尤度関数が無限大に発散する（特異性）ことを示す。

## 解答

データ集合 $X = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ が与えられたときの混合ガウスモデルの対数尤度関数は次のように表されます。

$$
\ln p(X|\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \sum_{n=1}^N \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
$$

### 特異性の発生
ここで、あるコンポーネント（例として $k=j$）の平均ベクトル $\boldsymbol{\mu}_j$ が、ちょうどデータ集合の中の特定のデータ点 $\mathbf{x}_m$ に一致していると仮定します。すなわち、$\boldsymbol{\mu}_j = \mathbf{x}_m$ とします。

コンポーネント $j$ の共分散行列を単純な等方的なもの $\boldsymbol{\Sigma}_j = \sigma_j^2 \mathbf{I}$ と置きます。
データ点 $\mathbf{x}_m$ に対する尤度の項を考えると、和の内部には $k=j$ の項が含まれます。

$$
\mathcal{N}(\mathbf{x}_m|\boldsymbol{\mu}_j, \sigma_j^2 \mathbf{I}) = \frac{1}{(2\pi \sigma_j^2)^{D/2}} \exp \left( -\frac{1}{2\sigma_j^2} \|\mathbf{x}_m - \boldsymbol{\mu}_j\|^2 \right)
$$

$\boldsymbol{\mu}_j = \mathbf{x}_m$ なので、指数部の $\|\mathbf{x}_m - \boldsymbol{\mu}_j\|^2$ は0になります。したがって指数関数部分は $\exp(0) = 1$ となります。

$$
\mathcal{N}(\mathbf{x}_m|\boldsymbol{\mu}_j, \sigma_j^2 \mathbf{I}) = \frac{1}{(2\pi \sigma_j^2)^{D/2}}
$$

データ点 $\mathbf{x}_m$ からの寄与を含む対数尤度の全体は、次のように下から抑えられます。

$$
\ln p(X) \ge \ln \left( \pi_j \frac{1}{(2\pi \sigma_j^2)^{D/2}} \right) + \sum_{n \neq m} \ln \left( \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_n|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
$$

### 分散が0に近づく極限
ここで、他のパラメータを固定したまま、$\sigma_j \to 0$ の極限をとります。
第1項について：
$$
\lim_{\sigma_j \to 0} \ln \left( \frac{\pi_j}{(2\pi \sigma_j^2)^{D/2}} \right) = +\infty
$$
となります（$\pi_j > 0$ のとき）。

残りの $n \neq m$ のデータ点については、$\mathbf{x}_n \neq \boldsymbol{\mu}_j$ であれば $\sigma_j \to 0$ とともに $k=j$ の項は急速に0に近づきますが、他のコンポーネント $k \neq j$ からの寄与が残るため、対数の中身は0にはならず有限の値を保ちます（データ点が完全に重なっていない限り）。

結果として、尤度関数の全体は $\sigma_j \to 0$ とともに $+\infty$ に発散します。

### 結論と意味
この結果は、混合ガウスモデルにおける最尤推定が、単純に尤度を最大化するアプローチでは病的な解（特異点）に陥る危険性があることを示しています。一つのデータ点に特化した分散0のコンポーネントを作ることで尤度を無限に大きくできるため、大域的な最大値が意味を持たなくなります。これを避けるためには、分散に下限を設けるか、ベイズ的アプローチ（事前分布の導入）によりパラメータを正則化する必要があります。
