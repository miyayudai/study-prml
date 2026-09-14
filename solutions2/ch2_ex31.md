## Exercise 2.31 (事前分散の有効観測数解釈)

$\sigma_0^2 = \sigma^2 / \nu_0$ と置くことで、事前分布を仮想的な $\nu_0$ 個の観測値と解釈できることを示せ。

**【解答】**

ガウス分布に従うデータセット $\mathbf{x} = (x_1, \dots, x_N)^T$ が与えられたとき、平均 $\mu$ が未知であり、分散 $\sigma^2$ が既知であるとする。平均 $\mu$ の事前分布をガウス分布 $p(\mu) = \mathcal{N}(\mu | \mu_0, \sigma_0^2)$ と設定する。
事後分布もガウス分布 $p(\mu | \mathbf{x}) = \mathcal{N}(\mu | \mu_N, \sigma_N^2)$ となり、そのパラメータは以下のように更新される（教科書の式 (2.141), (2.142)）：
$$ \mu_N = \frac{\sigma^2}{N\sigma_0^2 + \sigma^2} \mu_0 + \frac{N\sigma_0^2}{N\sigma_0^2 + \sigma^2} \mu_{\mathrm{ML}} $$
$$ \frac{1}{\sigma_N^2} = \frac{1}{\sigma_0^2} + \frac{N}{\sigma^2} $$
ここで、$\mu_{\mathrm{ML}} = \frac{1}{N} \sum_{n=1}^N x_n$ は標本平均（最尤推定量）である。

ここで、問題の指示に従い、事前分散 $\sigma_0^2$ を $\sigma_0^2 = \frac{\sigma^2}{\nu_0}$ と置く。これを事後分散の逆数（精度）の更新式に代入すると、
$$ \frac{1}{\sigma_N^2} = \frac{\nu_0}{\sigma^2} + \frac{N}{\sigma^2} = \frac{\nu_0 + N}{\sigma^2} $$
したがって、事後分散 $\sigma_N^2$ は次のように表される：
$$ \sigma_N^2 = \frac{\sigma^2}{\nu_0 + N} $$
これは、初期の仮想的な観測数 $\nu_0$ と実際のデータ数 $N$ の和が、全体の有効なデータ数として分散のスケールを決定していることを明確に示している。

次に、この置き換えを事後平均 $\mu_N$ の式にも適用する。$\sigma_0^2 = \sigma^2 / \nu_0$ を代入すると、
$$ \mu_N = \frac{\sigma^2}{N(\sigma^2/\nu_0) + \sigma^2} \mu_0 + \frac{N(\sigma^2/\nu_0)}{N(\sigma^2/\nu_0) + \sigma^2} \mu_{\mathrm{ML}} $$
分母と分子に $\nu_0 / \sigma^2$ を掛けると、
$$ \mu_N = \frac{\nu_0}{\nu_0 + N} \mu_0 + \frac{N}{\nu_0 + N} \mu_{\mathrm{ML}} $$
となる。
また、$\mu_{\mathrm{ML}} = \frac{1}{N}\sum_{n=1}^N x_n$ であることから、上式は以下のように展開できる：
$$ \mu_N = \frac{\nu_0 \mu_0 + \sum_{n=1}^N x_n}{\nu_0 + N} $$
この式は、全体の平均（事後平均）が、**値が $\mu_0$ である仮想的な観測値が $\nu_0$ 個**存在し、そこに実際の観測値 $x_1, \dots, x_N$ が $N$ 個追加されたときの、計 $\nu_0 + N$ 個の観測値の標本平均に他ならないことを示している。

以上より、事前分散パラメータを $\sigma_0^2 = \sigma^2 / \nu_0$ と表すとき、$\nu_0$ は事前分布が持つ情報量を「有効観測数 (effective number of observations)」として解釈できることが示された。
