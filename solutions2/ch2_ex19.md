## 演習 2.19: 多変量ガウス分布の平均値

**問題:**
$D$ 次元多変量ガウス分布 $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma})$ に関して、その期待値（平均値）が $\mathbb{E}[\mathbf{x}] = \boldsymbol{\mu}$ であることを示せ。

**解答と解説:**

多変量ガウス分布における $\mathbf{x}$ の期待値は、次のように定義されます。
$$ \mathbb{E}[\mathbf{x}] = \int \mathbf{x} \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) \, \mathrm{d}\mathbf{x} $$

変数変換 $\mathbf{z} = \mathbf{x} - \boldsymbol{\mu}$ を導入します。これにより、$\mathbf{x} = \mathbf{z} + \boldsymbol{\mu}$ となり、積分要素は $\mathrm{d}\mathbf{x} = \mathrm{d}\mathbf{z}$ となります。
これを期待値の式に代入します。
$$ \mathbb{E}[\mathbf{x}] = \int (\mathbf{z} + \boldsymbol{\mu}) \mathcal{N}(\mathbf{z} + \boldsymbol{\mu} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) \, \mathrm{d}\mathbf{z} $$

ここで、ガウス分布の確率密度関数の平行移動を考えます。
$$ \mathcal{N}(\mathbf{z} + \boldsymbol{\mu} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left\{ -\frac{1}{2} (\mathbf{z} + \boldsymbol{\mu} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{z} + \boldsymbol{\mu} - \boldsymbol{\mu}) \right\} $$
$$ = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left\{ -\frac{1}{2} \mathbf{z}^T \boldsymbol{\Sigma}^{-1} \mathbf{z} \right\} $$
これは、平均 0 のガウス分布 $\mathcal{N}(\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma})$ に等しくなります。

したがって、期待値の積分は次のように2つの項に分けることができます。
$$ \mathbb{E}[\mathbf{x}] = \int \mathbf{z} \mathcal{N}(\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma}) \, \mathrm{d}\mathbf{z} + \int \boldsymbol{\mu} \mathcal{N}(\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma}) \, \mathrm{d}\mathbf{z} $$

**第1項の評価:**
被積分関数 $\mathbf{f}(\mathbf{z}) = \mathbf{z} \mathcal{N}(\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma})$ について考えます。
指数部の二次形式 $\mathbf{z}^T \boldsymbol{\Sigma}^{-1} \mathbf{z}$ は、$\mathbf{z}$ を $-\mathbf{z}$ に置き換えても値が変わりません（$(-\mathbf{z})^T \boldsymbol{\Sigma}^{-1} (-\mathbf{z}) = \mathbf{z}^T \boldsymbol{\Sigma}^{-1} \mathbf{z}$）。したがって、確率密度関数は偶関数、すなわち $\mathcal{N}(-\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma}) = \mathcal{N}(\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma})$ です。
これに $\mathbf{z}$（奇関数）を掛けた関数 $\mathbf{f}(\mathbf{z})$ は、奇関数となります。すなわち、
$$ \mathbf{f}(-\mathbf{z}) = (-\mathbf{z}) \mathcal{N}(-\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma}) = -\mathbf{z} \mathcal{N}(\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma}) = -\mathbf{f}(\mathbf{z}) $$
奇関数を原点対称な全空間 $[-\infty, \infty]^D$ で積分すると、その値は $\mathbf{0}$（ゼロベクトル）になります。
$$ \int \mathbf{z} \mathcal{N}(\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma}) \, \mathrm{d}\mathbf{z} = \mathbf{0} $$

**第2項の評価:**
$\boldsymbol{\mu}$ は積分変数 $\mathbf{z}$ に依存しない定数ベクトルであるため、積分の外に出すことができます。
$$ \int \boldsymbol{\mu} \mathcal{N}(\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma}) \, \mathrm{d}\mathbf{z} = \boldsymbol{\mu} \int \mathcal{N}(\mathbf{z} | \mathbf{0}, \boldsymbol{\Sigma}) \, \mathrm{d}\mathbf{z} $$
演習 2.18 で示した通り、ガウス分布の全空間における積分は 1 です（規格化条件）。
したがって、この項は $\boldsymbol{\mu} \cdot 1 = \boldsymbol{\mu}$ となります。

以上の結果を合わせると、
$$ \mathbb{E}[\mathbf{x}] = \mathbf{0} + \boldsymbol{\mu} = \boldsymbol{\mu} $$
となり、多変量ガウス分布の期待値がパラメータ $\boldsymbol{\mu}$ に等しいことが証明されました。
