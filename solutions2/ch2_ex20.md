## 演習 2.20: 多変量ガウス分布の二次モーメント

**問題:**
$D$ 次元多変量ガウス分布 $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma})$ において、二次モーメント $\mathbb{E}[\mathbf{x}\mathbf{x}^T]$ が次式で与えられることを証明せよ。
$$ \mathbb{E}[\mathbf{x}\mathbf{x}^T] = \boldsymbol{\mu}\boldsymbol{\mu}^T + \boldsymbol{\Sigma} $$

**解答と解説:**

二次モーメント $\mathbb{E}[\mathbf{x}\mathbf{x}^T]$ は次のように定義されます。
$$ \mathbb{E}[\mathbf{x}\mathbf{x}^T] = \int \mathbf{x}\mathbf{x}^T \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) \, \mathrm{d}\mathbf{x} $$

ここで、ベクトル $\mathbf{x}$ を平均からの偏差を用いて $\mathbf{x} = (\mathbf{x} - \boldsymbol{\mu}) + \boldsymbol{\mu}$ と展開します。
これを二次形式 $\mathbf{x}\mathbf{x}^T$ に代入して展開します。

$$
\begin{aligned}
\mathbf{x}\mathbf{x}^T &= ( (\mathbf{x} - \boldsymbol{\mu}) + \boldsymbol{\mu} ) ( (\mathbf{x} - \boldsymbol{\mu}) + \boldsymbol{\mu} )^T \\
&= ( (\mathbf{x} - \boldsymbol{\mu}) + \boldsymbol{\mu} ) ( (\mathbf{x} - \boldsymbol{\mu})^T + \boldsymbol{\mu}^T ) \\
&= (\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T + (\mathbf{x} - \boldsymbol{\mu})\boldsymbol{\mu}^T + \boldsymbol{\mu}(\mathbf{x} - \boldsymbol{\mu})^T + \boldsymbol{\mu}\boldsymbol{\mu}^T
\end{aligned}
$$

期待値の線形性（積分は線形演算子であること）を利用して、各項の期待値を個別に計算します。

**第1項の期待値:**
$$ \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T] $$
これはまさに共分散行列の定義そのものです。ガウス分布の共分散は $\boldsymbol{\Sigma}$ であるため、この項は $\boldsymbol{\Sigma}$ になります。
（より厳密には、$\mathbf{z} = \mathbf{x} - \boldsymbol{\mu}$ と置換し、直交変換 $\mathbf{y} = \mathbf{U}^T \mathbf{z}$ を行うことで $\mathbb{E}[\mathbf{z}\mathbf{z}^T] = \mathbf{U} \mathbb{E}[\mathbf{y}\mathbf{y}^T] \mathbf{U}^T = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^T = \boldsymbol{\Sigma}$ と導かれます。）

**第2項および第3項の期待値（交差項）:**
$$ \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})\boldsymbol{\mu}^T] = \mathbb{E}[\mathbf{x} - \boldsymbol{\mu}] \boldsymbol{\mu}^T $$
$$ \mathbb{E}[\boldsymbol{\mu}(\mathbf{x} - \boldsymbol{\mu})^T] = \boldsymbol{\mu} \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})^T] $$

演習 2.19 で証明したように、ガウス分布の平均は $\mathbb{E}[\mathbf{x}] = \boldsymbol{\mu}$ です。したがって、偏差の期待値はゼロベクトルになります。
$$ \mathbb{E}[\mathbf{x} - \boldsymbol{\mu}] = \mathbb{E}[\mathbf{x}] - \boldsymbol{\mu} = \boldsymbol{\mu} - \boldsymbol{\mu} = \mathbf{0} $$
これにより、第2項と第3項は両方ともゼロ行列 $\mathbf{O}$ になります。

**第4項の期待値:**
$$ \mathbb{E}[\boldsymbol{\mu}\boldsymbol{\mu}^T] $$
$\boldsymbol{\mu}$ は定数ベクトルであるため、その期待値はそのまま $\boldsymbol{\mu}\boldsymbol{\mu}^T$ になります。

以上の4つの項の期待値をすべて足し合わせます。
$$
\begin{aligned}
\mathbb{E}[\mathbf{x}\mathbf{x}^T] &= \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T] + \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})\boldsymbol{\mu}^T] + \mathbb{E}[\boldsymbol{\mu}(\mathbf{x} - \boldsymbol{\mu})^T] + \mathbb{E}[\boldsymbol{\mu}\boldsymbol{\mu}^T] \\
&= \boldsymbol{\Sigma} + \mathbf{O} + \mathbf{O} + \boldsymbol{\mu}\boldsymbol{\mu}^T \\
&= \boldsymbol{\mu}\boldsymbol{\mu}^T + \boldsymbol{\Sigma}
\end{aligned}
$$

これにより、多変量ガウス分布の二次モーメントが $\mathbb{E}[\mathbf{x}\mathbf{x}^T] = \boldsymbol{\mu}\boldsymbol{\mu}^T + \boldsymbol{\Sigma}$ となることが証明されました。
