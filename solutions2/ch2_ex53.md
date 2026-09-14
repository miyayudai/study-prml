## Exercise 2.53 (フィッシャー情報量の正定値性)

あるパラメータ $\boldsymbol{\theta}$（ベクトル）で特徴付けられた確率分布 $p(\mathbf{x}|\boldsymbol{\theta})$ を考えます。このとき、フィッシャー情報行列 (Fisher Information Matrix) $\mathbf{I}(\boldsymbol{\theta})$ が半正定値行列 (positive semi-definite matrix) であることを示します。

### フィッシャー情報行列の定義
フィッシャー情報行列 $\mathbf{I}(\boldsymbol{\theta})$ は、対数尤度関数 $\ln p(\mathbf{x}|\boldsymbol{\theta})$ の勾配（スコア関数）の共分散行列として定義されます。スコア関数を $\nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta})$ とすると、フィッシャー情報行列は次のように表されます：
$$ \mathbf{I}(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x}} \left[ \left( \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right) \left( \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right)^T \right] $$
ここで、期待値 $\mathbb{E}_{\mathbf{x}}[\cdot]$ は真の分布 $p(\mathbf{x}|\boldsymbol{\theta})$ の下での期待値を意味します。

### 半正定値性の証明
行列 $\mathbf{I}(\boldsymbol{\theta})$ が半正定値であるための必要十分条件は、任意の非ゼロの列ベクトル $\mathbf{v}$ に対して、二次形式 $\mathbf{v}^T \mathbf{I}(\boldsymbol{\theta}) \mathbf{v}$ が $0$ 以上になることです：
$$ \mathbf{v}^T \mathbf{I}(\boldsymbol{\theta}) \mathbf{v} \ge 0 $$

フィッシャー情報行列の定義式に左から $\mathbf{v}^T$、右から $\mathbf{v}$ を掛けます。期待値の線形性を用いると、次のようになります：
$$
\begin{align*}
\mathbf{v}^T \mathbf{I}(\boldsymbol{\theta}) \mathbf{v} &= \mathbf{v}^T \mathbb{E}_{\mathbf{x}} \left[ \left( \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right) \left( \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right)^T \right] \mathbf{v} \\
&= \mathbb{E}_{\mathbf{x}} \left[ \mathbf{v}^T \left( \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right) \left( \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right)^T \mathbf{v} \right]
\end{align*}
$$

ここで、$s = \mathbf{v}^T \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta})$ というスカラー量を定義します。すると、右側の積は $\left( \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right)^T \mathbf{v} = s^T = s$（$s$ はスカラーなので転置と等しい）となります。

したがって、式は次のように書き直せます：
$$ \mathbf{v}^T \mathbf{I}(\boldsymbol{\theta}) \mathbf{v} = \mathbb{E}_{\mathbf{x}} \left[ s \cdot s \right] = \mathbb{E}_{\mathbf{x}} \left[ \left( \mathbf{v}^T \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right)^2 \right] $$

括弧の中身は実数の2乗であるため、どのような $\mathbf{x}$ に対しても常に非負（$0$ 以上）です。非負の関数の期待値もまた非負になるため、
$$ \mathbb{E}_{\mathbf{x}} \left[ \left( \mathbf{v}^T \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right)^2 \right] \ge 0 $$
が成り立ちます。

### 結論
以上のことから、任意のベクトル $\mathbf{v}$ に対して $\mathbf{v}^T \mathbf{I}(\boldsymbol{\theta}) \mathbf{v} \ge 0$ であることが示されました。これは、フィッシャー情報行列 $\mathbf{I}(\boldsymbol{\theta})$ が半正定値行列であることを証明しています。スコア関数の共分散行列としての性質を考えれば、任意の共分散行列が半正定値であることの直接的な帰結とも言えます。
