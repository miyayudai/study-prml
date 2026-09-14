# 演習問題 1.35

## 問題の概要
多変量ガウス分布の微分エントロピーを計算し、次の式になることを証明する。
$$
H[\mathbf{x}] = \frac{1}{2}\ln|\boldsymbol{\Sigma}| + \frac{D}{2}(1 + \ln(2\pi))
$$

## 証明
$D$ 次元の多変量ガウス分布 $p(\mathbf{x}) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma})$ は次のように定義される。
$$
p(\mathbf{x}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) \right)
$$

微分エントロピー $H[\mathbf{x}]$ は定義より次のように計算される。
$$
H[\mathbf{x}] = -\int p(\mathbf{x}) \ln p(\mathbf{x}) d\mathbf{x} = -\mathbb{E}[\ln p(\mathbf{x})]
$$
まず、$\ln p(\mathbf{x})$ を展開する。
$$
\ln p(\mathbf{x}) = \ln \left[ \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \right] - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})
$$
$$
= -\frac{D}{2}\ln(2\pi) - \frac{1}{2}\ln|\boldsymbol{\Sigma}| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})
$$
これをエントロピーの式に代入して期待値をとる。第1項と第2項は定数であるから、
$$
H[\mathbf{x}] = \frac{D}{2}\ln(2\pi) + \frac{1}{2}\ln|\boldsymbol{\Sigma}| + \frac{1}{2}\mathbb{E} \left[ (\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) \right]
$$
次に、期待値の部分 $\mathbb{E} \left[ (\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) \right]$ を計算する。
行列のトレース（跡）の性質 $\text{Tr}(AB) = \text{Tr}(BA)$、およびスカラー量はそれ自身のトレースと等しいこと（$a = \text{Tr}(a)$）を利用する。
$$
(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) = \text{Tr} \left( (\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) \right) = \text{Tr} \left( \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T \right)
$$
期待値とトレースの順序を交換すると、
$$
\mathbb{E} \left[ \text{Tr} \left( \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T \right) \right] = \text{Tr} \left( \boldsymbol{\Sigma}^{-1} \mathbb{E} \left[ (\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T \right] \right)
$$
ここで、共分散行列の定義より $\mathbb{E} \left[ (\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T \right] = \boldsymbol{\Sigma}$ である。
よって、
$$
\text{Tr} \left( \boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma} \right) = \text{Tr}(\mathbf{I}_D) = D
$$
となる（$\mathbf{I}_D$ は $D \times D$ の単位行列）。
したがって、期待値の項は $\frac{D}{2}$ となる。

最終的にこれをエントロピーの式に戻すと、
$$
H[\mathbf{x}] = \frac{D}{2}\ln(2\pi) + \frac{1}{2}\ln|\boldsymbol{\Sigma}| + \frac{D}{2}
$$
$$
= \frac{1}{2}\ln|\boldsymbol{\Sigma}| + \frac{D}{2}(1 + \ln(2\pi))
$$
となり、題意の式が証明された。
