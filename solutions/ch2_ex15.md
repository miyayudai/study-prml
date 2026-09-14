# 演習問題 2.15: 多変量ガウス分布のエントロピー

本演習問題では、$D$ 次元の多変量ガウス分布のエントロピーが共分散行列の行列式によってどのように表現されるかを導出します。

## 1. 問題の定式化

$D$ 次元の確率変数ベクトル $\mathbf{x}$ が平均 $\boldsymbol{\mu}$、共分散行列 $\boldsymbol{\Sigma}$ の多変量ガウス分布に従うとします。その確率密度関数は次のように与えられます。

$$
\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

この分布の微分エントロピー $H[\mathbf{x}]$ を求めます。

## 2. エントロピーの定義と展開

連続確率変数 $\mathbf{x}$ の微分エントロピーは以下で定義されます。

$$
H[\mathbf{x}] = -\int \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) \ln \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) d\mathbf{x} = -\mathbb{E}[\ln \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma})]
$$

まず、ガウス分布の対数 $\ln \mathcal{N}$ を展開します。

$$
\ln \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{D}{2} \ln(2\pi) - \frac{1}{2} \ln |\boldsymbol{\Sigma}| - \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})
$$

## 3. 期待値の計算

次に、上記で展開した各項に対する期待値 $\mathbb{E}[\cdot]$ を計算します。最初の2項は定数であるため、そのまま外に出ます。

$$
H[\mathbf{x}] = \frac{D}{2} \ln(2\pi) + \frac{1}{2} \ln |\boldsymbol{\Sigma}| + \frac{1}{2} \mathbb{E}\left[ (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right]
$$

ここで、第3項の二次形式の期待値を計算するために、トレースの巡回不変性（$\text{Tr}(AB) = \text{Tr}(BA)$）を利用します。スカラー値のトレースは元の値と等しいため、以下のようになります。

$$
(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = \text{Tr}\left( (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right) = \text{Tr}\left( \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T \right)
$$

期待値の線形性より、トレースの中に期待値を入れることができます。

$$
\mathbb{E}\left[ \text{Tr}\left( \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T \right) \right] = \text{Tr}\left( \boldsymbol{\Sigma}^{-1} \mathbb{E}\left[ (\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T \right] \right)
$$

共分散行列の定義 $\mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^T] = \boldsymbol{\Sigma}$ を代入すると、

$$
\text{Tr}\left( \boldsymbol{\Sigma}^{-1} \boldsymbol{\Sigma} \right) = \text{Tr}(\mathbf{I}_D) = D
$$

となります（$\mathbf{I}_D$ は $D$ 次の単位行列）。

## 4. 最終結果

得られた期待値 $D$ を元のエントロピーの式に代入します。

$$
H[\mathbf{x}] = \frac{D}{2} \ln(2\pi) + \frac{1}{2} \ln |\boldsymbol{\Sigma}| + \frac{D}{2}
$$

これを整理すると、多変量ガウス分布のエントロピーは次のように求まります。

$$
H[\mathbf{x}] = \frac{1}{2} \ln |\boldsymbol{\Sigma}| + \frac{D}{2} (1 + \ln(2\pi))
$$

## 幾何学的・物理的解釈
この結果は、ガウス分布のエントロピーが平均 $\boldsymbol{\mu}$ には依存せず、分布の広がりを表す共分散行列の行列式 $|\boldsymbol{\Sigma}|$ にのみ依存することを示しています。$|\boldsymbol{\Sigma}|$ は確率の「体積」のようなものを測る指標であり、分散が大きいほど不確実性が増し、結果としてエントロピーも増大します。
