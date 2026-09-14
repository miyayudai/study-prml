# 演習問題 2.13 (Exercise 2.13)

## 問題の目的
二つの多変量ガウス分布間のカルバック・ライブラー情報量 (Kullback-Leibler Divergence, KLダイバージェンス) を計算する公式を導出します。

## KLダイバージェンスの定義

二つの確率分布 $p(\mathbf{x})$ と $q(\mathbf{x})$ の間のKLダイバージェンスは以下のように定義されます。
$$ \text{KL}(p || q) = \int p(\mathbf{x}) \ln \frac{p(\mathbf{x})}{q(\mathbf{x})} d\mathbf{x} = \mathbb{E}_p[\ln p(\mathbf{x}) - \ln q(\mathbf{x})] $$

ここで、$p$ と $q$ をそれぞれ $D$ 次元のガウス分布とします。
$$ p(\mathbf{x}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p), \quad q(\mathbf{x}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q) $$

## 導出のステップ

対数確率密度関数の期待値を個別に計算します。
まず、$p(\mathbf{x})$ のエントロピー部分 $\mathbb{E}_p[\ln p(\mathbf{x})]$ は、
$$ \ln p(\mathbf{x}) = -\frac{D}{2} \ln(2\pi) - \frac{1}{2} \ln |\boldsymbol{\Sigma}_p| - \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_p)^T \boldsymbol{\Sigma}_p^{-1} (\mathbf{x} - \boldsymbol{\mu}_p) $$
トレースの性質 $\mathbf{x}^T \mathbf{A} \mathbf{x} = \text{Tr}(\mathbf{A} \mathbf{x} \mathbf{x}^T)$ を用いると、期待値は
$$ \mathbb{E}_p \left[ (\mathbf{x} - \boldsymbol{\mu}_p)^T \boldsymbol{\Sigma}_p^{-1} (\mathbf{x} - \boldsymbol{\mu}_p) \right] = \text{Tr} \left( \boldsymbol{\Sigma}_p^{-1} \mathbb{E}_p[(\mathbf{x} - \boldsymbol{\mu}_p)(\mathbf{x} - \boldsymbol{\mu}_p)^T] \right) = \text{Tr}(\boldsymbol{\Sigma}_p^{-1} \boldsymbol{\Sigma}_p) = \text{Tr}(\mathbf{I}) = D $$
よって、
$$ \mathbb{E}_p[\ln p(\mathbf{x})] = -\frac{D}{2} \ln(2\pi) - \frac{1}{2} \ln |\boldsymbol{\Sigma}_p| - \frac{D}{2} $$

次に、交差エントロピー部分 $\mathbb{E}_p[\ln q(\mathbf{x})]$ を計算します。
$$ \ln q(\mathbf{x}) = -\frac{D}{2} \ln(2\pi) - \frac{1}{2} \ln |\boldsymbol{\Sigma}_q| - \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_q)^T \boldsymbol{\Sigma}_q^{-1} (\mathbf{x} - \boldsymbol{\mu}_q) $$
二次形式の期待値は、$\mathbf{x} - \boldsymbol{\mu}_q = (\mathbf{x} - \boldsymbol{\mu}_p) + (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)$ と分解することで計算できます。
$$ \mathbb{E}_p \left[ (\mathbf{x} - \boldsymbol{\mu}_q)^T \boldsymbol{\Sigma}_q^{-1} (\mathbf{x} - \boldsymbol{\mu}_q) \right] = \text{Tr}(\boldsymbol{\Sigma}_q^{-1} \boldsymbol{\Sigma}_p) + (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^T \boldsymbol{\Sigma}_q^{-1} (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q) $$

これらを元のKLダイバージェンスの式に代入します。
$$ \text{KL}(p || q) = \mathbb{E}_p[\ln p(\mathbf{x})] - \mathbb{E}_p[\ln q(\mathbf{x})] $$
$$ = \frac{1}{2} \left( \ln \frac{|\boldsymbol{\Sigma}_q|}{|\boldsymbol{\Sigma}_p|} - D + \text{Tr}(\boldsymbol{\Sigma}_q^{-1} \boldsymbol{\Sigma}_p) + (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^T \boldsymbol{\Sigma}_q^{-1} (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q) \right) $$

この結果は、分布間の「距離」を測る指標として、情報幾何や変分推論において極めて重要な役割を果たします。
