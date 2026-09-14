## Exercise 2.54 (Jeffreys 事前分布の変数変換不変性)

ジェフリーズ事前分布 (Jeffreys prior) は、フィッシャー情報行列 $\mathbf{I}(\boldsymbol{\theta})$ の行列式の平方根に比例する事前分布として定義されます：
$$ p(\boldsymbol{\theta}) \propto \sqrt{|\mathbf{I}(\boldsymbol{\theta})|} $$
この分布の最大の特徴は、パラメータの任意の非特異な変数変換に対して不変な性質を持つことです。本項では、新しいパラメータ $\boldsymbol{\phi}$ への変換 $\boldsymbol{\theta} = \mathbf{h}(\boldsymbol{\phi})$ を考えたとき、ジェフリーズ事前分布が適切な変換則を満たすことを証明します。

### フィッシャー情報行列の変換
まず、パラメータ $\boldsymbol{\phi}$ による対数尤度の微分の連鎖律 (chain rule) を考えます：
$$ \frac{\partial \ln p(\mathbf{x}|\boldsymbol{\phi})}{\partial \phi_i} = \sum_{j} \frac{\partial \theta_j}{\partial \phi_i} \frac{\partial \ln p(\mathbf{x}|\boldsymbol{\theta})}{\partial \theta_j} $$
これを行列表記で表すと、スコア関数はヤコビ行列 $\mathbf{J}$ を用いて次のように書けます：
$$ \nabla_{\boldsymbol{\phi}} \ln p(\mathbf{x}|\boldsymbol{\phi}) = \mathbf{J}^T \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) $$
ここで、ヤコビ行列の各要素は $J_{ji} = \frac{\partial \theta_j}{\partial \phi_i}$ です。

パラメータ $\boldsymbol{\phi}$ に対するフィッシャー情報行列 $\mathbf{I}(\boldsymbol{\phi})$ を計算します：
$$
\begin{align*}
\mathbf{I}(\boldsymbol{\phi}) &= \mathbb{E}_{\mathbf{x}} \left[ \left( \nabla_{\boldsymbol{\phi}} \ln p(\mathbf{x}|\boldsymbol{\phi}) \right) \left( \nabla_{\boldsymbol{\phi}} \ln p(\mathbf{x}|\boldsymbol{\phi}) \right)^T \right] \\
&= \mathbb{E}_{\mathbf{x}} \left[ \left( \mathbf{J}^T \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right) \left( \mathbf{J}^T \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right)^T \right] \\
&= \mathbf{J}^T \mathbb{E}_{\mathbf{x}} \left[ \left( \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right) \left( \nabla_{\boldsymbol{\theta}} \ln p(\mathbf{x}|\boldsymbol{\theta}) \right)^T \right] \mathbf{J} \\
&= \mathbf{J}^T \mathbf{I}(\boldsymbol{\theta}) \mathbf{J}
\end{align*}
$$

### 行列式の計算
得られた $\mathbf{I}(\boldsymbol{\phi}) = \mathbf{J}^T \mathbf{I}(\boldsymbol{\theta}) \mathbf{J}$ の両辺の行列式をとります。行列式の積の性質 $|\mathbf{A}\mathbf{B}| = |\mathbf{A}||\mathbf{B}|$ と $|\mathbf{A}^T| = |\mathbf{A}|$ を用いると：
$$ |\mathbf{I}(\boldsymbol{\phi})| = |\mathbf{J}^T \mathbf{I}(\boldsymbol{\theta}) \mathbf{J}| = |\mathbf{J}^T| |\mathbf{I}(\boldsymbol{\theta})| |\mathbf{J}| = |\mathbf{J}|^2 |\mathbf{I}(\boldsymbol{\theta})| $$

この両辺の平方根をとります：
$$ \sqrt{|\mathbf{I}(\boldsymbol{\phi})|} = \sqrt{|\mathbf{J}|^2 |\mathbf{I}(\boldsymbol{\theta})|} = |\det(\mathbf{J})| \sqrt{|\mathbf{I}(\boldsymbol{\theta})|} $$

### 確率密度の変数変換則との一致
確率密度関数の変数変換の一般的な公式によれば、新しい変数 $\boldsymbol{\phi}$ での確率密度 $p(\boldsymbol{\phi})$ は、元の変数での密度 $p(\boldsymbol{\theta})$ とヤコビ行列の行列式の絶対値 $|\det(\mathbf{J})|$ を用いて次のように表されます：
$$ p(\boldsymbol{\phi}) = p(\boldsymbol{\theta}) \left| \det \left( \frac{\partial \boldsymbol{\theta}}{\partial \boldsymbol{\phi}} \right) \right| = p(\boldsymbol{\theta}) |\det(\mathbf{J})| $$

ジェフリーズ事前分布 $p(\boldsymbol{\theta}) \propto \sqrt{|\mathbf{I}(\boldsymbol{\theta})|}$ をこの変数変換の公式に代入すると：
$$ p(\boldsymbol{\phi}) \propto \sqrt{|\mathbf{I}(\boldsymbol{\theta})|} |\det(\mathbf{J})| $$
先ほど導いた関係式 $\sqrt{|\mathbf{I}(\boldsymbol{\phi})|} = |\det(\mathbf{J})| \sqrt{|\mathbf{I}(\boldsymbol{\theta})|}$ を用いると、
$$ p(\boldsymbol{\phi}) \propto \sqrt{|\mathbf{I}(\boldsymbol{\phi})|} $$
となり、変換後のパラメータ $\boldsymbol{\phi}$ に対するジェフリーズ事前分布の定義式そのものが導かれます。

### 結論
以上より、パラメータ空間でどのような座標系（再表現）を採用したとしても、ジェフリーズ事前分布の手続きを適用して得られる確率分布は、一意で一貫した確率測度を与えることが証明されました。
