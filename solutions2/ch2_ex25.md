## Exercise 2.25 (線形ガウスモデルの事後分布)

**問題:**
線形ガウスモデルの事後精度行列 $\boldsymbol{\Sigma}^{-1} = \boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A}$ を導出せよ。

**解答と解説:**

事前分布 $p(\mathbf{x}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1})$ と尤度関数 $p(\mathbf{y} | \mathbf{x}) = \mathcal{N}(\mathbf{y} | \mathbf{A}\mathbf{x} + \mathbf{b}, \mathbf{L}^{-1})$ が与えられたとき、事後分布 $p(\mathbf{x} | \mathbf{y})$ はベイズの定理によりこれら二つの積に比例します。
$$ p(\mathbf{x} | \mathbf{y}) \propto p(\mathbf{y} | \mathbf{x}) p(\mathbf{x}) $$

事後分布もガウス分布になるため、その指数部を展開し、$\mathbf{x}$ に関する二次項をまとめることで事後精度行列を特定することができます。

同時分布の対数（定数項を無視したもの）を $\mathbf{x}$ に着目して整理します。
$$ \ln p(\mathbf{x} | \mathbf{y}) = \ln p(\mathbf{y} | \mathbf{x}) + \ln p(\mathbf{x}) + \text{const} $$
$$ = -\frac{1}{2}(\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b})^T \mathbf{L} (\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b}) - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Lambda} (\mathbf{x} - \boldsymbol{\mu}) + \text{const} $$

この式の中で、$\mathbf{x}$ の二次形式を抽出します。
第1項から展開される $\mathbf{x}$ の二次項は：
$$ -\frac{1}{2} (-\mathbf{A}\mathbf{x})^T \mathbf{L} (-\mathbf{A}\mathbf{x}) = -\frac{1}{2} \mathbf{x}^T \mathbf{A}^T \mathbf{L} \mathbf{A} \mathbf{x} $$
第2項から展開される $\mathbf{x}$ の二次項は：
$$ -\frac{1}{2} \mathbf{x}^T \boldsymbol{\Lambda} \mathbf{x} $$

これらを足し合わせると、全体の $\mathbf{x}$ に関する二次項は、
$$ -\frac{1}{2} \mathbf{x}^T (\boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A}) \mathbf{x} $$
となります。

ガウス分布 $p(\mathbf{x} | \mathbf{y}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_{\mathbf{x}|\mathbf{y}}, \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}})$ の対数指数部の二次項は一般に $-\frac{1}{2} \mathbf{x}^T \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}}^{-1} \mathbf{x}$ の形を取ります。

したがって、係数行列を比較することにより、事後分布の精度行列 $\boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}}^{-1}$ が次のように求まります。
$$ \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}}^{-1} = \boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A} $$

（なお、一次の項も同様に比較することで、事後平均 $\boldsymbol{\mu}_{\mathbf{x}|\mathbf{y}}$ も導出可能です。）
この式は、事後精度（情報量）が「事前の精度 $\boldsymbol{\Lambda}$」と「観測から得られる精度 $\mathbf{A}^T \mathbf{L} \mathbf{A}$」の和で表されるという直感的な解釈を与えます。
