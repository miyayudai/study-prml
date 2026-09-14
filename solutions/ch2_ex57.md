# 演習問題 2.57

## 問題文（要約）
多変量ガウス分布が、指数型分布族の標準的な形式で表現できることを示せ。

## 解答と解説

指数型分布族の標準的な形式は以下のように定義されます：
$$ p(\mathbf{x}|\boldsymbol{\eta}) = h(\mathbf{x}) g(\boldsymbol{\eta}) \exp\{\boldsymbol{\eta}^T \mathbf{u}(\mathbf{x})\} $$

一方で、$D$ 次元の多変量ガウス分布の定義は以下の通りです：
$$ \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left\{ -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right\} $$

このガウス分布の指数部分を展開して、指数型分布族の形式に変形していきます。指数部分を展開すると以下のようになります。
$$ -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = -\frac{1}{2} \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} + \boldsymbol{\mu}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} - \frac{1}{2} \boldsymbol{\mu}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu} $$

したがって、ガウス分布は次のように書き換えることができます：
$$ \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} \boldsymbol{\mu}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu} \right) \exp\left( -\frac{1}{2} \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} + \boldsymbol{\mu}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} \right) $$

ここで、ベクトル化オペレータ $\text{vec}(\cdot)$ を導入します。これは行列の列を縦に並べて一つの列ベクトルにする線形変換です。トレースの性質 $\text{Tr}(\mathbf{A}^T\mathbf{B}) = \text{vec}(\mathbf{A})^T \text{vec}(\mathbf{B})$ および $\mathbf{x}^T \mathbf{A} \mathbf{x} = \text{Tr}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = \text{Tr}(\mathbf{A} \mathbf{x} \mathbf{x}^T)$ を用いると、二次形式の部分は次のように内積として表現できます：
$$ -\frac{1}{2} \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} = -\frac{1}{2} \text{vec}(\boldsymbol{\Sigma}^{-1})^T \text{vec}(\mathbf{x}\mathbf{x}^T) $$

これを用いて、ガウス分布の式を指数型分布族の標準形と比較すると、各要素は以下のように対応することがわかります。

1. **自然パラメータ $\boldsymbol{\eta}$**:
   $$ \boldsymbol{\eta} = \begin{pmatrix} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu} \\ -\frac{1}{2}\text{vec}(\boldsymbol{\Sigma}^{-1}) \end{pmatrix} $$

2. **十分統計量 $\mathbf{u}(\mathbf{x})$**:
   $$ \mathbf{u}(\mathbf{x}) = \begin{pmatrix} \mathbf{x} \\ \text{vec}(\mathbf{x}\mathbf{x}^T) \end{pmatrix} $$

3. **ベース測度 $h(\mathbf{x})$**:
   $$ h(\mathbf{x}) = (2\pi)^{-D/2} $$

4. **正規化係数 $g(\boldsymbol{\eta})$**:
   $$ g(\boldsymbol{\eta}) = |\boldsymbol{\Sigma}|^{-1/2} \exp\left( -\frac{1}{2} \boldsymbol{\mu}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu} \right) $$

（※ $g(\boldsymbol{\eta})$ は $\boldsymbol{\eta}$ のみの関数として陽に表現することも可能ですが、ここでは元のパラメータ $\boldsymbol{\mu}, \boldsymbol{\Sigma}$ を用いて表記しています。）

これにより、多変量ガウス分布が指数型分布族の標準的な形式にキャストできることが証明されました。
