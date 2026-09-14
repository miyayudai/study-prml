import os

os.makedirs('solutions', exist_ok=True)

ex8 = r"""# 演習問題 2.8 (Exercise 2.8)

## 問題の目的
多変量ガウス分布が与えられたとき、その同時確率分布 $p(\mathbf{x}_a, \mathbf{x}_b)$ から、変数の一部を積分消去した周辺分布 $p(\mathbf{x}_a)$ がどのような分布になるかを導出します。結果として周辺分布もガウス分布になり、その平均と分散が直感的な形（それぞれ全体平均の対応するブロック、全体共分散の対応するブロック）になることを示します。

## 導出のステップ

多変量ガウス分布の確率密度関数は以下のように定義されます。
$$ \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}} \frac{1}{|\boldsymbol{\Sigma}|^{1/2}} \exp\left\{ -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right\} $$

ここで、確率変数ベクトル $\mathbf{x}$、平均ベクトル $\boldsymbol{\mu}$、共分散行列 $\boldsymbol{\Sigma}$ を以下のようにブロック分割します。
$$ \mathbf{x} = \begin{pmatrix} \mathbf{x}_a \\ \mathbf{x}_b \end{pmatrix}, \quad \boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ \boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb} \end{pmatrix} $$

また、共分散行列の逆行列である精度行列 $\boldsymbol{\Lambda} \equiv \boldsymbol{\Sigma}^{-1}$ も同様に分割します。
$$ \boldsymbol{\Lambda} = \begin{pmatrix} \boldsymbol{\Lambda}_{aa} & \boldsymbol{\Lambda}_{ab} \\ \boldsymbol{\Lambda}_{ba} & \boldsymbol{\Lambda}_{bb} \end{pmatrix} $$

周辺分布 $p(\mathbf{x}_a)$ は、同時確率 $p(\mathbf{x}_a, \mathbf{x}_b)$ を $\mathbf{x}_b$ について積分することで得られます。
$$ p(\mathbf{x}_a) = \int p(\mathbf{x}_a, \mathbf{x}_b) d\mathbf{x}_b $$

指数部分（二次形式）のみに注目し、これを $-\frac{1}{2} \Delta$ とおきます。
$$ \Delta = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Lambda} (\mathbf{x} - \boldsymbol{\mu}) $$

$\mathbf{x}_a$ と $\mathbf{x}_b$ に分けて展開すると、
$$ \Delta = (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{aa} (\mathbf{x}_a - \boldsymbol{\mu}_a) + (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$
$$ + (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Lambda}_{ba} (\mathbf{x}_a - \boldsymbol{\mu}_a) + (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Lambda}_{bb} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$

この式を $\mathbf{x}_b$ について平方完成します。$\mathbf{x}_b$ に依存する項を取り出すと、
$$ \Delta = (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Lambda}_{bb} (\mathbf{x}_b - \boldsymbol{\mu}_b) + 2 (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Lambda}_{ba} (\mathbf{x}_a - \boldsymbol{\mu}_a) + \text{const} $$

ここで $\mathbf{m} = \boldsymbol{\mu}_b - \boldsymbol{\Lambda}_{bb}^{-1} \boldsymbol{\Lambda}_{ba} (\mathbf{x}_a - \boldsymbol{\mu}_a)$ とおくと、
$$ \Delta = (\mathbf{x}_b - \mathbf{m})^T \boldsymbol{\Lambda}_{bb} (\mathbf{x}_b - \mathbf{m}) - (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{ab} \boldsymbol{\Lambda}_{bb}^{-1} \boldsymbol{\Lambda}_{ba} (\mathbf{x}_a - \boldsymbol{\mu}_a) + (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{aa} (\mathbf{x}_a - \boldsymbol{\mu}_a) $$

$\mathbf{x}_b$ についての積分は、正規化されていないガウス積分の形になり、$\mathbf{x}_b$ に依存しない項だけが残ります。残った $\mathbf{x}_a$ に関する二次形式は、
$$ (\mathbf{x}_a - \boldsymbol{\mu}_a)^T (\boldsymbol{\Lambda}_{aa} - \boldsymbol{\Lambda}_{ab} \boldsymbol{\Lambda}_{bb}^{-1} \boldsymbol{\Lambda}_{ba}) (\mathbf{x}_a - \boldsymbol{\mu}_a) $$

ブロック行列の逆行列の公式により、$\boldsymbol{\Sigma}_{aa} = (\boldsymbol{\Lambda}_{aa} - \boldsymbol{\Lambda}_{ab} \boldsymbol{\Lambda}_{bb}^{-1} \boldsymbol{\Lambda}_{ba})^{-1}$ となるため、周辺分布もガウス分布となり、そのパラメータは以下で与えられます。
$$ p(\mathbf{x}_a) = \mathcal{N}(\mathbf{x}_a | \boldsymbol{\mu}_a, \boldsymbol{\Sigma}_{aa}) $$
"""

ex9 = r"""# 演習問題 2.9 (Exercise 2.9)

## 問題の目的
多変量ガウス分布において、一部の変数が観測された場合の残りの変数の条件付き分布 $p(\mathbf{x}_a | \mathbf{x}_b)$ を導出します。周辺分布と同様に、条件付き分布もまたガウス分布になることを示します。

## 導出のステップ

同時分布 $p(\mathbf{x}_a, \mathbf{x}_b)$ の指数部分の二次形式 $\Delta$ は、演習問題2.8の式から出発します。
$$ \Delta = (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{aa} (\mathbf{x}_a - \boldsymbol{\mu}_a) + 2 (\mathbf{x}_a - \boldsymbol{\mu}_a)^T \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) + (\mathbf{x}_b - \boldsymbol{\mu}_b)^T \boldsymbol{\Lambda}_{bb} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$

条件付き分布 $p(\mathbf{x}_a | \mathbf{x}_b)$ を考えるため、$\mathbf{x}_b$ は定数として扱い、$\mathbf{x}_a$ について平方完成を行います。$\mathbf{x}_a$ に依存する項をまとめると以下のようになります。
$$ \Delta_a = \mathbf{x}_a^T \boldsymbol{\Lambda}_{aa} \mathbf{x}_a - 2 \mathbf{x}_a^T (\boldsymbol{\Lambda}_{aa} \boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b)) + \text{const} $$

ここで、ガウス分布の一般的な二次形式 $(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} - 2 \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu} + \text{const}$ と係数比較を行います。

まず、2次の項 $\mathbf{x}_a^T \boldsymbol{\Lambda}_{aa} \mathbf{x}_a$ の係数から、条件付き共分散行列 $\boldsymbol{\Sigma}_{a|b}$ は精度行列の対応するブロックの逆行列であることがわかります。
$$ \boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Lambda}_{aa}^{-1} $$

次に、1次の項の係数を比較すると、
$$ \boldsymbol{\Sigma}_{a|b}^{-1} \boldsymbol{\mu}_{a|b} = \boldsymbol{\Lambda}_{aa} \boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$
$$ \boldsymbol{\Lambda}_{aa} \boldsymbol{\mu}_{a|b} = \boldsymbol{\Lambda}_{aa} \boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$

両辺に左から $\boldsymbol{\Lambda}_{aa}^{-1}$ を掛けることで、条件付き平均 $\boldsymbol{\mu}_{a|b}$ が得られます。
$$ \boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a - \boldsymbol{\Lambda}_{aa}^{-1} \boldsymbol{\Lambda}_{ab} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$

以上の結果から、条件付き分布 $p(\mathbf{x}_a | \mathbf{x}_b)$ は以下のパラメータを持つガウス分布となることが証明されました。
$$ p(\mathbf{x}_a | \mathbf{x}_b) = \mathcal{N}(\mathbf{x}_a | \boldsymbol{\mu}_{a|b}, \boldsymbol{\Sigma}_{a|b}) $$
"""

ex10 = r"""# 演習問題 2.10 (Exercise 2.10)

## 問題の目的
分割された行列の逆行列（ブロック行列の反転補題）を利用して、精度行列 $\boldsymbol{\Lambda}$ の要素を共分散行列 $\boldsymbol{\Sigma}$ の要素を用いて表現し、周辺分布と条件付き分布のパラメータの代替表現を導きます。

## 導出のステップ

ブロック行列 $\boldsymbol{\Sigma}$ とその逆行列 $\boldsymbol{\Lambda}$ の関係 $\boldsymbol{\Sigma} \boldsymbol{\Lambda} = \mathbf{I}$ より、以下の連立方程式が成り立ちます。
1. $\boldsymbol{\Sigma}_{aa}\boldsymbol{\Lambda}_{aa} + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Lambda}_{ba} = \mathbf{I}$
2. $\boldsymbol{\Sigma}_{aa}\boldsymbol{\Lambda}_{ab} + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Lambda}_{bb} = \mathbf{0}$
3. $\boldsymbol{\Sigma}_{ba}\boldsymbol{\Lambda}_{aa} + \boldsymbol{\Sigma}_{bb}\boldsymbol{\Lambda}_{ba} = \mathbf{0}$
4. $\boldsymbol{\Sigma}_{ba}\boldsymbol{\Lambda}_{ab} + \boldsymbol{\Sigma}_{bb}\boldsymbol{\Lambda}_{bb} = \mathbf{I}$

第2式から $\boldsymbol{\Lambda}_{ab}$ を解くと、
$$ \boldsymbol{\Lambda}_{ab} = -\boldsymbol{\Sigma}_{aa}^{-1} \boldsymbol{\Sigma}_{ab} \boldsymbol{\Lambda}_{bb} $$

これを第4式に代入します。
$$ -\boldsymbol{\Sigma}_{ba} \boldsymbol{\Sigma}_{aa}^{-1} \boldsymbol{\Sigma}_{ab} \boldsymbol{\Lambda}_{bb} + \boldsymbol{\Sigma}_{bb} \boldsymbol{\Lambda}_{bb} = \mathbf{I} $$
$$ (\boldsymbol{\Sigma}_{bb} - \boldsymbol{\Sigma}_{ba} \boldsymbol{\Sigma}_{aa}^{-1} \boldsymbol{\Sigma}_{ab}) \boldsymbol{\Lambda}_{bb} = \mathbf{I} $$
$$ \boldsymbol{\Lambda}_{bb} = (\boldsymbol{\Sigma}_{bb} - \boldsymbol{\Sigma}_{ba} \boldsymbol{\Sigma}_{aa}^{-1} \boldsymbol{\Sigma}_{ab})^{-1} $$
ここで得られた逆行列の括弧内の式は、シューア補行列 (Schur complement) と呼ばれます。

対称性から $\boldsymbol{\Lambda}_{aa}$ も同様に求められます。
$$ \boldsymbol{\Lambda}_{aa} = (\boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} \boldsymbol{\Sigma}_{ba})^{-1} $$

これを利用することで、演習問題2.9で求めた条件付き分布のパラメータを、精度行列 $\boldsymbol{\Lambda}$ ではなく共分散行列 $\boldsymbol{\Sigma}$ で書き換えることができます。
$$ \boldsymbol{\mu}_{a|b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} (\mathbf{x}_b - \boldsymbol{\mu}_b) $$
$$ \boldsymbol{\Sigma}_{a|b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} \boldsymbol{\Sigma}_{ba} $$

この表現は、共分散の形で情報が与えられている実用的なケース（カルマンフィルタなど）で非常に有用です。
"""

ex11 = r"""# 演習問題 2.11 (Exercise 2.11)

## 問題の目的
線形ガウスモデルにおいて、周辺分布 $p(\mathbf{x})$ と条件付き分布 $p(\mathbf{y}|\mathbf{x})$ が与えられたとき、同時分布 $p(\mathbf{x}, \mathbf{y})$ がガウス分布になることを示し、その平均と共分散を求めます。

## モデルの定義

以下のような線形ガウスモデルを考えます。
$$ p(\mathbf{x}) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1}) $$
$$ p(\mathbf{y} | \mathbf{x}) = \mathcal{N}(\mathbf{y} | \mathbf{A}\mathbf{x} + \mathbf{b}, \mathbf{L}^{-1}) $$

ここで $\mathbf{x}$ と $\mathbf{y}$ は確率変数ベクトル、$\mathbf{A}$ と $\mathbf{b}$ は線形変換のパラメータ、$\boldsymbol{\Lambda}$ と $\mathbf{L}$ はそれぞれの精度行列です。

## 同時分布の導出

同時分布は積の法則から $p(\mathbf{x}, \mathbf{y}) = p(\mathbf{y} | \mathbf{x}) p(\mathbf{x})$ で与えられます。両者がガウス分布であるため、対数同時確率は二次形式の和となります。
$$ \ln p(\mathbf{x}, \mathbf{y}) = -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Lambda} (\mathbf{x} - \boldsymbol{\mu}) - \frac{1}{2} (\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b})^T \mathbf{L} (\mathbf{y} - \mathbf{A}\mathbf{x} - \mathbf{b}) + \text{const} $$

変数を連結したベクトル $\mathbf{z} = \begin{pmatrix} \mathbf{x} \\ \mathbf{y} \end{pmatrix}$ について、この二次形式を整理し平方完成します。
まず2次の項を展開して抽出します。
$$ -\frac{1}{2} \left[ \mathbf{x}^T (\boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A}) \mathbf{x} - 2 \mathbf{x}^T \mathbf{A}^T \mathbf{L} \mathbf{y} + \mathbf{y}^T \mathbf{L} \mathbf{y} \right] $$

これを行列表記でまとめると、同時分布の精度行列 $\mathbf{R}$ が得られます。
$$ \mathbf{R} = \begin{pmatrix} \boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A} & -\mathbf{A}^T \mathbf{L} \\ -\mathbf{L} \mathbf{A} & \mathbf{L} \end{pmatrix} $$

同時分布の共分散行列 $\mathbf{C}$ は $\mathbf{R}$ の逆行列として求まります。ブロック行列の反転補題を用いると、
$$ \mathbf{C} = \begin{pmatrix} \boldsymbol{\Lambda}^{-1} & \boldsymbol{\Lambda}^{-1} \mathbf{A}^T \\ \mathbf{A} \boldsymbol{\Lambda}^{-1} & \mathbf{L}^{-1} + \mathbf{A} \boldsymbol{\Lambda}^{-1} \mathbf{A}^T \end{pmatrix} $$

次に1次の項の係数を比較することで、同時分布の平均 $\mathbb{E}[\mathbf{z}]$ が求まります。計算を進めると、
$$ \mathbb{E}[\mathbf{z}] = \begin{pmatrix} \boldsymbol{\mu} \\ \mathbf{A}\boldsymbol{\mu} + \mathbf{b} \end{pmatrix} $$

よって、同時分布は以下のガウス分布となることが証明されました。
$$ p(\mathbf{x}, \mathbf{y}) = \mathcal{N} \left( \begin{pmatrix} \mathbf{x} \\ \mathbf{y} \end{pmatrix} \middle| \begin{pmatrix} \boldsymbol{\mu} \\ \mathbf{A}\boldsymbol{\mu} + \mathbf{b} \end{pmatrix}, \begin{pmatrix} \boldsymbol{\Lambda}^{-1} & \boldsymbol{\Lambda}^{-1} \mathbf{A}^T \\ \mathbf{A} \boldsymbol{\Lambda}^{-1} & \mathbf{L}^{-1} + \mathbf{A} \boldsymbol{\Lambda}^{-1} \mathbf{A}^T \end{pmatrix} \right) $$
"""

ex12 = r"""# 演習問題 2.12 (Exercise 2.12)

## 問題の目的
演習問題2.11で導出した同時分布 $p(\mathbf{x}, \mathbf{y})$ を用いて、ベイズの定理に基づく逆方向の条件付き分布 $p(\mathbf{x}|\mathbf{y})$ および $\mathbf{y}$ の周辺分布 $p(\mathbf{y})$ を導出します。

## 周辺分布の導出

同時分布の共分散行列 $\mathbf{C}$ と平均から、$\mathbf{y}$ の周辺分布 $p(\mathbf{y})$ は、単純に $\mathbf{y}$ に対応するブロックを取り出すことで得られます（演習問題2.8の結果を利用）。
$$ p(\mathbf{y}) = \mathcal{N}(\mathbf{y} | \mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{L}^{-1} + \mathbf{A} \boldsymbol{\Lambda}^{-1} \mathbf{A}^T) $$

これは、入力 $\mathbf{x}$ の不確実性（分散 $\boldsymbol{\Lambda}^{-1}$）が、線形変換 $\mathbf{A}$ を通じて出力 $\mathbf{y}$ の分散に加算的に寄与することを示しています。

## 条件付き分布（事後分布）の導出

$p(\mathbf{x} | \mathbf{y})$ の導出には、演習問題2.9の条件付き分布の公式を適用します。
精度行列を利用した公式を用いると、計算がスムーズです。同時分布の精度行列 $\mathbf{R}$ のブロック要素は以下の通りでした。
$$ \mathbf{R}_{xx} = \boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A}, \quad \mathbf{R}_{xy} = -\mathbf{A}^T \mathbf{L} $$

条件付き分散 $\boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}}$ は、$\mathbf{R}_{xx}$ の逆行列です。
$$ \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}} = (\boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A})^{-1} $$

条件付き平均 $\boldsymbol{\mu}_{\mathbf{x}|\mathbf{y}}$ は、以下の式で与えられます。
$$ \boldsymbol{\mu}_{\mathbf{x}|\mathbf{y}} = \boldsymbol{\mu} - \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}} \mathbf{R}_{xy} (\mathbf{y} - (\mathbf{A}\boldsymbol{\mu} + \mathbf{b})) $$
$$ = \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}} \left( \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}}^{-1} \boldsymbol{\mu} + \mathbf{A}^T \mathbf{L} (\mathbf{y} - \mathbf{A}\boldsymbol{\mu} - \mathbf{b}) \right) $$
$$ = \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}} \left( (\boldsymbol{\Lambda} + \mathbf{A}^T \mathbf{L} \mathbf{A})\boldsymbol{\mu} + \mathbf{A}^T \mathbf{L} \mathbf{y} - \mathbf{A}^T \mathbf{L} \mathbf{A} \boldsymbol{\mu} - \mathbf{A}^T \mathbf{L} \mathbf{b} \right) $$
$$ = \boldsymbol{\Sigma}_{\mathbf{x}|\mathbf{y}} \left( \boldsymbol{\Lambda} \boldsymbol{\mu} + \mathbf{A}^T \mathbf{L} (\mathbf{y} - \mathbf{b}) \right) $$

この結果は、事前知識（平均 $\boldsymbol{\mu}$）と観測データ $\mathbf{y}$ の情報を、それぞれの精度行列を重みとして最適にブレンドした形になっています。ガウス分布のベイズ推論（線形回帰やカルマンフィルタなど）の基礎となる極めて重要な公式です。
"""

ex13 = r"""# 演習問題 2.13 (Exercise 2.13)

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
"""

ex14 = r"""# 演習問題 2.14 (Exercise 2.14)

## 問題の目的
多変量ガウス分布における共分散行列 $\boldsymbol{\Sigma}$ の対数尤度関数を微分し、最尤推定量の導出過程において、行列の微分公式を用いる方法を確認します。

## 尤度関数と対数尤度

データセット $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_N)$ が独立同分布 (i.i.d.) で $D$ 次元のガウス分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ から生成されているとします。
対数尤度関数は以下のようになります。
$$ \ln p(\mathbf{X} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{ND}{2} \ln(2\pi) - \frac{N}{2} \ln |\boldsymbol{\Sigma}| - \frac{1}{2} \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_n - \boldsymbol{\mu}) $$

簡略化のため、共分散の最尤推定を行うために $\boldsymbol{\mu}$ はすでに最尤推定量 $\boldsymbol{\mu}_{ML} = \frac{1}{N}\sum \mathbf{x}_n$ に設定されているとし、精度行列 $\boldsymbol{\Lambda} = \boldsymbol{\Sigma}^{-1}$ で微分することを考えます。

## 行列微分を用いた最大化

対数尤度関数を $\boldsymbol{\Lambda}$ について書き換えます。
$$ \ln p(\mathbf{X} | \boldsymbol{\Lambda}) = \frac{N}{2} \ln |\boldsymbol{\Lambda}| - \frac{1}{2} \text{Tr}(\boldsymbol{\Lambda} \mathbf{S}) + \text{const} $$
ここで $\mathbf{S}$ は非正規化の散布行列 $\mathbf{S} = \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^T$ です。

行列の微分公式 $\frac{\partial}{\partial \boldsymbol{\Lambda}} \ln |\boldsymbol{\Lambda}| = \boldsymbol{\Lambda}^{-1}$ （すなわち $\boldsymbol{\Sigma}$）および $\frac{\partial}{\partial \boldsymbol{\Lambda}} \text{Tr}(\boldsymbol{\Lambda} \mathbf{S}) = \mathbf{S}$ を用います。

対数尤度を $\boldsymbol{\Lambda}$ で微分し、ゼロとおきます。
$$ \frac{\partial \ln p}{\partial \boldsymbol{\Lambda}} = \frac{N}{2} \boldsymbol{\Sigma} - \frac{1}{2} \mathbf{S} = \mathbf{0} $$

これを解くことで、共分散行列の最尤推定量 $\boldsymbol{\Sigma}_{ML}$ が直ちに得られます。
$$ \boldsymbol{\Sigma}_{ML} = \frac{1}{N} \mathbf{S} = \frac{1}{N} \sum_{n=1}^N (\mathbf{x}_n - \boldsymbol{\mu}_{ML})(\mathbf{x}_n - \boldsymbol{\mu}_{ML})^T $$

この結果から、ガウス分布の分散の最尤推定量が標本分散（不偏分散ではない）となることが行列微分の観点から示されました。
"""

files = {
    'solutions/ch2_ex8.md': ex8,
    'solutions/ch2_ex9.md': ex9,
    'solutions/ch2_ex10.md': ex10,
    'solutions/ch2_ex11.md': ex11,
    'solutions/ch2_ex12.md': ex12,
    'solutions/ch2_ex13.md': ex13,
    'solutions/ch2_ex14.md': ex14,
}

for filepath, content in files.items():
    with open(filepath, 'w') as f:
        f.write(content)

print("Files created successfully.")
