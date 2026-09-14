# 演習問題 1.26

## 問題文
二乗損失関数の期待損失 $\mathbb{E}[L]$ に対して展開を行い、次の分解（テキストの式1.90）を導出せよ。
$$ \mathbb{E}[L] = \int \{y(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]\}^2 p(\mathbf{x}) \mathrm{d}\mathbf{x} + \int \{\mathbb{E}[t|\mathbf{x}] - t\}^2 p(\mathbf{x}, t) \mathrm{d}\mathbf{x} \mathrm{d}t $$

## 解答と解説
### 1. 二乗誤差項の展開
期待損失関数の定義は以下の通りです。
$$ \mathbb{E}[L] = \iint \{y(\mathbf{x}) - t\}^2 p(\mathbf{x}, t) \mathrm{d}\mathbf{x} \mathrm{d}t $$
被積分関数の中の二乗項に、意図的に条件付き期待値 $\mathbb{E}[t|\mathbf{x}]$ を足して引きます。
$$ \{y(\mathbf{x}) - t\}^2 = \{ (y(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]) + (\mathbb{E}[t|\mathbf{x}] - t) \}^2 $$
これを展開すると、以下の3つの項が得られます。
$$ \{y(\mathbf{x}) - t\}^2 = \{y(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]\}^2 + \{\mathbb{E}[t|\mathbf{x}] - t\}^2 + 2\{y(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]\}\{\mathbb{E}[t|\mathbf{x}] - t\} $$

### 2. 各項の積分
この展開を元の期待損失の式に代入し、各項について積分を計算します。

**第1項:**
$$ \iint \{y(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]\}^2 p(\mathbf{x}, t) \mathrm{d}\mathbf{x} \mathrm{d}t = \int \{y(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]\}^2 \left( \int p(\mathbf{x}, t) \mathrm{d}t \right) \mathrm{d}\mathbf{x} $$
$\int p(\mathbf{x}, t) \mathrm{d}t = p(\mathbf{x})$ より、
$$ = \int \{y(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]\}^2 p(\mathbf{x}) \mathrm{d}\mathbf{x} $$

**第2項:**
これはそのまま残ります。
$$ \iint \{\mathbb{E}[t|\mathbf{x}] - t\}^2 p(\mathbf{x}, t) \mathrm{d}\mathbf{x} \mathrm{d}t $$

**第3項（交差項）:**
$$ 2 \iint \{y(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]\}\{\mathbb{E}[t|\mathbf{x}] - t\} p(\mathbf{x}, t) \mathrm{d}\mathbf{x} \mathrm{d}t $$
ここで、$p(\mathbf{x}, t) = p(t | \mathbf{x}) p(\mathbf{x})$ を用いて内側の $t$ についての積分に注目します。$y(\mathbf{x})$ および $\mathbb{E}[t|\mathbf{x}]$ は $t$ に依存しないため積分の外に出せます。
$$ = 2 \int \{y(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]\} p(\mathbf{x}) \left[ \int \{\mathbb{E}[t|\mathbf{x}] - t\} p(t | \mathbf{x}) \mathrm{d}t \right] \mathrm{d}\mathbf{x} $$
大カッコの中の積分を評価します。
$$ \int \mathbb{E}[t|\mathbf{x}] p(t | \mathbf{x}) \mathrm{d}t - \int t p(t | \mathbf{x}) \mathrm{d}t = \mathbb{E}[t|\mathbf{x}] \cdot 1 - \mathbb{E}[t|\mathbf{x}] = 0 $$
したがって、交差項全体が 0 となることが分かります。

### 3. 結論
交差項が 0 になるため、残った第1項と第2項を足し合わせることで、目的の式が導出されます。
$$ \mathbb{E}[L] = \int \{y(\mathbf{x}) - \mathbb{E}[t|\mathbf{x}]\}^2 p(\mathbf{x}) \mathrm{d}\mathbf{x} + \iint \{\mathbb{E}[t|\mathbf{x}] - t\}^2 p(\mathbf{x}, t) \mathrm{d}\mathbf{x} \mathrm{d}t $$
この式の第1項は予測モデル $y(\mathbf{x})$ の依存部分であり、$y(\mathbf{x}) = \mathbb{E}[t|\mathbf{x}]$ で最小値 0 を取ります。第2項はデータそのものが持つ本質的なノイズの分散を表しており、$y(\mathbf{x})$ に依存しない達成不可能な最小誤差（ベイズ誤差）を示しています。
