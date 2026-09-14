# 演習問題 1.25

## 問題文
二乗損失関数の期待値
$$ \mathbb{E}[L] = \iint \{y(\mathbf{x}) - t\}^2 p(\mathbf{x}, t) \mathrm{d}\mathbf{x} \mathrm{d}t $$
を最小化する最適な予測関数 $y(\mathbf{x})$ が、条件付き期待値 $y(\mathbf{x}) = \mathbb{E}[t|\mathbf{x}]$ で与えられることを、変分法（または $y(\mathbf{x})$ による微分）を用いて示せ。

## 解答と解説
### 1. 汎関数積分の微分（変分法）
期待損失関数 $\mathbb{E}[L]$ は $y(\mathbf{x})$ を入力とする汎関数となっています。特定の入力 $\mathbf{x}$ に対する $y(\mathbf{x})$ の最適値を求めるために、被積分関数の中の $\mathbf{x}$ を固定し、各 $\mathbf{x}$ 独立に最小化を行います。

期待損失の式において、各 $\mathbf{x}$ について以下の積分（条件付き期待損失に等比例する部分）を最小化すればよいことが分かります。
$$ f(y(\mathbf{x})) = \int \{y(\mathbf{x}) - t\}^2 p(\mathbf{x}, t) \mathrm{d}t $$

### 2. $y(\mathbf{x})$ による微分と最適解の導出
$f(y(\mathbf{x}))$ を $y(\mathbf{x})$ について微分し、ゼロとおくことで極値を求めます。
$$ \frac{\partial f(y(\mathbf{x}))}{\partial y(\mathbf{x})} = \int 2 \{y(\mathbf{x}) - t\} p(\mathbf{x}, t) \mathrm{d}t = 0 $$
積分を分配すると、
$$ 2 y(\mathbf{x}) \int p(\mathbf{x}, t) \mathrm{d}t - 2 \int t p(\mathbf{x}, t) \mathrm{d}t = 0 $$
ここで周辺化の性質より $\int p(\mathbf{x}, t) \mathrm{d}t = p(\mathbf{x})$ です。したがって、
$$ y(\mathbf{x}) p(\mathbf{x}) = \int t p(\mathbf{x}, t) \mathrm{d}t $$
両辺を $p(\mathbf{x})$ で割ると（$p(\mathbf{x}) > 0$ を仮定）、
$$ y(\mathbf{x}) = \frac{\int t p(\mathbf{x}, t) \mathrm{d}t}{p(\mathbf{x})} = \int t \frac{p(\mathbf{x}, t)}{p(\mathbf{x})} \mathrm{d}t $$
乗法定理 $p(t | \mathbf{x}) = p(\mathbf{x}, t) / p(\mathbf{x})$ より、
$$ y(\mathbf{x}) = \int t p(t | \mathbf{x}) \mathrm{d}t = \mathbb{E}[t | \mathbf{x}] $$

### 3. 結論
これにより、二乗損失関数の下での最適な関数 $y(\mathbf{x})$ は、入力 $\mathbf{x}$ が与えられたときの目標値 $t$ の条件付き期待値（回帰関数）となることが厳密に示されました。
