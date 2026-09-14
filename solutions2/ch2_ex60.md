# Exercise 2.60

## 問題
Cover-Hartの定理（1-NN の漸近誤り率）について、データ数 $N \to \infty$ の極限において、1-最近傍法（1-NN）の分類誤り率がベイズ誤り率（理論的最小誤り率）の高々2倍以下に収まることを示せ。

## 解答と解説

分類問題において、ある入力 $\mathbf{x}$ に対する真のクラスの事後確率を $P(C_k | \mathbf{x})$ とします。
ベイズ決定則では最も事後確率が高いクラスを選ぶため、その誤り率（ベイズ誤り率） $P_B(\mathbf{x})$ は次のように書けます。
$$ P_B(\mathbf{x}) = 1 - \max_k P(C_k | \mathbf{x}) $$
最大確率を $p_m = \max_k P(C_k | \mathbf{x})$ とおくと、$P_B = 1 - p_m$ となります。

1-NN法では、未知のデータ点 $\mathbf{x}$ に対して、学習データの中で最も近い点 $\mathbf{x}'$ を見つけ、その真のクラス $c'$ を予測クラスとします。
データ数 $N \to \infty$ の漸近的な極限では、十分な密度のデータが存在するため、最も近い点 $\mathbf{x}'$ は $\mathbf{x}$ に限りなく近づきます（$\mathbf{x}' \to \mathbf{x}$）。
分布が連続であれば、事後確率も近づくため $P(C_k | \mathbf{x}') \to P(C_k | \mathbf{x})$ と仮定できます。

このとき、1-NN法が誤分類する確率 $P_{1NN}(\mathbf{x})$ は、真のクラス $c$（分布 $P(C_k|\mathbf{x})$ に従う）と、予測クラス $c'$（近傍点での分布 $P(C_k|\mathbf{x}')$ に従う）が異なる確率です。クラスの抽出は独立であるため、
$$ P_{1NN}(\mathbf{x}) = \sum_{k} P(C_k | \mathbf{x}) \left( 1 - P(C_k | \mathbf{x}') \right) $$
極限 $\mathbf{x}' \to \mathbf{x}$ を適用すると、
$$ P_{1NN}(\mathbf{x}) \to \sum_{k} P(C_k | \mathbf{x}) \left( 1 - P(C_k | \mathbf{x}) \right) = 1 - \sum_{k} P(C_k | \mathbf{x})^2 $$

ここで、$P_{1NN}$ とベイズ誤り率 $P_B = 1 - p_m$ の関係を評価します。
クラスの確率の二乗和 $\sum_k P(C_k | \mathbf{x})^2$ の下限を考えます。和の中には必ず最大の確率 $p_m$ の項が含まれるため、
$$ \sum_{k} P(C_k | \mathbf{x})^2 = p_m^2 + \sum_{k \neq m} P(C_k | \mathbf{x})^2 \ge p_m^2 $$
が成り立ちます。

これを $P_{1NN}$ の漸近式の不等式に適用すると、
$$ P_{1NN}(\mathbf{x}) \le 1 - p_m^2 $$
右辺を因数分解すると、
$$ 1 - p_m^2 = (1 - p_m)(1 + p_m) $$
事後確率の最大値 $p_m$ は明らかに $p_m \le 1$ であるため、$(1 + p_m) \le 2$ となります。
$$ (1 - p_m)(1 + p_m) \le (1 - p_m) \times 2 = 2 (1 - p_m) $$
ここで $1 - p_m = P_B(\mathbf{x})$ であるため、
$$ P_{1NN}(\mathbf{x}) \le 2 P_B(\mathbf{x}) $$
が得られます。

また、どのような分類器であれベイズ誤り率を下回ることはできないため $P_B(\mathbf{x}) \le P_{1NN}(\mathbf{x})$ が成り立ちます。
以上より、データ数が無限大の極限において、1-NNの漸近誤り率はベイズ誤り率の2倍以下に抑えられることが示されました。
$$ P_B \le P_{1NN} \le 2P_B $$
