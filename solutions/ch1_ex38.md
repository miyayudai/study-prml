## Exercise 1.38: イェンセンの不等式と数学的帰納法 (Jensen's Inequality and Mathematical Induction)

### 問題
数学的帰納法を用いて、凸関数の不等式 (1.114) がイェンセンの不等式 (1.115) を意味することを示せ。

### 解答と解説

関数 $f(x)$ を凸関数とします。凸関数の定義（式1.114）から、$0 \leq \lambda \leq 1$ に対して次の不等式が成り立ちます：
$$ f(\lambda a + (1 - \lambda)b) \leq \lambda f(a) + (1 - \lambda)f(b) \quad \text{--- (A)} $$

証明すべきイェンセンの不等式（式1.115）は、非負で和が1となる係数 $\lambda_1, \ldots, \lambda_M$ （すなわち $\lambda_i \geq 0$, $\sum_{i=1}^M \lambda_i = 1$）に対して次が成り立つというものです：
$$ f\left( \sum_{i=1}^M \lambda_i x_i \right) \leq \sum_{i=1}^M \lambda_i f(x_i) \quad \text{--- (B)} $$

この命題 (B) を、変数の数 $M$ に対する数学的帰納法を用いて証明します。

**1. $M=1$ および $M=2$ の場合**

- $M=1$ のとき、条件 $\sum_{i=1}^1 \lambda_i = 1$ より $\lambda_1 = 1$ となります。
  $$ f(\lambda_1 x_1) = f(x_1) \leq 1 \cdot f(x_1) $$
  となり、明らかに成立します。
- $M=2$ のとき、$\lambda_1 + \lambda_2 = 1$ より $\lambda_2 = 1 - \lambda_1$ です。したがって、
  $$ f(\lambda_1 x_1 + \lambda_2 x_2) = f(\lambda_1 x_1 + (1 - \lambda_1)x_2) \leq \lambda_1 f(x_1) + (1 - \lambda_1)f(x_2) = \lambda_1 f(x_1) + \lambda_2 f(x_2) $$
  これは凸関数の定義 (A) そのものであり、成立します。

**2. $M=k$ のときの仮定**

ある整数 $k \geq 2$ に対して、命題 (B) が成り立つと仮定します。すなわち、$\sum_{i=1}^k \eta_i = 1$ かつ $\eta_i \geq 0$ を満たす任意の $\{\eta_i\}$ に対して、次が成り立つとします：
$$ f\left( \sum_{i=1}^k \eta_i x_i \right) \leq \sum_{i=1}^k \eta_i f(x_i) \quad \text{--- (C)} $$

**3. $M=k+1$ の場合の証明**

$M = k+1$ の場合を考えます。和が1となる係数 $\lambda_1, \ldots, \lambda_{k+1}$ に対する関数の値は、次のように書けます：
$$ f\left( \sum_{i=1}^{k+1} \lambda_i x_i \right) = f\left( \sum_{i=1}^k \lambda_i x_i + \lambda_{k+1} x_{k+1} \right) $$

ここで、$S_k = \sum_{i=1}^k \lambda_i$ と置きます。$\sum_{i=1}^{k+1} \lambda_i = 1$ より、$S_k + \lambda_{k+1} = 1$ です。
もし $S_k = 0$ ならば、$\lambda_1 = \dots = \lambda_k = 0, \lambda_{k+1} = 1$ となり自明に成立するため、$S_k > 0$ と仮定して進めます。

総和を次のように変形します：
$$ \sum_{i=1}^k \lambda_i x_i + \lambda_{k+1} x_{k+1} = S_k \left( \sum_{i=1}^k \frac{\lambda_i}{S_k} x_i \right) + \lambda_{k+1} x_{k+1} $$

$S_k + \lambda_{k+1} = 1$ であるため、$M=2$ の場合の不等式（凸関数の定義）を適用することができます：
$$ f\left( S_k \left( \sum_{i=1}^k \frac{\lambda_i}{S_k} x_i \right) + \lambda_{k+1} x_{k+1} \right) \leq S_k f\left( \sum_{i=1}^k \frac{\lambda_i}{S_k} x_i \right) + \lambda_{k+1} f(x_{k+1}) $$

次に、右辺の第1項に注目します。係数 $\eta_i = \frac{\lambda_i}{S_k}$ を考えると、$\sum_{i=1}^k \eta_i = \sum_{i=1}^k \frac{\lambda_i}{S_k} = \frac{S_k}{S_k} = 1$ を満たします。
したがって、帰納法の仮定 (C) を適用できます：
$$ f\left( \sum_{i=1}^k \frac{\lambda_i}{S_k} x_i \right) \leq \sum_{i=1}^k \frac{\lambda_i}{S_k} f(x_i) $$

これを先ほどの式に代入します：
$$ S_k f\left( \sum_{i=1}^k \frac{\lambda_i}{S_k} x_i \right) + \lambda_{k+1} f(x_{k+1}) \leq S_k \left( \sum_{i=1}^k \frac{\lambda_i}{S_k} f(x_i) \right) + \lambda_{k+1} f(x_{k+1}) $$
$$ = \sum_{i=1}^k \lambda_i f(x_i) + \lambda_{k+1} f(x_{k+1}) = \sum_{i=1}^{k+1} \lambda_i f(x_i) $$

以上の結果をまとめると、
$$ f\left( \sum_{i=1}^{k+1} \lambda_i x_i \right) \leq \sum_{i=1}^{k+1} \lambda_i f(x_i) $$
となり、$M=k+1$ の場合でも命題 (B) が成立することが示されました。

したがって、数学的帰納法により、すべての自然数 $M$ に対してイェンセンの不等式が成り立つことが証明されました。
