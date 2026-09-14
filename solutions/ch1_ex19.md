# 演習問題 1.19

## 問題設定
$D$ 次元空間における半径 $r$ の球について、大部分の体積が表面付近の薄い殻（シェル）に集中していることを示せ。

## 解答
半径 $r$ の $D$ 次元球の体積を $V(r)$ と表す。前問の結果より、球の体積は半径の $D$ 乗に比例する。
$$ V(r) \propto r^D $$

ここで、球の表面から厚さ $\epsilon$ （$0 < \epsilon \ll r$）の薄い殻を考える。
この殻の体積は、半径 $r$ の球の体積から、半径 $r-\epsilon$ の球の体積を引いたものである。
$$ V_{\text{shell}} = V(r) - V(r-\epsilon) $$

殻の体積が全体積に占める割合（比率）を $f$ とすると、
$$ f = \frac{V(r) - V(r-\epsilon)}{V(r)} = 1 - \frac{V(r-\epsilon)}{V(r)} $$
体積が $r^D$ に比例することを用いると、
$$ f = 1 - \left( \frac{r-\epsilon}{r} \right)^D = 1 - \left( 1 - \frac{\epsilon}{r} \right)^D $$

次元 $D$ が非常に大きい場合を考える。
$$ \lim_{D \to \infty} \left( 1 - \frac{\epsilon}{r} \right)^D $$
$\epsilon > 0$ であるため、$0 < 1 - \frac{\epsilon}{r} < 1$ となる。
したがって、$D$ が大きくなるにつれて、この項は急速に $0$ に近づく。
$$ \lim_{D \to \infty} \left( 1 - \frac{\epsilon}{r} \right)^D = 0 $$

よって、高次元においては
$$ f \approx 1 $$
となり、球の体積の大部分が表面付近の極めて薄い殻に集中することが示された。
これは「次元の呪い (Curse of Dimensionality)」の幾何学的な現れの一つである。
