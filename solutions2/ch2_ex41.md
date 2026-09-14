## Exercise 2.41 (Student's t分布のEMアルゴリズム)

Student's t分布は、外れ値（アウトライアー）に対してロバストな確率分布として広く用いられます。この演習問題では、Student's t分布を無限個のガウス分布の重ね合わせ（混合ガウスモデルの連続版）として表現した際、EMアルゴリズムのEステップにおける潜在変数の事後期待値が、どのように外れ値の重みを抑制するかを数学的に証明し、その物理的な解釈を与えます。

### 1. Student's t分布の無限混合モデル表現

PRML 2.3.7節で議論されているように、自由度 $\nu$、平均 $\mu$、精度パラメータ $\tau$ を持つ1次元のStudent's t分布 $St(x|\mu, \tau, \nu)$ は、精度がガンマ分布に従うガウス分布の無限混合として定式化できます。
各データ点 $x_n$ に対して、潜在変数（隠れ変数）となる精度スケーリング係数 $\eta_n$ を導入します。モデルの生成過程は以下のように記述されます。

まず、潜在変数 $\eta_n$ がパラメータ $\nu/2, \nu/2$ のガンマ分布から生成されます：
$$ p(\eta_n|\nu) = \mathrm{Gam}\left(\eta_n \middle| \frac{\nu}{2}, \frac{\nu}{2}\right) $$
ここで、ガンマ分布の定義は以下の通りです：
$$ \mathrm{Gam}(\eta|a, b) = \frac{b^a}{\Gamma(a)} \eta^{a-1} \exp(-b\eta) $$

次に、観測データ $x_n$ が、平均 $\mu$、精度 $\eta_n \tau$ のガウス分布から生成されます：
$$ p(x_n|\eta_n, \mu, \tau) = \mathcal{N}\left(x_n \middle| \mu, (\eta_n \tau)^{-1}\right) = \sqrt{\frac{\eta_n \tau}{2\pi}} \exp\left( -\frac{\eta_n \tau}{2} (x_n - \mu)^2 \right) $$

これらを周辺化（積分消去）することで、観測データの周辺分布としてStudent's t分布が得られます：
$$ p(x_n|\mu, \tau, \nu) = \int_0^\infty p(x_n|\eta_n, \mu, \tau) p(\eta_n|\nu) d\eta_n = St(x_n|\mu, \tau, \nu) $$

### 2. Eステップ：潜在変数の事後分布の導出

EMアルゴリズムにおいて、Eステップでは現在のパラメータ推定値 $\mu, \tau, \nu$ が与えられた下での潜在変数 $\eta_n$ の事後分布 $p(\eta_n|x_n, \mu, \tau, \nu)$ を計算します。ベイズの定理より、この事後分布は尤度と事前分布の積に比例します：

$$ p(\eta_n|x_n, \mu, \tau, \nu) \propto p(x_n|\eta_n, \mu, \tau) p(\eta_n|\nu) $$

具体的な関数形を代入すると：
$$ p(\eta_n|x_n, \mu, \tau, \nu) \propto \left[ \eta_n^{1/2} \exp\left( -\frac{\eta_n \tau}{2} (x_n - \mu)^2 \right) \right] \times \left[ \eta_n^{\nu/2 - 1} \exp\left( -\frac{\nu}{2}\eta_n \right) \right] $$

$\eta_n$ について項をまとめると：
$$ p(\eta_n|x_n, \mu, \tau, \nu) \propto \eta_n^{(\nu+1)/2 - 1} \exp\left( -\eta_n \left( \frac{\nu}{2} + \frac{\tau}{2}(x_n - \mu)^2 \right) \right) $$

この形状は、明らかにガンマ分布 $\mathrm{Gam}(\eta_n|a_N, b_N)$ の関数形 $\eta_n^{a_N-1}\exp(-b_N \eta_n)$ と一致しています（ガウス分布の精度に対するガンマ事前分布の共役性）。したがって、事後分布のパラメータは以下のようになります：
$$ a_N = \frac{\nu + 1}{2} $$
$$ b_N = \frac{\nu + \tau(x_n - \mu)^2}{2} $$
すなわち、事後分布は次のように求まります：
$$ p(\eta_n|x_n, \mu, \tau, \nu) = \mathrm{Gam}\left( \eta_n \middle| \frac{\nu+1}{2}, \frac{\nu + \tau(x_n - \mu)^2}{2} \right) $$

### 3. 潜在変数の事後期待値とロバスト性の解釈

EMアルゴリズムのMステップでは、完全データの対数尤度の期待値を最大化しますが、その際に必要となるのは潜在変数 $\eta_n$ の事後分布における期待値 $\mathbb{E}[\eta_n]$ です。ガンマ分布 $\mathrm{Gam}(\eta|a, b)$ の期待値は $a/b$ で与えられるため、これを計算すると：

$$ \mathbb{E}[\eta_n] = \frac{a_N}{b_N} = \frac{\frac{\nu+1}{2}}{\frac{\nu + \tau(x_n - \mu)^2}{2}} = \frac{\nu + 1}{\nu + \tau(x_n - \mu)^2} $$

この式こそが、Student's t分布のロバスト性の源泉を表しています。この結果を詳細に解釈してみましょう。

1. **外れ値の重み抑制機構**: 
   データ点 $x_n$ が現在の平均 $\mu$ から大きく離れている（つまり外れ値である）とします。このとき、二乗誤差 $(x_n - \mu)^2$ は非常に大きな値になります。分母にこの項が含まれているため、極限 $\lim_{|x_n - \mu| \to \infty} \mathbb{E}[\eta_n] = 0$ となります。
   $\eta_n$ は、データ点 $x_n$ が生成されたガウス分布の「精度スケーリング係数」です。これが0に近づくということは、「このデータ点は精度が非常に低い（分散が無限大に近い）分布から生成された信頼性の低いデータである」とモデルが自己評価し、Mステップでの平均 $\mu$ や精度 $\tau$ の更新において、この外れ値の影響力（重み）を自動的にゼロに近づけて無視することに相当します。

2. **正常なデータ点の扱い**:
   逆に $x_n \approx \mu$ の場合、$\mathbb{E}[\eta_n] \approx \frac{\nu+1}{\nu} \approx 1$ （$\nu$が十分大きい場合）となり、通常のガウス分布と同様の重みを与えます。

3. **自由度 $\nu$ の役割**:
   $\nu \to \infty$ の極限では、$\mathbb{E}[\eta_n] \to 1$ となり、すべてのデータ点に均等な重みが与えられます。これはStudent's t分布が標準的なガウス分布に帰着することに対応しています。逆に $\nu$ が小さいほど、少しの外れ値に対しても敏感に $\mathbb{E}[\eta_n]$ が低下し、より強力なロバスト性を発揮します。

以上より、Student's t分布を用いた推論では、潜在変数の事後期待値がデータ点までのマハラノビス距離（に比例する値）の二乗に反比例して減衰することで、外れ値に対する強靭なロバスト性が自然な形で組み込まれていることが示されました。
