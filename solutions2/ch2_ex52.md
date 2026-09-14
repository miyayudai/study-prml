## Exercise 2.52 (拡大縮小不変性と尺度母数事前分布)

確率密度関数が尺度パラメータ (scale parameter) $\sigma$ を持つ場合、すなわち $p(x|\sigma) = \frac{1}{\sigma}f\left(\frac{x}{\sigma}\right)$ の形式で表される場合を考えます。このとき、測定単位（スケール）の変更に対する不変性を要請することで、$\sigma$ の事前分布が $p(\sigma) \propto \frac{1}{\sigma}$ となることを示します。

### スケール変換の定義
元の観測変数 $x$ に対して、定数 $c > 0$ を掛けてスケール変換した新しい変数 $\widehat{x} = cx$ を考えます。このスケール変換に対応して、尺度パラメータも $\widehat{\sigma} = c\sigma$ に変換されると仮定します。

元の変数と新しい変数の確率密度の間には、変数変換の公式により次の関係が成り立ちます：
$$ p(x|\sigma) \, dx = p(\widehat{x}|\widehat{\sigma}) \, d\widehat{x} $$
実際に $d\widehat{x} = c \, dx$ および $\widehat{\sigma} = c\sigma$ を用いると、元の分布の定義と整合することが確認できます。

### 事前分布に対する不変性の要請
無情報事前分布 (non-informative prior) を構築するため、尺度パラメータの事前分布 $p(\sigma)$ もスケール変換に対して不変であるべきだと考えます。すなわち、特定の区間に $\sigma$ が含まれる確率は、スケール変換後の区間に $\widehat{\sigma}$ が含まれる確率と等しくなければなりません：
$$ p(\sigma) \, d\sigma = p(\widehat{\sigma}) \, d\widehat{\sigma} $$

ここで、$\widehat{\sigma} = c\sigma$ より、微分の関係は $d\widehat{\sigma} = c \, d\sigma$ となります。これを上の条件式に代入すると、
$$ p(\sigma) \, d\sigma = p(c\sigma) \, c \, d\sigma $$
したがって、以下の関係式が得られます：
$$ p(\sigma) = c \, p(c\sigma) $$

### 事前分布の導出
得られた等式 $p(\sigma) = c \, p(c\sigma)$ は、任意の $\sigma > 0$ および $c > 0$ に対して成立する必要があります。ここで、特別に $c = \frac{1}{\sigma}$ と選ぶと、
$$ p(\sigma) = \frac{1}{\sigma} p(1) $$
となります。$p(1)$ は定数であるため、これを比例定数として無視すれば、
$$ p(\sigma) \propto \frac{1}{\sigma} $$
という結果が得られます。

### 対数スケールでの解釈
この結果は、対数変換を行った変数について考えるとより直感的です。$\eta = \ln \sigma$ とおくと、$d\eta = \frac{1}{\sigma} d\sigma$ となります。これを事前分布の確率要素に代入すると、
$$ p(\sigma) \, d\sigma \propto \frac{1}{\sigma} d\sigma = d\eta $$
となり、$\eta = \ln \sigma$ に関する事前分布 $p(\eta)$ が一様分布（定数）になることを示しています。つまり、対数スケール上で「平坦」であることと、元のスケールで $1/\sigma$ に比例することは等価であり、これをジェフリーズ事前分布 (Jeffreys prior) と呼びます。
