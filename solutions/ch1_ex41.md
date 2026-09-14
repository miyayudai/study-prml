## Exercise 1.41: 相互情報量とエントロピーの関係 (Mutual Information and Entropy)

### 問題
確率の加法定理 (sum rule) と乗法定理 (product rule) を用いて、相互情報量 $I[x, y]$ が関係式 (1.121) を満たすことを示せ。
$$ I[x, y] = H[x] - H[x|y] = H[y] - H[y|x] = H[x] + H[y] - H[x, y] $$

### 解答と解説

相互情報量 $I[x, y]$ の定義式から出発します。
$$ I[x, y] = \iint p(x, y) \ln \frac{p(x, y)}{p(x)p(y)} \, dx \, dy $$

対数の性質 $\ln(A / B) = \ln A - \ln B$ および $\ln(AB) = \ln A + \ln B$ を用いて展開します。
$$ \ln \frac{p(x, y)}{p(x)p(y)} = \ln p(x, y) - \ln p(x) - \ln p(y) $$

これを相互情報量の積分に代入し、3つの項に分割します：
$$ I[x, y] = \iint p(x, y) \ln p(x, y) \, dx \, dy - \iint p(x, y) \ln p(x) \, dx \, dy - \iint p(x, y) \ln p(y) \, dx \, dy $$

各項をエントロピーの定義に当てはめて計算していきます。

**第1項:**
これは同時エントロピー $H[x, y]$ の定義の符号を反転させたものです。
$$ \iint p(x, y) \ln p(x, y) \, dx \, dy = -H[x, y] $$

**第2項:**
積分順序を交換し、確率の加法定理（周辺化：$\int p(x, y) \, dy = p(x)$）を用います。
$$ - \iint p(x, y) \ln p(x) \, dx \, dy = - \int \left( \int p(x, y) \, dy \right) \ln p(x) \, dx $$
$$ = - \int p(x) \ln p(x) \, dx = H[x] $$

**第3項:**
同様に、確率の加法定理（$\int p(x, y) \, dx = p(y)$）を用います。
$$ - \iint p(x, y) \ln p(y) \, dx \, dy = - \int \left( \int p(x, y) \, dx \right) \ln p(y) \, dy $$
$$ = - \int p(y) \ln p(y) \, dy = H[y] $$

これらを足し合わせると、関係式の3つ目の形が得られます。
$$ I[x, y] = -H[x, y] + H[x] + H[y] = H[x] + H[y] - H[x, y] \quad \text{--- (1)} $$

さらに、確率の乗法定理 $p(x, y) = p(y|x)p(x)$ を用いて導出された式 (1.112) の関係 $H[x, y] = H[y|x] + H[x]$ を式(1)に代入します。
$$ I[x, y] = H[x] + H[y] - (H[y|x] + H[x]) $$
$$ = H[y] - H[y|x] \quad \text{--- (2)} $$

同様に、$x$ と $y$ の対称性から $p(x, y) = p(x|y)p(y)$ より得られる関係 $H[x, y] = H[x|y] + H[y]$ を式(1)に代入します。
$$ I[x, y] = H[x] + H[y] - (H[x|y] + H[y]) $$
$$ = H[x] - H[x|y] \quad \text{--- (3)} $$

以上、(1), (2), (3) により、求める関係式がすべて示されました。
$$ I[x, y] = H[x] - H[x|y] = H[y] - H[y|x] = H[x] + H[y] - H[x, y] $$

**物理的・情報論的解釈**：
相互情報量 $I[x, y]$ は、$y$ を知ることで $x$ の不確実性がどれだけ減少するか（またはその逆）を表します。式 $H[x] - H[x|y]$ は、「$x$ が元々持っていた不確実性」から「$y$ を知った後でも残る $x$ の不確実性」を引いたものであり、これが $y$ によって得られた $x$ に関する情報量に他ならないことを数式で示しています。
