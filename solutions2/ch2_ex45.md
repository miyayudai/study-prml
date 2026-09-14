## Exercise 2.45 (集中度大でのフォン・ミーゼス分布のガウス近似)

フォン・ミーゼス分布は「円周上のガウス分布」としばしば呼ばれます。その理由を数学的に裏付けるため、集中度パラメータ $m$ が十分に大きい（$m \gg 1$）極限において、フォン・ミーゼス分布が通常の1次元ガウス分布（正規分布）に近似できることを示します。

### 1. テーラー展開による指数部の近似

フォン・ミーゼス分布の定義は以下の通りです：

$$ p(\theta|\theta_0, m) = \frac{1}{2\pi I_0(m)} \exp\left( m \cos(\theta - \theta_0) \right) $$

集中度 $m$ が非常に大きい場合、分布は平均方向 $\theta_0$ の周りに極めて鋭く集中します。つまり、$p(\theta|\theta_0, m)$ は $\theta \approx \theta_0$ の近傍以外では事実上ゼロになります。
そこで、$\theta$ が $\theta_0$ に非常に近いと仮定し、微小な変位 $\epsilon = \theta - \theta_0$ を導入します。

コサイン関数 $\cos(\epsilon)$ を $\epsilon = 0$ の周りでテーラー展開（マクローリン展開）すると、以下のようになります：

$$ \cos(\epsilon) = 1 - \frac{\epsilon^2}{2!} + \frac{\epsilon^4}{4!} - \dots $$

$\epsilon$ は微小であるため、2次までの項を残して高次項を無視する近似（小角近似）が有効です：

$$ \cos(\theta - \theta_0) \approx 1 - \frac{(\theta - \theta_0)^2}{2} $$

これをフォン・ミーゼス分布の指数部の項に代入します：

$$ \exp\left( m \cos(\theta - \theta_0) \right) \approx \exp\left( m \left( 1 - \frac{(\theta - \theta_0)^2}{2} \right) \right) $$
$$ = \exp(m) \exp\left( -\frac{m}{2} (\theta - \theta_0)^2 \right) $$

### 2. 変形ベッセル関数の漸近展開

次に、規格化定数に含まれる第1種変形ベッセル関数 $I_0(m)$ の振る舞いを評価します。
大きな $m$ に対して、$I_0(m)$ は次のような漸近展開を持つことが知られています：

$$ I_0(m) \approx \frac{\exp(m)}{\sqrt{2\pi m}} \quad \text{for} \quad m \gg 1 $$

### 3. ガウス分布への帰着

1と2で導出した近似式を、元のフォン・ミーゼス分布に代入して整理します。

$$ p(\theta|\theta_0, m) \approx \frac{1}{2\pi \left( \frac{\exp(m)}{\sqrt{2\pi m}} \right)} \times \exp(m) \exp\left( -\frac{m}{2} (\theta - \theta_0)^2 \right) $$

定数項の $\exp(m)$ が分子と分母で打ち消し合います：

$$ p(\theta|\theta_0, m) \approx \frac{\sqrt{2\pi m}}{2\pi} \exp\left( -\frac{m}{2} (\theta - \theta_0)^2 \right) $$

係数を整理すると $\frac{\sqrt{m}}{\sqrt{2\pi}}$ となるため、最終的な形は以下のようになります：

$$ p(\theta|\theta_0, m) \approx \frac{1}{\sqrt{2\pi (1/m)}} \exp\left( -\frac{(\theta - \theta_0)^2}{2(1/m)} \right) $$

### 4. 結論とパラメータの対応付け

得られた式の右辺は、明らかに1次元ガウス分布 $\mathcal{N}(\theta | \mu, \sigma^2)$ の確率密度関数

$$ \mathcal{N}(\theta | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(\theta - \mu)^2}{2\sigma^2} \right) $$

と完全に一致しています。パラメータを比較すると、以下の対応関係が成り立ちます：

- **平均** $\mu = \theta_0$
- **分散** $\sigma^2 = \frac{1}{m}$

**物理的な解釈**:
集中度 $m$ が無限大に近づくと、分散 $\sigma^2 = 1/m$ は0に近づきます。これは、データが平均方向 $\theta_0$ のごく近傍に密集し、円周（周期性）の影響が無視できるほど狭い範囲に局在化することを意味します。この局所的な領域内では、円周上の曲線はほぼ直線（接空間）とみなせるため、周期分布であるフォン・ミーゼス分布が局所的なガウス分布へと漸近するのです。この性質により、高集中度の角度データに対してはガウス分布の理論を近似的に適用することが正当化されます。
