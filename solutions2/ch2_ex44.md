## Exercise 2.44 (フォン・ミーゼス分布の最尤推定)

フォン・ミーゼス分布のパラメータ（平均方向 $\theta_0$ と集中度 $m$）を、与えられた観測データから推定する問題を考えます。この演習では、最尤推定（Maximum Likelihood Estimation）を用いて平均方向 $\theta_0$ を求めた結果が、前回の演習（Exercise 2.42）で幾何学的に導出した「周期変数の平均方向」と完全に一致することを示します。

### 1. 対数尤度関数の立式

独立同分布（i.i.d.）に従う $N$ 個の角度データ $\mathcal{D} = \{\theta_1, \theta_2, \dots, \theta_N\}$ が与えられたとします。フォン・ミーゼス分布 $p(\theta|\theta_0, m) = \frac{1}{2\pi I_0(m)} \exp(m \cos(\theta - \theta_0))$ を仮定すると、尤度関数は各データの確率の積となります。

$$ p(\mathcal{D}|\theta_0, m) = \prod_{n=1}^N \frac{1}{2\pi I_0(m)} \exp\left( m \cos(\theta_n - \theta_0) \right) $$

計算を容易にするため、対数尤度関数 $\ln p(\mathcal{D}|\theta_0, m)$ をとります：

$$ \ln p(\mathcal{D}|\theta_0, m) = \sum_{n=1}^N \left[ -\ln(2\pi) - \ln I_0(m) + m \cos(\theta_n - \theta_0) \right] $$
$$ = -N \ln(2\pi) - N \ln I_0(m) + m \sum_{n=1}^N \cos(\theta_n - \theta_0) $$

### 2. 平均方向 $\theta_0$ に関する対数尤度の最大化

最尤推定量 $\theta_0^{ML}$ を見つけるため、対数尤度関数を未知パラメータ $\theta_0$ で偏微分し、それを0と置きます（停留点条件）。

$$ \frac{\partial}{\partial \theta_0} \ln p(\mathcal{D}|\theta_0, m) = m \sum_{n=1}^N \frac{\partial}{\partial \theta_0} \cos(\theta_n - \theta_0) $$

合成関数の微分則より $\frac{\partial}{\partial \theta_0} \cos(\theta_n - \theta_0) = \sin(\theta_n - \theta_0)$ となるため：

$$ m \sum_{n=1}^N \sin(\theta_n - \theta_0) = 0 $$

ここで、$m > 0$ であると仮定すれば両辺を $m$ で割ることができます：

$$ \sum_{n=1}^N \sin(\theta_n - \theta_0) = 0 $$

### 3. 三角関数の加法定理を用いた展開と整理

この方程式を解くために、サイン関数の加法定理 $\sin(\alpha - \beta) = \sin\alpha \cos\beta - \cos\alpha \sin\beta$ を用いて $\sin(\theta_n - \theta_0)$ を展開します：

$$ \sum_{n=1}^N \left( \sin\theta_n \cos\theta_0 - \cos\theta_n \sin\theta_0 \right) = 0 $$

和の記号 $\sum$ は $n$ に依存しない定数項 $\cos\theta_0$ と $\sin\theta_0$ を外に出すことができます：

$$ \cos\theta_0 \sum_{n=1}^N \sin\theta_n - \sin\theta_0 \sum_{n=1}^N \cos\theta_n = 0 $$

項を移項して等式を整理します：

$$ \cos\theta_0 \sum_{n=1}^N \sin\theta_n = \sin\theta_0 \sum_{n=1}^N \cos\theta_n $$

両辺を $\cos\theta_0$ と $\sum_{n=1}^N \cos\theta_n$ で割ることで、$ \frac{\sin\theta_0}{\cos\theta_0} $ の形を作ります：

$$ \frac{\sin\theta_0}{\cos\theta_0} = \frac{\sum_{n=1}^N \sin\theta_n}{\sum_{n=1}^N \cos\theta_n} $$

したがって、最尤推定量 $\theta_0^{ML}$ に関する次の方程式が得られます：

$$ \tan\theta_0^{ML} = \frac{\sum_{n=1}^N \sin\theta_n}{\sum_{n=1}^N \cos\theta_n} $$

### 4. 幾何学的な平均方向との一致

得られた方程式は、Exercise 2.42 で導出した「単位ベクトルの和から求めた平均方向」 $\bar{\theta}$ の方程式と完全に一致しています。
前回の演習と同様に、角度が属する正しい象限を特定するためには $\mathrm{arctan2}$ 関数を使用する必要があります：

$$ \theta_0^{ML} = \mathrm{arctan2}\left( \sum_{n=1}^N \sin\theta_n, \sum_{n=1}^N \cos\theta_n \right) $$

**物理的な解釈**:
この結果は非常に美しく、重要な事実を示唆しています。ガウス分布において「対数尤度を最大化する平均パラメータは、データの算術平均になる」という事実と同様に、フォン・ミーゼス分布においても「対数尤度を最大化する平均方向は、データを単位ベクトルとみなしたときのベクトル和の方向になる」のです。
最尤推定という確率論・推計統計学的なアプローチと、ベクトル和の方向を求めるという幾何学的なアプローチが、全く同じ結論を導き出すことが数学的に証明されました。
