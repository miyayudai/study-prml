## Exercise 2.42 (周期変数と平均方向)

角度や時刻、カレンダーの月などの「周期性」を持つ変数（周期変数）を扱う場合、通常のユークリッド空間上の算術平均を適用することは不適切です。例えば、角度 $1^\circ$ と $359^\circ$ の単純な平均は $180^\circ$ になってしまいますが、幾何学的には明らかに $0^\circ$ （または $360^\circ$）が妥当な「平均方向」です。

この演習問題では、観測された周期変数を2次元平面上の単位ベクトルとして表現し、それらのベクトル和の方向から適切な平均方向の式を導出します。

### 1. 周期変数のベクトル表現

周期変数（角度）の観測値の集合を $\mathcal{D} = \{\theta_1, \theta_2, \dots, \theta_N\}$ とします。それぞれの角度は $0 \le \theta_n < 2\pi$ で定義されているとします。
これらを適切に扱うため、各角度 $\theta_n$ を2次元デカルト座標系における単位円上のベクトル $\mathbf{x}_n$ にマッピングします：

$$ \mathbf{x}_n = \begin{pmatrix} \cos\theta_n \\ \sin\theta_n \end{pmatrix} $$

このように表現することで、角度の周期性という厄介な性質が、2次元平面上の自然な幾何学的構造として扱えるようになります。

### 2. ベクトル和に基づく平均方向の導出

この単位ベクトルの集合に対して、算術平均ベクトル（重心） $\bar{\mathbf{x}}$ を計算します：

$$ \bar{\mathbf{x}} = \frac{1}{N}\sum_{n=1}^N \mathbf{x}_n = \frac{1}{N} \begin{pmatrix} \sum_{n=1}^N \cos\theta_n \\ \sum_{n=1}^N \sin\theta_n \end{pmatrix} $$

この重心ベクトル $\bar{\mathbf{x}}$ は一般に単位ベクトルではありません（長さが1未満になります）。このベクトルの「方向」こそが、私たちが求めるべき真の平均方向 $\bar{\theta}$ になります。
そこで、重心ベクトル $\bar{\mathbf{x}}$ を、長さ $\bar{r}$ と角度 $\bar{\theta}$ を用いた極座標系で表します：

$$ \bar{\mathbf{x}} = \begin{pmatrix} \bar{r} \cos\bar{\theta} \\ \bar{r} \sin\bar{\theta} \end{pmatrix} $$

これら2つの $\bar{\mathbf{x}}$ の表現を要素ごとに等置すると、以下の関係式が得られます：

$$ \bar{r} \cos\bar{\theta} = \frac{1}{N}\sum_{n=1}^N \cos\theta_n \quad \cdots \text{(Eq. 1)} $$
$$ \bar{r} \sin\bar{\theta} = \frac{1}{N}\sum_{n=1}^N \sin\theta_n \quad \cdots \text{(Eq. 2)} $$

我々の目的は平均方向 $\bar{\theta}$ を求めることです。Eq. 2をEq. 1で割ることで、未知の長さ $\bar{r}$ を消去します：

$$ \frac{\bar{r} \sin\bar{\theta}}{\bar{r} \cos\bar{\theta}} = \frac{\frac{1}{N}\sum_{n=1}^N \sin\theta_n}{\frac{1}{N}\sum_{n=1}^N \cos\theta_n} $$

これを整理すると、$\tan \bar{\theta}$ に関する式が得られます：

$$ \tan\bar{\theta} = \frac{\sum_{n=1}^N \sin\theta_n}{\sum_{n=1}^N \cos\theta_n} $$

### 3. arctan2関数による厳密な角度の決定

上記の式から $\bar{\theta}$ を求めるために逆正接関数（$\arctan$）を適用することを考えますが、単純な $\arctan$ 関数では値域が $(-\pi/2, \pi/2)$ に制限されるため、ベクトルの存在する象限（Quadrant）を正しく判定できません。

例えば、$x > 0, y > 0$ の第一象限のベクトルと、$x < 0, y < 0$ の第三象限のベクトルは、共に正の $\tan$ 値を持ちますが、方向は $180^\circ$ 異なります。

これを解決するために、プログラミングや数学で広く用いられる $\mathrm{arctan2}(y, x)$ 関数（四象限逆正接関数）を使用します。この関数は引数 $x$ と $y$ の符号を個別に考慮することで、$[-\pi, \pi]$ の正しい範囲の角度を返します。

したがって、最終的な平均方向 $\bar{\theta}$ は次のように表現されます：

$$ \bar{\theta} = \mathrm{arctan2}\left( \sum_{n=1}^N \sin\theta_n, \sum_{n=1}^N \cos\theta_n \right) $$

この導出は、周期変数の平均を求めるという直感的な操作が、ユークリッド空間上の単位ベクトルを用いた重心計算という幾何学的に明確な枠組みによって正当化されることを示しています。
