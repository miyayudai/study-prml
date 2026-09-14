# 演習問題 2.21: フォン・ミーゼス分布の正規化

本演習問題では、角度を扱う周期的な確率分布であるフォン・ミーゼス分布（von Mises distribution）の確率密度関数が、全区間で積分すると1になる（正規化されている）ことを示します。

## 1. フォン・ミーゼス分布の定義

角度 $\theta \in [0, 2\pi)$ に対するフォン・ミーゼス分布の確率密度関数は次のように定義されます。

$$
p(\theta | \theta_0, m) = \frac{1}{2\pi I_0(m)} \exp\left( m \cos(\theta - \theta_0) \right)
$$

ここで：
- $\theta_0$ は平均方向を表すパラメータです。
- $m > 0$ は集中度パラメータ（ガウス分布の分散の逆数に相当）です。
- $I_0(m)$ は第0種の変形ベッセル関数（modified Bessel function of the first kind of order zero）です。

## 2. 積分が1であることの証明

この分布が適切な確率密度関数であるためには、区間 $[0, 2\pi]$ での積分が1になる必要があります。
$$
\int_{0}^{2\pi} p(\theta | \theta_0, m) d\theta = 1
$$
これを示すために、左辺の積分を計算します。

$$
\int_{0}^{2\pi} \frac{1}{2\pi I_0(m)} \exp\left( m \cos(\theta - \theta_0) \right) d\theta
$$

定数係数を積分の外に出します。
$$
\frac{1}{2\pi I_0(m)} \int_{0}^{2\pi} \exp\left( m \cos(\theta - \theta_0) \right) d\theta
$$

変数変換を行います。$\phi = \theta - \theta_0$ と置くと、$d\phi = d\theta$ となります。積分区間は $[-\theta_0, 2\pi - \theta_0]$ となります。
$$
\int_{-\theta_0}^{2\pi - \theta_0} \exp\left( m \cos\phi \right) d\phi
$$

被積分関数 $\exp(m \cos\phi)$ は $\phi$ に関して周期 $2\pi$ を持つ周期関数です。周期関数の1周期分の積分は、積分区間の開始位置に依存しません。したがって、積分区間を $[0, 2\pi]$ にシフトさせることができます。

$$
\int_{0}^{2\pi} \exp\left( m \cos\phi \right) d\phi
$$

## 3. 変形ベッセル関数の利用

ここで、第0種の変形ベッセル関数 $I_0(m)$ の積分表示を思い出します。$I_0(m)$ は次のように定義されています。
$$
I_0(m) = \frac{1}{2\pi} \int_{0}^{2\pi} \exp\left( m \cos\phi \right) d\phi
$$

したがって、積分の部分は $2\pi I_0(m)$ に等しくなります。
$$
\int_{0}^{2\pi} \exp\left( m \cos\phi \right) d\phi = 2\pi I_0(m)
$$

## 4. 結論

これを元の式に代入します。

$$
\frac{1}{2\pi I_0(m)} \times 2\pi I_0(m) = 1
$$

以上により、フォン・ミーゼス分布が区間 $[0, 2\pi]$ で正しく正規化されている（積分して1になる）ことが証明されました。これは、フォン・ミーゼス分布が周期的な変数に対する適切な確率分布であることを示しています。
