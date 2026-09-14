# PRML Exercise 2.53

## 問題
von Mises分布の対数尤度関数を $\theta_0$ について最大化する条件から得られる方程式
$$ \sum_{n=1}^N \sin(\theta_n - \theta_0) = 0 $$
（PRML式 2.182）を変形し、$\theta_0$ の最尤推定量 $\theta_0^{\text{ML}}$ が次式を満たすことを示せ。
$$ \tan \theta_0^{\text{ML}} = \frac{\sum_{n=1}^N \sin \theta_n}{\sum_{n=1}^N \cos \theta_n} $$

## 解答と解説
与えられた方程式の左辺に対して、三角関数の加法定理 $\sin(A - B) = \sin A \cos B - \cos A \sin B$ を適用します。

$$ \sum_{n=1}^N \sin(\theta_n - \theta_0) = \sum_{n=1}^N (\sin \theta_n \cos \theta_0 - \cos \theta_n \sin \theta_0) = 0 $$

和の演算は $n$ に関するものなので、$\theta_0$ に依存する項（$\cos \theta_0$ と $\sin \theta_0$）は和の外に出すことができます。
$$ \cos \theta_0 \sum_{n=1}^N \sin \theta_n - \sin \theta_0 \sum_{n=1}^N \cos \theta_n = 0 $$

この式を移項して整理します。
$$ \cos \theta_0 \sum_{n=1}^N \sin \theta_n = \sin \theta_0 \sum_{n=1}^N \cos \theta_n $$

両辺を $\cos \theta_0 \sum_{n=1}^N \cos \theta_n$ で割ります（$\cos \theta_0 \neq 0$ および $\sum \cos \theta_n \neq 0$ を仮定）。
$$ \frac{\sin \theta_0}{\cos \theta_0} = \frac{\sum_{n=1}^N \sin \theta_n}{\sum_{n=1}^N \cos \theta_n} $$

$\tan \theta_0 = \frac{\sin \theta_0}{\cos \theta_0}$ であるため、最尤推定量 $\theta_0^{\text{ML}}$ は以下の関係を満たします。
$$ \tan \theta_0^{\text{ML}} = \frac{\sum_{n=1}^N \sin \theta_n}{\sum_{n=1}^N \cos \theta_n} $$

### 幾何学的な解釈
この結果は直感的にも理解できます。各データ点 $\theta_n$ を単位円上のベクトル $(\cos \theta_n, \sin \theta_n)$ とみなしたとき、それらのベクトルの和は
$$ \left( \sum_{n=1}^N \cos \theta_n, \sum_{n=1}^N \sin \theta_n \right) $$
となります。$\tan \theta_0^{\text{ML}}$ はこの合成ベクトルの $y$ 成分と $x$ 成分の比を表しており、最尤推定量 $\theta_0^{\text{ML}}$ はデータ点に対応するベクトル群の**平均的な方向（角度）**に一致することを意味しています。
