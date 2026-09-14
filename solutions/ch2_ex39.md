# 演習問題 2.39

## 問題設定
連続型確率変数のエントロピー（微分エントロピー）について、線形変換 $y = \Delta x$ がエントロピーに与える影響を示せ。

## 解答と証明

確率変数 $x$ の微分エントロピー $H[x]$ は次のように定義されます。
$$ H[x] = -\int p(x) \ln p(x) dx $$

新しい変数 $y = \Delta x$ を考えます。変数変換の公式より、確率密度関数 $p(y)$ は以下のように表されます。
$$ p(y) = p(x) \left| rac{dx}{dy} ight| = rac{1}{\Delta} p\left(rac{y}{\Delta}ight) $$

変数 $y$ の微分エントロピー $H[y]$ を計算します。
$$
egin{align*}
H[y] &= -\int p(y) \ln p(y) dy \
&= -\int p(x) \left| rac{dx}{dy} ight| \ln \left( p(x) \left| rac{dx}{dy} ight| ight) \left| rac{dy}{dx} ight| dx \
&= -\int p(x) \ln \left( rac{p(x)}{\Delta} ight) dx \
&= -\int p(x) (\ln p(x) - \ln \Delta) dx \
&= -\int p(x) \ln p(x) dx + \ln \Delta \int p(x) dx \
&= H[x] + \ln \Delta
\end{align*}
$$

以上により、$y = \Delta x$ というスケール変換を行った場合、微分エントロピーは元のエントロピーに $\ln \Delta$ を加えたものになることが示されました。離散エントロピーとは異なり、微分エントロピーは変数変換によって値が変化し、負の値を取ることも可能であるという性質を明確に表しています。
