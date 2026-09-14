# 演習問題 2.41

## 問題設定
カーネル密度推定において、確率密度関数が正規化されていること（積分して1になること）を示せ。

## 解答と証明

カーネル密度推定法（Kernel Density Estimation; KDE）において、データ点 $\mathbf{x}_1, \dots, \mathbf{x}_N$ に基づく未知の確率密度関数 $p(\mathbf{x})$ の推定量は、次のように定義されます。
$$ p(\mathbf{x}) = rac{1}{N} \sum_{n=1}^N rac{1}{h^D} k\left( rac{\mathbf{x} - \mathbf{x}_n}{h} ight) $$
ここで、$D$ は空間の次元、$h$ は平滑化パラメータ（バンド幅）、$k(\mathbf{u})$ はカーネル関数です。

カーネル関数 $k(\mathbf{u})$ は、それ自体が正当な確率密度関数としての性質を満たしている必要があります。すなわち、
$$ k(\mathbf{u}) \geq 0 $$
$$ \int k(\mathbf{u}) d\mathbf{u} = 1 $$

推定された確率密度関数 $p(\mathbf{x})$ が空間全体で正規化されている（積分が1になる）ことを確認するため、全空間にわたって積分を行います。
$$
egin{align*}
\int p(\mathbf{x}) d\mathbf{x} &= \int \left( rac{1}{N} \sum_{n=1}^N rac{1}{h^D} k\left( rac{\mathbf{x} - \mathbf{x}_n}{h} ight) ight) d\mathbf{x} \
&= rac{1}{N} \sum_{n=1}^N rac{1}{h^D} \int k\left( rac{\mathbf{x} - \mathbf{x}_n}{h} ight) d\mathbf{x}
\end{align*}
$$

ここで、変数変換 $\mathbf{u} = rac{\mathbf{x} - \mathbf{x}_n}{h}$ を導入します。ヤコビアンの行列式は $d\mathbf{x} = h^D d\mathbf{u}$ となります。
これを代入すると、
$$
egin{align*}
\int p(\mathbf{x}) d\mathbf{x} &= rac{1}{N} \sum_{n=1}^N rac{1}{h^D} \int k(\mathbf{u}) (h^D d\mathbf{u}) \
&= rac{1}{N} \sum_{n=1}^N \int k(\mathbf{u}) d\mathbf{u}
\end{align*}
$$

カーネル関数の正規化条件 $\int k(\mathbf{u}) d\mathbf{u} = 1$ を用いると、
$$
egin{align*}
\int p(\mathbf{x}) d\mathbf{x} &= rac{1}{N} \sum_{n=1}^N 1 \
&= rac{1}{N} \cdot N \
&= 1
\end{align*}
$$

以上により、カーネル関数 $k(\mathbf{u})$ が正規化されていれば、カーネル密度推定量 $p(\mathbf{x})$ もまた正規化された正しい確率密度関数となることが証明されました。
