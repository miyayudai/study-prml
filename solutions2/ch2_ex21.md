## Exercise 2.21 (ガウス分布の4次モーメントとIsserlisの定理)

**問題:**
平均 0 のガウス分布において $\mathbb{E}[x_i x_j x_k x_l] = \Sigma_{ij}\Sigma_{kl} + \Sigma_{ik}\Sigma_{jl} + \Sigma_{il}\Sigma_{jk}$ を示せ。

**解答と解説:**

平均が $\boldsymbol{\mu} = \mathbf{0}$ の多変量ガウス分布において、確率変数 $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})$ のモーメント母関数（あるいは特性関数）を用いて高次のモーメントを導出します。

モーメント母関数 $M(\mathbf{t})$ は次のように与えられます。
$$ M(\mathbf{t}) = \mathbb{E}[\exp(\mathbf{t}^T \mathbf{x})] = \exp\left( \frac{1}{2} \mathbf{t}^T \boldsymbol{\Sigma} \mathbf{t} \right) = \exp\left( \frac{1}{2} \sum_{a,b} t_a \Sigma_{ab} t_b \right) $$

$x_i x_j x_k x_l$ の期待値は、モーメント母関数を $t_i, t_j, t_k, t_l$ で偏微分し、$\mathbf{t} = \mathbf{0}$ とおくことで得られます。
$$ \mathbb{E}[x_i x_j x_k x_l] = \left. \frac{\partial^4 M(\mathbf{t})}{\partial t_i \partial t_j \partial t_k \partial t_l} \right|_{\mathbf{t}=\mathbf{0}} $$

まず、$t_i$ について1階微分を計算します。
$$ \frac{\partial M(\mathbf{t})}{\partial t_i} = \left( \sum_{a} \Sigma_{ia} t_a \right) \exp\left( \frac{1}{2} \mathbf{t}^T \boldsymbol{\Sigma} \mathbf{t} \right) $$

次に、$t_j$ について微分します。
$$ \frac{\partial^2 M(\mathbf{t})}{\partial t_i \partial t_j} = \Sigma_{ij} \exp\left( \frac{1}{2} \mathbf{t}^T \boldsymbol{\Sigma} \mathbf{t} \right) + \left( \sum_{a} \Sigma_{ia} t_a \right) \left( \sum_{b} \Sigma_{jb} t_b \right) \exp\left( \frac{1}{2} \mathbf{t}^T \boldsymbol{\Sigma} \mathbf{t} \right) $$

さらに $t_k$ について微分します。
$$ \frac{\partial^3 M(\mathbf{t})}{\partial t_i \partial t_j \partial t_k} = \Sigma_{ij} \left( \sum_{c} \Sigma_{kc} t_c \right) M(\mathbf{t}) + \Sigma_{ik} \left( \sum_{b} \Sigma_{jb} t_b \right) M(\mathbf{t}) + \Sigma_{jk} \left( \sum_{a} \Sigma_{ia} t_a \right) M(\mathbf{t}) + \left( \sum_{a} \Sigma_{ia} t_a \right) \left( \sum_{b} \Sigma_{jb} t_b \right) \left( \sum_{c} \Sigma_{kc} t_c \right) M(\mathbf{t}) $$

最後に、$t_l$ について微分し、$\mathbf{t}=\mathbf{0}$ とおきます。$\mathbf{t}=\mathbf{0}$ のとき、$t$ に比例する項はすべて 0 になるため、$t_l$ で微分されて定数（共分散成分）となる項のみが残ります。$M(\mathbf{0}) = 1$ に注意すると、
$$ \left. \frac{\partial^4 M(\mathbf{t})}{\partial t_i \partial t_j \partial t_k \partial t_l} \right|_{\mathbf{t}=\mathbf{0}} = \Sigma_{ij}\Sigma_{kl} + \Sigma_{ik}\Sigma_{jl} + \Sigma_{jk}\Sigma_{il} $$
となります（対称行列の性質 $\Sigma_{ab} = \Sigma_{ba}$ を用いています）。

したがって、
$$ \mathbb{E}[x_i x_j x_k x_l] = \Sigma_{ij}\Sigma_{kl} + \Sigma_{ik}\Sigma_{jl} + \Sigma_{il}\Sigma_{jk} $$
が導かれました。この結果はIsserlisの定理（あるいはWickの定理）として知られており、ゼロ平均のガウス変数の高次モーメントが2次モーメント（共分散）のすべての可能なペアの積の和で表されることを示しています。この場合、$4! / (2^2 \cdot 2!) = 3$ 通りの組み合わせが現れます。
