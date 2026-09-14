# 演習問題 1.5

## 問題
分散の定義 (1.38) を用いて、
$$ \text{var}[f(x)] = \mathbb{E}[f(x)^2] - \mathbb{E}[f(x)]^2 $$
となることを示せ。

## 解答・解説

関数 $f(x)$ の分散の定義 (1.38) は以下の通りです。
$$ \text{var}[f(x)] = \mathbb{E}\left[ (f(x) - \mathbb{E}[f(x)])^2 \right] $$
ここで、$\mathbb{E}[f(x)]$ は $f(x)$ の期待値（平均）であり、期待値の線形性を持つ演算子です。表記を簡潔にするため、$\mu = \mathbb{E}[f(x)]$ とおきます。$\mu$ は $x$ に依存しない定数です。

定義式の中の二乗を展開します。
$$ (f(x) - \mu)^2 = f(x)^2 - 2\mu f(x) + \mu^2 $$

これを元の分散の定義式に代入します。
$$ \text{var}[f(x)] = \mathbb{E}\left[ f(x)^2 - 2\mu f(x) + \mu^2 \right] $$

期待値演算子 $\mathbb{E}[\cdot]$ の線形性（$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$）を用います。ここで、$\mu$ は定数であるため、$\mathbb{E}[\mu^2] = \mu^2$ となることに注意してください。
$$ \text{var}[f(x)] = \mathbb{E}[f(x)^2] - \mathbb{E}[2\mu f(x)] + \mathbb{E}[\mu^2] $$
$$ \text{var}[f(x)] = \mathbb{E}[f(x)^2] - 2\mu \mathbb{E}[f(x)] + \mu^2 $$

ここで、最初に定義した $\mu = \mathbb{E}[f(x)]$ を再び代入します。
$$ \text{var}[f(x)] = \mathbb{E}[f(x)^2] - 2\mu \cdot \mu + \mu^2 $$
$$ \text{var}[f(x)] = \mathbb{E}[f(x)^2] - 2\mu^2 + \mu^2 $$
$$ \text{var}[f(x)] = \mathbb{E}[f(x)^2] - \mu^2 $$

$\mu$ を $\mathbb{E}[f(x)]$ に戻すことで、求める式が得られます。
$$ \text{var}[f(x)] = \mathbb{E}[f(x)^2] - \mathbb{E}[f(x)]^2 $$

これは、「（二乗の平均）引く（平均の二乗）」として広く知られる、分散を計算するための非常に有用な公式です。
