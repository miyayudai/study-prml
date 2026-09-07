import numpy as np
import scipy.special as sp_special

def tanh(a):
    return np.tanh(a)

def dtanh(a):
    return 1.0 - np.tanh(a)**2

def softmax(a, axis=-1):
    a_max = np.max(a, axis=axis, keepdims=True)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a, axis=axis, keepdims=True)

class MLPRegressor:
    """
    2層フィードフォワードニューラルネットワーク (回帰用, PRML 5.1-5.3節)
    入力 -> 隠れ層 (tanh) -> 出力層 (線形)
    """
    def __init__(self, n_in, n_hidden, n_out, weight_decay=0.0, lr=0.01, random_state=42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.weight_decay = weight_decay
        self.lr = lr
        
        rng = np.random.RandomState(random_state)
        # 重みの初期化 (Xavier / He に準拠したスケーリング)
        self.W1 = rng.randn(n_in, n_hidden) / np.sqrt(n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, n_out) / np.sqrt(n_hidden)
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        """
        X: shape (N, D)
        戻り値:
          a1: (N, M) 隠れ層入力
          z1: (N, M) 隠れ層出力
          a2: (N, K) 出力層入力
          y:  (N, K) ネットワーク出力
        """
        a1 = X @ self.W1 + self.b1
        z1 = np.tanh(a1)
        a2 = z1 @ self.W2 + self.b2
        y = a2 # 線形出力
        return a1, z1, a2, y

    def predict(self, X):
        _, _, _, y = self.forward(X)
        return y

    def compute_loss_and_grads(self, X, T):
        """
        二乗和誤差 E = 0.5 * sum_n ||y_n - t_n||^2 + 0.5 * weight_decay * ||w||^2
        および解析的勾配の逆伝播計算
        """
        N = len(X)
        a1, z1, a2, y = self.forward(X)
        
        # 二乗和誤差
        data_loss = 0.5 * np.sum((y - T)**2)
        reg_loss = 0.5 * self.weight_decay * (np.sum(self.W1**2) + np.sum(self.W2**2))
        loss = data_loss + reg_loss
        
        # 誤差逆伝播 (Backpropagation)
        # 出力層の delta_k = y_k - t_k
        delta2 = (y - T) # (N, K)
        grad_W2 = z1.T @ delta2 + self.weight_decay * self.W2
        grad_b2 = np.sum(delta2, axis=0)
        
        # 隠れ層の delta_j = (1 - z_j^2) * sum_k W_jk delta_k
        delta1 = (delta2 @ self.W2.T) * (1.0 - z1**2) # (N, M)
        grad_W1 = X.T @ delta1 + self.weight_decay * self.W1
        grad_b1 = np.sum(delta1, axis=0)
        
        grads = {
            'W1': grad_W1, 'b1': grad_b1,
            'W2': grad_W2, 'b2': grad_b2
        }
        return loss, grads

    def fit(self, X, T, n_epochs=1000, lr=None, verbose=False):
        if lr is not None:
            self.lr = lr
        loss_history = []
        for epoch in range(n_epochs):
            loss, grads = self.compute_loss_and_grads(X, T)
            loss_history.append(loss)
            
            # 勾配降下更新
            self.W1 -= self.lr * grads['W1']
            self.b1 -= self.lr * grads['b1']
            self.W2 -= self.lr * grads['W2']
            self.b2 -= self.lr * grads['b2']
            
            if verbose and (epoch % (n_epochs // 10) == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:4d}/{n_epochs}: Loss = {loss:.6f}")
        return loss_history

    def compute_jacobian(self, x):
        """
        単一サンプル x (shape: (D,)) に対するヤコビ行列 J_ki = dy_k / dx_i (PRML 5.3.4節)
        戻り値: shape (K, D)
        """
        a1 = x @ self.W1 + self.b1 # (M,)
        z1 = np.tanh(a1) # (M,)
        diag_deriv = (1.0 - z1**2) # (M,)
        J = (self.W2.T * diag_deriv) @ self.W1.T
        return J

    def get_params_flat(self):
        """全パラメータを1次元ベクトルに平坦化して取得 (shape: (W,))"""
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel()
        ])

    def set_params_flat(self, w):
        """1次元ベクトル w から各層の重みとバイアスを展開して更新"""
        idx = 0
        w1_size = self.n_in * self.n_hidden
        self.W1 = w[idx:idx + w1_size].reshape(self.n_in, self.n_hidden)
        idx += w1_size
        
        b1_size = self.n_hidden
        self.b1 = w[idx:idx + b1_size]
        idx += b1_size
        
        w2_size = self.n_hidden * self.n_out
        self.W2 = w[idx:idx + w2_size].reshape(self.n_hidden, self.n_out)
        idx += w2_size
        
        b2_size = self.n_out
        self.b2 = w[idx:idx + b2_size]

    def compute_param_grad_flat(self, X, T):
        """全パラメータに関する平坦化された解析的勾配ベクトル (shape: (W,)) を計算"""
        loss, grads = self.compute_loss_and_grads(X, T)
        grad_flat = np.concatenate([
            grads['W1'].ravel(), grads['b1'].ravel(),
            grads['W2'].ravel(), grads['b2'].ravel()
        ])
        return loss, grad_flat

    def compute_hessian_exact(self, X, T, eps=1e-5):
        """
        全パラメータに関する厳密なヘッセ行列 H (shape: (W, W)) を計算
        解析的勾配に対する中心差分摂動により高精度 (O(eps^2)) で算出
        """
        orig_w = self.get_params_flat()
        W = len(orig_w)
        H = np.zeros((W, W))
        for i in range(W):
            e_i = np.zeros(W)
            e_i[i] = eps
            self.set_params_flat(orig_w + e_i)
            _, g_plus = self.compute_param_grad_flat(X, T)
            self.set_params_flat(orig_w - e_i)
            _, g_minus = self.compute_param_grad_flat(X, T)
            H[:, i] = (g_plus - g_minus) / (2.0 * eps)
        self.set_params_flat(orig_w)
        return 0.5 * (H + H.T)

    def compute_param_jacobian(self, x):
        """
        単一データ点 x (shape: (D,)) に対する、出力 y (shape: (K,)) の全パラメータ w に対するヤコビアン
        J_w = dy / dw (shape: (K, W))
        """
        a1 = x @ self.W1 + self.b1
        z1 = np.tanh(a1)
        K = self.n_out
        W_len = len(self.get_params_flat())
        J = np.zeros((K, W_len))
        for k in range(K):
            dW2 = np.zeros((self.n_hidden, self.n_out))
            dW2[:, k] = z1
            db2 = np.zeros(self.n_out)
            db2[k] = 1.0
            da1 = self.W2[:, k] * (1.0 - z1**2)
            db1 = da1
            dW1 = np.outer(x, da1)
            J[k, :] = np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])
        return J

    def compute_hessian_gauss_newton(self, X):
        """
        外積近似 (Gauss-Newton) ヘッセ行列 H_GN = sum_n J_n^T J_n (PRML 5.4.2節 式 5.84)
        常に半正定値 (固有値 >= 0)
        """
        W = len(self.get_params_flat())
        H_gn = np.zeros((W, W))
        for x_n in X:
            J_n = self.compute_param_jacobian(x_n)
            H_gn += J_n.T @ J_n
        return H_gn

    def hessian_vector_product(self, X, T, v, eps=1e-5):
        """
        Pearlmutter の R{.} 演算子による高速ヘッセ・ベクトル積 H v (PRML 5.4.7節)
        O(W) の計算量で評価可能
        """
        orig_w = self.get_params_flat()
        self.set_params_flat(orig_w + eps * v)
        _, g_plus = self.compute_param_grad_flat(X, T)
        self.set_params_flat(orig_w - eps * v)
        _, g_minus = self.compute_param_grad_flat(X, T)
        self.set_params_flat(orig_w)
        return (g_plus - g_minus) / (2.0 * eps)


def gradient_check(model, X, T, eps=1e-5):
    """
    誤差逆伝播の勾配と数値微分（有限差分）の整合性を検証する (PRML 5.3.3節)
    """
    _, ana_grads = model.compute_loss_and_grads(X, T)
    
    for param_name in ['W1', 'b1', 'W2', 'b2']:
        param = getattr(model, param_name)
        num_grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            orig_val = param[idx]
            
            param[idx] = orig_val + eps
            loss_plus, _ = model.compute_loss_and_grads(X, T)
            
            param[idx] = orig_val - eps
            loss_minus, _ = model.compute_loss_and_grads(X, T)
            
            param[idx] = orig_val
            num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            it.iternext()
            
        ana_grad = ana_grads[param_name]
        diff = np.linalg.norm(ana_grad - num_grad)
        denom = np.linalg.norm(ana_grad) + np.linalg.norm(num_grad) + 1e-12
        rel_diff = diff / denom
        assert rel_diff < 1e-4, f"Gradient check failed for {param_name}! rel_diff={rel_diff:.2e}"
    return True


class MixtureDensityNetwork:
    """
    混合密度ネットワーク (Mixture Density Network: MDN, PRML 5.6節)
    入力 x から、条件付き分布 p(t|x) = sum_k pi_k(x) N(t | mu_k(x), sigma_k^2(x)) の
    混合重み pi_k、平均 mu_k、分散 sigma_k^2 を予測する。
    """
    def __init__(self, n_in, n_hidden, n_components, random_state=42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.K = n_components # ガウス分布の数
        
        rng = np.random.RandomState(random_state)
        self.W1 = rng.randn(n_in, n_hidden) / np.sqrt(n_in)
        self.b1 = np.zeros(n_hidden)
        
        # 隠れ層 -> 3つのパラメータグループ
        # 1. pi: K次元 (softmax)
        self.W_pi = rng.randn(n_hidden, self.K) / np.sqrt(n_hidden)
        self.b_pi = np.zeros(self.K)
        # 2. sigma: K次元 (exp)
        self.W_sig = rng.randn(n_hidden, self.K) / np.sqrt(n_hidden)
        self.b_sig = np.zeros(self.K)
        # 3. mu: K次元 (linear)
        self.W_mu = rng.randn(n_hidden, self.K) / np.sqrt(n_hidden)
        self.b_mu = rng.randn(self.K)

    def forward(self, X):
        a1 = X @ self.W1 + self.b1
        z = np.tanh(a1) # (N, n_hidden)
        
        # パラメータ出力 (PRML 式 5.145 - 5.148)
        a_pi = z @ self.W_pi + self.b_pi
        pi = softmax(a_pi, axis=-1)
        
        a_sig = z @ self.W_sig + self.b_sig
        sigma = np.exp(np.clip(a_sig, -10, 10)) # 正値制約
        
        a_mu = z @ self.W_mu + self.b_mu
        mu = a_mu # 線形
        return z, pi, sigma, mu, a_sig

    def compute_loss_and_grads(self, X, T):
        """
        負の対数尤度誤差 E = - sum_n ln [ sum_k pi_k N(t_n | mu_k, sigma_k^2) ]
        および各重みに対する解析的勾配 (PRML 式 5.153 - 5.157)
        """
        N = len(X)
        T = T.reshape(N, 1) # (N, 1)
        z, pi, sigma, mu, a_sig = self.forward(X)
        
        # 各成分の正規分布密度 N(t_n | mu_k, sigma_k^2)
        var = sigma**2 + 1e-12
        diff = T - mu # (N, K)
        # N(t | mu, sigma^2) = (2 pi sigma^2)^(-1/2) exp(- (t - mu)^2 / (2 sigma^2))
        norm_const = 1.0 / np.sqrt(2.0 * np.pi * var)
        exp_term = np.exp(- 0.5 * (diff**2) / var)
        comp_dens = norm_const * exp_term + 1e-30 # (N, K)
        
        # 結合確率密度 p(t_n | x_n) = sum_k pi_k N(...)
        joint = np.sum(pi * comp_dens, axis=1, keepdims=True) # (N, 1)
        nll = -np.sum(np.log(joint + 1e-30))
        
        # 責任度 (responsibilities) gamma_nk = pi_k N(...) / joint (PRML 式 5.154)
        gamma = (pi * comp_dens) / (joint + 1e-30) # (N, K)
        
        # 出力層のデルタ (PRML 式 5.155 - 5.157)
        delta_pi = pi - gamma                     # (N, K)
        delta_mu = gamma * (-diff / var)          # (N, K)
        delta_sig = gamma * (1.0 - (diff**2)/var) # (N, K)
        
        # 隠れ層への勾配
        grad_W_pi = z.T @ delta_pi; grad_b_pi = np.sum(delta_pi, axis=0)
        grad_W_mu = z.T @ delta_mu; grad_b_mu = np.sum(delta_mu, axis=0)
        grad_W_sig = z.T @ delta_sig; grad_b_sig = np.sum(delta_sig, axis=0)
        
        # 隠れ層入力へのデルタ
        delta_z = delta_pi @ self.W_pi.T + delta_mu @ self.W_mu.T + delta_sig @ self.W_sig.T
        delta_hidden = delta_z * (1.0 - z**2)
        grad_W1 = X.T @ delta_hidden; grad_b1 = np.sum(delta_hidden, axis=0)
        
        grads = {
            'W1': grad_W1, 'b1': grad_b1,
            'W_pi': grad_W_pi, 'b_pi': grad_b_pi,
            'W_mu': grad_W_mu, 'b_mu': grad_b_mu,
            'W_sig': grad_W_sig, 'b_sig': grad_b_sig
        }
        return nll, grads

    def fit(self, X, T, n_epochs=1000, lr=0.005, verbose=False):
        loss_history = []
        for epoch in range(n_epochs):
            loss, grads = self.compute_loss_and_grads(X, T)
            loss_history.append(loss)
            
            self.W1 -= lr * grads['W1']
            self.b1 -= lr * grads['b1']
            self.W_pi -= lr * grads['W_pi']
            self.b_pi -= lr * grads['b_pi']
            self.W_mu -= lr * grads['W_mu']
            self.b_mu -= lr * grads['b_mu']
            self.W_sig -= lr * grads['W_sig']
            self.b_sig -= lr * grads['b_sig']
            
            if verbose and (epoch % (n_epochs // 10) == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:4d}/{n_epochs}: NLL = {loss:.4f}")
        return loss_history


class MLPClassifier:
    """
    2層フィードフォワードニューラルネットワーク (分類用, PRML 5.2-5.3節)
    二値分類 (ロジスティックシグモイド + 二値交差エントロピー)
    多クラス分類 (ソフトマックス + 多クラス交差エントロピー)
    """
    def __init__(self, n_in, n_hidden, n_classes, weight_decay=0.0, lr=0.01, random_state=42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        self.lr = lr
        
        # 二値分類 (n_classes=2 or 1) なら出力次元1、多クラスなら n_classes 次元
        self.n_out = 1 if n_classes <= 2 else n_classes
        
        rng = np.random.RandomState(random_state)
        self.W1 = rng.randn(n_in, n_hidden) / np.sqrt(n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, self.n_out) / np.sqrt(n_hidden)
        self.b2 = np.zeros(self.n_out)

    def forward(self, X):
        a1 = X @ self.W1 + self.b1
        z1 = np.tanh(a1)
        a2 = z1 @ self.W2 + self.b2
        if self.n_out == 1:
            y = 1.0 / (1.0 + np.exp(-np.clip(a2, -50, 50)))
        else:
            y = softmax(a2, axis=-1)
        return a1, z1, a2, y

    def predict_proba(self, X):
        _, _, _, y = self.forward(X)
        if self.n_out == 1:
            p1 = y.ravel()
            return np.column_stack([1.0 - p1, p1])
        return y

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def compute_loss_and_grads(self, X, y_target):
        N = len(X)
        a1, z1, a2, y = self.forward(X)
        
        if self.n_out == 1:
            T = y_target.reshape(N, 1).astype(float)
            eps = 1e-15
            y_clip = np.clip(y, eps, 1.0 - eps)
            data_loss = -np.sum(T * np.log(y_clip) + (1.0 - T) * np.log(1.0 - y_clip))
            delta2 = y - T
        else:
            if y_target.ndim == 1 or (y_target.ndim == 2 and y_target.shape[1] == 1):
                T = np.zeros((N, self.n_classes))
                T[np.arange(N), y_target.ravel().astype(int)] = 1.0
            else:
                T = y_target.astype(float)
            eps = 1e-15
            data_loss = -np.sum(T * np.log(np.clip(y, eps, 1.0)))
            delta2 = y - T
            
        reg_loss = 0.5 * self.weight_decay * (np.sum(self.W1**2) + np.sum(self.W2**2))
        loss = data_loss + reg_loss
        
        grad_W2 = z1.T @ delta2 + self.weight_decay * self.W2
        grad_b2 = np.sum(delta2, axis=0)
        
        delta1 = (delta2 @ self.W2.T) * (1.0 - z1**2)
        grad_W1 = X.T @ delta1 + self.weight_decay * self.W1
        grad_b1 = np.sum(delta1, axis=0)
        
        grads = {
            'W1': grad_W1, 'b1': grad_b1,
            'W2': grad_W2, 'b2': grad_b2
        }
        return loss, grads

    def fit(self, X, y, n_epochs=1000, lr=None, verbose=False):
        if lr is not None:
            self.lr = lr
        loss_history = []
        for epoch in range(n_epochs):
            loss, grads = self.compute_loss_and_grads(X, y)
            loss_history.append(loss)
            self.W1 -= self.lr * grads['W1']
            self.b1 -= self.lr * grads['b1']
            self.W2 -= self.lr * grads['W2']
            self.b2 -= self.lr * grads['b2']
            if verbose and (epoch % (n_epochs // 10) == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:4d}/{n_epochs}: Loss = {loss:.6f}")
        return loss_history


class BayesianMLPRegressor:
    """
    ベイズニューラルネットワーク (回帰用, PRML 5.7節)
    MAP推定 + ラプラス近似による重み事後分布および予測分布
    p(w|D) ~ N(w_map, A^{-1}), A = beta * H_GN + alpha * I
    p(t|x, D) = N(t | y(x, w_map), sigma^2(x))
    sigma^2(x) = beta^{-1} + g(x)^T A^{-1} g(x)
    """
    def __init__(self, n_in, n_hidden, n_out=1, alpha=1.0, beta=1.0, random_state=42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.mlp = MLPRegressor(
            n_in=n_in, n_hidden=n_hidden, n_out=n_out,
            weight_decay=alpha / beta, random_state=random_state
        )
        self.A_inv = None
        self.log_det_A = None

    def fit(self, X, T, n_epochs=1000, lr=0.01, verbose=False):
        if T.ndim == 1:
            T = T[:, np.newaxis]
        self.mlp.weight_decay = self.alpha / self.beta
        self.mlp.fit(X, T, n_epochs=n_epochs, lr=lr, verbose=verbose)
        
        # ガウス・ニュートン外積近似ヘッセ行列
        H_gn = self.mlp.compute_hessian_gauss_newton(X)
        W_dim = len(self.mlp.get_params_flat())
        # ヘッセ行列 A = beta * H_gn + alpha * I (PRML 式 5.166)
        A = self.beta * H_gn + self.alpha * np.eye(W_dim)
        
        sign, logdet = np.linalg.slogdet(A)
        self.log_det_A = logdet if sign > 0 else np.nan
        self.A_inv = np.linalg.pinv(A)
        return self

    def predict(self, X, return_std=True):
        """
        予測平均 y(x, w_map) および予測標準偏差 sigma(x) を算出
        """
        y_mean = self.mlp.predict(X)
        if not return_std:
            return y_mean
            
        N = len(X)
        variances = np.zeros(N)
        for i, x_n in enumerate(X):
            J = self.mlp.compute_param_jacobian(x_n)
            g = J[0, :]
            var_extra = g @ self.A_inv @ g
            variances[i] = (1.0 / self.beta) + var_extra
            
        stds = np.sqrt(np.maximum(variances, 1e-12))
        return y_mean, stds

    def compute_evidence(self, X, T):
        """
        ハイパーパラメータ alpha, beta に関する対数エビデンス ln p(D | alpha, beta) (PRML 5.7.3節 式 5.175)
        """
        if T.ndim == 1:
            T = T[:, np.newaxis]
        N = len(X)
        y = self.mlp.predict(X)
        E_D = 0.5 * np.sum((y - T)**2)
        w = self.mlp.get_params_flat()
        E_W = 0.5 * np.sum(w**2)
        W_dim = len(w)
        
        log_ev = (
            - self.beta * E_D
            - self.alpha * E_W
            - 0.5 * self.log_det_A
            + 0.5 * W_dim * np.log(self.alpha)
            + 0.5 * N * np.log(self.beta)
            - 0.5 * N * np.log(2.0 * np.pi)
        )
        return log_ev

