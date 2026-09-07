import numpy as np

def check_d_separation(adj_list, source_nodes, target_nodes, cond_nodes):
    """
    有向非巡回グラフ (DAG) における d-分離 (d-separation) の判定アルゴリズム (PRML 8.2.2節)
    adj_list: dict, ノード -> 子ノードのリスト
    source_nodes: 集合 A
    target_nodes: 集合 B
    cond_nodes: 条件付け集合 C
    """
    source_nodes = set(source_nodes)
    target_nodes = set(target_nodes)
    cond_nodes = set(cond_nodes)
    
    # 1. 全ノードの親関係と子関係の構築
    parents = {}
    children = {}
    all_nodes = set(adj_list.keys())
    for u in adj_list:
        for v in adj_list[u]:
            all_nodes.add(v)
            
    for u in all_nodes:
        parents[u] = set()
        children[u] = set()
        
    for u in adj_list:
        for v in adj_list[u]:
            children[u].add(v)
            parents[v].add(u)
            
    # 2. 条件付け集合 C の祖先 (ancestors) を探索
    ancestors_of_cond = set(cond_nodes)
    to_visit = list(cond_nodes)
    while to_visit:
        curr = to_visit.pop()
        for p in parents[curr]:
            if p not in ancestors_of_cond:
                ancestors_of_cond.add(p)
                to_visit.append(p)
                
    # 3. グラフ上の全無向パスを深さ優先探索し、各パスがブロックされているか判定
    # 無向近傍: parents + children
    # パス上のノード v に対し:
    # - v が head-to-head (u -> v <- w): v もしくはその子孫が cond_nodes に含まれなければブロック
    # - v が それ以外 (u -> v -> w, u <- v <- w, u <- v -> w): v が cond_nodes に含まれればブロック
    def is_path_active(path):
        for i in range(1, len(path) - 1):
            prev_node = path[i-1]
            curr_node = path[i]
            next_node = path[i+1]
            
            # head-to-head 判定: prev -> curr <- next
            is_h2h = (curr_node in children[prev_node]) and (curr_node in children[next_node])
            if is_h2h:
                # curr_node またはその子孫が cond_nodes に含まれていなければブロック
                if curr_node not in ancestors_of_cond:
                    return False
            else:
                # non-head-to-head: curr_node が cond_nodes に含まれていればブロック
                if curr_node in cond_nodes:
                    return False
        return True

    # A の任意のノードから B の任意のノードへのパスを探索
    for start in source_nodes:
        stack = [[start]]
        while stack:
            path = stack.pop()
            curr = path[-1]
            if curr in target_nodes and len(path) > 1:
                if is_path_active(path):
                    # アクティブな（ブロックされていない）パスが存在する -> d-分離されていない
                    return False
            
            neighbors = parents[curr].union(children[curr])
            for nxt in neighbors:
                if nxt not in path:
                    stack.append(path + [nxt])
                    
    # すべてのパスがブロックされている -> d-分離されている
    return True

def denoise_image_icm(noisy_img, h=0.0, beta=1.0, eta=2.1, max_iter=10):
    """
    マルコフ確率場 (MRF) と ICM (Iterated Conditional Modes) による画像ノイズ除去 (PRML 8.3.3節)
    noisy_img: 2次元配列 {-1, +1}
    エネルギー: E(x, y) = h * sum(x_i) - beta * sum_{<i,j>} x_i * x_j - eta * sum_i x_i * y_i
    """
    X = noisy_img.copy().astype(float)
    Y = noisy_img.copy().astype(float)
    H, W = X.shape
    
    energies = []
    
    for iteration in range(max_iter):
        # 現在のエネルギーの計算
        # 相互作用項: 隣接ペア (i, j)
        diff_h = X[:, :-1] * X[:, 1:]
        diff_v = X[:-1, :] * X[1:, :]
        energy = h * np.sum(X) - beta * (np.sum(diff_h) + np.sum(diff_v)) - eta * np.sum(X * Y)
        energies.append(energy)
        
        changed = 0
        # 各ピクセルを順次更新
        for i in range(H):
            for j in range(W):
                # 4近傍の和
                neighbors_sum = 0.0
                if i > 0: neighbors_sum += X[i-1, j]
                if i < H - 1: neighbors_sum += X[i+1, j]
                if j > 0: neighbors_sum += X[i, j-1]
                if j < W - 1: neighbors_sum += X[i, j+1]
                
                # x_ij = +1 の局所エネルギーと x_ij = -1 の局所エネルギーの差分
                # E(x=+1) - E(x=-1) = 2 * (h - beta * neighbors_sum - eta * Y[i, j])
                delta_E = 2.0 * (h - beta * neighbors_sum - eta * Y[i, j])
                
                # エネルギーを低くする方を選択 (delta_E < 0 なら +1, delta_E > 0 なら -1)
                new_val = 1.0 if delta_E < 0 else -1.0
                if new_val != X[i, j]:
                    X[i, j] = new_val
                    changed += 1
                    
        if changed == 0:
            break
            
    return X, energies

class SimpleFactorGraphChain:
    """
    一次元連鎖構造に対する厳密な Sum-Product アルゴリズム (PRML 8.4.1, 8.4.4節)
    各ノードは K 状態の離散変数
    """
    def __init__(self, node_states, transition_matrices, emission_potentials):
        self.K = node_states # 状態数
        self.N = len(emission_potentials)
        self.trans = transition_matrices # リスト: N-1 個の (K, K) 行列
        self.emiss = emission_potentials # リスト: N 個の (K,) 観測ポテンシャル

    def forward_backward_marginals(self):
        # 前方向メッセージ mu_forward
        alpha = np.zeros((self.N, self.K))
        alpha[0] = self.emiss[0]
        alpha[0] /= np.sum(alpha[0])
        
        for n in range(1, self.N):
            # alpha_n(x_n) = emiss_n(x_n) * sum_{x_{n-1}} trans(x_{n-1}, x_n) * alpha_{n-1}(x_{n-1})
            msg = self.trans[n-1].T @ alpha[n-1]
            alpha[n] = self.emiss[n] * msg
            alpha[n] /= np.sum(alpha[n]) # 規格化
            
        # 後方向メッセージ mu_backward
        beta = np.zeros((self.N, self.K))
        beta[-1] = np.ones(self.K)
        
        for n in range(self.N - 2, -1, -1):
            # beta_n(x_n) = sum_{x_{n+1}} trans(x_n, x_{n+1}) * emiss_{n+1}(x_{n+1}) * beta_{n+1}(x_{n+1})
            next_msg = self.emiss[n+1] * beta[n+1]
            beta[n] = self.trans[n] @ next_msg
            beta[n] /= np.sum(beta[n]) # 規格化
            
        # 周辺確率 p(x_n)
        marginals = alpha * beta
        marginals /= np.sum(marginals, axis=1, keepdims=True)
        return marginals

def noisy_or(x, mu, mu_0=0.0):
    """
    Noisy-OR 分布 (PRML 8.1.3節, 演習 8.6)
    x: (M,) 0または1の二値配列
    mu: (M,) 各原因が単独で活性化する確率
    mu_0: 背景（自発）発症確率
    戻り値: p(y=1|x)
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    fail_prob = (1.0 - mu_0) * np.prod((1.0 - mu) ** x)
    return float(1.0 - fail_prob)

def linear_gaussian_moments(W, b, v):
    """
    線形ガウスDAGにおける平均ベクトルと共分散行列の再帰的計算 (PRML 8.1.4節, 式8.15-8.18)
    W: (D, D) 重み行列 (トポロジカル順序で下三角、対角成分は0)
    b: (D,) バイアスベクトル
    v: (D,) 各変数の局所ノイズ分散
    戻り値:
        mean: (D,) 平均ベクトル
        cov: (D, D) 共分散行列
    """
    D = len(b)
    mean = np.zeros(D)
    for i in range(D):
        mean[i] = np.dot(W[i], mean) + b[i]
        
    cov = np.zeros((D, D))
    for j in range(D):
        for i in range(j + 1):
            c_val = sum(W[j, k] * cov[i, k] for k in range(j))
            if i == j:
                c_val += v[i]
            cov[i, j] = c_val
            cov[j, i] = c_val
            
    return mean, cov

