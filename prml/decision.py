"""
PRML Chapter 1.5: Decision Theory
決定理論のコアモデルとユーティリティ関数
- 最小誤識別率および期待損失最小化 (Bayes Decision Classifier)
- 棄却オプション (Reject Option Classifier)
- ミンコフスキー損失関数 (Minkowski Loss)
- ROC曲線およびAUC評価 (compute_roc_curve)
"""

import numpy as np


def minkowski_loss(y_true, y_pred, q=2.0):
    """
    PRML 式 (1.89) ミンコフスキー損失関数
    E_q(y, t) = |y - t|^q
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_pred - y_true) ** q)


class BayesDecisionClassifier:
    """
    PRML 1.5.1, 1.5.2: ベイズ決定理論に基づく最適分類器
    
    事後確率 P(C_k | x) と損失行列 L_{kj} (真のクラスが k で、クラス j と予測したときの損失)
    に基づき、期待損失を最小化する決定規則:
        j^*(x) = argmin_j \\sum_{k} L_{kj} P(C_k | x)
    
    L が単位行列の反転 (L_{kj} = 1 - \\delta_{kj}) の場合は最小誤識別率基準:
        j^*(x) = argmax_k P(C_k | x)
    """

    def __init__(self, loss_matrix=None, priors=None):
        """
        loss_matrix: 形状 (K, K) の行列。loss_matrix[k, j] は真がkでjと予測したときの損失。
                     None の場合は 0-1 損失 (最小誤識別率)。
        priors: 形状 (K,) の事前確率ベクトル。None の場合は一様事前分布またはデータ準拠。
        """
        self.loss_matrix = np.asarray(loss_matrix) if loss_matrix is not None else None
        self.priors = np.asarray(priors) if priors is not None else None
        self.classes_ = None

    def fit_posteriors(self, probas):
        """
        事後確率行列 probas: (N, K)
        """
        probas = np.asarray(probas)
        n_samples, n_classes = probas.shape
        self.classes_ = np.arange(n_classes)
        if self.loss_matrix is None:
            # 0-1 loss: L_{kj} = 0 if k == j else 1
            self.loss_matrix = 1.0 - np.eye(n_classes)
        return self

    def predict(self, probas):
        """
        与えられた事後確率 P(C_k | x) に対して、期待損失を最小化するクラス予測を返す。
        probas: (N, K)
        """
        probas = np.asarray(probas)
        n_samples, n_classes = probas.shape
        if self.loss_matrix is None:
            self.loss_matrix = 1.0 - np.eye(n_classes)
        
        # 期待損失: (N, K) = probas (N, K) @ loss_matrix (K, K)
        # expected_loss[n, j] = sum_k probas[n, k] * loss_matrix[k, j]
        expected_losses = np.dot(probas, self.loss_matrix)
        return np.argmin(expected_losses, axis=1)

    def expected_loss(self, probas):
        """
        各クラスを選択した場合の期待損失 (N, K) を返す。
        """
        probas = np.asarray(probas)
        n_classes = probas.shape[1]
        if self.loss_matrix is None:
            self.loss_matrix = 1.0 - np.eye(n_classes)
        return np.dot(probas, self.loss_matrix)


class RejectOptionClassifier:
    """
    PRML 1.5.3: 棄却オプション付き分類器
    
    確信度 max_k P(C_k | x) が閾値 theta を下回るサンプルを棄却 (-1) する。
    """

    def __init__(self, theta=0.5, reject_label=-1):
        """
        theta: 棄却閾値 (1/K <= theta < 1.0)
        reject_label: 棄却されたサンプルに付与するラベル (-1)
        """
        self.theta = float(theta)
        self.reject_label = reject_label

    def predict(self, probas):
        """
        probas: (N, K) の事後確率行列
        returns: (N,) のクラスラベル (棄却された場合は reject_label)
        """
        probas = np.asarray(probas)
        max_probas = np.max(probas, axis=1)
        predictions = np.argmax(probas, axis=1)
        
        # 閾値未満を棄却
        rejected = max_probas < self.theta
        predictions[rejected] = self.reject_label
        return predictions

    def evaluate_tradeoff(self, y_true, probas, thetas=None):
        """
        閾値 thetas を変化させたときの「誤り率」と「棄却率」のトレードオフを計算する。
        PRML Figure 1.26 の再現用。
        """
        if thetas is None:
            thetas = np.linspace(1.0 / probas.shape[1], 0.99, 100)
        
        y_true = np.asarray(y_true)
        probas = np.asarray(probas)
        max_probas = np.max(probas, axis=1)
        raw_preds = np.argmax(probas, axis=1)
        
        reject_rates = []
        error_rates = []
        
        for th in thetas:
            accepted = max_probas >= th
            n_accepted = np.sum(accepted)
            reject_rate = 1.0 - (n_accepted / len(y_true))
            reject_rates.append(reject_rate)
            
            if n_accepted > 0:
                errors = np.sum(raw_preds[accepted] != y_true[accepted])
                error_rate = errors / n_accepted
            else:
                error_rate = 0.0
            error_rates.append(error_rate)
            
        return np.array(thetas), np.array(reject_rates), np.array(error_rates)


def compute_roc_curve(y_true, y_score):
    """
    2値分類の ROC 曲線 (False Positive Rate, True Positive Rate) および AUC を計算する。
    
    y_true: {0, 1} の正解ラベル
    y_score: クラス 1 である確率やスコア
    returns: fpr, tpr, thresholds, auc
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # スコアの降順にソート
    desc_order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_order]
    y_score_sorted = y_score[desc_order]
    
    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.concatenate([distinct_indices, [len(y_score_sorted) - 1]])
    
    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative samples must be present.")
        
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    thresholds = np.concatenate([[y_score_sorted[0] + 1e-5], y_score_sorted[threshold_idxs]])
    
    # 台形公式による AUC
    if hasattr(np, "trapezoid"):
        auc = np.trapezoid(tpr, fpr)
    elif hasattr(np, "trapz"):
        auc = np.trapz(tpr, fpr)
    else:
        auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) * 0.5
    return fpr, tpr, thresholds, float(auc)
