import numpy as np

def hit_k(y_pred, y_true, k):
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    return int(y_true in y_pred_indices)

def ndcg_k(y_pred, y_true, k):
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    if y_true in y_pred_indices:
        idx = y_pred_indices.index(y_true)
        return 1 / np.log2(idx + 2)
    return 0

def batch_performance(batch_y_pred, batch_y_true, k):
    batch_size = batch_y_pred.size(0)
    batch_recall = 0
    batch_ndcg = 0
    for idx in range(batch_size):
        batch_recall += hit_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_ndcg += ndcg_k(batch_y_pred[idx], batch_y_true[idx], k)
    recall = batch_recall / batch_size
    ndcg = batch_ndcg / batch_size
    return recall, ndcg 