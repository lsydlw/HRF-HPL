import numpy as np
import scipy.sparse as sp
import torch

def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

def normalized_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum + 1e-8, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv @ adj @ d_mat_inv
    return norm_adj

def transform_csr_matrix_to_tensor(csr_mat):
    csr = csr_mat.tocoo()
    indices = np.vstack((csr.row, csr.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(csr.data)
    shape = csr.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape)) 

def get_hyper_deg(incidence_matrix):
    """计算超图的节点度和边度
    Args:
        incidence_matrix: 超图入射矩阵，形状为[num_nodes, num_edges]
    Returns:
        node_deg: 节点度向量，形状为[num_nodes]
        edge_deg: 边度向量，形状为[num_edges]
    """
    if isinstance(incidence_matrix, torch.Tensor):
        # 计算节点度（每行的和）
        node_deg = torch.sparse.sum(incidence_matrix, dim=1).to_dense()
        # 计算边度（每列的和）
        edge_deg = torch.sparse.sum(incidence_matrix, dim=0).to_dense()
    else:
        # 对于scipy稀疏矩阵
        node_deg = np.array(incidence_matrix.sum(axis=1)).flatten()
        edge_deg = np.array(incidence_matrix.sum(axis=0)).flatten()
        node_deg = torch.FloatTensor(node_deg)
        edge_deg = torch.FloatTensor(edge_deg)
    
    # 添加小值防止除零
    node_deg = node_deg + 1e-8
    edge_deg = edge_deg + 1e-8
    
    return node_deg, edge_deg

def csr_matrix_drop_edge(csr_matrix, drop_rate=0.1):
    """随机删除CSR矩阵中的边，用于数据增强"""
    if drop_rate <= 0:
        return csr_matrix
    
    # 转换为COO格式以方便操作
    coo_matrix = csr_matrix.tocoo()
    nnz = coo_matrix.nnz
    
    # 随机选择要删除的边
    keep_mask = np.random.rand(nnz) > drop_rate
    
    # 创建新的稀疏矩阵
    new_data = coo_matrix.data[keep_mask]
    new_row = coo_matrix.row[keep_mask]
    new_col = coo_matrix.col[keep_mask]
    
    return sp.csr_matrix((new_data, (new_row, new_col)), shape=csr_matrix.shape)

def gen_sparse_H(num_nodes, edges):
    """根据边列表生成稀疏超图入射矩阵"""
    row = []
    col = []
    data = []
    
    for e_idx, nodes_in_edge in enumerate(edges):
        for node in nodes_in_edge:
            if 0 <= node < num_nodes:
                row.append(node)
                col.append(e_idx)
                data.append(1.0)
    
    shape = (num_nodes, len(edges))
    return sp.csr_matrix((data, (row, col)), shape=shape)