import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import haversine_distance, normalized_adj, transform_csr_matrix_to_tensor, get_hyper_deg, csr_matrix_drop_edge, gen_sparse_H
import scipy.sparse as sp

class POIDataset(Dataset):
    def __init__(self, data_filename, pois_coos_filename, num_users, num_pois, padding_idx, device):
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)
        self.sessions_dict = data[0]
        self.labels_dict = data[1]
        with open(pois_coos_filename, 'rb') as f:
            self.pois_coos_dict = pickle.load(f)
        self.num_users = num_users
        self.num_pois = num_pois
        self.padding_idx = padding_idx
        self.device = device
        self.users_trajs_dict = self.get_user_trajs(self.sessions_dict)
        self.poi_geo_adj = self.gen_poi_geo_adj(num_pois, self.pois_coos_dict)
        self.poi_geo_graph = transform_csr_matrix_to_tensor(normalized_adj(self.poi_geo_adj)).to(device)
        self.cooccur_poi_adj = self.gen_cooccur_poi_adj(self.users_trajs_dict, num_pois, min_cooccur=2)
        # Collaborative co-occurrence graph (global, built from training trajectories)
        self.cooccur_poi_graph = transform_csr_matrix_to_tensor(normalized_adj(self.cooccur_poi_adj)).to(device)
        # Backward compatibility (older code used this name)
        self.user_poi_graph = self.cooccur_poi_graph
        self.all_user_seqs = self.get_all_user_seqs(self.users_trajs_dict)
        self.pad_all_user_seqs = pad_sequence(self.all_user_seqs, batch_first=True, padding_value=padding_idx).to(device)
        self.max_seq_len = self.pad_all_user_seqs.size(1)
        # 构建传统超图（限制边数以节省内存）
        self.geo_incidence = self.build_geo_hyper_incidence(num_pois, self.pois_coos_dict, max_edges=1000)
        self.seq_incidence = self.build_seq_hyper_incidence(self.users_trajs_dict, num_pois, max_edges=2000)
        self.co_incidence = self.build_co_hyper_incidence(self.users_trajs_dict, num_pois, max_edges=1500)
        # 计算超图度信息（用于多语义超图）
        self.node_deg, self.edge_deg = get_hyper_deg(self.seq_incidence)
        # 构建有向超图（限制边数以节省内存）
        self.src_incidence, self.tar_incidence = self.build_directed_hyper_incidence(self.users_trajs_dict, num_pois, max_edges=1000)
        # 构建用户-POI二分图（用于多语义超图融合）
        self.HG_up, self.HG_pu = self.build_user_poi_bipartite_graph(self.users_trajs_dict, num_users, num_pois)

        # POI popularity counts (used by popularity-aware attention)
        poi_counts = np.zeros((num_pois,), dtype=np.float32)
        for traj in self.users_trajs_dict.values():
            for poi in traj:
                if 0 <= poi < num_pois:
                    poi_counts[poi] += 1.0
        self.poi_popularity = torch.tensor(poi_counts, device=device)
        
    def get_user_trajs(self, sessions_dict):
        users_trajs = {}
        for uid, sessions in sessions_dict.items():
            # Flatten sessions more efficiently
            traj = [poi for session in sessions for poi in session]
            users_trajs[uid] = traj
        return users_trajs
        
    def gen_poi_geo_adj(self, num_pois, pois_coos_dict, distance_threshold=3.0):
        adj = np.zeros((num_pois, num_pois))
        for i in range(num_pois):
            lat1, lon1 = pois_coos_dict[i]
            for j in range(i, num_pois):
                lat2, lon2 = pois_coos_dict[j]
                dist = haversine_distance(lon1, lat1, lon2, lat2)
                if dist <= distance_threshold:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return sp.csr_matrix(adj)
        
    def gen_cooccur_poi_adj(self, users_trajs_dict, num_pois, min_cooccur=2):
        adj = np.zeros((num_pois, num_pois))
        from collections import defaultdict
        cooccur = defaultdict(int)
        # Vectorized co-occurrence calculation
        for traj in users_trajs_dict.values():
            traj_array = np.array(traj)
            unique_pois = np.unique(traj_array)
            for i, a in enumerate(unique_pois):
                for b in unique_pois[i+1:]:
                    if a != b:
                        cooccur[(min(a, b), max(a, b))] += 1
        for (a, b), cnt in cooccur.items():
            if cnt >= min_cooccur:
                adj[a, b] = 1
                adj[b, a] = 1
        return sp.csr_matrix(adj)
        
    def get_all_user_seqs(self, users_trajs_dict):
        return [torch.tensor(traj) for traj in users_trajs_dict.values()]
        
    def __len__(self):
        return self.num_users
        
    def __getitem__(self, user_idx):
        user_seq = self.users_trajs_dict[user_idx]
        label = self.labels_dict[user_idx]
        sample = {
            'user_idx': torch.tensor(user_idx).to(self.device),
            'user_seq': torch.tensor(user_seq).to(self.device),
            'label': torch.tensor(label).to(self.device),
        }
        return sample
        
    def build_geo_hyper_incidence(self, num_pois, pois_coos_dict, distance_threshold=3.0, max_edges=1000):
        # 每个POI为中心的地理簇为一个超边（限制边数）
        clusters = []
        edge_count = 0
        for i in range(num_pois):
            if edge_count >= max_edges:
                break
            cluster = [i]
            lat1, lon1 = pois_coos_dict[i]
            for j in range(num_pois):
                if i != j and edge_count < max_edges:
                    lat2, lon2 = pois_coos_dict[j]
                    dist = haversine_distance(lon1, lat1, lon2, lat2)
                    if dist <= distance_threshold:
                        cluster.append(j)
            if len(cluster) > 1:  # Only add non-trivial clusters
                clusters.append(cluster)
                edge_count += 1
        row, col, data = [], [], []
        for e, nodes in enumerate(clusters):
            for n in nodes:
                row.append(n)
                col.append(e)
                data.append(1)
        shape = (num_pois, len(clusters))
        if len(clusters) == 0:  # Handle empty case
            row, col, data = [0], [0], [1]
            shape = (num_pois, 1)
        indices = torch.LongTensor([row, col])
        values = torch.FloatTensor(data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()

    def build_seq_hyper_incidence(self, users_trajs_dict, num_pois, max_edges=2000):
        # 每个用户的历史轨迹为一个超边（限制边数）
        row, col, data = [], [], []
        edge_count = 0
        for e, traj in enumerate(users_trajs_dict.values()):
            if edge_count >= max_edges:
                break
            for n in traj:
                if n < num_pois:
                    row.append(n)
                    col.append(e)
                    data.append(1)
            edge_count += 1
        shape = (num_pois, min(len(users_trajs_dict), max_edges))
        if shape[1] == 0:  # Handle empty case
            row, col, data = [0], [0], [1]
            shape = (num_pois, 1)
        indices = torch.LongTensor([row, col])
        values = torch.FloatTensor(data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()

    def build_co_hyper_incidence(self, users_trajs_dict, num_pois, min_cooccur=2, max_edges=1500):  # 增加min_cooccur阈值
        # 协同超图：共现次数大于阈值的POI集合为超边（限制边数）
        from collections import defaultdict
        cooccur = defaultdict(list)
        for traj in users_trajs_dict.values():
            for i in range(len(traj)):
                for j in range(i+1, len(traj)):
                    a, b = traj[i], traj[j]
                    if a != b:
                        cooccur[(min(a, b), max(a, b))].append((a, b))
        row, col, data = [], [], []
        e = 0
        for (a, b), pairs in cooccur.items():
            if len(pairs) >= min_cooccur and e < max_edges:
                row.extend([a, b])
                col.extend([e, e])
                data.extend([1, 1])
                e += 1
        shape = (num_pois, e)
        if e == 0:
            row = list(range(min(num_pois, 100)))  # Limit initial nodes for memory
            col = list(range(len(row)))
            data = [1]*len(row)
            shape = (num_pois, len(row))
        indices = torch.LongTensor([row, col])
        values = torch.FloatTensor(data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()
        
    def build_directed_hyper_incidence(self, users_trajs_dict, num_pois, max_edges=1000):
        """构建有向超图入射矩阵（限制边数）
        每个用户的轨迹序列作为有向超边，源节点是序列中的前一个位置，目标节点是序列中的后一个位置
        """
        # 构建源节点和目标节点的入射矩阵
        src_row, src_col, src_data = [], [], []
        tar_row, tar_col, tar_data = [], [], []
        
        edge_idx = 0
        # 为每个轨迹中的连续对构建有向超边（限制边数）
        for traj in users_trajs_dict.values():
            if edge_idx >= max_edges:
                break
            for i in range(len(traj) - 1):
                src_node = traj[i]
                tar_node = traj[i+1]
                
                if src_node < num_pois and tar_node < num_pois:
                    # 源节点入射矩阵
                    src_row.append(src_node)
                    src_col.append(edge_idx)
                    src_data.append(1)
                    
                    # 目标节点入射矩阵
                    tar_row.append(tar_node)
                    tar_col.append(edge_idx)
                    tar_data.append(1)
                    
                    edge_idx += 1
                    if edge_idx >= max_edges:
                        break
        
        # 创建源节点入射矩阵
        src_shape = (num_pois, edge_idx)
        if edge_idx == 0:  # Handle empty case
            src_row, src_col, src_data = [0], [0], [1]
            src_shape = (num_pois, 1)
        src_indices = torch.LongTensor([src_row, src_col])
        src_values = torch.FloatTensor(src_data)
        src_incidence = torch.sparse_coo_tensor(src_indices, src_values, torch.Size(src_shape)).coalesce()
        
        # 创建目标节点入射矩阵
        tar_shape = (num_pois, edge_idx)
        if edge_idx == 0:  # Handle empty case
            tar_row, tar_col, tar_data = [0], [0], [1]
            tar_shape = (num_pois, 1)
        tar_indices = torch.LongTensor([tar_row, tar_col])
        tar_values = torch.FloatTensor(tar_data)
        tar_incidence = torch.sparse_coo_tensor(tar_indices, tar_values, torch.Size(tar_shape)).coalesce()
        
        return src_incidence, tar_incidence
        
    def build_user_poi_bipartite_graph(self, users_trajs_dict, num_users, num_pois):
        """构建用户-POI二分图，用于多语义超图融合"""
        # 构建用户-POI交互矩阵
        row, col, data = [], [], []
        
        for user_id, traj in users_trajs_dict.items():
            if user_id >= num_users:
                continue
            # 统计用户访问的POI及其频次
            poi_counts = {}
            for poi in traj:
                if poi < num_pois:
                    poi_counts[poi] = poi_counts.get(poi, 0) + 1
            
            # 添加用户-POI边
            for poi, count in poi_counts.items():
                row.append(user_id)
                col.append(poi)
                data.append(count)
        
        # 创建用户到POI的稀疏矩阵 HG_up
        hg_up_shape = (num_users, num_pois)
        hg_up_indices = torch.LongTensor([row, col])
        hg_up_values = torch.FloatTensor(data)
        HG_up = torch.sparse_coo_tensor(hg_up_indices, hg_up_values, torch.Size(hg_up_shape)).coalesce()
        
        # 创建POI到用户的稀疏矩阵 HG_pu（转置）
        hg_pu_shape = (num_pois, num_users)
        hg_pu_indices = torch.LongTensor([col, row])
        hg_pu_values = torch.FloatTensor(data)
        HG_pu = torch.sparse_coo_tensor(hg_pu_indices, hg_pu_values, torch.Size(hg_pu_shape)).coalesce()
        
        return HG_up, HG_pu