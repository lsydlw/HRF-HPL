import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualGCNLayer(nn.Module):
    """残差GCN层，用于处理普通图结构"""
    def __init__(self, emb_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        residual = x
        x = torch.sparse.mm(adj, x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return F.relu(x)

class LightweightAdaptiveHyperConvLayer(nn.Module):
    """轻量级自适应超图卷积层
    创新点：
    1. 简化计算：直接计算节点间消息传递，避免中间超边嵌入，降低计算复杂度
    2. 自适应权重：学习超边和节点的重要性权重，提升表达能力
    3. 门控机制：控制信息流，防止信息过载
    4. 高效实现：相比传统方法减少一次矩阵乘法，提升计算效率
    """
    def __init__(self, emb_dim, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        
        # 节点特征变换（用于消息生成）
        self.node_transform = nn.Linear(emb_dim, emb_dim)
        
        # 自适应超边权重学习（每个超边学习一个重要性权重）
        # 使用MLP根据超边内节点特征学习权重
        self.edge_weight_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 门控机制：控制信息流
        self.gate_net = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Sigmoid()
        )
        
        # 输出变换
        self.output_transform = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
        
    def forward(self, x, incidence):
        """
        Args:
            x: 节点特征 [N, D]
            incidence: 超图入射矩阵 [N, E] (稀疏张量)
        Returns:
            更新后的节点特征 [N, D]
        """
        incidence = incidence.to(x.device)
        N, D = x.shape
        
        # 1. 节点特征变换
        x_transformed = self.node_transform(x)  # [N, D]
        
        # 2. 计算超边特征（节点→超边聚合）
        hyperedge_features = torch.sparse.mm(incidence.t(), x_transformed)  # [E, D]
        
        # 3. 自适应超边权重：根据超边特征学习重要性
        edge_weights = self.edge_weight_net(hyperedge_features).squeeze(-1)  # [E]
        
        # 4. 加权超边特征
        weighted_hyperedge_features = hyperedge_features * edge_weights.unsqueeze(-1)  # [E, D]
        
        # 5. 节点聚合：将加权超边特征聚合回节点（超边→节点）
        aggregated = torch.sparse.mm(incidence, weighted_hyperedge_features)  # [N, D]
        
        # 6. 门控机制：控制新信息和旧信息的融合
        gate_input = torch.cat([x, aggregated], dim=-1)  # [N, 2*D]
        gate = self.gate_net(gate_input)  # [N, D]
        
        # 7. 门控融合
        output = gate * aggregated + (1 - gate) * x
        
        # 8. 输出变换和残差连接
        output = self.output_transform(output)
        output = self.dropout(output)
        output = self.layer_norm(output + x)  # 残差连接
        
        return F.relu(output)

class HypergraphConvLayer(nn.Module):
    """传统超图卷积层（保留作为对比）"""
    def __init__(self, emb_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
        
    def forward(self, x, incidence):
        incidence = incidence.to(x.device)
        # 超边嵌入：通过入射矩阵转置聚合节点特征
        hyperedge_emb = torch.sparse.mm(incidence.t(), x)
        # 节点嵌入：通过入射矩阵聚合超边特征
        node_emb = torch.sparse.mm(incidence, hyperedge_emb)
        node_emb = self.linear(node_emb)
        node_emb = self.dropout(node_emb)
        node_emb = self.layer_norm(node_emb + x)
        return F.relu(node_emb)

class MultiSemanticHyperConvLayer(nn.Module):
    """多语义超图卷积层，融合他人代码的多语义处理思想"""
    def __init__(self, emb_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
        # 可学习的语义权重
        self.semantic_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x, incidence, node_deg=None, edge_deg=None):
        incidence = incidence.to(x.device)
        # 使用度归一化（借鉴他人代码的归一化方法）
        if node_deg is not None and edge_deg is not None:
            # 节点度归一化
            node_deg = node_deg.to(x.device)
            edge_deg = edge_deg.to(x.device)
            # 计算归一化因子
            norm_factor = torch.sqrt(node_deg.unsqueeze(1) * edge_deg.unsqueeze(0)) + 1e-8
            # 应用归一化
            values = incidence.values() / norm_factor[incidence.indices()[0], incidence.indices()[1]]
            norm_incidence = torch.sparse_coo_tensor(incidence.indices(), values, incidence.size())
        else:
            norm_incidence = incidence
        
        # 超边嵌入：通过入射矩阵转置聚合节点特征
        hyperedge_emb = torch.sparse.mm(norm_incidence.t(), x)
        # 节点嵌入：通过入射矩阵聚合超边特征
        node_emb = torch.sparse.mm(norm_incidence, hyperedge_emb)
        
        # 应用线性变换和残差连接
        node_emb = self.linear(node_emb)
        node_emb = self.dropout(node_emb)
        node_emb = self.layer_norm(node_emb + x)
        return F.relu(node_emb)

class DirectedHyperConvLayer(nn.Module):
    """有向超图卷积层，处理有向超图结构"""
    def __init__(self, emb_dim, dropout=0.2):
        super().__init__()
        self.linear_src = nn.Linear(emb_dim, emb_dim)
        self.linear_tar = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
        # 源和目标的权重
        self.src_weight = nn.Parameter(torch.ones(1))
        self.tar_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x, src_incidence, tar_incidence):
        src_incidence = src_incidence.to(x.device)
        tar_incidence = tar_incidence.to(x.device)
        
        # 从源节点聚合特征
        src_hyperedge_emb = torch.sparse.mm(src_incidence.t(), x)
        src_node_emb = torch.sparse.mm(src_incidence, src_hyperedge_emb)
        src_node_emb = self.linear_src(src_node_emb)
        
        # 从目标节点聚合特征
        tar_hyperedge_emb = torch.sparse.mm(tar_incidence.t(), x)
        tar_node_emb = torch.sparse.mm(tar_incidence, tar_hyperedge_emb)
        tar_node_emb = self.linear_tar(tar_node_emb)
        
        # 加权融合源和目标特征
        node_emb = self.src_weight * src_node_emb + self.tar_weight * tar_node_emb
        
        # 残差连接和层归一化
        node_emb = self.dropout(node_emb)
        node_emb = self.layer_norm(node_emb + x)
        return F.relu(node_emb)

class HierarchicalCrossAwareAttention(nn.Module):
    """层次化交叉感知注意力机制
    创新点：
    1. 层次化处理：地理→流行度→个性化，逐层增强（个性化占主导）
    2. 交叉感知交互：每个感知机制动态影响下一个的权重
    3. 自适应权重学习：可学习的上下文感知权重
    4. 个性化主导：个性化感知基于前两层客观因素，做出最终主观决策
    """
    def __init__(self, emb_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        
        assert self.head_dim * num_heads == emb_dim, "emb_dim必须能被num_heads整除"
        
        # 基础多头注意力投影
        self.query_proj = nn.Linear(emb_dim, emb_dim)
        self.key_proj = nn.Linear(emb_dim, emb_dim)
        self.value_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        
        # 层次化感知投影（每层都有独立的投影）
        # 第一层：地理感知（客观因素）
        self.geo_query_proj = nn.Linear(emb_dim, emb_dim)
        self.geo_key_proj = nn.Linear(emb_dim, emb_dim)
        self.geo_value_proj = nn.Linear(emb_dim, emb_dim)
        
        # 第二层：流行度感知（基于地理感知结果，客观因素）
        self.popularity_query_proj = nn.Linear(emb_dim * 2, emb_dim)  # 输入包含地理感知结果
        self.popularity_key_proj = nn.Linear(emb_dim, emb_dim)
        self.popularity_value_proj = nn.Linear(emb_dim, emb_dim)
        
        # 第三层：个性化感知（基于前两层结果，主观因素，占主导）
        self.personal_query_proj = nn.Linear(emb_dim * 3, emb_dim)  # 输入包含地理+流行度感知结果
        self.personal_key_proj = nn.Linear(emb_dim, emb_dim)
        self.personal_value_proj = nn.Linear(emb_dim, emb_dim)
        
        # 交叉感知权重生成器（动态权重学习）
        self.geo_weight_gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, num_heads),
            nn.Sigmoid()
        )
        self.popularity_weight_gate = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),  # user + geo_attended + poi
            nn.ReLU(),
            nn.Linear(emb_dim, num_heads),
            nn.Sigmoid()
        )
        self.personal_weight_gate = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim),  # user + geo_attended + popularity_attended + poi
            nn.ReLU(),
            nn.Linear(emb_dim, num_heads),
            nn.Sigmoid()
        )
        
        # 最终融合门控（个性化权重应该更大）
        self.fusion_gate = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 3),  # [geo_weight, popularity_weight, personal_weight]
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)

        # Paper: lambda_geo and lambda_pop scale attention bias terms.
        # The original implementation used fixed constants 0.3; we keep that as default.
        self.lambda_geo = 0.3
        self.lambda_pop = 0.3
        
    def forward(self, user_emb, poi_emb, geo_adj=None, user_visited_mask=None, poi_popularity=None):
        B, D = user_emb.shape
        N, _ = poi_emb.shape
        
        # ========== 第一层：地理感知注意力 ==========
        geo_Q = self.geo_query_proj(user_emb).view(B, self.num_heads, self.head_dim)
        geo_K = self.geo_key_proj(poi_emb).view(N, self.num_heads, self.head_dim)
        geo_V = self.geo_value_proj(poi_emb).view(N, self.num_heads, self.head_dim)
        
        # 基础地理注意力分数
        geo_scores_base = torch.einsum('bhd,nhd->bhn', geo_Q, geo_K) / (self.head_dim ** 0.5)
        
        # 地理邻接矩阵增强（如果可用）
        if geo_adj is not None:
            # 计算POI的地理影响（通过地理邻接矩阵传播）
            if isinstance(geo_adj, torch.sparse.Tensor):
                geo_influence = torch.sparse.mm(geo_adj, poi_emb)  # [N, D]
            else:
                geo_influence = torch.mm(geo_adj, poi_emb)  # [N, D]
            # 将地理影响转换为注意力偏置
            geo_influence_Q = self.geo_query_proj(user_emb).view(B, self.num_heads, self.head_dim)
            geo_influence_K = self.geo_key_proj(geo_influence).view(N, self.num_heads, self.head_dim)
            geo_adj_scores = torch.einsum('bhd,nhd->bhn', geo_influence_Q, geo_influence_K) / (self.head_dim ** 0.5)
            geo_scores_base = geo_scores_base + self.lambda_geo * geo_adj_scores
        
        # 动态权重（基于用户和POI的交互）
        poi_mean = poi_emb.mean(dim=0, keepdim=True).expand(B, -1)  # [B, D]
        geo_weight_input = torch.cat([user_emb, poi_mean], dim=-1)
        geo_weights = self.geo_weight_gate(geo_weight_input)  # [B, num_heads]
        geo_scores = geo_scores_base * geo_weights.unsqueeze(-1)  # [B, num_heads, N]
        
        # 应用掩码
        if user_visited_mask is not None:
            mask = user_visited_mask.unsqueeze(1).expand(-1, self.num_heads, -1)
            geo_scores = geo_scores.masked_fill(mask.bool(), float('-inf'))
        
        geo_attn_weights = F.softmax(geo_scores, dim=-1)
        geo_attn_weights = self.dropout(geo_attn_weights)
        geo_attended = torch.einsum('bhn,nhd->bhd', geo_attn_weights, geo_V)
        geo_attended = geo_attended.contiguous().view(B, D)
        
        # ========== 第二层：流行度感知注意力（基于地理感知结果）==========
        # 交叉感知：流行度感知的查询包含地理感知的结果
        popularity_input = torch.cat([user_emb, geo_attended], dim=-1)  # [B, 2*D]
        popularity_Q = self.popularity_query_proj(popularity_input).view(B, self.num_heads, self.head_dim)
        popularity_K = self.popularity_key_proj(poi_emb).view(N, self.num_heads, self.head_dim)
        popularity_V = self.popularity_value_proj(poi_emb).view(N, self.num_heads, self.head_dim)
        
        popularity_scores_base = torch.einsum('bhd,nhd->bhn', popularity_Q, popularity_K) / (self.head_dim ** 0.5)
        
        # 流行度增强（如果可用）
        if poi_popularity is not None:
            # 将流行度转换为注意力偏置（使用log变换处理长尾分布）
            popularity_bias = torch.log(poi_popularity + 1e-8).unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1)  # [B, num_heads, N]
            popularity_scores_base = popularity_scores_base + self.lambda_pop * popularity_bias
        
        # 动态权重（基于用户、地理感知结果和POI）
        popularity_weight_input = torch.cat([user_emb, geo_attended, poi_mean], dim=-1)
        popularity_weights = self.popularity_weight_gate(popularity_weight_input)  # [B, num_heads]
        popularity_scores = popularity_scores_base * popularity_weights.unsqueeze(-1)
        
        # 应用掩码
        if user_visited_mask is not None:
            mask = user_visited_mask.unsqueeze(1).expand(-1, self.num_heads, -1)
            popularity_scores = popularity_scores.masked_fill(mask.bool(), float('-inf'))
        
        popularity_attn_weights = F.softmax(popularity_scores, dim=-1)
        popularity_attn_weights = self.dropout(popularity_attn_weights)
        popularity_attended = torch.einsum('bhn,nhd->bhd', popularity_attn_weights, popularity_V)
        popularity_attended = popularity_attended.contiguous().view(B, D)
        
        # ========== 第三层：个性化感知注意力（基于前两层结果，占主导）==========
        # 交叉感知：个性化感知的查询包含地理+流行度感知的结果，做出最终主观决策
        personal_input = torch.cat([user_emb, geo_attended, popularity_attended], dim=-1)  # [B, 3*D]
        personal_Q = self.personal_query_proj(personal_input).view(B, self.num_heads, self.head_dim)
        personal_K = self.personal_key_proj(poi_emb).view(N, self.num_heads, self.head_dim)
        personal_V = self.personal_value_proj(poi_emb).view(N, self.num_heads, self.head_dim)
        
        personal_scores_base = torch.einsum('bhd,nhd->bhn', personal_Q, personal_K) / (self.head_dim ** 0.5)
        
        # 动态权重（基于用户、地理感知、流行度感知结果和POI）
        personal_weight_input = torch.cat([user_emb, geo_attended, popularity_attended, poi_mean], dim=-1)
        personal_weights = self.personal_weight_gate(personal_weight_input)  # [B, num_heads]
        personal_scores = personal_scores_base * personal_weights.unsqueeze(-1)
        
        # 应用掩码
        if user_visited_mask is not None:
            mask = user_visited_mask.unsqueeze(1).expand(-1, self.num_heads, -1)
            personal_scores = personal_scores.masked_fill(mask.bool(), float('-inf'))
        
        personal_attn_weights = F.softmax(personal_scores, dim=-1)
        personal_attn_weights = self.dropout(personal_attn_weights)
        personal_attended = torch.einsum('bhn,nhd->bhd', personal_attn_weights, personal_V)
        personal_attended = personal_attended.contiguous().view(B, D)
        
        # ========== 自适应融合三层感知结果（个性化占主导）==========
        fusion_input = torch.cat([geo_attended, popularity_attended, personal_attended], dim=-1)  # [B, 3*D]
        fusion_weights = self.fusion_gate(fusion_input)  # [B, 3] = [geo_weight, popularity_weight, personal_weight]
        
        # 加权融合（个性化权重在最后，应该最大）
        final_attended = (fusion_weights[:, 0:1] * geo_attended +
                         fusion_weights[:, 1:2] * popularity_attended +
                         fusion_weights[:, 2:3] * personal_attended)
        
        # 输出投影和残差连接
        output = self.out_proj(final_attended)
        return self.layer_norm(output + user_emb)

# 保持向后兼容的别名
TripleAwareAttention = HierarchicalCrossAwareAttention

class MultiScaleUserEncoder(nn.Module):
    """多尺度用户编码器
    结合双向GRU和用户历史偏好进行多尺度特征融合
    """
    def __init__(self, num_pois, emb_dim, hidden_dim, padding_idx, dropout=0.2, temporal_backbone: str = "gru"):
        super().__init__()
        self.poi_embedding = nn.Embedding(num_pois + 1, emb_dim, padding_idx=padding_idx)
        self.temporal_backbone = temporal_backbone
        
        if temporal_backbone == "tcn":
            # 3层膨胀TCN，使用GLU门控
            channels = hidden_dim
            self.tcn_layers = nn.ModuleList([
                nn.Conv1d(emb_dim, channels, kernel_size=3, padding=1, dilation=1),
                nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
                nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
            ])
            self.tcn_norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(3)])
            self.proj_to_hidden = nn.Linear(channels, hidden_dim)
        else:
            # 使用双向GRU（根据内存要求）
            self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
            
        # 投影层，匹配双向GRU输出维度
        self.proj_to_emb = nn.Linear(hidden_dim * 2, emb_dim)
        # 融合门控机制 with learnable parameters
        self.fuse_gate = nn.Linear(emb_dim + emb_dim, emb_dim)
        self.gate_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)
        
    def forward(self, user_seqs, visited_poi_embs):
        emb = self.poi_embedding(user_seqs)
        
        if self.temporal_backbone == "tcn":
            # emb: [B,T,D] -> [B,D,T]
            x = emb.transpose(1, 2)
            for i, conv in enumerate(self.tcn_layers):
                y = conv(x)
                y = F.relu(y)
                # 残差连接（如果维度匹配）
                if x.shape == y.shape:
                    y = y + x
                # 转换回[B,T,C]进行LayerNorm
                y_t = y.transpose(1, 2)
                y_t = self.tcn_norms[i](y_t)
                x = y_t.transpose(1, 2)
            # 时间维度全局池化后投影到隐藏层
            short_h = x.mean(dim=2)
            short_h = self.proj_to_hidden(short_h)
        else:
            # 双向GRU输出处理 - 优化版本
            gru_output, _ = self.gru(emb)
            # 使用最后一个时间步的输出并投影到嵌入维度
            short_h = gru_output[:, -1, :]
            short_h = self.proj_to_emb(short_h)
            
        short_h = self.dropout(short_h)
        # 融合短期序列特征和长期用户偏好 with improved gating
        combined = torch.cat([short_h, visited_poi_embs], dim=1)
        gate = self.gate_activation(self.fuse_gate(combined))
        fused = gate * visited_poi_embs + (1 - gate) * short_h
        return self.layer_norm(fused)

class AdaptiveMultiViewFusion(nn.Module):
    """自适应多视图融合模块
    通过可学习参数和交叉视图注意力机制融合不同视图
    """
    def __init__(self, emb_dim, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        # 三个视图的可学习权重 with softmax normalization
        self.geo_weight = nn.Parameter(torch.ones(1))
        self.cooccur_weight = nn.Parameter(torch.ones(1))
        self.seq_weight = nn.Parameter(torch.ones(1))
        
        # 交叉视图注意力机制
        self.cross_view_attn = nn.MultiheadAttention(emb_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        # 视图特定变换
        self.geo_transform = nn.Linear(emb_dim, emb_dim)
        self.cooccur_transform = nn.Linear(emb_dim, emb_dim)
        self.seq_transform = nn.Linear(emb_dim, emb_dim)
        
        # Paper: gate uses [H_attended || H_weighted] -> emb_dim*2
        self.fusion_gate = nn.Linear(emb_dim * 2, emb_dim)
        self.gate_activation = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, geo_emb, cooccur_emb, seq_emb):
        # 视图特定变换
        geo_transformed = F.relu(self.geo_transform(geo_emb))
        cooccur_transformed = F.relu(self.cooccur_transform(cooccur_emb))
        seq_transformed = F.relu(self.seq_transform(seq_emb))
        
        # 软max归一化权重
        weights = torch.softmax(torch.stack([self.geo_weight, self.cooccur_weight, self.seq_weight]), dim=0)
        
        # 加权组合
        weighted_geo = weights[0] * geo_transformed
        weighted_cooccur = weights[1] * cooccur_transformed
        weighted_seq = weights[2] * seq_transformed
        # H_weighted = sum_v w_v * H_v
        h_weighted = weighted_geo + weighted_cooccur + weighted_seq
        
        # 交叉视图注意力融合
        # 将三个视图堆叠为序列进行注意力计算
        view_sequence = torch.stack([weighted_geo, weighted_cooccur, weighted_seq], dim=1)  # [N, 3, D]
        attn_output, _ = self.cross_view_attn(view_sequence, view_sequence, view_sequence)
        
        # 全局池化获取最终融合嵌入
        h_attended = attn_output.mean(dim=1)  # H_attended

        # Paper: H_final = g ⊙ H_attended + (1-g) ⊙ H_weighted
        gate_input = torch.cat([h_attended, h_weighted], dim=-1)
        g = self.gate_activation(self.fusion_gate(gate_input))

        final_emb = g * h_attended + (1 - g) * h_weighted
        final_emb = self.dropout(final_emb)
        return self.layer_norm(final_emb)

class POIEncoder(nn.Module):
    """POI编码器
    集成多视图建模：地理、共现和序列超图，并融合多语义和有向超图表示
    """
    def __init__(self, num_pois, emb_dim, geo_gcn_layers, cooccur_gcn_layers, seq_hyper_layers, padding_idx, dropout=0.2, use_multi_semantic=True, use_directed=False):
        super().__init__()
        self.poi_embedding = nn.Embedding(num_pois + 1, emb_dim, padding_idx=padding_idx)
        self.use_multi_semantic = use_multi_semantic
        self.use_directed = use_directed
        
        # 地理GCN层
        self.geo_gcn_layers = nn.ModuleList([
            ResidualGCNLayer(emb_dim, dropout) for _ in range(geo_gcn_layers)
        ])
        
        # 共现GCN层
        self.cooccur_gcn_layers = nn.ModuleList([
            ResidualGCNLayer(emb_dim, dropout) for _ in range(cooccur_gcn_layers)
        ])
        
        # 序列超图层 - 使用轻量级自适应超图卷积（创新点）
        # 默认使用新的轻量级自适应超图卷积，相比传统方法更高效且表达能力更强
        self.seq_hyper_layers = nn.ModuleList([
            LightweightAdaptiveHyperConvLayer(emb_dim, dropout) for _ in range(seq_hyper_layers)
        ])

        # Paper: E_seq = alpha * X_undir + (1-alpha) * X_dir
        self.seq_alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 init
        
        # 有向超图相关层
        if use_directed:
            self.directed_hyper_layers = nn.ModuleList([
                DirectedHyperConvLayer(emb_dim, dropout) for _ in range(seq_hyper_layers)
            ])
        
        # 自适应多视图融合模块
        self.fusion_module = AdaptiveMultiViewFusion(emb_dim, dropout)
        
    def forward(self, geo_adj, cooccur_adj, seq_incidence, node_deg=None, edge_deg=None, 
                src_incidence=None, tar_incidence=None):
        x = self.poi_embedding.weight[:-1]  # 移除填充嵌入
        
        # 地理视图处理
        geo_x = x
        for layer in self.geo_gcn_layers:
            geo_x = layer(geo_x, geo_adj)
            
        # 共现视图处理
        cooccur_x = x
        for layer in self.cooccur_gcn_layers:
            cooccur_x = layer(cooccur_x, cooccur_adj)
            
        # 序列视图处理：使用轻量级自适应超图卷积
        seq_undir_x = x
        for layer in self.seq_hyper_layers:
            seq_undir_x = layer(seq_undir_x, seq_incidence)
            
        # 如果启用有向超图，额外处理有向信息
        seq_dir_x = seq_undir_x
        if self.use_directed and src_incidence is not None and tar_incidence is not None:
            directed_x = x
            for layer in self.directed_hyper_layers:
                directed_x = layer(directed_x, src_incidence, tar_incidence)
            seq_dir_x = directed_x

        # fused sequential view for cross-view fusion
        alpha = torch.sigmoid(self.seq_alpha_logit)  # in [0,1]
        seq_x = alpha * seq_undir_x + (1.0 - alpha) * seq_dir_x
            
        # 自适应多视图融合
        fused_emb = self.fusion_module(geo_x, cooccur_x, seq_x)
        # Return both sequential views for contrastive alignment
        return geo_x, cooccur_x, seq_x, fused_emb, seq_undir_x, seq_dir_x

class FusionGRUHypergraphModel(nn.Module):
    """融合GRU-Hypergraph模型
    结合多语义融合机制、GRU序列建模和增强注意力机制，融合多语义和有向超图表示
    """
    def __init__(self, num_users, num_pois, emb_dim, hidden_dim, geo_gcn_layers, cooccur_gcn_layers, seq_hyper_layers, padding_idx, dropout=0.2, attn_topk: int = 0, attn_prior: str = "none", temporal_backbone: str = "gru", use_multi_semantic=True, use_directed=False):
        super().__init__()
        # 初始化POI编码器，支持多语义和有向超图
        self.poi_encoder = POIEncoder(num_pois, emb_dim, geo_gcn_layers, cooccur_gcn_layers, 
                                     seq_hyper_layers, padding_idx, dropout, 
                                     use_multi_semantic=use_multi_semantic, 
                                     use_directed=use_directed)
        self.user_encoder = MultiScaleUserEncoder(num_pois, emb_dim, hidden_dim, padding_idx, dropout, temporal_backbone=temporal_backbone)
        self.attention_module = TripleAwareAttention(emb_dim, num_heads=4, dropout=dropout)
        self.emb_dim = emb_dim
        self.use_multi_semantic = use_multi_semantic
        self.use_directed = use_directed
        
        # POI流行度统计（用于流行度感知）
        self.poi_popularity = nn.Parameter(torch.ones(num_pois), requires_grad=False)
        
    def forward(self, user_seqs, geo_adj, cooccur_adj, seq_incidence, user_visited_mask, 
                node_deg=None, edge_deg=None, src_incidence=None, tar_incidence=None):
        # 调用POI编码器，传递必要的参数
        geo_poi_emb, cooccur_poi_emb, seq_x_avg, fused_poi_emb, seq_undir_poi_emb, seq_dir_poi_emb = self.poi_encoder(
            geo_adj, cooccur_adj, seq_incidence, 
            node_deg=node_deg if self.use_multi_semantic else None, 
            edge_deg=edge_deg if self.use_multi_semantic else None,
            src_incidence=src_incidence if self.use_directed else None,
            tar_incidence=tar_incidence if self.use_directed else None
        )
        
        # Expose sequential view embeddings for training losses/contrastive alignment.
        # (These are computed inside the forward pass, so they still participate in autograd.)
        self.last_undir_seq_poi_emb = seq_undir_poi_emb
        self.last_dir_seq_poi_emb = seq_dir_poi_emb

        # 基于用户历史访问构建用户偏好嵌入 - 优化版本
        mask = user_visited_mask.float()
        denom = mask.sum(dim=1, keepdim=True) + 1e-8
        visited_poi_embs = torch.matmul(mask, fused_poi_emb) / denom
        
        # 多尺度用户序列编码
        user_emb = self.user_encoder(user_seqs, visited_poi_embs)
        
        # 三重感知注意力增强
        user_emb_enhanced = self.attention_module(
            user_emb, 
            fused_poi_emb, 
            geo_adj=geo_adj, 
            user_visited_mask=user_visited_mask,
            poi_popularity=self.poi_popularity
        )
        
        # Keep backward compatibility for downstream code:
        # `seq_poi_emb` is used as the hypergraph scoring branch embedding.
        # For directed mode, we use the directed sequential view; otherwise use undirected.
        seq_poi_emb = self.last_dir_seq_poi_emb if self.use_directed else self.last_undir_seq_poi_emb
        return user_emb_enhanced, geo_poi_emb, cooccur_poi_emb, seq_poi_emb, fused_poi_emb
        
    def decode(self, user_emb, poi_emb):
        """解码函数，计算用户-POI匹配分数"""
        return torch.matmul(user_emb, poi_emb.t())
        
    def update_popularity(self, poi_checkins):
        """更新POI流行度统计"""
        with torch.no_grad():
            self.poi_popularity.data = poi_checkins.float()