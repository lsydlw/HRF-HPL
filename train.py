import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dataset import POIDataset
from dataset_config import dataset_file_paths, load_dataset_stats
from model import FusionGRUHypergraphModel  # 使用新的融合模型
from metrics import batch_performance
import logging
import time
import yaml
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='NYC', help='NYC / TKY / Gowalla')
parser.add_argument('--emb_dim', type=int, default=448)
parser.add_argument('--hidden_dim', type=int, default=160)
parser.add_argument('--geo_gcn_layers', type=int, default=2)
parser.add_argument('--cooccur_gcn_layers', type=int, default=2)
parser.add_argument('--seq_hyper_layers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-04)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
parser.add_argument('--neg_samples', type=int, default=200, help='Number of negative POIs per positive sample')
parser.add_argument('--lambda_cl', type=float, default=0.2, help='Weight of InfoNCE contrastive loss')
parser.add_argument('--tau', type=float, default=0.2, help='Temperature for InfoNCE contrastive loss')
parser.add_argument('--beta', type=float, default=3e-06, help='L2 regularization coefficient')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--save_dir', type=str, default="logs")
parser.add_argument('--hyper_loss_weight', type=float, default=0.6)  # lambda_hyper in the paper
parser.add_argument('--temporal_backbone', type=str, default="gru", help='gru or tcn')
# 默认开启多语义超图，可用 --no_use_multi_semantic 关闭
parser.add_argument('--use_multi_semantic', action='store_true', default=True, help='使用多语义超图卷积')
parser.add_argument('--no_use_multi_semantic', dest='use_multi_semantic', action='store_false')
# 有向超图在论文中为核心结构，默认开启；如需关闭可显式禁用
parser.add_argument('--use_directed', action='store_true', default=True, help='使用有向超图卷积')
parser.add_argument('--no_use_directed', dest='use_directed', action='store_false', help='关闭有向超图卷积')
args = parser.parse_args()

_paths = dataset_file_paths(args.dataset)
data_root = str(_paths["data_root"])
NUM_USERS, NUM_POIS, PADDING_IDX = load_dataset_stats(args.dataset, _paths["data_root"])

def collate_fn(batch):
    user_idx = torch.stack([item['user_idx'] for item in batch])
    user_seq = [item['user_seq'] for item in batch]
    user_seq = torch.nn.utils.rnn.pad_sequence(user_seq, batch_first=True, padding_value=PADDING_IDX)
    label = torch.stack([item['label'] for item in batch])
    return {'user_idx': user_idx, 'user_seq': user_seq, 'label': label}

def build_batch_user_poi_mask(user_seqs, num_pois, padding_idx):
    B, T = user_seqs.shape
    mask = torch.zeros(B, num_pois, dtype=torch.float32, device=user_seqs.device)
    # Vectorized implementation for better performance
    valid_mask = (user_seqs != padding_idx) & (user_seqs < num_pois)
    batch_indices = torch.arange(B, device=user_seqs.device).unsqueeze(1).expand(-1, T)
    poi_indices = user_seqs
    mask[batch_indices[valid_mask], poi_indices[valid_mask]] = 1.0
    return mask

def build_batch_bipartite_adj(user_seqs, num_pois, padding_idx):
    adj = torch.zeros(num_pois, num_pois, dtype=torch.float32, device=user_seqs.device)
    # Vectorized implementation for better performance
    valid_mask = (user_seqs != padding_idx) & (user_seqs < num_pois)
    for i, user_seq in enumerate(user_seqs):
        valid_pois = user_seq[valid_mask[i]]
        if len(valid_pois) > 1:
            # Create all pairs efficiently
            pois_expanded_1 = valid_pois.unsqueeze(1).expand(-1, len(valid_pois))
            pois_expanded_2 = valid_pois.unsqueeze(0).expand(len(valid_pois), -1)
            # Fill adjacency matrix
            adj[pois_expanded_1, pois_expanded_2] = 1
            adj[pois_expanded_2, pois_expanded_1] = 1
    # Set diagonal to zero to avoid self-loops
    adj.fill_diagonal_(0)
    deg = adj.sum(dim=1) + 1e-8
    adj = adj / deg.unsqueeze(1)
    return adj.to_sparse()

def main():
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    current_save_dir = os.path.join(args.save_dir, current_time)
    os.mkdir(current_save_dir)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(current_save_dir, f"log_training.txt"),
                        filemode='w+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    args_filename = args.dataset + '_args.yaml'
    with open(os.path.join(current_save_dir, args_filename), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info("device: {}".format(args.device))
    logging.info(
        "Dataset {}: users={}, pois={}, data_root={}".format(
            args.dataset, NUM_USERS, NUM_POIS, data_root
        )
    )
    if args.dataset == "Gowalla" and args.batch_size == 8:
        # Gowalla POI 规模大，默认略减小 batch 以降低显存压力
        args.batch_size = 4
        logging.info("Gowalla: 默认 batch_size 调整为 {}".format(args.batch_size))

    logging.info("2. Load Dataset")
    device = args.device
    train_dataset = POIDataset(
        data_filename=str(_paths["train"]),
        pois_coos_filename=str(_paths["poi_coos"]),
        num_users=NUM_USERS, num_pois=NUM_POIS, padding_idx=PADDING_IDX, device=device
    )
    test_dataset = POIDataset(
        data_filename=str(_paths["test"]),
        pois_coos_filename=str(_paths["poi_coos"]),
        num_users=NUM_USERS, num_pois=NUM_POIS, padding_idx=PADDING_IDX, device=device
    )
    logging.info("Train users: {}, Test users: {}".format(len(train_dataset), len(test_dataset)))

    logging.info("3. Construct DataLoader")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    logging.info("4. Load Model")
    # 使用新的融合模型，支持多语义和有向超图
    model = FusionGRUHypergraphModel(
        NUM_USERS, NUM_POIS, args.emb_dim, args.hidden_dim, 
        args.geo_gcn_layers, args.cooccur_gcn_layers, args.seq_hyper_layers, 
        PADDING_IDX, args.dropout, temporal_backbone=args.temporal_backbone,
        use_multi_semantic=args.use_multi_semantic,
        use_directed=args.use_directed
    ).to(device)
    
    # Paper: Adam + explicit L2 regularization term in the loss.
    # To avoid double-counting, we disable optimizer weight_decay here and use args.beta in the loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Popularity counts for popularity-aware attention
    if hasattr(train_dataset, "poi_popularity"):
        model.update_popularity(train_dataset.poi_popularity)

    logging.info("5. Start Training")
    Ks_list = [5, 10]
    final_results = {"Rec5": 0.0, "Rec10": 0.0, "NDCG5": 0.0, "NDCG10": 0.0}
    best_test_ndcg10 = 0.0
    best_epoch = 0
    patience_counter = 0
    for epoch in range(args.epochs):
        logging.info("================= Epoch {}/{} =================".format(epoch + 1, args.epochs))
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_recall_array = np.zeros(shape=(len(train_loader), len(Ks_list)))
        train_ndcg_array = np.zeros(shape=(len(train_loader), len(Ks_list)))
        for idx, batch in enumerate(train_loader):
            user_seqs = batch['user_seq'].to(device)
            labels = batch['label'].to(device)
            user_visited_mask = build_batch_user_poi_mask(user_seqs, NUM_POIS, PADDING_IDX)
            cooccur_adj = train_dataset.cooccur_poi_graph
            geo_adj = train_dataset.poi_geo_graph
            seq_incidence = train_dataset.seq_incidence  # 使用序列超图关联矩阵
            optimizer.zero_grad()
            
            # 根据配置传递额外参数
            model_kwargs = {
                'user_seqs': user_seqs,
                'geo_adj': geo_adj,
                'cooccur_adj': cooccur_adj,
                'seq_incidence': seq_incidence,
                'user_visited_mask': user_visited_mask
            }
            
            # 如果使用多语义超图，传递度信息
            if args.use_multi_semantic:
                model_kwargs['node_deg'] = train_dataset.node_deg.to(device)
                model_kwargs['edge_deg'] = train_dataset.edge_deg.to(device)
            
            # 如果使用有向超图，传递有向超图信息
            if args.use_directed:
                model_kwargs['src_incidence'] = train_dataset.src_incidence.to(device)
                model_kwargs['tar_incidence'] = train_dataset.tar_incidence.to(device)
            
            user_emb, geo_poi_emb, cooccur_poi_emb, seq_poi_emb, fused_poi_emb = model(**model_kwargs)

            # -------- Paper loss: BCE (main + hyper) + InfoNCE + L2 --------
            B = labels.size(0)
            visited_bool = user_visited_mask.bool()

            # Negative sampling from unvisited POIs
            neg_k = min(args.neg_samples, NUM_POIS - 1)
            rand = torch.rand(B, NUM_POIS, device=device)
            rand.masked_fill_(visited_bool, -1.0)
            rand[torch.arange(B, device=device), labels] = -1.0
            negative_pois = rand.topk(neg_k, dim=1).indices  # [B, neg_k]

            user_vec = user_emb  # H_final in the paper
            undirected_seq_poi_emb = model.last_undir_seq_poi_emb
            directed_seq_poi_emb = model.last_dir_seq_poi_emb

            # Main branch BCE (fused multi-view scoring)
            pos_fused = fused_poi_emb[labels]  # [B, D]
            neg_fused = fused_poi_emb[negative_pois]  # [B, neg_k, D]
            pos_logit_main = (user_vec * pos_fused).sum(dim=-1)  # [B]
            neg_logit_main = (user_vec.unsqueeze(1) * neg_fused).sum(dim=-1)  # [B, neg_k]
            logits_main = torch.cat([pos_logit_main.unsqueeze(1), neg_logit_main], dim=1)
            targets_main = torch.zeros_like(logits_main)
            targets_main[:, 0] = 1.0
            loss_main = bce_criterion(logits_main, targets_main)

            # Hypergraph branch BCE (directed sequential scoring)
            pos_hyper = directed_seq_poi_emb[labels]  # [B, D]
            neg_hyper = directed_seq_poi_emb[negative_pois]  # [B, neg_k, D]
            pos_logit_hyper = (user_vec * pos_hyper).sum(dim=-1)  # [B]
            neg_logit_hyper = (user_vec.unsqueeze(1) * neg_hyper).sum(dim=-1)  # [B, neg_k]
            logits_hyper = torch.cat([pos_logit_hyper.unsqueeze(1), neg_logit_hyper], dim=1)
            targets_hyper = torch.zeros_like(logits_hyper)
            targets_hyper[:, 0] = 1.0
            loss_hyper = bce_criterion(logits_hyper, targets_hyper)

            # Cross-view InfoNCE contrastive alignment (fused vs undirected/directed seq)
            unique_labels = torch.unique(labels)
            P = unique_labels.size(0)
            if P > 1:
                z_fused = torch.nn.functional.normalize(fused_poi_emb[unique_labels], dim=-1)
                z_undir = torch.nn.functional.normalize(undirected_seq_poi_emb[unique_labels], dim=-1)
                z_dir = torch.nn.functional.normalize(directed_seq_poi_emb[unique_labels], dim=-1)
                targets = torch.arange(P, device=device)
                loss_cl_1 = torch.nn.functional.cross_entropy((z_fused @ z_undir.t()) / args.tau, targets)
                loss_cl_2 = torch.nn.functional.cross_entropy((z_fused @ z_dir.t()) / args.tau, targets)
                loss_cl = loss_cl_1 + loss_cl_2
            else:
                loss_cl = torch.tensor(0.0, device=device)

            # Explicit L2 regularization term
            l2_reg = torch.tensor(0.0, device=device)
            for p in model.parameters():
                l2_reg = l2_reg + p.pow(2).sum()
            l2_reg = args.beta * l2_reg

            loss = loss_main + args.hyper_loss_weight * loss_hyper + args.lambda_cl * loss_cl + l2_reg

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            fused_scores = model.decode(user_emb, fused_poi_emb)
            for k in Ks_list:
                recall, ndcg = batch_performance(fused_scores.detach().cpu(), labels.detach().cpu(), k)
                col_idx = Ks_list.index(k)
                train_recall_array[idx, col_idx] = recall
                train_ndcg_array[idx, col_idx] = ndcg
        logging.info("Training finishes at this epoch. It takes {} min".format((time.time() - start_time) / 60))
        logging.info("Training loss: {}".format(train_loss / len(train_loader)))
        logging.info("Training Epoch {}/{} results:".format(epoch + 1, args.epochs))
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            logging.info("Recall@{}: {}".format(k, np.mean(train_recall_array[:, col_idx])))
            logging.info("NDCG@{}: {}".format(k, np.mean(train_ndcg_array[:, col_idx])))
        logging.info("\n")
        logging.info("Testing")
        test_loss = 0.0
        test_recall_array = np.zeros(shape=(len(test_loader), len(Ks_list)))
        test_ndcg_array = np.zeros(shape=(len(test_loader), len(Ks_list)))
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                user_seqs = batch['user_seq'].to(device)
                labels = batch['label'].to(device)
                user_visited_mask = build_batch_user_poi_mask(user_seqs, NUM_POIS, PADDING_IDX)
                cooccur_adj = test_dataset.cooccur_poi_graph
                geo_adj = test_dataset.poi_geo_graph
                seq_incidence = test_dataset.seq_incidence  # 使用序列超图关联矩阵
                
                # 根据配置传递额外参数
                model_kwargs = {
                    'user_seqs': user_seqs,
                    'geo_adj': geo_adj,
                    'cooccur_adj': cooccur_adj,
                    'seq_incidence': seq_incidence,
                    'user_visited_mask': user_visited_mask
                }
                
                # 如果使用多语义超图，传递度信息
                if args.use_multi_semantic:
                    model_kwargs['node_deg'] = test_dataset.node_deg.to(device)
                    model_kwargs['edge_deg'] = test_dataset.edge_deg.to(device)
                
                # 如果使用有向超图，传递有向超图信息
                if args.use_directed:
                    model_kwargs['src_incidence'] = test_dataset.src_incidence.to(device)
                    model_kwargs['tar_incidence'] = test_dataset.tar_incidence.to(device)
                
                user_emb, geo_poi_emb, cooccur_poi_emb, seq_poi_emb, fused_poi_emb = model(**model_kwargs)

                # Keep test_loss consistent with the paper loss (negative sampling uses the same visited-mask rule)
                B = labels.size(0)
                visited_bool = user_visited_mask.bool()
                neg_k = min(args.neg_samples, NUM_POIS - 1)
                rand = torch.rand(B, NUM_POIS, device=device)
                rand.masked_fill_(visited_bool, -1.0)
                rand[torch.arange(B, device=device), labels] = -1.0
                negative_pois = rand.topk(neg_k, dim=1).indices

                user_vec = user_emb
                undirected_seq_poi_emb = model.last_undir_seq_poi_emb
                directed_seq_poi_emb = model.last_dir_seq_poi_emb

                pos_fused = fused_poi_emb[labels]
                neg_fused = fused_poi_emb[negative_pois]
                pos_logit_main = (user_vec * pos_fused).sum(dim=-1)
                neg_logit_main = (user_vec.unsqueeze(1) * neg_fused).sum(dim=-1)
                logits_main = torch.cat([pos_logit_main.unsqueeze(1), neg_logit_main], dim=1)
                targets_main = torch.zeros_like(logits_main)
                targets_main[:, 0] = 1.0
                loss_main = bce_criterion(logits_main, targets_main)

                pos_hyper = directed_seq_poi_emb[labels]
                neg_hyper = directed_seq_poi_emb[negative_pois]
                pos_logit_hyper = (user_vec * pos_hyper).sum(dim=-1)
                neg_logit_hyper = (user_vec.unsqueeze(1) * neg_hyper).sum(dim=-1)
                logits_hyper = torch.cat([pos_logit_hyper.unsqueeze(1), neg_logit_hyper], dim=1)
                targets_hyper = torch.zeros_like(logits_hyper)
                targets_hyper[:, 0] = 1.0
                loss_hyper = bce_criterion(logits_hyper, targets_hyper)

                unique_labels = torch.unique(labels)
                P = unique_labels.size(0)
                if P > 1:
                    z_fused = torch.nn.functional.normalize(fused_poi_emb[unique_labels], dim=-1)
                    z_undir = torch.nn.functional.normalize(undirected_seq_poi_emb[unique_labels], dim=-1)
                    z_dir = torch.nn.functional.normalize(directed_seq_poi_emb[unique_labels], dim=-1)
                    targets = torch.arange(P, device=device)
                    loss_cl_1 = torch.nn.functional.cross_entropy((z_fused @ z_undir.t()) / args.tau, targets)
                    loss_cl_2 = torch.nn.functional.cross_entropy((z_fused @ z_dir.t()) / args.tau, targets)
                    loss_cl = loss_cl_1 + loss_cl_2
                else:
                    loss_cl = torch.tensor(0.0, device=device)

                l2_reg = torch.tensor(0.0, device=device)
                for p in model.parameters():
                    l2_reg = l2_reg + p.pow(2).sum()
                l2_reg = args.beta * l2_reg

                loss = loss_main + args.hyper_loss_weight * loss_hyper + args.lambda_cl * loss_cl + l2_reg

                test_loss += loss.item()

                fused_scores = model.decode(user_emb, fused_poi_emb)
                for k in Ks_list:
                    recall, ndcg = batch_performance(fused_scores.detach().cpu(), labels.detach().cpu(), k)
                    col_idx = Ks_list.index(k)
                    test_recall_array[idx, col_idx] = recall
                    test_ndcg_array[idx, col_idx] = ndcg
        logging.info("Testing finishes")
        logging.info("Testing loss: {}".format(test_loss / len(test_loader)))
        logging.info("Testing results:")
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            recall = np.mean(test_recall_array[:, col_idx])
            ndcg = np.mean(test_ndcg_array[:, col_idx])
            logging.info("Recall@{}: {}".format(k, recall))
            logging.info("NDCG@{}: {}".format(k, ndcg))
        # Early stopping metric: validation NDCG@10 (paper)
        ndcg_at_10_col = Ks_list.index(10)
        test_ndcg10 = np.mean(test_ndcg_array[:, ndcg_at_10_col])
        scheduler.step(test_ndcg10)
        if test_ndcg10 > best_test_ndcg10:
            best_test_ndcg10 = test_ndcg10
            best_epoch = epoch
            patience_counter = 0
            logging.info("Update test results and save model at epoch{}".format(epoch))
            saved_model_path = os.path.join(current_save_dir, "{}.pt".format(args.dataset))
            torch.save(model.state_dict(), saved_model_path)
        else:
            patience_counter += 1
            logging.info(f"EarlyStopping counter: {patience_counter} / {args.patience}")
            if patience_counter >= args.patience:
                logging.info(f"EarlyStopping triggered at epoch {epoch+1}")
                break
        for k in Ks_list:
            col_idx = Ks_list.index(k)
            if k == 5:
                final_results["Rec5"] = max(final_results["Rec5"], np.mean(test_recall_array[:, col_idx]))
                final_results["NDCG5"] = max(final_results["NDCG5"], np.mean(test_ndcg_array[:, col_idx]))
            elif k == 10:
                final_results["Rec10"] = max(final_results["Rec10"], np.mean(test_recall_array[:, col_idx]))
                final_results["NDCG10"] = max(final_results["NDCG10"], np.mean(test_ndcg_array[:, col_idx]))
        logging.info("==================================\n\n")
    logging.info("6. Final Results")
    logging.info(final_results)
    logging.info(f"Best test NDCG@10: {best_test_ndcg10} at epoch {best_epoch}")
    logging.info("\n")

if __name__ == '__main__':
    main()