# !/usr/bin/env python3
import sys, os, math, random, shutil, json, time, warnings
import numpy as np
import pandas as pd
import networkx as nx
import argparse
from copy import deepcopy
from collections import defaultdict
from rdkit.ML.Scoring.Scoring import CalcEnrichment
from sklearn.decomposition import PCA
from scipy.stats import rankdata
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import sparse
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix
from tqdm import tqdm
from statsmodels.stats.multitest import local_fdr
from scipy.stats import zscore
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import gaussian_kde, norm   


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)  
print("Running on device:", device)

CONV_THRESHOLD = 1e-6

def seed_everything(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
seed_everything(1)

def calculate_enrichment_factor(node_scores, val_seeds, top_fractions=[0.01]):
    node_scores = sorted(node_scores, key=lambda x:x[1], reverse=True)
    scores_for_rdkit = [[score, 1 if node in val_seeds else 0] for (node, score) in node_scores]
    label_col_idx = 1
    ef_list = CalcEnrichment(scores_for_rdkit, label_col_idx, top_fractions)
    return ef_list

def split_train_val(lst_seed, train_ratio=0.8, used_val_seeds=None):
    if used_val_seeds is None:
        used_val_seeds = set()
    available_val_seeds = [s for s in lst_seed if s not in used_val_seeds]
    if len(available_val_seeds) < len(lst_seed) * (1 - train_ratio):
        raise ValueError("Not enough seeds left for validation.")
    random.shuffle(available_val_seeds)
    val_size = int(len(lst_seed) * (1 - train_ratio))
    val_seeds = available_val_seeds[:val_size]
    train_seeds = [s for s in lst_seed if s not in val_seeds]
    return train_seeds, val_seeds

def calculate_max_samples(lst_seed, train_ratio=0.8):
    used_val_seeds=set()
    max_samples=0
    while True:
        available=[s for s in lst_seed if s not in used_val_seeds]
        if len(available)< len(lst_seed)*(1-train_ratio):
            break
        random.shuffle(available)
        val_size=int(len(lst_seed)*(1-train_ratio))
        val_seeds=available[:val_size]
        used_val_seeds.update(val_seeds)
        max_samples+=1
    return max_samples

class OptimizedWalker:
    def __init__(self, original_ppi, constantWeight=False, absWeight=False, addBidirectionEdge=False):
        self._build_matrices(original_ppi, constantWeight, absWeight, addBidirectionEdge)    
        self.A_coo = self._convert_to_coo(self.adjacency_matrix)
        self.N = self.A_coo.shape[0]
        print(f"[OptimizedWalker] Node={self.N}")
        
        edge_index_np = np.vstack((self.A_coo.row, self.A_coo.col))
        self.edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
        self.edge_weight = torch.tensor(self.A_coo.data, dtype=torch.float32, device=device)
        print(f"[OptimizedWalker] edge_index={self.edge_index.shape}, edge_weight={self.edge_weight.shape}")
        
    def run_exp(self, seed2weight, restart_prob, normalize=False, node_list=[]):
        p_0 = self._set_up_p0(seed2weight)
        if normalize and np.sum(p_0) > 0:
            p_0 = p_0 / np.sum(p_0)
        
        p_t = p_0.copy()
        arr_p = np.empty((len(p_t), 1))
        arr_p[:, 0] = p_t
        
        diff_norm = 1.0
        iteration = 0
        CONV_THRESHOLD = 1e-6
        
        while diff_norm > CONV_THRESHOLD:
            iteration += 1
            p_t_1 = self._calculate_next_p(p_t, p_0, restart_prob)
            if normalize and np.sum(p_t_1) > 0:
                p_t_1 = p_t_1 / np.sum(p_t_1)
            diff_norm = np.linalg.norm(p_t_1 - p_t, 1)
            p_t = p_t_1
            arr_p = np.c_[arr_p, p_t]
            if iteration >= 50000:
                print("[NP] Reached 50,000 iterations, stopping early...")
                break
        print(f"[NP finished] {iteration} iterations, final diff_norm={diff_norm:.6e}")
        total_idx = {node: i for i, node in enumerate(sorted(self.dic_node2idx))}
        output = []
        if node_list:
            for node in node_list:
                i = total_idx[node]
                output.append([node, p_t[i], arr_p[i, :].tolist()])
        else:
            for node in sorted(self.dic_node2idx):
                i = total_idx[node]
                output.append([node, p_t[i], arr_p[i, :].tolist()])
        return output
    
    def _calculate_next_p(self, p_t, p_0, restart_prob):
        no_restart = self.A_coo.dot(p_t) * (1 - restart_prob)
        restart = p_0 * restart_prob
        return no_restart + restart
    
    def _set_up_p0(self, seed2weight):
        p_0 = np.zeros(self.N, dtype=np.float32)
        for seed, weight in seed2weight.items():
            if seed in self.dic_node2idx:
                idx = self.dic_node2idx[seed]
                p_0[idx] = weight
        return p_0
    
    def _build_matrices(self, original_ppi, constantWeight, absWeight, addBidirectionEdge):
        nodes = set()
        edges = []
        with open(original_ppi, 'r') as f:
            for line in f:
                src, tgt, *weight = line.strip().split('\t')
                weight = float(weight[0]) if weight else 1.0
                if constantWeight:
                    weight = 1.0
                if absWeight:
                    weight = abs(weight)
                nodes.update([src, tgt])
                edges.append((src, tgt, weight))
                if addBidirectionEdge:
                    edges.append((tgt, src, weight))
        self.dic_node2idx = {node: idx for idx, node in enumerate(sorted(nodes))}
        size = len(self.dic_node2idx)
        row, col, data = [], [], []
        for src, tgt, weight in edges:
            i, j = self.dic_node2idx[src], self.dic_node2idx[tgt]
            row.append(i)
            col.append(j)
            data.append(weight)
        self.adjacency_matrix = self._normalize_cols(row, col, data, size)
    def _normalize_cols(self, row, col, data, size):
        matrix = coo_matrix((data, (row, col)), shape=(size, size))
        matrix = normalize(matrix, norm='l1', axis=0)
        return matrix
    def _convert_to_coo(self, matrix):
        return matrix.tocoo()


class RankGCN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, embed_dim=32, dropout=0.2):
        super(RankGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.residual1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        reduced_dim = hidden_dim // 2
        self.conv2 = GCNConv(hidden_dim, reduced_dim)
        self.residual2 = nn.Linear(hidden_dim, reduced_dim)  
        self.ln2 = nn.LayerNorm(reduced_dim)
        self.fc = nn.Linear(reduced_dim, 1)  
        self.proj = nn.Linear(reduced_dim, embed_dim)  
        
    def forward(self, x, edge_index, edge_weight):
        x1 = self.conv1(x, edge_index, edge_weight) + self.residual1(x)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv2(x1, edge_index, edge_weight) + self.residual2(x1)
        x2 = self.ln2(x2)
        x2 = F.relu(x2)
        
        ranking_logits = self.fc(x2).squeeze(-1)
        embeddings = F.normalize(self.proj(x2), dim=1)
        
        return ranking_logits, embeddings

def composite_loss(ranking_logits, embeddings, np_features, val_idx, non_val_idx, temperature=0.07, lambda_rank=0.3, lambda_contrast=0.6, gamma_np=0.5):
    device = ranking_logits.device
    target = torch.zeros_like(ranking_logits, device=device)
    target[val_idx] = 1.0
    num_pos = len(val_idx)
    num_neg = ranking_logits.shape[0] - num_pos
    pos_weight = torch.tensor(num_neg / (num_pos + 1e-6), device=device)
    bce_loss_raw = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(ranking_logits, target)
    weight_np = 1 + gamma_np * np_features
    bce_loss = (bce_loss_raw * weight_np).mean()
    
    margin = 0.5
    ranknet_losses = []
    non_val_idx_list = list(non_val_idx)
    for pos in val_idx:
        sampled_neg = non_val_idx_list
        pos_score = ranking_logits[pos]
        neg_scores = ranking_logits[sampled_neg]
        losses = F.relu(margin - (pos_score - neg_scores))
        ranknet_losses.append(losses.mean())
    ranknet_loss = torch.stack(ranknet_losses).mean() if ranknet_losses else torch.tensor(0.0, device=device)
    
    if len(val_idx) < 2:
        contrast_loss = torch.tensor(0.0, device=device)
    else:
        anchor_embeddings = embeddings[val_idx]  
        sim_matrix = torch.matmul(anchor_embeddings, anchor_embeddings.T)  
        sim_matrix = sim_matrix / temperature
        contrast_loss = 0.0
        num_val = len(val_idx)
        
        for i in range(num_val):
            mask = torch.ones(num_val, dtype=torch.bool, device=device)
            mask[i] = False
            sims = sim_matrix[i][mask]  
            pos_sim, pos_idx = torch.max(sims, dim=0)
            negatives = torch.cat([sims[:pos_idx], sims[pos_idx+1:]])
            numerator = torch.exp(pos_sim)
            denominator = numerator + torch.sum(torch.exp(negatives)) + 1e-6
            loss_i = -torch.log(numerator / denominator)
            contrast_loss += loss_i
        contrast_loss = contrast_loss / num_val
    
    total_loss = (1 - lambda_rank)*bce_loss + lambda_rank * ranknet_loss + lambda_contrast * contrast_loss
    return total_loss, bce_loss, ranknet_loss, contrast_loss

def update_seeds_LFDR(
    rank_gcn,
    x_tensor,
    edge_index,
    edge_weight,
    nodes_order,
    train_seeds,
    augmented_seeds,
    local_seed2weight,
    beta,
    epoch,
    fdr_add_threshold=0.1,
    baseline=0.5,
    central_width=1.0,
    min_kde_samples=50,
):
    with torch.no_grad():
        logits, _ = rank_gcn(x_tensor, edge_index, edge_weight)
    logits_np = logits.cpu().numpy()
    z_scores = zscore(logits_np)
    try:
        fdr_array = local_fdr(z_scores, nbins=30)
    except Exception as e:
        if len(logits_np) < min_kde_samples:
            return augmented_seeds, local_seed2weight
        z_scores = (logits_np - logits_np.mean()) / (logits_np.std() + 1e-12)
        kde = gaussian_kde(z_scores)
        f_z = kde(z_scores) + 1e-12
        
        core_mask = np.abs(z_scores) < central_width
        if core_mask.sum() < 5:
            core_mask = np.argsort(np.abs(z_scores))[:max(5, len(z_scores) // 10)]
        mu0 = z_scores[core_mask].mean()
        sigma0 = z_scores[core_mask].std(ddof=1) + 1e-12
        f0_z = norm.pdf(z_scores, loc=mu0, scale=sigma0)
        
        pi0 = np.clip(np.mean(f0_z / f_z), 0, 1)
        fdr_array = np.clip(pi0 * f0_z / f_z, 0, 1)
    
    new_augmented = set(augmented_seeds)
    add_cnt = remove_cnt = update_cnt = 0

    for i, node in enumerate(nodes_order):
        lfdr_i = fdr_array[i]
        prob_i = torch.sigmoid(logits[i]).item()
        
        if node in train_seeds:
            local_seed2weight[node] += beta * (prob_i - baseline)
            continue
        
        if node not in new_augmented:
            if lfdr_i < fdr_add_threshold:
                new_augmented.add(node)
                local_seed2weight[node] = 1.0
                add_cnt += 1
        else:
            if lfdr_i > fdr_add_threshold:
                new_augmented.remove(node)
                local_seed2weight.pop(node, None)
                remove_cnt += 1
            else:
                local_seed2weight[node] += beta * (prob_i - baseline)
                update_cnt += 1
                
    print(f"[LFDR] added={add_cnt}, removed={remove_cnt}, updated={update_cnt}")
    return new_augmented, local_seed2weight

def train_rank(
    rank_gcn,               
    local_seed2weight,      
    augmented_seeds,        
    wk,                     
    node2sim,               
    hybrid_scores,          
    pt_cb_dict,             
    fp_dict,                
    np_dict,                
    train_seeds,            
    val_seeds,              
    optimizer,
    num_epochs=5,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    beta=0.7,               
    lambda_rank=0.3,
    lambda_contrast=0.6,
    gamma_np=0.5,
):
    edge_index = wk.edge_index
    edge_weight = wk.edge_weight
    nodes_order = sorted(wk.dic_node2idx.keys())
    
    val_indices = [wk.dic_node2idx[n] for n in val_seeds if n in wk.dic_node2idx]
    non_val_indices = [i for i in range(wk.N) if i not in val_indices]
    if len(val_indices) == 0:
        raise ValueError("No validation node exists in the graph.")
    
    pt_cb_dim = pt_cb_dict[list(pt_cb_dict.keys())[0]].shape[0]
    fp_dim = fp_dict[list(fp_dict.keys())[0]].shape[0]
    
    rank_gcn.train()
    scaler = GradScaler()
    
    final_loss = 0.0
    for epoch in range(num_epochs):
        x_list = []
        for node in nodes_order:
            w = local_seed2weight.get(node, 0.0)
            s = node2sim.get(node, 0.0)
            h = hybrid_scores.get(node, 0.0)
            n = np_dict.get(node, 0.0)
            pt_cb = pt_cb_dict.get(node, np.zeros(pt_cb_dim)).tolist()
            fp = fp_dict.get(node, np.zeros(fp_dim)).tolist()
            x_list.append([w, s, h, n] + pt_cb + fp)
        x_tensor = torch.tensor(x_list, dtype=torch.float32, device=device)
        optimizer.zero_grad()
        with autocast():  
            ranking_logits, embeddings = rank_gcn(x_tensor, edge_index, edge_weight)
            np_feautre = x_tensor[:,3]
            loss, loss_bce, loss_ranknet, loss_contrast = composite_loss(
                ranking_logits, embeddings, np_feautre, val_indices, non_val_indices,
                temperature=0.07, lambda_rank=lambda_rank, lambda_contrast=lambda_contrast, gamma_np=gamma_np
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        final_loss = loss.item()
        print(f"[train_rank] epoch={epoch+1}/{num_epochs}, Total_loss={final_loss:.4f}, BCE={loss_bce.item():.4f}, RankNet={loss_ranknet.item():.4f}, Contrast={loss_contrast.item():.4f}")
        
        augmented_seeds, local_seed2weight = update_seeds_LFDR(
            rank_gcn, x_tensor, edge_index, edge_weight,
            nodes_order,
            train_seeds,
            augmented_seeds,
            local_seed2weight,
            beta,
            epoch,
            fdr_add_threshold=0.1,
            baseline=0.5,
            central_width=1.0,
            min_kde_samples=50,
        )
    return local_seed2weight, final_loss

def main_propagation(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_graphs', nargs='+')
    parser.add_argument('seed')
    parser.add_argument('-o', required=True)
    parser.add_argument('-e','--restart_prob', type=float, default=0.1)
    parser.add_argument('-constantWeight', default='False', choices=['True','False'])
    parser.add_argument('-absoluteWeight', default='False', choices=['True','False'])
    parser.add_argument('-addBidirectionEdge', default='False', choices=['True','False'])
    parser.add_argument('-normalize', default='False', choices=['True','False'])
    parser.add_argument('-train_ratio', type=float, default=0.5)
    parser.add_argument('-data_SEED', required=True)
    parser.add_argument('-data_ACTIVE', required=True)
    parser.add_argument('-data_DECOY', required=True)
    parser.add_argument('-fp_SEED', required=True)
    parser.add_argument('-fp_ACTIVE', required=True)
    parser.add_argument('-fp_DECOY', required=True)
    parser.add_argument('-PT_CB_SEED', required=True)
    parser.add_argument('-PT_CB_ACTIVE', required=True)
    parser.add_argument('-PT_CB_DECOY', required=True)
    args = parser.parse_args()
    
    print(f"[INFO] train ratio: {args.train_ratio}")
    
    with open(args.seed, "r") as fp:
        lst_seed = [line.strip().split()[0] for line in fp]
    print(f"[INFO] Total {len(lst_seed)} seed nodes loaded.")
    
    DATA_SEED = pd.read_csv(args.data_SEED, sep=',')
    DATA_ACTIVE = pd.read_csv(args.data_ACTIVE, sep=',')
    DATA_DECOY = pd.read_csv(args.data_DECOY, sep=',')
    
    FP_SEED = pd.read_pickle(args.fp_SEED)
    FP_ACTIVE = pd.read_pickle(args.fp_ACTIVE)
    FP_DECOY = pd.read_pickle(args.fp_DECOY)
    
    PT_CB_SEED = pd.read_pickle(args.PT_CB_SEED)
    PT_CB_ACTIVE = pd.read_pickle(args.PT_CB_ACTIVE)
    PT_CB_DECOY = pd.read_pickle(args.PT_CB_DECOY)
    
    input_graph_filename = os.path.basename(args.input_graphs[0])
    keyword = input_graph_filename.split("_")[-1].replace('.txt','')
    print(f"[INFO] Keyword: {keyword}")
    data_seed = FP_SEED[keyword]
    data_ACTIVE = FP_ACTIVE[keyword]
    data_DECOY = FP_DECOY[keyword]
    
    seed_ids = DATA_SEED['cid'].tolist()
    ACTIVE_ids = DATA_ACTIVE['CHEMBLID'].tolist()
    DECOY_ids = DATA_DECOY['DECOYID'].tolist()
    
    TOTAL_NODES = seed_ids + ACTIVE_ids + DECOY_ids
    PT_CB_combine = np.vstack([PT_CB_SEED, PT_CB_ACTIVE, PT_CB_DECOY])
    PT_CB_DICT = {node: PT_CB_combine[i] for i, node in enumerate(TOTAL_NODES)}
    pt_cb_dim = PT_CB_combine.shape[1]
    
    FP_combine = np.vstack([FP_SEED[keyword], FP_ACTIVE[keyword], FP_DECOY[keyword]])
    FP_DICT = {node: FP_combine[i] for i, node in enumerate(TOTAL_NODES)}
    fp_dim = FP_combine.shape[1]
    
    wk = OptimizedWalker(
        args.input_graphs[0],
        constantWeight=(args.constantWeight=='True'),
        absWeight=(args.absoluteWeight=='True'),
        addBidirectionEdge=(args.addBidirectionEdge=='True')
    )
    
    max_samples = calculate_max_samples(lst_seed, train_ratio=args.train_ratio)
    print(f"[INFO] N: {max_samples}")
    
    def precompute_pca_similarity_parzen(
        data_seed,
        data_ACTIVE,
        data_DECOY,
        seed_ids,
        ACTIVE_ids,
        DECOY_ids,
        train_seeds,
        n_components=2,
        gamma=1.0,           
        aggregator='mean'    
    ):
        data_combined = np.vstack([data_seed, data_ACTIVE, data_DECOY])
        node_ids = seed_ids + ACTIVE_ids + DECOY_ids
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(data_combined)
        node2pca = {node: X_pca[i] for i, node in enumerate(node_ids)}
        train_coords = np.array([node2pca[s] for s in train_seeds if s in node2pca])
        node2sim = {}

        if len(train_coords) == 0:
            for node in node_ids:
                node2sim[node] = 0.0
            return node2sim
        for node, coords in node2pca.items():
            diffs = train_coords - coords  
            dists_sq = np.sum(diffs**2, axis=1)  
            sims = np.exp(-gamma * dists_sq)     
            
            if aggregator == 'mean':
                node2sim[node] = np.mean(sims)
            elif aggregator == 'max':
                node2sim[node] = np.max(sims)
            else:
                node2sim[node] = np.mean(sims)
        return node2sim

    input_dim = 4 + pt_cb_dim + fp_dim 
    rank_gcn = RankGCN(input_dim=input_dim, hidden_dim=64, embed_dim=32).to(device)
    def init_lin(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    rank_gcn.apply(init_lin)
    optimizer = optim.Adam(rank_gcn.parameters(), lr=0.0008, weight_decay=1.57e-5)

    final_weights_list = []
    used_val_seeds = set()
    hybrid_alpha = 0.5

    for sample_idx in tqdm(range(max_samples), desc="Processing train/val splits"):
        print(f"\n[N {sample_idx+1}/{max_samples}]")
        try:
            train_seeds, val_seeds = split_train_val(lst_seed, train_ratio=args.train_ratio, used_val_seeds=used_val_seeds)
            used_val_seeds.update(val_seeds)
            
            local_seed2weight = {s: 1.0 for s in train_seeds}
            
            node2sim = precompute_pca_similarity_parzen(data_seed, data_ACTIVE, data_DECOY, seed_ids, ACTIVE_ids, DECOY_ids, train_seeds, n_components=2, gamma=1, aggregator='mean')
            pca_scores = np.array([node2sim.get(node, 0.0) for node in node2sim])
            pca_ranks = rankdata(-pca_scores)
            augmented_seeds = set()
            
            best_ef = -1e9
            best_seed2weight = None
            best_model_state = None
            old_loss = 1e9
            new_loss = old_loss - 1.0
            train_count = 0
            MAX_ITER = 7
            patience = 3  
            
            while train_count < MAX_ITER and patience > 0:
                if train_count >= MAX_ITER:
                    print(f"[SUBDYVE] Reach maximum iterations ({MAX_ITER}), end.")
                    break
                old_loss_current = new_loss
                
                rwr_result = wk.run_exp(local_seed2weight, args.restart_prob, normalize=(args.normalize=='True'))
                node_scores = [(n, sc) for (n, sc, _) in rwr_result]
                node_scores.sort(key=lambda x: x[1], reverse=True)
                np_scores = np.array([sc for _, sc in node_scores])
                np_ranks = rankdata(-np_scores)
                np_dict = {node: score for node, score in node_scores}
                
                hybrid_scores={}
                for i, (node, _) in enumerate(node_scores):
                    hy = hybrid_alpha * np_ranks[i] + (1 - hybrid_alpha) * pca_ranks[i]
                    hybrid_scores[node] = hy
                
                efs_current = calculate_enrichment_factor(node_scores, val_seeds, [0.1])
                if efs_current[0] > best_ef:
                    best_ef = efs_current[0]
                    best_seed2weight = deepcopy(local_seed2weight)
                
                print(f"[SUBDYVE] NP EF={efs_current[0]:.4f}")
                
                local_seed2weight, new_loss = train_rank(
                    rank_gcn=rank_gcn,
                    local_seed2weight=local_seed2weight,
                    augmented_seeds=augmented_seeds,
                    wk=wk,
                    node2sim=node2sim,
                    hybrid_scores=hybrid_scores,
                    pt_cb_dict=PT_CB_DICT,
                    fp_dict=FP_DICT,
                    np_dict=np_dict,
                    train_seeds=train_seeds,
                    val_seeds=val_seeds,
                    optimizer=optimizer,
                    num_epochs=50, 
                    device=device,
                    beta=0.7,
                    lambda_rank=0.3,
                    lambda_contrast=0.6,
                    gamma_np=5,
                )
                print(f"[SUBDYVE] step={train_count}, old_loss={old_loss_current:.4f}, new_loss={new_loss:.4f}")
                
                if new_loss < old_loss_current - 1e-6:
                    patience = 3  
                else:
                    patience -= 1
                    print(f"[EarlyStopping] patience reduced to {patience}")
                old_loss = new_loss
                train_count += 1
                
            print(f"[Split {sample_idx}] best EF={best_ef:.4f}")
            final_weights_list.append(best_seed2weight if best_seed2weight else deepcopy(local_seed2weight))
        except Exception as e:
            continue
        
    def aggregate_seed_weights(seed_weights_list, method="max"):
        final_seed2weight = defaultdict(float)
        if method == "max":
            for sw in seed_weights_list:
                for seed, w in sw.items():
                    final_seed2weight[seed] = max(final_seed2weight[seed], w)
        return dict(final_seed2weight)
    final_seed2weight = aggregate_seed_weights(final_weights_list, method="max")
    final_run = wk.run_exp(final_seed2weight, args.restart_prob, normalize=(args.normalize=='True'))
    final_node_scores = [(n, sc) for (n, sc, _) in final_run]
    final_node_scores.sort(key=lambda x: x[1], reverse=True)

    with open(args.o, 'w') as of:
        for (n, sc) in final_node_scores:
            of.write(f"{n}\t{sc}\n")
    print("[INFO] Final Network Propagation Done.")

if __name__=='__main__':
    main_propagation(sys.argv)
