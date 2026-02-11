import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (  
    SAGEConv,
    DenseSAGEConv,
    dense_diff_pool,
    global_mean_pool,
    global_max_pool,
    AttentionalAggregation,
    TopKPooling
)
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse

class GNNsimple(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers=2,
        pooling="mean"
    ):
        super().__init__()

        # --- GNN layers --- (SAGEConv layers!)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim)) #always at least one layer
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.bn = nn.BatchNorm1d(hidden_dim)

        # --- Pooling ---
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            self.pool = AttentionalAggregation(gate_nn)
        else:
            raise ValueError("Unknown pooling")

        # --- MLP classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),            #eventual sweep also on the dropout rate
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            #x = self.bn(x)
            x = F.relu(x)
        graph_emb = self.pool(x, batch)
        out = self.classifier(graph_emb).squeeze(-1)
        return out

class GNNtopk(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers=2,
        topk_ratio=0.3,
        aggr="mean"
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.pool = TopKPooling(
            hidden_dim,
            ratio=topk_ratio        #k = ceil(ratio * N)
        )

        if aggr == "mean":
            self.aggr = global_mean_pool
        elif aggr == "max":
            self.aggr = global_max_pool
        elif aggr == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            self.aggr = AttentionalAggregation(gate_nn)
        else:
            raise ValueError("Unknown pooling")
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, batch=batch)
        graph_emb = self.aggr(x, batch)

        return self.classifier(graph_emb).squeeze(-1)

def diffpool_manual(z, s, batch):
    """
    z: [N, F] node embeddings
    s: [N, C] assignment matrix
    batch: [N] batch vector
    """

    s = torch.softmax(s, dim=-1)

    out = []
    for b in batch.unique():
        mask = batch == b
        z_b = z[mask]      # [N_b, F]
        s_b = s[mask]      # [N_b, C]

        x_pool = s_b.t() @ z_b  # [C, F]
        out.append(x_pool)

    return torch.cat(out, dim=0)


class GNNcluster(nn.Module):
    def __init__(self, in_dim=1024, 
                 hidden_dim=256, 
                 num_clusters=1
                 ):
        super().__init__()

        self.num_clusters = num_clusters

        self.gnn_embed = nn.ModuleList([
            SAGEConv(in_dim, hidden_dim),
            SAGEConv(hidden_dim, hidden_dim)
        ])

        self.gnn_assign = nn.ModuleList([
            SAGEConv(in_dim, hidden_dim),
            SAGEConv(hidden_dim, num_clusters)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_clusters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        z = x
        for conv in self.gnn_embed:
            z = F.relu(conv(z, edge_index))

        s = x
        for conv in self.gnn_assign:
            s = conv(s, edge_index)

        x_pool = diffpool_manual(z, s, batch)
        x_pool = x_pool.view(batch.max() + 1, -1)

        return self.classifier(x_pool).squeeze(-1)

def diffpool_dense(z, s, edge_index, num_nodes):
    """
    z: [N, F] node embeddings
    s: [N, C] assignment matrix
    edge_index: sparse adjacency
    """

    # Dense adjacency
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [N, N]

    s = torch.softmax(s, dim=-1)

    # Pooled features and adjacency
    X_pool = s.T @ z              # [C, F]
    A_pool = s.T @ A @ s          # [C, C]

    return X_pool, A_pool
        

class DiffPoolGNNMIL(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256, C=1):
        super().__init__()

        self.C = C
        self.hidden_dim = hidden_dim

        # 1️⃣ Node embedding GNN
        self.gnn_embed = SAGEConv(in_dim, hidden_dim)

        # 2️⃣ Assignment GNN
        self.gnn_assign = SAGEConv(in_dim, C)

        # 3️⃣ Second embedding on pooled graph
        self.gnn_embed2 = SAGEConv(hidden_dim, hidden_dim)

        # 4️⃣ Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * C, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        Z = F.relu(self.gnn_embed(x, edge_index))
        S = self.gnn_assign(x, edge_index)

        pooled_X = []
        pooled_A = []

        for b in batch.unique():
            mask = batch == b

            z_b = Z[mask]
            s_b = S[mask]

            # edges belonging to this bag
            node_idx = torch.where(mask)[0]
            edge_mask = torch.isin(edge_index[0], node_idx) & torch.isin(edge_index[1], node_idx)
            edge_b = edge_index[:, edge_mask]
            edge_b = edge_b - node_idx.min()  # reindex

            X_pool, A_pool = diffpool_dense(
                z_b, s_b, edge_b, z_b.size(0)
            )

            pooled_X.append(X_pool)
            pooled_A.append(A_pool)

        # concatenate pooled graphs
        X_pool = torch.cat(pooled_X, dim=0)   # [(B·C), F]

        # build pooled edge_index
        A_pool = torch.block_diag(*pooled_A)
        edge_index_pool, _ = dense_to_sparse(A_pool)

        # batch vector for pooled graph
        batch_pool = torch.repeat_interleave(
            torch.arange(len(pooled_X), device=x.device), self.C
        )

        # 3️⃣ Second GNN
        Z_pool = F.relu(self.gnn_embed2(X_pool, edge_index_pool))

        # 4️⃣ Concatenate clusters per bag
        graph_emb = Z_pool.view(len(pooled_X), -1)

        return self.classifier(graph_emb).squeeze(-1)


class GNNpaper(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 num_clusters
                 ):
        super().__init__()

        self.gnn_embd = SAGEConv(in_dim, hidden_dim)

        self.gnn_pool = SAGEConv(hidden_dim, num_clusters)
        self.mlp = nn.Linear(num_clusters, num_clusters)

        self.gnn_embd2 = DenseSAGEConv(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_clusters, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        z = F.leaky_relu(self.gnn_embd(x, edge_index), negative_slope=0.01)
        s = F.leaky_relu(self.gnn_pool(z, edge_index), negative_slope=0.01)
        s = self.mlp(s)

        s, _ = to_dense_batch(s, batch)
        z, mask = to_dense_batch(z, batch)
        a = to_dense_adj(edge_index, batch)
        z, a, link_loss, ent_loss = dense_diff_pool(z, a, s, mask)
        
        x = F.leaky_relu(self.gnn_embd2(z, a))
        x = x.reshape(x.size(0), -1)
        return self.classifier(x).squeeze(-1),  link_loss + ent_loss

