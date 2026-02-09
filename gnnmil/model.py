import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (  
    SAGEConv,
    global_mean_pool,
    global_max_pool,
    AttentionalAggregation,
    TopKPooling
)

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

        
