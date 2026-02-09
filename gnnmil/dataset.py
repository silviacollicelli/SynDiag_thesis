import torch
from torchmil.datasets import BinaryClassificationDataset
from torch_geometric.utils import to_undirected
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F

def build_edge_index(x, k):
    num_nodes = x.size(0)
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)    # single node: no edges
    #x = F.normalize(x, p=2, dim=1)
    dist = torch.cdist(x, x)        # pairwise Euclidean distances
    dist.fill_diagonal_(float('inf'))       # ignore self-distance
    knn = dist.topk(k=min(k, num_nodes - 1), largest=False).indices         # k nearest neighbors
    row = torch.arange(num_nodes).unsqueeze(1).repeat(1, knn.size(1))
    edge_index = torch.stack([row.flatten(), knn.flatten()], dim=0)

    return edge_index

class GraphMILDataset(Dataset):
    def __init__(self, mil_dataset, k):
        super().__init__()
        self.mil_dataset = mil_dataset
        self.k = k

    def len(self):
        return len(self.mil_dataset)

    def get(self, idx):
        x = self.mil_dataset[idx]['X'].float()
        y = self.mil_dataset[idx]['Y']

        edge_index = build_edge_index(x, self.k)
        edge_index = to_undirected(edge_index)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([y], dtype=torch.long)
        )
        return data

#features_path = "C:\\Users\\utente\\Documents\\UNI\\MAGISTRALE\\tesi\\raw_dataset\\artifacts\\embeddings\\features"
#labels_path = "C:\\Users\\utente\\Documents\\UNI\\MAGISTRALE\\tesi\\raw_dataset\\artifacts\\embeddings\\labels"
#mil_dataset = BinaryClassificationDataset(features_path, labels_path, bag_keys=["X", "Y"], load_at_init=False, verbose=False)
#graph_dataset = GraphMILDataset(mil_dataset, k=2)
#print(graph_dataset[0]['x'].size(1))