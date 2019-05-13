# try_gcn.py

# example 1
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([
    [0,1,1,2],
    [1,0,2,1]
    ], dtype=torch.long)
# from a to b 0 -> 1 etc

x = torch.tensor([[-1],[0],[1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# example 2
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())

print(data.keys)

print(data['x'])

for key, item in data:
    print('%s found in data'%key)

'edge_attr' in data

data.num_nodes
data.num_edges
data.num_features

data.contains_isolated_nodes()
data.contains_self_loops()
data.is_directed()

# device = torch.device('cuda')
# data = data.to(device)
