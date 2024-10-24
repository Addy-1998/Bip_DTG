
import torch
from torch.nn import Linear, ReLU, Sigmoid, LogSoftmax
from torch_geometric.nn import GATConv, APPNP, global_mean_pool
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=None):
        super(MLP, self).__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        if activation == 'relu':
            self.act = ReLU()
        elif activation == 'sigmoid':
            self.act = Sigmoid()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, K=5, alpha=0.1, edge_attr_dim=2, gat_edge_dim=4):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.edge_weight_mlp = MLP(input_dim=edge_attr_dim, hidden_dim=4, output_dim=1, activation='sigmoid')
        self.edge_attr_mlp_1 = MLP(input_dim=edge_attr_dim, hidden_dim=8, output_dim=gat_edge_dim, activation='relu')
        self.edge_attr_mlp_2 = MLP(input_dim=edge_attr_dim, hidden_dim=8, output_dim=gat_edge_dim, activation='relu')
        self.appnp = APPNP(K, alpha)
        self.conv1 = GATConv((-1, -1), hidden_channels[0], heads=6, concat=False)
        self.conv2 = GATConv((-1, -1), hidden_channels[1], heads=6, concat=False)
        self.lin = Linear(hidden_channels[1], num_classes)
        self.log_softmax = LogSoftmax(dim=1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_weight = torch.sigmoid(self.edge_weight_mlp(edge_attr).squeeze())
        modified_edge_attr_1 = self.edge_attr_mlp_1(edge_attr)
        modified_edge_attr_2 = self.edge_attr_mlp_2(edge_attr)
        
        x = self.appnp(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv1(x, edge_index, modified_edge_attr_1)
        x = F.relu(x)
        x = self.conv2(x, edge_index, modified_edge_attr_2)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(x)
        x = self.lin(x)
        x = self.log_softmax(x)
        return x
