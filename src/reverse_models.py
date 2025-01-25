import torch
import torch.nn as nn
import math
from torch import Tensor


def uniform(size, value):
    if isinstance(value, Tensor):
        bound = 1. / math.sqrt(size)
        value.data.uniform_(-bound, bound)
        

class GatedGraphConv(nn.Module):
    def __init__(self, out_channels, num_layers=1, bias=True, device='cpu', **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.device = device
        self.weight = nn.Parameter(torch.randn(num_layers, self.out_channels, self.out_channels))
        self.rnn = nn.GRUCell(out_channels, out_channels, bias=bias)
        
    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()
        
    def forward(self, x, edge_index, *args):
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')
        
        if x.size(-1) < self.out_channels:
            x = torch.cat((x, x.new_zeros(x.size(0), self.out_channels - x.size(-1), device=self.device)), dim=1)
    
        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            
            node_feature_padded = torch.cat((m, torch.zeros(1, self.out_channels, device=self.device)))
            neigh_feature = node_feature_padded[edge_index]
            m = neigh_feature.sum(1)
            
            x = self.rnn(m, x)
        
        return x
    
class GraphPooling(nn.Module):
    def __init__(self, in_channels, out_channels, n_dim=1, bias=True, aggr='mean', **kwargs):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.aggr = aggr
        
        self.net = nn.Sequential(nn.Linear(self.in_channels, self.out_channels, bias=bias),
                                 nn.ELU(),
                                 nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                 nn.ELU(),)
        
        self.readout = nn.Sequential(nn.Linear(self.out_channels, self.out_channels, bias=bias),
                                     nn.ELU(),
                                     nn.Linear(self.out_channels, n_dim, bias=bias),)
    
    def forward(self, x):
        output = self.net(x)
        if self.aggr == 'mean':
            output = torch.mean(output, dim=-2, keepdim=True)
        elif self.aggr == 'sum':
            output = torch.sum(output, dim=-2, keepdim=True)
        elif self.aggr == 'max':
            output = torch.max(output, dim=-2, keepdim=True)
        else:
            raise NotImplementedError
        
        return self.readout(output) 
        
        

class GNNModel(nn.Module):         
    def __init__(self, ntips, n_dim, hidden_dim=100, num_layers=1, gnn_type='gcn', aggr='sum', project=False, bias=True, device='cpu', **kwargs):
        super().__init__()
        self.ntips = ntips
        self.device=device
        
        if gnn_type == 'ggnn':
            self.gnn = GatedGraphConv(hidden_dim, num_layers=num_layers, bias=bias, device=device).to(device=device)
        else:
            raise NotImplementedError
                   
        self.pooling_net = GraphPooling(hidden_dim, hidden_dim, n_dim=n_dim, bias=bias, aggr=aggr, device=device).to(device=device)
        
    def mp_forward(self, node_features, edge_index):
        batch_size, nnodes = edge_index.shape[0], edge_index.shape[1]
        compact_node_features = node_features.view(-1, node_features.shape[-1])
        compact_edge_index = torch.where(edge_index>-1, edge_index + torch.arange(0, batch_size, device=self.device)[:,None,None]*nnodes, -1)
        compact_edge_index = compact_edge_index.view(-1, compact_edge_index.shape[-1])

        compact_node_features = self.gnn(compact_node_features, compact_edge_index)
        node_features = compact_node_features.view(batch_size, nnodes, compact_node_features.shape[-1])
        out = self.pooling_net(node_features).squeeze(1)

        return out
    