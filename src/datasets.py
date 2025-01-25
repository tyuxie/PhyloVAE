import torch
import numpy as np
import os
from torch.utils.data import Dataset
from src.utils import mcmc_treeprob, namenum, summary
from src.vector_representation import tree2vec

def node_embedding(tree, ntips):
    leaf_features = torch.eye(ntips)
    for node in tree.traverse('postorder'):
        if node.is_leaf():
            node.c = 0
            node.d = leaf_features[node.name]
        else:
            child_c, child_d = 0., 0.
            for child in node.children:
                child_c += child.c
                child_d += child.d
            node.c = 1./(3. - child_c) 
            node.d = node.c * child_d
    
    node_features, node_idx_list, edge_index = [], [], []            
    for node in tree.traverse('preorder'):
        neigh_idx_list = []
        if not node.is_root():
            node.d = node.c * node.up.d + node.d
            neigh_idx_list.append(node.up.name)
            
            if not node.is_leaf():
                neigh_idx_list.extend([child.name for child in node.children])
            else:
                neigh_idx_list.extend([-1, -1])              
        else:
            neigh_idx_list.extend([child.name for child in node.children])
        
        edge_index.append(neigh_idx_list)                
        node_features.append(node.d)
        node_idx_list.append(node.name)
    
    branch_idx_map = torch.sort(torch.tensor(node_idx_list).long(), dim=0, descending=False)[1]
    edge_index = torch.tensor(edge_index).long()

    return torch.index_select(torch.stack(node_features), 0, branch_idx_map), edge_index[branch_idx_map]

def process_data(dataset, repo):
    tree_dict, tree_names, tree_wts = mcmc_treeprob('data/short_run_data/' + str(dataset) + '/rep_{}/'.format(repo) + str(dataset) + '.trprobs', 'nexus', taxon='keep')
    taxa = sorted(list(tree_dict.values())[0].get_leaf_names())
    ntips = len(taxa)
    trees, wts = [tree_dict[name] for name in tree_names], tree_wts
    wts = np.array(wts) / np.sum(wts)
    path = os.path.join('embed_data',dataset,'repo{}'.format(repo))
    os.makedirs(path, exist_ok=True)

    np.save(os.path.join(path, 'wts.npy'), wts)
    np.save(os.path.join(path, 'taxa.npy'), taxa)
    node_features_tensor, edge_index_tensor, vec_tensor = [], [], []
    for tree in trees:
        tree_cp = tree.copy()
        namenum(tree_cp, taxa)
        node_features, edge_index = node_embedding(tree_cp, ntips)
        vec = torch.tensor(tree2vec(tree_cp))
        del tree_cp
        node_features_tensor.append(node_features)
        edge_index_tensor.append(edge_index)
        vec_tensor.append(vec)
    node_features_tensor, edge_index_tensor, vec_tensor = torch.stack(node_features_tensor), torch.stack(edge_index_tensor), torch.stack(vec_tensor)
    torch.save(node_features_tensor, os.path.join(path, 'node_features.pt'))
    torch.save(edge_index_tensor, os.path.join(path, 'edge_index.pt'))
    torch.save(vec_tensor, os.path.join(path, 'vec.pt'))

    return

def process_empFreq(dataset):
    if 'DS' in dataset:
        ground_truth_path, samp_size = 'data/raw_data/', 750001
        tree_dict_total, tree_names_total, tree_wts_total = summary(dataset, ground_truth_path, samp_size=samp_size)
        emp_tree_freq = {tree_dict_total[tree_name]:tree_wts_total[i] for i, tree_name in enumerate(tree_names_total)}

    wts = list(emp_tree_freq.values())
    taxa = sorted(list(emp_tree_freq.keys())[0].get_leaf_names())
    ntips = len(taxa)
    path = os.path.join('embed_data',dataset,'emp_tree_freq')
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'wts.npy'), wts)
    np.save(os.path.join(path, 'taxa.npy'), taxa)

    node_features_tensor, edge_index_tensor, vec_tensor = [], [], []
    for tree in emp_tree_freq.keys():
        tree_cp = tree.copy()
        namenum(tree_cp, taxa)
        node_features, edge_index = node_embedding(tree_cp, ntips)
        vec = torch.tensor(tree2vec(tree_cp))
        del tree_cp
        node_features_tensor.append(node_features)
        edge_index_tensor.append(edge_index)
        vec_tensor.append(vec)
    node_features_tensor, edge_index_tensor, vec_tensor = torch.stack(node_features_tensor), torch.stack(edge_index_tensor), torch.stack(vec_tensor)
    torch.save(node_features_tensor, os.path.join(path, 'node_features.pt'))
    torch.save(edge_index_tensor, os.path.join(path, 'edge_index.pt'))
    torch.save(vec_tensor, os.path.join(path, 'vec.pt'))

    return

class EmbedData(Dataset):
    def __init__(self, dataset, repo='emp') -> None:
        super().__init__()
        if repo == 'emp':
            self.path = os.path.join('embed_data',dataset,'emp_tree_freq')
        else:
            self.path = os.path.join('embed_data',dataset,'repo{}'.format(repo))
        
        self.wts = np.load(os.path.join(self.path, 'wts.npy'))
        self.length = len(self.wts)

        self.node_features = torch.load(os.path.join(self.path, 'node_features.pt'))
        self.edge_index = torch.load(os.path.join(self.path, 'edge_index.pt'))
        self.vec = torch.load(os.path.join(self.path, 'vec.pt'))

    def __getitem__(self, index):
        return self.node_features[index], self.edge_index[index], self.vec[index]
    
    def __len__(self):
        return self.length
    