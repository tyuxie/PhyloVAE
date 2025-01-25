import numpy as np
from ete3 import Tree
import warnings
warnings.simplefilter('always', UserWarning)
import pdb


def init(tree, branch=None, name='all', scale=0.1, display=False, return_map=False):
    if return_map: idx2node = {}
    i, j = 0, len(tree)
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            if name != 'interior':
                node.name, i = i, i+1
            else:
                node.name = int(node.name)
        else:
            node.name, j = j, j+1
        if not node.is_root():
            if isinstance(branch, basestring) and branch =='random':
                node.dist = np.random.exponential(scale)
            elif branch is not None:
                node.dist = branch[node.name]
        else:
            node.dist = 0.0
            
        if return_map: idx2node[node.name] = node
        if display:
            print(node.name, node.dist)
        
    if return_map: return idx2node
    
    
def create(ntips, branch='random', scale=0.1):
    tree = Tree()
    tree.populate(ntips)
    tree.unroot()
    init(tree, branch=branch, scale=scale)
    
    return tree

def namenum(tree, taxon, nodetosplitMap=None):
    taxon2idx = {}
    j = len(taxon)
    if nodetosplitMap:
        idx2split = ['']*(2*j-3)
    for i, name in enumerate(taxon):
        taxon2idx[name] = i
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            if not isinstance(node.name, str):
                warnings.warn("The taxon names are not strings, please check if they are already integers!")
            else:
                node.name = taxon2idx[node.name]
                if nodetosplitMap:
                    idx2split[node.name] = nodetosplitMap[node]
        else:
            node.name, j = j, j+1
            if nodetosplitMap and not node.is_root():
                idx2split[node.name] = nodetosplitMap[node]
    
    if nodetosplitMap:
        return idx2split
