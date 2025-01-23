from ete3 import Tree

def vec2tree(vec):
    ntips = len(vec) + 3
    root = Tree()
    root.add_child(name=0)
    root.add_child(name=1)
    root.add_child(name=2)
    root.name = 2*ntips - 3
    idx2node = {node.name:node for node in root.children}
    for i,num in enumerate(vec):
        node = idx2node[num]
        parent = node.up
        
        node.detach()
        new_node = parent.add_child(name=i+ntips)
        new_node.add_child(node)
        new_tip_node = new_node.add_child(name=i+3)
        
        idx2node[i+ntips] = new_node
        idx2node[i+3] = new_tip_node

    return root


def tree2vec(tree):
    ntips = len(tree)
    edges = []
    nodes = []
    copy_tree = tree.copy('newick')
    idx2node = {int(node.name):node for node in copy_tree.traverse('postorder')}
    root = copy_tree
    
    for i in range(ntips-1, 2, -1):
        node = idx2node[i]
        parent = node.up
        if not parent.is_root():
            sister = node.get_sisters()[0]
            grandparent = parent.up
            parent.detach()
            sister.detach()
            grandparent.add_child(sister)
            edges.append((sister.name, grandparent.name))
        else:
            sister_1, sister_2 = node.get_sisters()
            sister_1.detach()
            sister_2.detach()
            if not sister_2.is_leaf():
                sister_2.add_child(sister_1)
                root = sister_2
            else:
                sister_1.add_child(sister_2)
                root=sister_1         
            
            edges.append((sister_1.name, sister_2.name))
        nodes.append(parent)
            
    edge2node = {}
    nodes = nodes[::-1]
    for child in root.children:
        edge2node[(child.name, root.name)] = child
        edge2node[(root.name, child.name)] = child
    
    vec = []
    for i, edge in enumerate(edges[::-1]):
        node = edge2node[edge]
        parent = node.up
        new_node = nodes[i]
        for child in new_node.children:
            edge2node[(new_node.name, child.name)] = child
            edge2node[(child.name, new_node.name)] = child
        
        node.detach()
        new_node.add_child(node)
        new_node.name_ = i+ntips
        parent.add_child(new_node)
        
        edge2node[(node.name, new_node.name)] = node
        edge2node[(new_node.name, node.name)] = node
        edge2node[(parent.name, new_node.name)] = new_node
        edge2node[(new_node.name, parent.name)] = new_node
        
        if node.is_leaf():
            vec.append(int(node.name))
        else:
            vec.append(node.name_)          
    
    return vec
