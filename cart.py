import numpy as np

class Node:
    def __init__(self, idx, val, groups,score):
        self.index_attr = idx
        self.val_attr = val 
        self.groups = groups
        self.score = score
        self.left_node = None
        self.right_node = None


def get_impurity(groups, unique_labels):
    
    nrows = 0

    for g in groups:
        nrows += len(g)
    
    weighted_gini = 0.
    for g in groups:
        grows = float(len(g))

        if grows > 0:
            score = 0.
            for l in unique_labels:
                p = (g[:,-1]==l).sum() / grows
                score+= p**2
            weighted_gini += (grows/nrows)*(1.-score)
    
    return weighted_gini

def split(attr_idx, attr_val, data):
    left_idxs = data[:,attr_idx] <=attr_val
    
    return data[left_idxs], data[~left_idxs]


def best_split(data):
    unique_labels = np.unique(data[:,-1])
    best_index = 999
    best_val = 999
    best_score = 999
    best_groups = None

    n_attribute = data.shape[1]-1

    for idx in range(n_attribute):
        unique_attr_vals = np.unique(data[:,idx])
        for attr_val in unique_attr_vals:
            gps = split(idx,attr_val ,data)
            score = get_impurity(gps, unique_labels)
            
            if score < best_score:
                best_index = idx
                best_val = attr_val
                best_score = score
                best_groups = gps

        return Node(best_index,best_val,best_groups,best_score)

def terminal_node(group):
    class_labels, count = np.unique(group[:,-1], return_counts= True)
    return class_labels[np.argmax(count)]

def split_branch(node, depth, settings):
    left_node , right_node = node.groups
    node.groups = None
    if not isinstance(left_node,np.ndarray) or not isinstance(right_node,np.ndarray):
        node.left_node = terminal_node(left_node + right_node)
        node.right_node = terminal_node(left_node + right_node)
        return

    if depth >= settings['max_depth']:
        node.left_node = terminal_node(left_node)
        node.right_node = terminal_node(right_node)
        return

    if len(left_node) <= settings['min_splits']:
        node.left_node  = terminal_node(left_node)
    else:
        node.left_node = best_split(left_node)
        split_branch(node.left_node,depth + 1,settings)


    if len(right_node) <= settings['min_splits']:
        node.right_node = terminal_node(right_node)
    else:
        node.right_node = best_split(right_node)
        split_branch(node.right_node,depth + 1,settings)

class DecisionTree:
    def __init__(self,settings):
        self.settings = settings
    
    def fit(self, _feature, _label):
        self.feature = _feature
        self.label = _label
        self.train_data = np.column_stack((self.feature,self.label))
        self.build_tree()

    def build_tree(self):
        """
        build tree recursively with help of split_branch function
        - Create a root node
        - call recursive split_branch to build the complete tree
        :return:
        """
        self.root = best_split(self.train_data)
        split_branch(self.root, 1,self.settings)
        return self.root

    def _predict(self, node, row):
        if row[node.index_attr] < node.val_attr:
            if isinstance(node.left_node, Node):
                return self._predict(node.left_node, row)
            else:
                return node.left_node

        else:
            if isinstance(node.right_node,Node):
                return self._predict(node.right_node,row)
            else:
                return node.right_node

    def predict(self, test_data):
        """
        predict the set of data point
        :param test_data:
        :return:
        """
        self.predicted_label = np.array([])
        for idx in test_data:
            self.predicted_label = np.append(self.predicted_label, self._predict(self.root,idx))

        return self.predicted_label