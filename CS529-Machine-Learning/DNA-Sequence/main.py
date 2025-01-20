from stats import environment
from stats import gain, chi_square
from stats.purity import mce, entropy, gini
from data import read_file, convert_to_labels
from tree import Node, LeafNode
import numpy as np
from typing import List, Union, Dict, Callable, Set
from time import localtime, strftime
from sys import argv
from getopt import getopt, GetoptError
from pathlib import Path

calls = 0
stop = 59

def _process_params(args: list = argv[1:]):
    chi_square=[1.0]
    cost_function=[None]
    train_only=False
    sample_split=-1
    strats=[0]
    try:
        _opts, _args = getopt(args, "c:f:t:s:")
        #print(_opts)
        #print(_args)
        for o,a in _opts:
            if o == '-c':
                chi_square=[float(x) for x in a.split(",")]
            elif o == '-f':
                cost_function=[]
                for x in a.split(","):
                    if x == 'mce':
                        cost_function.append(mce)
                    elif x == 'entropy':
                        cost_function.append(entropy)
                    elif x == 'gini':
                        cost_function.append(gini)
                    else:
                        raise ValueError('Unknown cost function.')
            elif o == '-t':
                train_only = True
                sample_split = float(a)
            elif o == '-s':
                strats = []
                for x in [int(y) for y in a.split(",")]:
                    if x < 3:
                        strats.append(x)
                    else:
                        raise ValueError('Unknown strategy')
    except GetoptError as err:
        print(err)
        exit(1)
    return chi_square, cost_function, strats, train_only, sample_split


def get_label(data_set: np.ndarray):
    return np.bincount(data_set[:, -1]).argmax()


def build_tree(data_set: np.ndarray, attributes: Set[int], error_tolerance: float = 0.0,
               cost_function: Callable[[np.ndarray], float] = None, chi2_confidence: float = 1.0,
               value_list: Union[Dict[str, int], List[str]] = None,
               class_list: Union[Dict[str, int], List[str]] = None):
    global calls
    global stop
    alpha = 1 - chi2_confidence
    calls += 1
    if calls == stop:
        print()
    try:
        if cost_function is None:
            cost_function = mce
        if cost_function(data_set) <= error_tolerance:
            return LeafNode(get_label(data_set))
        if len(attributes):
            max_gain = 0
            max_gain_arg = -1
            subsets = None
            for attr in attributes:
                g, s = gain(data_set, attr, cost_function)
                if g >= max_gain:
                    max_gain = g
                    max_gain_arg = attr
                    subsets = s
            if max_gain_arg != -1 and chi_square(data_set, max_gain_arg, alpha):  # PUT CHI2 CHECK HERE
                node = Node(max_gain_arg)
                rem_attributes = attributes.difference({max_gain_arg})
                for k, v in subsets.items():
                    if v.shape[0] == 0:
                        node.add_child(k, LeafNode(get_label(data_set)))
                        continue
                    node.add_child(k,
                                   build_tree(subsets[k], rem_attributes, error_tolerance, cost_function,
                                              chi2_confidence,
                                              value_list, class_list))
                return node
        return LeafNode(get_label(data_set))
    except AttributeError as e:
        '''
        print(e)
        print(data_set)
        print(max_gain_arg)
        print(subsets)
        print(rem_attributes)
        exit(0)
        '''

def classify(node: Node, sample: np.ndarray):
    if node.attr == -1:
        return node.label
    return classify(node.children[sample[node.attr]], sample)

def write_output(node: Node, data_set: np.ndarray, file_name: str = None):
    output = np.zeros((data_set.shape[0], 2), dtype=np.int)
    output[:, 0] = data_set[:, 0]
    data_set = data_set[:, 1:]
    if isinstance(node, list):
        for i in range(data_set.shape[0]):
            votes = np.bincount(np.asarray([classify(n, data_set[i]) for n in node]))
            output[i, 1] = np.where(votes == votes.max())[0].min()
    else:
        for i in range(data_set.shape[0]):
            output[i,1] = classify(node, data_set[i])
    if file_name is None:
        p = Path(strftime("Attempt_%h%d_%H%M%S.csv", localtime()))
    else:
        p = Path(file_name)
    p_ = p
    i = 0
    while(p_.exists() and i <= 1000):
        i+= 1
        p_ = p.with_name(p.stem + '_' + str(i) + p.suffix)
    with open(p_, "w") as outfile:
        outfile.write("id,class\n")
        for i, l in zip(output[:,0].tolist(), convert_to_labels(output[:, 1].tolist())):
            outfile.write(f"{i},{l}\n")

def max_depth(node: Node):
    if node.attr == -1:
        return 0
    else:
        return 1 + max([max_depth(n) for n in node.children.values()])

if __name__ == '__main__':
    _c, _f, _cs, _t, _s = _process_params()
    '''
    print(_c)
    print(_f)
    print(_cs)
    print(_t)
    print(_s)
    '''
    environment.set_value_list(list(range(8)))
    environment.set_class_list(list(range(3)))
    data_sets = [read_file('training.csv', conversion_strat = s) for s in _cs]
    train_size = data_sets[0].shape[0]
    if _t:
        train_size = int(train_size * _s)
    root = [build_tree(d[:train_size, :], set(list(range(60))), cost_function=f, error_tolerance=0.05, chi2_confidence=c) for d in data_sets for c in _c for f in _f]
    if _t:
        for r in root:
            classification = np.asarray([classify(r, data_set[0]) for i in range(train_size, data_set.shape[0])])
            table = np.zeros(classification.shape[0], dtype=np.int)
            table[data_set[train_size:, -1] == classification] = 1
            print(f'Success Rate: {table.sum() / table.shape[0]}')
    else:
        test_set = read_file('testing.csv', labeled=False, remove_id=False)
        fs = list(map(lambda x: 'mce' if x == mce else 'entropy' if x == entropy else 'gini' if x == gini else 'UNKOWN' , _f)) 
        css = list(map(lambda x: 'none' if x == 0 else 'uniform' if x == 1 else 'max' if x == 2 else 'UNKNOWN', _cs))
        _i = 0
        for cs in css:
            for c in _c:
                for f in fs:
                    print(f"Maximum Tree Depth: Attempt_{f}_{c}_{cs}.csv is {max_depth(root[_i])}")
                    write_output(root[_i], test_set, f"Attempt_{f}_{c}_{cs}.csv")
                    print("Use this file in Kaggle to get the classification rate")
                    _i += 1
        write_output(root, test_set, f"Attempt_All{len(_c) * len(css) * len(fs)}.csv")
