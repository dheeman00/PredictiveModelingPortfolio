import numpy as np
from typing import List, Set, Callable
from . import environment
from typing import Dict
from scipy.stats import chi2

# This is the function that is used for Gini Index where 
# the inputs are: dataset, attributes, cost function
# the output is: value of the gini index
def gain(data_set: np.ndarray, attribute: int, cost_function: Callable[[np.ndarray], float]) -> (float, Dict):
    children = {}
    cost = 0
    for v in environment._value_list:
        children[v] = data_set[np.where(data_set[:, attribute] == v)]
        cost += cost_function(children[v]) * (children[v].shape[0] / data_set.shape[0])
    return cost_function(data_set) - cost, children

# This is the function that is used to calculate Chi-Square
# the inputs: dataset, attributes, alpha value
# the output: the chi sqaure value after comparing with the table
def chi_square(data_set: np.ndarray, attribute: int, alpha: float):
    # sum_classes sum_values (actual vs expected)^2 / expected
    if alpha <= 0:
        return True
    partition = {}
    actual = {}
    expected = {}
    for v in environment._value_list:
        partition[v] = len(np.where(data_set[:, attribute] == v)[0])
        actual[v] = {}
        expected[v] = {}
        for c in environment._class_list:
            actual[v][c] = len(np.where(np.logical_and(data_set[:, attribute] == v, data_set[:, -1] == c))[0])
            expected[v][c] = partition[v] * len(np.where(data_set[:, -1] == c)[0]) / data_set.shape[0]
    value = 0
    for v in environment._value_list:
        for c in environment._class_list:
            if expected[v][c] != 0:
                value += (actual[v][c] - expected[v][c]) ** 2 / expected[v][c]
    return chi2.ppf(alpha, (len(environment._value_list) - 1) * (len(environment._class_list) - 1)) < value
