import numpy as np

#This is function MSE
def mce(data_set: np.ndarray) -> float:
    if data_set.shape[0] < 2:
        return 0
    label_values = data_set[:, -1]
    _, counts = np.unique(label_values, return_counts=True)
    distribution = counts / label_values.shape[0]
    return 1 - np.max(distribution)

#This function is used to calculate the entropy
def entropy(data_set: np.ndarray) -> float:
    if data_set.shape[0] < 2:
        return 1
    label_values = data_set[:, -1]
    _, counts = np.unique(label_values, return_counts=True)
    distribution = counts / label_values.shape[0]
    return (-distribution * np.log2(distribution)).sum()

#This function is used to calculate the gini
def gini(data_set: np.ndarray) -> float:
    if data_set.shape[0] < 2:
        return 1
    label_values = data_set[:, -1]
    _, counts = np.unique(label_values, return_counts=True)
    distribution = counts / label_values.shape[0]
    return 1 - (distribution ** 2).sum()
