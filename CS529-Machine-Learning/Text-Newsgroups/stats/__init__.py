import numpy as np
from scipy.sparse import find, csr_matrix as csr
from scipy.sparse.data import _data_matrix
from typing import Union


class Delta:
    def __init__(self, data_set = None):
        """
        A one hot encoded matrix for classification.

        Parameters
        data_set = None
            The data_set for which the matrix is generated.
        """
        self.matrix = self.create_delta_matrix(data_set) if data_set is not None else None
    
    def create_delta_matrix(data_set):
        """
        Creates the one hot encoded matrix for classification. Each column of the matrix represents a single sample. This class assumes a 2D input matrix where the final column contains the numerical label values for the samples. The values are converted to integers before processing. Column 0 represents the minimum class value amongst the labels. The number of rows is equivalent to the difference of the minimum and maximum integer label value.

        Parameters
        data_set
            The data_set for which the matrix is generated.

        Returns
            A one hot encoded sparse matrix for data_set[:, -1]
        """
        labels = data_set[:,-1].toarray().astype(np.int)
        labels -= labels.min()
        classes = labels.max() + 1
        print(f"Creating delta from: {labels}")
        delta = np.zeros((int(classes), labels.shape[0]), dtype=np.float)
        delta[labels.reshape(1, -1), range(labels.shape[0])] = 1
        return csr(delta)

#This is the class for MLE
class MaximumLikelyhoodEstimator:
    def __init__(self, data_set = None):
        """
        Maximum Log Likelyhood Estimator.

        Parameters
        data_set
            The data_set for which the estimator is generated.
        """
        self.matrix = self.mle(data_set) if data_set is not None else None

    #generate the matrix from the outcome of the mle()
    def set_matrix(self, data_set, process = True):
        """
        Sets the estimator.

        Parameters
        data_set
            Data for the estimator.
        process
            If True the data is processed and a matrix is generated from the data otherwise the data_set is assumed to be an already processed estimator parameter set. 
        """
        if process:
            self.matrix = self.mle(data_set)
        else:
            self.matrix = data_set

    #this function is used to calculate the MLE
    def mle(self, data_set):
        """
        Computes the estimator parameters from the given data.

        Parameters
        data_set
            The data_set for which the parameters are estimated. The labels are assumed to be column -1 of a given 2D matrix.

        Returns
            The parameter matrix.
        """
        #the outcome is a log value instead of probability
        return np.log2(np.unique(data_set.getcol(-1).data, return_counts=True)[1] / data_set.shape[0]).reshape((-1, 1))


class MaximumAPosterioriEstimator:
    def __init__(self, data_set = None, beta = None):
        """
        Maximum Log A Posteriori Estimator.

        Parameters
        data_set
            The data_set for which the estimator is generated.
        """
        self.matrix = self.map(data_set, beta) if data_set is not None else None

    #generate the matrix from the outcome of the map()
    def set_matrix(self, data_set, beta = None, process = True):
        """
        Sets the estimator.

        Parameters
        data_set
            Data for the estimator.
        process
            If True the data is processed and a matrix is generated from the data otherwise the data_set is assumed to be an already processed estimator parameter set. 
        """
        if process:
            self.matrix = self.map(data_set, beta)
        else:
            self.matrix = data_set

    def map(self, data_set, beta = None):
        """
        Computes the estimator parameters from the given data.

        Parameters
        data_set
            The data_set for which the parameters are estimated. The function assumes the data is labeled with column 0 as the id column and column -1 as labels in a given 2D matrix.
        beta
            The beta value for computation. If set to None it defaults to 1/vocabulary where vocabulary is data_set.shape[1] - 2 i.e. Number of features.
        Returns
            The parameter matrix.
        """
        features = data_set[:, 1:-1]
        vocabulary = features.shape[1]
        if beta is None:
            beta = 1/vocabulary
        classes = data_set.getcol(-1)
        num_classes = 20
        matrix = np.zeros((num_classes, features.shape[1]))
        total_words = np.zeros(num_classes, dtype=np.int)
        for k in range(num_classes):
            print(f'Processing Class {k}')
            rows = find(classes[:,-1] == k + 1)[0]
            tmp = features[rows]
            for i in range(matrix.shape[1]):
                matrix[k,i] = tmp.getcol(i).sum()
            total_words[k] = tmp.sum()
            matrix[k] = (matrix[k] + beta) / (beta * vocabulary + total_words[k])
        #the outcome is a log value instead of probability
        return np.log2(matrix)

def l1_norm(data_set: Union[_data_matrix, np.ndarray], axis: int = 1):
    """
    Computes the L1 norm of a 2D matrix along the axis.

    Parameters
    data_set : Union[scipy.sparse._data_matrix, numpy.ndarray]
        Matrix to normalize.
    axis : int
        The direction along which to normalize the matrix. Axis 0 divides every value by the sum of the column. Axis 1 divides each value by the sum of the row.

    Returns
        Normalized matrix

    Raises
    TypeError
        If data_set is not a scipy.sparse._data_matrix or numpy.ndarray.
    """
    out = data_set.copy()
    if axis == 0:
        for i in range(data_set.shape[1]):
            s = data_set[:, i].sum()
            if s != 0:
                out[:, i] = data_set[:, i] / s
    else:
        for i in range(data_set.shape[0]):
            s = data_set[i, :].sum()
            if s != 0:
                out[i, :] = data_set[i, :] / s
    if isinstance(data_set, _data_matrix):
        return csr(out)
    elif isinstance(data_set, np.ndarray):
        return out
    else:
        raise TypeError(f"Unknown input type {type(data_set)}")

#function for the soft max
def soft_max(data_set, axis = 0):
    """
    Computes the soft_max function of the data using numpy.expm1.
    
    Paramters
    data_set
        The matrix to be process.

    axis
        The axis along which the data is normalized.
    """
    return l1_norm(np.expm1(data_set), axis = axis)
