import numpy as np
from scipy.sparse import csr_matrix as csr, hstack

from stats import MaximumAPosterioriEstimator as Map, MaximumLikelyhoodEstimator as Mle, Delta, soft_max, l1_norm

#this is the overall class for the classifier
class Classifier:
    """
    Classifier Base Class
    """
    def __init__(self):
        pass

    def classify(self, data):
        """
        Classifies the input samples.
        
        Parameters
        data
            Sample data.
        """
        pass

#this is the class for the Naive Bayes Classifier
class NaiveBayesClassifier(Classifier):
    def __init__(self, data_set = None, beta = None):
        """
        A Naive Bayes Classifier Implementation.

        Parameters
        data_set
            The training data for the classifier.
        beta
            The beta value for the MaximumAPosterioriEstimator
        """
        super().__init__()
        if data_set is not None:
            self.mlle = Mle(data_set)
            self.mlap = Map(data_set, beta)
        else:
            self.mlle = None
            self.mlap = None

    #set function for the MLE
    def set_mlle(self, mlle):
        """
        Set a precomupted MaximumLikelyhoodEstimator.

        Parameters
        mlle
            The precomputed ersimator.
        """
        self.mlle = mlle
    #set function for the MAP
    def set_mlap(self, mlap):
        """
        Set a precomputed MaximumAPosterioriEstimator.
        
        Parameters
        mlap
            The precomputed estimator.
        """
        self.mlap = mlap

    #classification algorithm for Y using augmax
    def classify(self, data):
        if self.mlle is None or self.mlap is None:
            raise RuntimeError('MLE or MAP not initialized')
        # 20 x 1 + 20 x 61188 * 61188 x 1
        result = self.mlap.matrix * data.T
        result = self.mlle.matrix + result
        return np.argmax(result)

#this is the class for the Linear Regression
class LinearRegressorClassifier(Classifier):
    def __init__(self, param_ada: float, param_lambda: float, k:int, n:int, train_data = None, eval_data = None, seed = None):
        """
        Linear Regressor Classifier that can be trained by Stochastic Gradient Descent.

        Parameters
        param_ada : float
            The training step size.

        param_lambda : float
            The training penalty.
        k : int
            Number of classes.
        n : int
            Number of features.
        train_data = None
            The data for training. It is assumed that the first column in data is id column.
        eval_data = None
            The data for computing training loss. It is assumed that the first column in data is id column.
        seed = None
            The numpy random number generator seed.
        """
        super().__init__()
        self.epoch = 0
        self.ada = param_ada
        self.lam = param_lambda
        self.k = k
        self.n = n
        self.min_class = None

        if seed is not None:
            np.random.seed(seed)

        #the weights are selected with a random number between 0 and 1
        self.weights = np.random.uniform(0, 1, (k, n + 1))
        print(f"Weight Shape: {self.weights.shape}")
        print(f"Init Weights: {self.weights}")

        self.train_features = None
        self.train_delta = None
        self.train_logits = None
        self.eval_features = None
        self.eval_delta = None
        self.eval_logits = None
        if train_data is not None:
            self.set_train_data(train_data)
        if eval_data is not None:
            self.set_eval_data(eval_data)

    #set function for determining the minimum value of the class labels
    def set_min_class(self, data_set):
        """
        Determine the minimum class value from data.

        Parameters
        data_set
            The data from which the min class is determined.
        """
        min_class = int(data_set[:, -1].min())
        if self.min_class is None:
            self.min_class = min_class
        elif self.min_class != min_class:
            raise ValueError('Class label values not in expected range.')

    #function that takes a portion of the training dataset for validation purpose
    def set_eval_data(self, data_set, has_id = True):
        """
        Processed the evaluation data set.

        Parameters
        data_set
            The sample data set.

        has_id
            When True the first column of the data set is ignored.
        """
        start_column = 1 if has_id else 0
        # extracting the features that is the dataset
        print(f"Normalizing Eval Features")
        self.eval_features = data_set[:, start_column:-1]

        if self.eval_features.shape[1] != self.n:
            raise ValueError(f"Features shape {self.eval_features.shape}. Expected (None, {self.n})")

        self.eval_features = csr(hstack((np.ones((self.eval_features.shape[0], 1)), self.eval_features)))
        self.eval_delta = Delta.create_delta_matrix(data_set)

        if self.eval_delta.shape[0] != self.k or self.eval_delta.shape[1] != data_set.shape[0]:
            raise ValueError(f"Delta shape {self.eval_delta.shape}. Expected ({self.k},{data_set.shape[0]})")

        self.eval_logits = np.zeros((self.k, self.eval_features.shape[0]))
        self.set_min_class(data_set)

    #the function will take a portion of the training dataset for training the classifier
    def set_train_data(self, data_set, has_id = True):
        start_column = 1 if has_id else 0
        #extracting the features form the dataset
        print("Normalizing Train Features")
        self.train_features = data_set[:, start_column:-1]
        #self.train_features = l1_norm(data_set[:, start_column:-1])

        if self.train_features.shape[1] != self.n:
            raise ValueError(f"Features shape {self.train_features.shape}. Expected (None, {self.n})")

        self.train_features = csr(hstack((np.ones((self.train_features.shape[0], 1)), self.train_features)))
        print(f"Train Features: {repr(self.train_features)}")
        self.train_delta = Delta.create_delta_matrix(data_set)

        if self.train_delta.shape[0] != self.k or self.train_delta.shape[1] != data_set.shape[0]:
            raise ValueError(f"Delta shape {self.train_delta.shape}. Expected ({self.k},{data_set.shape[0]})")

        self.train_logits = np.zeros((self.k, self.train_features.shape[0]))
        self.set_min_class(data_set)

    #function is performing the equation: P(Y | X,W) ~ exp(W X^T)
    def logits(self, features, output = None):
        """
        Feed forward operation. Computes the prediction values based on current weights. It is activated by a soft_max activation function.
        
        Parameters
        features
            The sample feature set.
        output = None
            NOTE: NOT YET IMPLEMENTED.
        """
        return soft_max(l1_norm(self.weights * features.T, axis = 0))

    #the function is used for the training of the classifier
    def train(self, train_data = None):
        """
        Single training epoch. Updates the weights based on current or given training samples.

        Parameters
        train_data
            If provided the existing training data is replaced by new data for subsequent calls.
        """
        self.epoch += 1
        if train_data is not None:
            self.set_train_data(train_data)

        #performing the logit() function over here
        self.train_logits = self.logits(self.train_features, self.train_logits)

        #the outcome of the logit is used for updating the weights
        self.weights += self.ada * ((self.train_delta - self.train_logits) * self.train_features - self.lam * self.weights)

    #the function is used for validating the classifer
    def eval(self, eval_data = None):
        """
        Computes the Mean Square Error of prection using current weights.

        Parameters
        eval_data
            If provided the exisiting evalution data is replaced by new data for subsequence calls.
        """
        if eval_data is not None:
            self.set_eval_data(eval_data)

        self.eval_logits = self.logits(self.eval_features, self.eval_logits)
        diff = np.array(self.eval_logits - self.eval_delta)
        diff = np.square(diff)
        diff = diff.sum(axis=0)
        avg = np.average(diff)
        return avg

    def classify(self, data, had_id = True):
        """
        Classifies given samples.

        Parameters
        data
            The sample features.
        has_id
            If True the first column of data is ignored.
        """
        if had_id:
            features = csr(hstack((np.ones((data.shape[0], 1)), data[:, 1:])))
        else:
            features = csr(hstack((np.ones((data.shape[0], 1)), data)))
        logits = self.logits(features)
        return np.argmax(logits, axis=0) + self.min_class

