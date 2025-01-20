from data import Loader
from model import NaiveBayesClassifier as NBC
from stats import MaximumLikelyhoodEstimator, MaximumAPosterioriEstimator
from sys import argv
from typing import Union, Iterable
from getopt import getopt, GetoptError
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse

#The main file is used just to load the raw data and convert into the desired for format

def get_params(args: Iterable[str] = argv[1:]):
    params = {'operation': None, 'train_file': None, 'test_file': None, 'model': None, 'data_file': None,
              'output_file': None, 'verbose': False}
    opts, args = getopt(args, 'o:d:t:e:f:m:v',
                        ['save-npz', 'model=', 'operation=', 'train=', 'test=', 'data=', 'output='])
    try:
        for o, a in opts:
            if o in ['-o', '--operation']:
                if a in ['train', 'save-npz']:
                    params['operation'] = a
            elif o in ['-d', '--data']:
                params['data_file'] = a
            elif o in ['-t', '--train']:
                params['train_file'] = a
            elif o in ['-e', '--test']:
                params['test_file'] = a
            elif o in ['-f', '--output']:
                params['output_file'] = a
            elif o in ['-m', '--model']:
                if a in ['regression', 'lr']:
                    params['model'] = 'lr'
                elif a in ['bayes', 'naive', 'nb']:
                    params['model'] = 'nb'
            elif o == '-v':
                params['verbose'] = True
    except GetoptError as e:
        print(e.msg)
        exit(1)

    return params


if __name__ == '__main__':
    p = get_params()
    if p['operation'] == 'save-npz':
        l = Loader('.', p['verbose'])
        if p['data_file'] is not None:
            d = l.load(p['data_file'])
            if isinstance(d, np.ndarray):
                d = l.get_sparse(d)
        else:
            raise ValueError('Data file path not provided. Use -d or --data parameter.')
        l.save(d, p['output_file'], True)

    ##This portion of the code is just used for the testing purpose
"""
    # loading the sparse dataset
    loader = Loader('data')
    training_data = loader.load('training.npz')

    testing_data = loader.load('testing.npz')
    # loading the class and vocabulary
    vocabulary = pd.read_csv('vocabulary.txt', header = None)
    class_type = pd.read_csv('newsgrouplabels.txt', header = None)


    #initial Dirichlet to estimate P(X|Y)
    length_vocabulary = len(vocabulary)
    beta = 1/length_vocabulary
    alpha = [1 + beta] * len(class_type)




    #testing
#    test = MaximumLikelyhoodEstimator()
#    test.mle(training_data)

#    test = MaximumAPosterioriEstimator()
#    test.map(training_data)

    #this is not working currently need to fix it up
    '''
    classifier = NBC(training_data[:-5])
    for i in range(-5, 0):
        print(f'Testing sample[-5]: Classifier returned {1 + classifier.classify(training_data[i,1:-1])} vs {training_data[i, -1]}')
    '''
"""
