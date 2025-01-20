from sys import argv
from getopt import getopt, GetoptError
from typing import Iterable
from data import Loader
from stats import MaximumLikelyhoodEstimator, MaximumAPosterioriEstimator
from model import NaiveBayesClassifier as NBC
from pathlib import Path
import numpy as np

#this script make use of some input paramaters on the terminal and create the predict outcome

#considering the number of arguments from the array position 1 till the end
args = argv[1:]
num_args = len(argv[1:])

if num_args:
    operation = args[0]
else:
    operation = ''

if operation == 'train' and num_args >= 3:
    #argument for considering the dataset i.e. training or testing npz files
    file_path = Path(args[1])
    if not file_path.is_file():
        print(f'Error accessing {file_path}.')
        exit(1)
    try:
        # input the beta value
        beta = float(args[2])
    except ValueError:
        print(f"Argument 3 must be a float value. Given {args[2]}.")
        exit(1)

        #the argument will used for naming the network. Better to use the name: training
    if num_args > 3:
        network_name = args[3]
    else:
        network_name = 'naive_bayes'

        #the argument will create the path to save the outcome
    if num_args > 4:
        dir_path = Path(args[4])
        if not dir_path.exists():
            print(f"Creating directory {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        if dir_path.is_file():
            print(f"{dir_path} is not a directory.")
            exit(1)
    else:
        dir_path = Path('.')

elif operation == 'predict' and num_args == 4:
    file_path = Path(args[1])
    if not file_path.is_file():
        print(f'Error accessing {file_path}.')
        exit(1)
    network_name = args[2]
    dir_path = Path(args[3])
    if not dir_path.exists():
        print(f"{dir_path} does not exist.")
        exit(1)
    else:
        mle_path = dir_path.joinpath(Path(f"{network_name}_mle.npz"))
        if not mle_path.exists():
            print(f"Could not locate file {mle_path}.")
            exit(1)
        map_path = dir_path.joinpath(Path(f"{network_name}_map.npz"))
        if not map_path.exists():
            print(f"Could not locate file {map_path}.")
            exit(1)
elif operation == 'rank' and num_args >= 2:
    network_name = args[1]
    if num_args > 2:
        dir_path = Path(args[2])
    else:
        dir_path = Path('.')
    if num_args > 3:
        vocab = Path(args[3])
    else:
        vocab = Path('vocabulary.txt')

    if not vocab.is_file():
        print(f"Warning: Could not located {vocab}")
        vocab = None
    if not dir_path.exists():
        print(f"{dir_path} does not exist.")
        exit(1)
    else:
        mle_path = dir_path.joinpath(Path(f"{network_name}_mle.npz"))
        if not mle_path.exists():
            print(f"Could not locate file {mle_path}.")
            exit(1)
        map_path = dir_path.joinpath(Path(f"{network_name}_map.npz"))
        if not map_path.exists():
            print(f"Could not locate file {map_path}.")
            exit(1)

else:
    tstring = f"\tpython3 {argv[0]} train data_file beta [network_name] [save_dir]\n"
    pstring = f"\tpython3 {argv[0]} predict data_file network_name network_dir\n"
    rstring = f"\tpython3 {argv[0]} rank network_name network_dir [vocabulary_file]\n"
    print(f"Usage:\n{tstring if operation not in ['predict', 'rank'] else ''}{pstring if operation not in ['train', 'rank'] else ''}{rstring if operation not in ['train', 'predict'] else ''}")
    exit(1)

#training
if operation == 'train':
    loader = Loader('.', True)
    saver = Loader(dir_path, True)
    if 0 <= beta <= 1:
        training_data = loader.load(file_path)
        # Create and save MLE
        print('Computing MLE')
        train_mle = MaximumLikelyhoodEstimator(training_data)
        print(train_mle.matrix.shape)
        if isinstance(train_mle.matrix, np.ndarray):
            train_mle.set_matrix(loader.get_sparse(train_mle.matrix), process=False)
        saver.save(train_mle.matrix, f"{network_name}_{beta}_mle.npz", True)
        print('Computing MAP')
        train_map = MaximumAPosterioriEstimator(training_data, beta)
        if isinstance(train_map.matrix, np.ndarray):
            train_map.set_matrix(loader.get_sparse(train_map.matrix), process=False)
        saver.save(train_map.matrix, f"{network_name}_{beta}_map.npz", True)
    else:
        ValueError('Beta Value is between is 0.00001 and 1.0')

#testing
if operation == 'predict':
    data_loader = Loader('.')
    network_loader = Loader(dir_path)
    test_data = data_loader.load(file_path)
    print("Loading MLE.")
    mle_matrix = network_loader.load(f"{network_name}_mle.npz")
    print(f"Loaded {mle_matrix.shape} matrix.")
    print("Loading MAP.")
    map_matrix = network_loader.load(f"{network_name}_map.npz")
    print(f"Loaded {map_matrix.shape} matrix.")
    classifier = NBC()
    c_mle = MaximumLikelyhoodEstimator()
    c_mle.set_matrix(mle_matrix, process=False)
    c_map = MaximumAPosterioriEstimator()
    c_map.set_matrix(map_matrix, process=False)
    classifier.set_mlle(c_mle)
    classifier.set_mlap(c_map)
    results = np.zeros((test_data.shape[0], 2))
    print("Classifying.")
    for i in range(test_data.shape[0]):
        if i % 100 == 99 and i:
            print(f"Processing sample {i+1}")
        results[i] = [test_data[i,0], classifier.classify(test_data[i,1:]) + 1]
    print(f"Saving Result.")
    data_loader.save(results, f"{file_path.stem}_{str(network_name).replace('.', '_')}_output", True)

if operation == 'rank':
    network_loader = Loader(dir_path)
    print("Loading MLE.")
    mle_matrix = network_loader.load(f"{network_name}_mle.npz")
    print(f"Loaded {mle_matrix.shape} matrix.")
    print("Loading MAP.")
    map_matrix = network_loader.load(f"{network_name}_map.npz")
    print(f"Loaded {map_matrix.shape} matrix.")
    print(type(mle_matrix))
    print(type(map_matrix))
    rank_keys = np.asarray(((map_matrix.T * mle_matrix).T).sum(axis = 0) / map_matrix.shape[0])
    rank_keys = 2 ** rank_keys
    #print(rank_keys.tolist())
    print(rank_keys.shape)
    if vocab is not None:
        with open(vocab, 'r') as vfile:
            words = list(filter(lambda x: len(x) > 0, [s.strip().rstrip('\n') for s in vfile.readlines()]))
            if len(words) > map_matrix.shape[1]:
                print(f'Warning: {len(words)} words found in vocabulary file. Using the first {map_matrix.shape[1]}')
            elif len(words) < map_matrix.shape[1]:
                print('Number of word in vocabulary file is less than the features of the map.')
                exit(1)
            ranks = list(zip(words[:map_matrix.shape[1]], rank_keys.tolist()[0]))
    else:
        ranks = list(zip(range(map_matrix.shape[1]), rank_keys.tolist()[0]))
    
    ranks.sort(key=lambda x : -x[1])
    print(f"Top 100 words and importance:\n{ranks[:100]}")
    print(f"Top 100 words:\n{','.join([str(x[0]) for x in ranks[:100]])}")
    
