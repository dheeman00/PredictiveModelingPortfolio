from sys import argv
from getopt import getopt, GetoptError
from typing import Iterable
from data import Loader
from model import LinearRegressorClassifier as LRC
from pathlib import Path
from scipy.sparse import csr_matrix as csr
import numpy as np

#considering the number of arguments from the array position 1 till the end
args = argv[1:]
num_args = len(argv[1:])

#if no arguments are passed through
if not num_args:
    tstring = f"\tpython3 {argv[0]} train data_file ada lambda epochs max_tries split [network_name] [save_dir]\n"
    pstring = f"\tpython3 {argv[0]} predict data_file network_name network_dir\n"
    print(f"Usage:\n{tstring}{pstring}")
    exit(1)

#taking the 1st argument from the argv
operation = args[0]
if operation not in ['train', 'predict']:
    print(f'Unknown operation {operation}')
    exit(1)

print(operation)
print(num_args)
#considering the training argument and looking into the different arguments
if operation == 'train' and num_args > 6:
    if '-r' in args:
        args.remove('-r')
        restore = True
        num_args -= 1
    else:
        restore = False

    #argument for considering the dataset i.e. training or testing npz files
    file_path = Path(args[1])
    if not file_path.is_file():
        print(f'Error accessing {file_path}.')
        exit(1)
    try:
        #argument for considering eta
        ada = float(args[2])
    except ValueError:
        print(f"Ada argument must be a float value. Given {args[2]}.")
        exit(1)

    try:
        #argument for considering lambda
        lam = float(args[3])

    except ValueError:
        print(f"Lambda argument must be a float value. Given {args[3]}.")
        exit(1)

    try:
        #argument epoch for the number of iteration
        epochs = int(args[4])
    except ValueError:
        print(f"Epochs argument must be an int value. Given {args[4]}.")
        exit(1)

    try:
        #argument max_tries for the window in which better weights should be found since the last best weights or the program exits disregarding epoch.
        max_tries = int(args[5])
    except ValueError:
        print(f"MaxTries argument must be an int value. Given {args[5]}.")
        exit(1)

    try:
        #the argument split will consider the amount for training and validation dataset
        split = float(args[6])
    except ValueError:
        print(f"Split argument must be an float value. Given {args[6]}.")
        exit(1)

        #the argument will used for naming the network. Better to use the name: training
    if num_args > 7:
        network_name = args[7]
    else:
        network_name = 'network'
        #the argument will create the path to save the outcome
    if num_args > 8:
        dir_path = Path(args[8])
        if not dir_path.exists():
            print(f"Creating directory {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        if dir_path.is_file():
            print(f"{dir_path} is not a directory.")
            exit(1)
    else:
        dir_path = Path('.')

elif num_args == 4:
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
    tstring = f"\tpython3 {argv[0]} train data_file ada lambda epochs max_tries split [network_name] [save_dir]\n"
    pstring = f"\tpython3 {argv[0]} predict data_file network_name network_dir\n"
    print(f"Usage:\n{tstring if argv[1] != 'predict' else ''}{pstring if argv[1] != 'train' else ''}")
    exit(1)

#training: with the arguements that are all passed through
if operation == 'train':
    loader = Loader('.', True)
    saver = Loader(dir_path, True)
    
    training_data = loader.load(file_path)
    if split != 0:
        s = 1 / training_data.shape[0]
        if s < split < 1 - s:
            split = int(training_data.shape[0] * split)
            eval_data = training_data[:split,:]
            training_data = training_data[split:, :]
        else:
            print('Bad split for training data. Split does not produce well formed sets.')

    with open(dir_path.joinpath(Path(f"{network_name}_{ada}_{lam}.log")), 'w+') as logger:
        logger.write(f'Netowrk Name: {network_name}\nAda: {ada}\nLambda: {lam}\nTrain File: {file_path} with {training_data.shape[0] + eval_data.shape[0]} samples\nTraining Size: {training_data.shape[0]}\nEvaluation Size: {eval_data.shape[0]}\nMax Epochs: {epochs}\nMax Tries: {max_tries}\n')
        k = int(max(training_data[:, -1].max() - training_data[: , -1].min() + 1, eval_data[:, -1].max() - eval_data[:, -1].min() + 1))
        n = training_data.shape[1] - 2
        classifier = LRC(ada, lam, k, n, training_data, eval_data)
        num = 0
        if restore:
            files = list(dir_path.glob(f'{network_name}*.npz'))
            wfile = None
            for f in files:
                if f.stem.startswith(network_name):
                    nnum = f.stem[:f.stem.rindex('_')]
                    try:
                        nnum = int(nnum[nnum.rindex('_') + 1:])
                    except:
                        nnum = -1
                    if nnum > num:
                        num = nnum
                        wfile = f
            print(f"Loading from {f}")
            classifier.weights = saver.load(Path(f)).toarray()
            classifier.epoch = num

        print(f"Training Features:\n{classifier.train_features.toarray()}")
        print(f"Training Labels:\n{classifier.train_delta.toarray()}")
        print(f"Evaluation Features:\n{classifier.eval_features.toarray()}")
        print(f"Evaluation Labels:\n{classifier.eval_delta.toarray()}")
        errors = [classifier.eval()]
        best_weights = np.copy(classifier.weights)
        best = errors[-1]
        best_epoch = num
        tries = 0
        for i in range(epochs):
            logger.write(f"Epoch {i} loss: {errors[-1]}\n")
            if tries > max_tries:
                break
            classifier.train()
            errors.append(classifier.eval())
            if errors[-1] <= best:
                best = errors[-1]
                tries = 0
                best_weights = np.copy(classifier.weights)
                best_epoch = classifier.epoch
            else:
                tries += 1
        logger.write(f"loss = {errors}")
        if len(errors) > 1:
            logger.write(f"Epoch {len(errors) - 1} loss: {errors[-1]}\n")
    print(classifier.weights)
    saver.save(csr(classifier.weights), f"{network_name}_{ada}_{lam}_{classifier.epoch}_weights.npz", True)
    classifier.weights = best_weights
    saver.save(csr(classifier.weights), f"{network_name}_{ada}_{lam}_{best_epoch}_best_weights.npz", True)
    a1 = eval_data[:, [0, -1]]
    print(a1.shape)
    cls = classifier.classify(eval_data[:, :-1]).reshape((-1, 1))
    print(cls.shape)
    results = np.concatenate((a1.toarray(), cls), axis = 1)
    saver.save(results, f"{network_name}_{file_path.stem}_{ada}_{lam}_finaleval.csv", True)

#testing: with the arguements that are all passed through
if operation == 'predict':
    data_loader = Loader('.')
    network_loader = Loader(dir_path)
    test_data = data_loader.load(file_path)
    print("Loading weights.")
    weights = network_loader.load(f"{network_name}_weights.npz").toarray()
    print(f"Loaded {weights.shape} matrix.")
    classifier = LRC(None, None, 0, 0)
    classifier.weights = weights
    classifier.min_class = 1
    print("Classifying.")
    classification = classifier.classify(test_data).reshape((-1, 1))
    results = np.concatenate((test_data[:, 0].toarray(), classification), axis = 1)
    data_loader.save(results, f"{file_path.stem}_{str(network_name).replace('.', '_')}_output", True)

