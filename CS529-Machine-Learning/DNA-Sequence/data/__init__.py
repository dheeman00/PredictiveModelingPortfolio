import numpy as np
from typing import List, Dict

#this function is used for reading the input file 
def read_file(path, samples_dict: Dict[str, int] = None, labels_dict: Dict[str, int] = None, ignore_unknown: bool = False, labeled: bool = True, remove_id: bool = True, conversion_strat: int = 0):
    with open(path, 'r') as file:
        data = file.read().splitlines()
        data = [x.split(",") for x in data]
        tmp = []
        [tmp.extend(x) for x in data]
        data = tmp
        if not remove_id:
            if labeled:
                ids = np.asarray(data[0::3]).reshape((1,-1))
            else:
                ids = np.asarray(data[0::2]).reshape((1,-1))
            print(ids)
            print(ids.shape, flush=True)
        if labeled:
            samples = _convert_samples(data[1::3], samples_dict, ignore_unknown)
        else:
            samples = _convert_samples(data[1::2], samples_dict, ignore_unknown)
            print(samples)
            print(samples.shape, flush=True)
        if labeled:
            labels = _convert_labels(data[2::3], labels_dict, ignore_unknown)
        #over there we have considered the case of other characters indicate ambiguity among the standard characters
        if conversion_strat:
            if samples_dict is None:
                dictionary = {4: [0, 1, 2, 3], 5: [0, 2], 6: [1, 2], 7: [0, 2, 3]}
            else:
                print('Cannot do conversion with custom dict.')
                exit(1)
            if conversion_strat == 1: # Replace any ambigugous value with a valid value (for the symbol) uniformly at random
                np.vectorize(lambda x: np.random.choice(dictionary[x]) if x in dictionary.keys() else x)(samples)
            if conversion_strat == 2: # Replace any ambigugous value with the maximum occurring value in the column
                dist_table = np.zeros((4, samples.shape[1]), dtype=np.int)
                for j in range(samples.shape[1]):
                    counts = np.bincount(samples[:, j])[:4]
                    for i in range(4):
                        ci = counts[dictionary[i + 4]]
                        dist_table[i,j] = dictionary[i + 4][np.where(ci == ci.max())[0][0]]
                for i in range(samples.shape[0]):
                    for j in range(samples.shape[1]):
                        if samples[i, j] > 3:
                            samples[i, j] = dist_table[samples[i, j] - 4, j]
        if labeled:
            samples = np.insert(samples, samples.shape[1], labels, 1)
        if not remove_id:
            samples = np.insert(samples, 0, ids, 1)
        return samples

#converted the common features into numberic form
def _convert_samples(input_list: List[str], dictionary: Dict[str, int] = None, ignore_unknown: bool = False):
    if dictionary is None:
        dictionary = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'R': 5, 'S': 6, 'D': 7}
    if ignore_unknown:
        return np.asarray(
            list(map(lambda x: [dictionary[y] if y in dictionary.keys() else -1 for y in list(x)], input_list)))
    else:
        return np.asarray(list(map(lambda x: [dictionary[y] for y in list(x)], input_list)))

#converting the labels into the numeric form 
def _convert_labels(input_list: List[str], dictionary: Dict[str, int] = None, ignore_unknown: bool = False):
    if dictionary is None:
        dictionary = {'EI': 0, 'IE': 1, 'N': 2}
    if ignore_unknown:
        return np.asarray(list([dictionary[y] if y in dictionary.keys() else -1 for y in input_list]))
    else:
        return np.asarray(list([dictionary[y] for y in input_list]))
#convert the numeric labels back into character form
def convert_to_labels(data_set: List[int], dictionary: Dict[int, str] = None, ignore_unknown: bool = False):
    if dictionary is None:
        dictionary = {0: 'EI', 1: 'IE', 2: 'N'}
    if ignore_unknown:
        return [dictionary[x] if x in dictionary.keys() else '' for x in data_set]
    else:
        return [dictionary[x] for x in data_set]

