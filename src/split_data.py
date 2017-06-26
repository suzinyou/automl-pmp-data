"""
Split train data from AutoML Challenge and pickle.dump as a 4-tuple

    train_X, train_y, test_X, test_y
    
"""

import argparse
import os

import pandas as pd
import numpy as np
from path import *


parser = argparse.ArgumentParser()

parser.add_argument("name", help="Name of dataset", type=str)

args = parser.parse_args()

if __name__ == "__main__":

    dataset = args.name

    data_file = DATASETS_FROM + dataset + "/" + dataset + "_train.data"
    solution_file = DATASETS_FROM + dataset + "/" + dataset + "_train.solution"

    print("Reading files from %s..." % (DATASETS_FROM + dataset))

    df = pd.read_csv(data_file, sep=" ", header=None)
    target = pd.read_csv(solution_file, sep=" ", header=None)

    if len(df) != len(target):
        raise ValueError("Number of samples in %s(%d) and %s(%d) don't match" % (data_file, len(df), solution_file, len(target)))

    n_samples = len(df)
    print("Number of samples: %d." % n_samples)

    shuffle_idx = np.random.permutation(n_samples)
    df = df.iloc(shuffle_idx)
    target = target.iloc(shuffle_idx)

    split_idx = int(0.7 * n_samples)
    train_X = df[:split_idx]
    train_y = target[:split_idx]
    test_X = df[split_idx:]
    test_y = target[split_idx:]

    if not os.path.exists("%s" % DATASETS_TO):
        os.makedirs("%s" % DATASETS_TO)

    prefix = "%s/%s" % (DATASETS_TO, dataset)

    train_X.to_csv("%s_train.data" % prefix, sep=" ", header=False, index=False)
    train_y.to_csv("%s_train.solution" % prefix, sep=" ", header=False, index=False)
    test_X.to_csv("%s_test.data" % prefix, sep=" ", header=False, index=False)
    test_y.to_csv("%s_test.solution" % prefix, sep=" ", header=False, index=False)

    print("Successfully saved files in %s (%s_...)." % (DATASETS_TO, dataset))

