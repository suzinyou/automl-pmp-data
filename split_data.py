"""
Split train data from AutoML Challenge and pickle.dump as a 4-tuple

    train_X, train_y, test_X, test_y
    
"""

import argparse
import os

import pandas as pd
import numpy as np
import pickle


parser = argparse.ArgumentParser()

parser.add_argument("name", help="Name of dataset", type=str)

args = parser.parse_args()

DATASETS_PATH = "/Users/suzinyou/Google Drive/XBrain/01_ML/01_Auto_ML/AutoML_Datasets/"


if __name__ == "__main__":

    dataset = args.name

    data_file = DATASETS_PATH + dataset + "/" + dataset + "_train.data"
    solution_file = DATASETS_PATH + dataset + "/" + dataset + "_train.solution"

    print("Reading files from %s..." % (DATASETS_PATH + dataset))

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

    if not os.path.exists("datasets"):
        os.makedirs("datasets")

    train_X.to_csv("datasets/%s_train.data" % dataset, sep=" ", header=False, index=False)
    train_y.to_csv("datasets/%s_train.solution" % dataset, sep=" ", header=False, index=False)
    test_X.to_csv("datasets/%s_test.data" % dataset, sep=" ", header=False, index=False)
    test_y.to_csv("datasets/%s_test.solution" % dataset, sep=" ", header=False, index=False)

    print("Successfully saved files in datasets (%s_...)." % dataset)

