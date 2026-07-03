import os
import h5py
import numpy as np

def load_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    train_path = os.path.join(base_dir, "datasets/trainset.hdf5")
    test_path  = os.path.join(base_dir, "datasets/testset.hdf5")

    train_dataset = h5py.File(train_path, "r")
    X_train = np.array(train_dataset["X_train"][:])
    y_train = np.array(train_dataset["Y_train"][:])

    test_dataset = h5py.File(test_path, "r")
    X_test = np.array(test_dataset["X_test"][:])
    y_test = np.array(test_dataset["Y_test"][:])

    return X_train, y_train, X_test, y_test
