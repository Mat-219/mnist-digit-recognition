import gzip
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _images(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28, 1)


def _labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)


def load_mnist(data_dir=DATA_DIR):
    """Loads MNIST from the original idx .gz files. Pixels scaled to [0, 1]."""
    x_train = _images(os.path.join(data_dir, "train-images-idx3-ubyte.gz")).astype("float32") / 255.0
    y_train = _labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    x_test = _images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")).astype("float32") / 255.0
    y_test = _labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    return (x_train, y_train), (x_test, y_test)