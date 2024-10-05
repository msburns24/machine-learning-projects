'''
CIFAR10 Dataset

Contains 60,000 images from 10 classes.
Images are 32x32, RGB --> Each shape = (32, 32, 3)

classes = [
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

A Note on Raw Data
------------------
The data is broken into batches ('data_batch_1', 'data_batch_2', ...),
along with a test set 'test_batch'. Each batch contains 10k.
50k train, 10k test. The data comes as a pickled Python dict,
with keys 'data' (X, np.ndarry) and 'labels' (y, list).
The shape of each batch's X is (10000, 3072), with each example as a
row, and columns representing the rows/columns/channels of the image.

0    - 1023 : Red Channel (32x32)
1024 - 2047 : Green Channel (32x32)
2048 - 3071 : Blue Channel (32x32)

Pixel values range from 0-255 over each channel.

'''
import pickle
import pathlib as pl
import numpy as np


def read_batch_data(name:str) -> tuple[np.ndarray, list]:
    data_dir = pl.Path(__file__).parent / 'data'
    path = data_dir / name
    with open(path, 'rb') as pkl:
        data = pickle.load(pkl, encoding='bytes')
    return data


def get_raw_data() -> tuple[np.ndarray]:
    filenames = [f'data_batch_{i}' for i in range(1, 6)]
    filenames.append('test_batch')
    data = [read_batch_data(name) for name in filenames]
    ftres = [batch[b'data'] for batch in data]
    X = np.concatenate(ftres, axis=0)
    labels = [d[b'labels'] for d in data]
    y = np.array([val for _list in labels for val in _list])
    return X, y


def transform_data(
        X:np.ndarray,
        y:np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    red =   X[:,     :1024].reshape(-1, 32, 32, 1)
    green = X[:, 1024:2048].reshape(-1, 32, 32, 1)
    blue =  X[:, 2048:    ].reshape(-1, 32, 32, 1)
    X = np.concatenate([red, green, blue], axis=3)
    return X, y


def load_data() -> tuple[tuple[np.ndarray]]:
    X_raw, y_raw = get_raw_data()
    X, y = transform_data(X_raw, y_raw)
    X_train, y_train = X[:50000], y[:50000]
    X_test, y_test   = X[50000:], y[50000:]
    return (X_train, y_train), (X_test, y_test)


