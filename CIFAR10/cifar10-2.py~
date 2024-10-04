'''
CIFAR10 Dataset

Contains 60,000 images from 10 classes. 
Images are 32x32, RGB --> Each shape = (32, 32, 3) 


Basic CNN architecture from HOML
--------------------------------




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


added in this way, each of the batch files contains a dictionary with the 
following elements:
    
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores 
            a 32x32 colour image. The first 1024 entries contain the red 
            channel values, the next 1024 the green, and the final 1024 the 
            blue. The image is stored in row-major order, so that the first 
            32 entries of the array are the red channel values of the first 
            row of the image.
    
    labels -- a list of 10000 numbers in the range 0-9. The number at 
              index i indicates the label of the ith image in the array data.

'''

from functools import partial
import pathlib as pl
import numpy as np
import keras
import pickle


def read_file(name:str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = pl.Path(__file__).parent / 'data'
    path = data_dir / name
    with open(path, 'rb') as pkl:
        data:dict[str,np.ndarray] = pickle.load(pkl, encoding='bytes')
    X = data[b'data']
    y = data[b'labels']
    return X, y


def transform_data(X:np.ndarray, y:list) -> tuple[np.ndarray, np.ndarray]:
    red =   X[:, :1024].reshape(-1, 32, 32, 1)
    green = X[:, 1024:2048].reshape(-1, 32, 32, 1)
    blue =  X[:, 2048:].reshape(-1, 32, 32, 1)
    X = np.concatenate([red, green, blue], axis=-1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def get_data() -> tuple[tuple[np.ndarray, np.ndarray]]:
    train_filenames = [f'data_batch_{i}' for i in range(1, 6)]
    test_filename = 'test_batch'

    X_train_batches = []
    y_train_batches = []
    for filename in train_filenames:
        X_batch_i, y_batch_i = read_file(filename)
        X_train_batches.append(X_batch_i)
        y_train_batches.extend(y_batch_i)
    
    X_train_flat = np.concatenate(X_train_batches, axis=0)
    X_train, y_train = transform_data(X_train_flat, y_train_batches)

    X_test, y_test = read_file(test_filename)
    X_test, y_test = transform_data(X_test, y_test)

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    return (X_train, y_train), (X_test, y_test)



(X_train, y_train), (X_test, y_test) = get_data()

print(X_train[0, :5, :5, 0])
exit()


DefaultConv2D = partial(
    keras.layers.Conv2D, 
    kernel_size=3, 
    activation='relu', 
    padding='SAME', 
)

model = keras.Sequential([
    keras.layers.Input(shape=(32, 32, 3)),
    keras.layers.BatchNormalization(), 
    DefaultConv2D(filters=64, kernel_size=5),
    keras.layers.MaxPooling2D(pool_size=2), 
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128), 
    keras.layers.MaxPooling2D(pool_size=2), 
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2), 
    keras.layers.Flatten(), 
    keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'), 
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'), 
    keras.layers.Dropout(0.5), 
    keras.layers.Dense(units=10, activation='softmax') 
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'], 
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2, 
    epochs=100, 
    verbose=2
)

results = model.evaluate(X_test, y_test, return_dict=True)
print("\n\n")
print("Results:")
print(results)





























### End of File
