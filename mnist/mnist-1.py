import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import keras_tuner

import warnings
warnings.simplefilter('ignore')


def load_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = load_mnist_data()


def add_dense_layer(
        model:keras.Sequential, 
        hp:keras_tuner.HyperParameters, 
        layer_num:int, 
) -> keras.Sequential:
    units = int(64 / layer_num)
    factor = 0.1 if layer_num == 2 else 1
    l2 = hp.Float(
        f'l2_{layer_num}', 
        min_value=0.0010 * factor, 
        max_value=0.0014 * factor, 
        step=0.0002 * factor, 
    )
    kreg = keras.regularizers.l2(l2)

    model.add(keras.layers.Dense(units, 'relu', kernel_regularizer=kreg))
    return model


def add_dropout_layer(
        model:keras.Sequential, 
        hp:keras_tuner.HyperParameters, 
        layer_num:int, 
) -> keras.Sequential:
    dropout_rate = hp.Choice(f'dropout_rate_{layer_num}', [0.0, 0.3])
    if dropout_rate == 0:
        return model
    model.add(keras.layers.Dropout(dropout_rate))
    return model


def add_batch_norm_layer(model:keras.Sequential) -> keras.Sequential:
    model.add(keras.layers.BatchNormalization())
    return model


def build_model(hp:keras_tuner.HyperParameters) -> keras.Sequential:
    model = keras.Sequential()
    
    model = add_dense_layer(model, hp, 1)
    model = add_dropout_layer(model, hp, 1)
    model = add_batch_norm_layer(model)
    model = add_dense_layer(model, hp, 2)
    model = add_dropout_layer(model, hp, 2)
    model = add_batch_norm_layer(model)
    model.add(keras.layers.Dense(10, activation='softmax'))

    lr = 0.0002
    opt = keras.optimizers.Adam(lr)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    return model



'''
# New round
┌─────────────────┬───────────┬───────────┬─────────┬───┐
│ Hyperparameter  │ min_value │ max_value │ step    │ # │
├─────────────────┼───────────┼───────────┼─────────┼───┤
│ dropout_rate_1  │                      [0.0, 0.3] │ 2 │
│ l2_1            │ 0.0010    │ 0.0014    │ 0.0002  │ 3 │
│ dropout_rate_2  │                      [0.0, 0.3] │ 2 │
│ l2_2            │ 0.00010   │ 0.00014   │ 0.00002 │ 3 │
└─────────────────┴───────────┴───────────┴─────────┴───┘
Total Models: 36


Best So Far:

┌─────────────────┬────────────┐
│ Hyperparameter  │ Best       │
├─────────────────┼────────────┤
│ units_1         │ 64         │
│ l2_1            │ 0.001      │
│ units_2         │ 32         │
│ l2_2            │ 0.012589   │ 0.012
│ learning_rate   │ 0.00019953 │ 0.0002
│ dropout_rate_2  │ 0.2        │
└─────────────────┴────────────┘

Models Checked:              675
Total Time:          18h 56m 36s
Average Time:             1m 41s
'''




if __name__ == '__main__':
    X_train_norm = X_train / 255
    X_test_norm = X_test / 255

    hp = keras_tuner.HyperParameters()
    es_callback = keras.callbacks.EarlyStopping(patience=5)
    grid = keras_tuner.GridSearch(build_model, objective='val_accuracy')
    grid.search(
        X_train, y_train, 
        epochs=30, 
        validation_split=0.2, 
        verbose=1
    )
    best_model:keras.Sequential = grid.get_best_models()
    best_model.summary()