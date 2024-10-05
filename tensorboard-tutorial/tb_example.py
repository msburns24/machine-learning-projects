import datetime as dt
import tensorflow as tf
import tensorboard as tb
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import (Input,
                          BatchNormalization,
                          Flatten,
                          Dense,
                          Dropout)


def make_model():
    model = Sequential([
        Input(shape=(28,28), name='Input'),
        BatchNormalization(name='InputNorm'),
        Flatten(name='InputFlatten'),
        Dense(512, activation='relu', name='Dense'),
        Dropout(0.2, name='Dropout20p'),
        Dense(10, activation='softmax', name='Output')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train, valid):
    X_train, y_train = train
    X_valid, y_valid = valid
    datecode = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = 'logs/fit/' + datecode
    model.fit(
        *train,
        epochs=5,
        validation_data=(X_valid, y_valid),
        callbacks=[TensorBoard(log_dir=log_dir, histogram_freq=1)]
    )
    return


if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    train, valid = mnist.load_data()
    model = make_model()
    train_model(model, train, valid)



