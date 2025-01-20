import os
from keras import Sequential
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    BatchNormalization
)
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from dataset import load_data


# Ignore GPU warnings from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess_image(X):
    X = X.astype('float32') / 255.0
    X = tf.image.per_image_standardization(X):


(X_train, y_train), (X_test, y_test) = load_data()
X_train = X_train.astype('float32') / 255
X_test  = X_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test, num_classes=10)




# Model
model = Sequential([
    Input(shape=(32,32,3)),
    Conv2D(
        filters=128,
        kernel_size=(7,7),
        strides=(4,4),
        activation='relu',
        name='C1'
    ),
    BatchNormalization(name='BN1'),
    MaxPooling2D(pool_size=(2,2), name='P1'),
    Conv2D(
        filters=256,
        kernel_size=(5,5),
        strides=(1,1),
        activation='relu',
        padding="same",
        name='C2'
    ),
    BatchNormalization(name='BN2'),
    MaxPooling2D(pool_size=(3,3), name='P2'),
    Conv2D(
        filters=256,
        kernel_size=(3,3),
        strides=(1,1),
        activation='relu',
        padding='same',
        name='C3',
    ),
    BatchNormalization(name='BN3'),
    Conv2D(
        filters=256,
        kernel_size=(1,1),
        strides=(1,1),
        activation='relu',
        padding="same",
        name='C4',
    ),
    BatchNormalization(name='BN4'),
    Conv2D(
        filters=256,
        kernel_size=(1,1),
        strides=(1,1),
        activation='relu',
        padding="same",
        name='C5',
    ),
    BatchNormalization(name='BN5'),
    MaxPooling2D(pool_size=(2,2), name='P5'),
    Flatten(name='F'),
    Dense(1024, activation='relu', name='D1'),
    Dropout(0.5, name='D-O1'),
    Dense(1024, activation='relu', name='D2'),
    Dropout(0.5, name='D-O2'),
    Dense(10, activation='softmax', name='O')
])

model.compile(
     optimizer=SGD(learning_rate=0.001),
     loss='categorical_crossentropy',
     metrics=['accuracy'],
)

print("Model Compiled. Summary:\n")
print(model.summary())

# print("\n\nStarting training...\n")
# history = model.fit(
#     X_train, y_train,
#     batch_size=32,
#     epochs=50,
#     verbose=2,
#     validation_data=(X_test, y_test),
#     validation_freq=1,
# )
# 
# results = model.evaluate(X_test, y_test, return_dict=True, verbose=0)
# print("\n")
# print("Results:")
# print(results)





























### End of File
