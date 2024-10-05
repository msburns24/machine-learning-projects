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

'''
import os
import pathlib as pl
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
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import load_data


# Ignore GPU warnings from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(X_train, y_train), (X_test, y_test) = load_data()
X_train = X_train.astype('float32') / 255
X_test  = X_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test, num_classes=10)


# Layer Arguments
input_shape = X_train.shape[1:]
conv_kwargs = dict(
    kernel_size=(3,3),
    padding='same',
    kernel_initializer='he_normal',
    activation='relu',
)
pool_size=(2,2)
do1_rate = 0.25
do2_rate = 0.5
dense1_kwargs = dict(
    units=312,
    activation='relu',
    kernel_initializer='he_normal',
)
dense2_kwargs = dict(
    units=10,
    activation='softmax',
    kernel_initializer='glorot_uniform',
)


# Model
model = Sequential()
model.add(Input(shape=input_shape))
for n_f in (32, 64):
    model.add(Conv2D(n_f, **conv_kwargs))
    model.add(BatchNormalization()),
    model.add(Conv2D(n_f, **conv_kwargs))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(do1_rate))
    model.add(BatchNormalization())
# FLATTEN -> DENSE -> RELU -> DROPOUT
model.add(Flatten())
model.add(Dense(**dense1_kwargs))
model.add(BatchNormalization())
# DENSE -> SOFTMAX
model.add(Dense(**dense2_kwargs))


model.compile(
     optimizer=Adam(learning_rate=0.005),
     loss='categorical_crossentropy',
     metrics=['accuracy'],
 )

history = model.fit(
    X_train, y_train,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    epochs=100,
    verbose=2,
)

results = model.evaluate(X_test, y_test, return_dict=True)
print("\n\n")
print("Results:")
print(results)





























### End of File
