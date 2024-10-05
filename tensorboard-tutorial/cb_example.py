import datetime as dt
from keras.datasets import mnist
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import (Input,
                          BatchNormalization,
                          Flatten,
                          Dense,
                          Dropout)


#   inoremap {<CR> {<CR>}<C-o>O
# 
#   inoremap <C-Return> <CR><CR><C-o>k<Tab>

'''
my_list = [

my_very_long_name_that_may_take_up_the_whole_line = [

------------->_
#23456789012345678901234567890
#        1         2         3
#        1         2         3
#23456789012345678901234567890

---->_
'''



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
    model.fit(
        *train,
        epochs=5,
        validation_data=(X_valid, y_valid),
    )
    return


if __name__ == '__main__':
    train, valid = mnist.load_data()
    model = make_model()
    train_model(model, train, valid)



