import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, AveragePooling1D
from keras.utils import to_categorical

from utils import *

def build_model_2D(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(48, (2, 2), activation='relu'))
    model.add(Conv2D(120, (2, 2), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(output_shape, activation='softmax'))
    
    return model

def build_model_1D(input_shape, output_shape):
    model = Sequential()
    model.add(Conv1D(128, 2, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv1D(128, 2, activation='relu', padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv1D(256, 2, activation='relu', padding='same'))
    model.add(AveragePooling1D(2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))
    
    return model

if __name__ == "__main__":
    # Loading train set and test set
    x_train, x_test, y_train, y_test = load_dataset()
    
    # Reshaping to perform 2D convolution
    #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    #x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1)
    # one-hot-encoded
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test) 

    print(x_train.shape, x_test.shape)
    print(y_train_hot.shape, y_test_hot.shape)

    epochs = 100
    batch_size = 128
    verbose = 1

    input_shape = x_train.shape[1:]
    print(input_shape)
    output_shape = y_train_hot.shape[1]
    model = build_model_1D(input_shape=input_shape, output_shape=output_shape)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    # train
    history = model.fit(x_train, y_train_hot, 
              batch_size=batch_size, 
              epochs=epochs, 
              verbose=verbose, 
              validation_data=(x_test, y_test_hot))
    save_model(model, model_name='model_1')

    # plot loss (save in images/loss)
    file_name = 'loss'
    plot_loss(history, file_name)

    # plot accuracy (save in images/loss)
    file_name = 'accuracy'
    plot_accuracy(history, file_name)
