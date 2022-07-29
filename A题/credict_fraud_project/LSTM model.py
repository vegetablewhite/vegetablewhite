import numpy as np
from matplotlib import pyplot as plt

from date_processing import date_processing

# from keras.callbacks import EarlyStopping
# from keras.utils import np_utils
# from keras.callbacks import ModelCheckpoint
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report, confusion_matrix
# from keras.models import Sequential
# from keras.utils import np_utils

from keras.layers import LSTM, Dense, Embedding, Dropout, Input, Attention, Layer, Concatenate, Permute, Dot, Multiply, Flatten
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.models import Sequential
from keras import backend as K, regularizers, Model, metrics
# from keras.backend import cast

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(attention, self).build(input_shape)

    def call(self, x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at, axis=-1)
        output=x*at
        return K.sum(output, axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention, self).get_config()


def plot_learningCurve(history, epoch):
    # Plot training & validation accuracy values
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


def LSTM_building():
    np.random.seed(7)
    c_path = 'C:\\Users\\123\\Desktop\\2022年首届钉钉杯大学生大数据挑战赛初赛题目\\A题\\数据集\\card_transdata.csv'
    train_x, test_x, train_y, test_y = date_processing(c_path=c_path)

    # X_train et X_val sont des dataframe qui contient les features
    train_LSTM_X = train_x
    val_LSTM_X = test_x

    ## Reshape input to be 3D [samples, timesteps, features] (format requis par LSTM)
    train_LSTM_X = train_LSTM_X.values.reshape((train_LSTM_X.shape[0], 1, train_LSTM_X.shape[1]))
    val_LSTM_X = val_LSTM_X.values.reshape((val_LSTM_X.shape[0], 1, val_LSTM_X.shape[1]))

    ## Recuperation des labels
    train_LSTM_y = train_y
    val_LSTM_y = test_y

    inputs1 = Input((1, 7))
    att_in = LSTM(50, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(inputs1)
    att_in_1 = LSTM(50, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(att_in)
    att_out = attention()(att_in_1)
    outputs1 = Dense(1, activation='sigmoid', trainable=True)(att_out)
    model1 = Model(inputs1, outputs1)
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model1.fit(train_LSTM_X, train_LSTM_y, epochs=20, batch_size=20000, validation_data=(val_LSTM_X, val_LSTM_y))
    return history


if __name__ == '__main__':
    import time

    start_time = time.time()
    epochs = 20
    history = LSTM_building()
    plot_learningCurve(history, epochs)



