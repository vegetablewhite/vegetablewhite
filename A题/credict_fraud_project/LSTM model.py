import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

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
    train_x, test_x, train_y, test_y, y, train_x_SMOTE, train_y_SMOTE, train_x_ada, train_y_ada = date_processing(
        c_path=c_path)




    ## Reshape input to be 3D [samples, timesteps, features] (format requis par LSTM)
    train_x = train_x.values.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.values.reshape((test_x.shape[0], 1, test_x.shape[1]))

    train_x_SMOTE = train_x_SMOTE.values.reshape((train_x_SMOTE.shape[0], 1, train_x_SMOTE.shape[1]))
    train_x_ada = train_x_ada.values.reshape((train_x_ada.shape[0], 1, train_x_ada.shape[1]))


    ## Recuperation des labels


    inputs1 = Input((1, 7))
    att_in = LSTM(50, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(inputs1)
    att_in_1 = LSTM(50, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(att_in)
    att_out = attention()(att_in_1)
    outputs1 = Dense(1, activation='sigmoid', trainable=True)(att_out)
    model1 = Model(inputs1, outputs1)
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # history = model1.fit(train_LSTM_X, train_LSTM_y, epochs=20, batch_size=20000, validation_data=(val_LSTM_X, val_LSTM_y))

    train_list = [
        {'X': train_x, 'Y': train_y, 'name': '原始数据'},
        {'X': train_x_SMOTE, 'Y': train_y_SMOTE, 'name': 'SMOTE后'},
        {'X': train_x_ada, 'Y': train_y_ada, 'name': 'ADASYN后'},
    ]

    final_dict_list = []
    for one_dict in train_list:
        name = one_dict['name']
        lstm = model1()
        lstm.fit(one_dict['X'], one_dict['Y'], epochs=20, batch_size=20000, validation_data=(test_x, test_y))
        lstm_pred = lstm.predict(test_x)
        one_dict['y_pred'] = lstm_pred
        one_dict['model'] = lstm
        n_errors = (lstm_pred != test_y).sum()
        one_dict['预测错误个数'] = n_errors
        acc = accuracy_score(test_y, lstm_pred)
        prec = precision_score(test_y, lstm_pred)
        rec = recall_score(test_y, lstm_pred)
        f1 = f1_score(test_y, lstm_pred)
        MCC = matthews_corrcoef(test_y, lstm_pred)
        one_final_dict = {
            'name': one_dict['name'],
            '预测错误个数': n_errors,
            '准确率': acc,
            '精确度': prec,
            '召回率': rec,
            'F1-Score': f1,
            'Matthews相关系数': MCC
        }

        final_dict_list.append(one_final_dict)
    print(final_dict_list)
    return final_dict_list




if __name__ == '__main__':
    import time

    start_time = time.time()
    epochs = 20
    history = LSTM_building()
    # plot_learningCurve(history, epochs)



