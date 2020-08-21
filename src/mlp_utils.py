from tensorflow.keras import backend as K, layers
import numpy as np
from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd

def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


def normalize(arr):
    shape = arr.shape
    its = 1
    size = shape[-1]
    temp_arr = arr.flatten()
    for shape_i in shape:
        its *= shape_i
    its //= size
    ind = None
    for i in range(its - 1):
        temp_slice = temp_arr[i*size : (i+1)*size]
        temp_arr[i*size : (i+1)*size] = (temp_slice - np.min(temp_slice))/(np.max(temp_slice) - np.min(temp_slice))
        ind = i
    temp_slice = temp_arr[(ind+1)*size:]
    temp_arr[(ind+1)*size:] = (temp_slice - np.min(temp_slice))/(np.max(temp_slice) - np.min(temp_slice))
    return temp_arr.reshape(*shape)


class EarlyStopDifference(keras.callbacks.Callback):
    # Custom callback that stops training if the difference between training and validation
    # loss function is more than delta for the past patience training epochs
    
    ### Parameters:
    ### min_delta: Integer, default=0; Minimal affordable difference between loss functions
    ### patience: Integer, default=0; Number of epochs it is tolerable to have difference greater than delta
    ### verbose: Integer, default=0; Prints output if it is 1
    
    def __init__(self, patience=0, min_delta=0, verbose=0):
        # Initializing parameters
        super(EarlyStopDifference, self).__init__()
        self.patience=patience
        self.min_delta=min_delta
        self.verbose=verbose
        self.counter=0
    
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # Recieving loss function values
        mse = logs['loss']
        val_mse = logs['val_loss']
        # Comparing them to delta
        if self.verbose == 1:
            print("Epoch: ", epoch, "; Value: ", abs(mse - val_mse))
        if (abs(mse - val_mse) <= self.min_delta):
            # Resetting counter
            self.wait = 0
        else:
            # Incrementing counter
            self.wait += 1
            if self.wait >= self.patience:
                # Stopping the model if wait >= patience
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.flag=1
            
    def on_train_end(self, logs=None):
        # Printing the epoch the model has stopped
        if ((self.stopped_epoch > 0) and (self.verbose == 1)):
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
            
    def on_train_end(self, logs=None):
        # Printing the epoch the model has stopped
        if ((self.stopped_epoch > 0) and (self.verbose == 1)):
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

def baseline_model(inputs=5, outputs=5, metric=coeff_determination):
    model = tf.keras.Sequential()
    model.add(layers.Dense(inputs, activation='linear'))
    model.add(layers.Dense(10, kernel_initializer='random_normal', bias_initializer='zeros', activation='tanh'))
    model.add(layers.Dense(10, kernel_initializer='random_normal', bias_initializer='zeros', activation='tanh'))
    model.add(layers.Dense(outputs, activation='linear'))
    # Компиляция модели
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.001), metrics=[metric])
    return model

def test_predict(train_src,
                 train_trgt,
                 val_src,
                 val_trgt,
                 dif_delta=0.001,
                 stop_delta=0.0001,
                 dif_patience=10,
                 stop_patience=15,
                 metric=coeff_determination
                 restore_best_weights=False):
    my_callbacks = [
        EarlyStopDifference(patience=dif_patience, min_delta=dif_delta,  verbose=0),
        tf.keras.callbacks.EarlyStopping(patience=stop_patience, min_delta=stop_delta, restore_best_weights=restore_best_weights),
    ]
    flag = 1
    while flag != 0:
        model = baseline_model(inputs=train_src.shape[-1], outputs=train_trgt.shape[-1], metric=metric)
        model.flag = 0
        hist = model.fit(
            train_src,
            train_trgt,
            validation_data=(val_src, val_trgt),
            epochs=300,
            callbacks=my_callbacks,
            batch_size=100,
            verbose=False
        )
        score = hist.history['val_coeff_determination'][-1]
        flag = model.flag
    return score


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(hist['epoch'], hist['val_coeff_determination'],
           label='Val Coefficient of determination')
    ax.plot(hist['epoch'], hist['coeff_determination'],
           label='Coefficient of determination')
    ax.legend()
    return fig, ax