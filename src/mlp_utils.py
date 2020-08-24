from tensorflow.keras import backend as K, layers, Sequential
import numpy as np
from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

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


class ResearchModel(keras.Sequential):
    def __init__(self):
        super().__init__()
        self.flag = 1

    def generate_weights(self, loc=0, scale=0.5):
        """
        Updates model weights without recompilation
        weights: list of numpy arrays with weights of the model
        """
        result = []
        for w in self.get_weights():
            arr = np.random.normal(loc, scale, size=w.shape)
            arr = arr.astype(np.float32)
            result.append(arr)
        self.set_weights(result)
        self.flag = 1


def baseline_model(inputs=5, outputs=5):
    # Creates a Keras neural network model
    model = ResearchModel()
    model.add(layers.Dense(inputs, activation='linear'))
    model.add(layers.Dense(10, kernel_initializer='random_normal', bias_initializer='random_normal', activation='tanh'))
    model.add(layers.Dense(10, kernel_initializer='random_normal', bias_initializer='random_normal', activation='tanh'))
    model.add(layers.Dense(outputs, activation='linear'))
    # Model Compilation
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.001), metrics=[coeff_determination])
    return model

def train_network(inputs, outputs, model, num_epochs=90000, my_callbacks=None, verbose=False):

    init_src, init_trgt, val_src, val_trgt = train_test_split(inputs, outputs, test_size=0.5, shuffle=True)

    i = 1
    r2 = 0

    while model.flag != 0:
        if verbose:
            print('Model: ', i, 'Training start!')
        model.flag = 0
        history = model.fit(init_src, init_trgt,
                            epochs=num_epochs, batch_size=64, callbacks=my_callbacks,
                            validation_data=(val_src, val_trgt),
                            verbose=0)
        r2 = history.history['val_coeff_determination'][-1]
        if verbose:
            print('Model: ', i, 'Training end, flag = ', model.flag, 'epoch: ', len(history.history['val_coeff_determination']))
        i+=1
    return r2


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