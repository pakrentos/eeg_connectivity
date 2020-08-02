from tensorflow.keras import backend as K
import numpy as np
from tensorflow import keras


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
    for i in range(its - 1):
        temp_slice = temp_arr[i*size : (i+1)*size]
        temp_arr[i*size : (i+1)*size] = (temp_slice - np.min(temp_slice))/(np.max(temp_slice) - np.min(temp_slice))
    temp_slice = temp_arr[(i+1)*size:]
    temp_arr[(i+1)*size:] = (temp_slice - np.min(temp_slice))/(np.max(temp_slice) - np.min(temp_slice))
    return temp_arr.reshape(*shape)


class EarlyStopDifference(keras.callbacks.Callback):
    # Custom callback that stops training if the difference between training and validation
    # loss function is more than delta for the past patience training epochs

    ### Parameters:
    ### delta: Integer, default=0; Minimal affordable difference between loss functions
    ### patience: Integer, default=0; Number of epochs it is tolerable to have difference greater than delta
    ### verbose: Integer, default=0; Prints output if it is 1

    def __init__(self, patience=0, delta=0, verbose=0):
        # Initializing parameters
        super(EarlyStopDifference, self).__init__()
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0

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
        if ((mse - val_mse) <= self.delta):
            # Resetting counter
            self.wait = 0
        else:
            # Incrementing counter
            self.wait += 1
            if self.wait >= self.patience:
                # Stopping the model if wait >= patience
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if (self.verbose != 0):
                    print("Model stopped because mse and val_mse differ for more than ", self.delta, " for the past ",
                          self.patience, " training epochs.")

    def on_train_end(self, logs=None):
        # Printing the epoch the model has stopped
        if (self.stopped_epoch > 0) & (self.verbose != 0):
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))