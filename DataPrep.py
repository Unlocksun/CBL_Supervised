# the loaded data type in MATLAB is struct with 2 fields: channels (100, 1, 1184923) and userLoc (1184923, 3)
# channels: (#ant, #subcarrier, #user), userLoc: (#user, 3)
# And also note that each element of channels is complex number.
import numpy as np
import h5py as h5

def dataPrep(inputName=None, valPerc=0.3, save_shuffled_idx=False):
    with h5.File(inputName, 'r') as f:
        fields = [k for k in f.keys()]
        deepMIMO_data = f[fields[1]]
        nested = [k for k in deepMIMO_data]
        data_channels = np.squeeze(np.array(deepMIMO_data[nested[0]])).T
        # shape: (#users, #ant), in #ant dim, it is a tuple with real and imag parts of original data (real, imag)
        # shape: (#users, #ant, 2), decoup[0,0,0]=real, decoup[0,0,1]=imag
        X = data_channels

    print('Creating training and validation inputs')
    numTrain = np.floor((1 - valPerc) * X.shape[0]).astype('int')
    numVal = np.floor(valPerc * X.shape[0]).astype('int')
    print('Size of training dataset: ' + str(numTrain) + '\n' + 'Size of validation dataset: ' + str(numVal))
    shuffled_idx = np.random.permutation(X.shape[0])
    if save_shuffled_idx:
        np.save('shuffled_ind', shuffled_idx)
    X_shuffled = X[shuffled_idx, :]
    train_inp = X_shuffled[0:numTrain, :]
    val_inp = X_shuffled[numTrain:, :]

    return train_inp, val_inp


def load_data(tr_set, val_set, tr_load_perc=1):
    train_inp = np.load(tr_set)      # shape: (numTrain, 2*#ant)
    val_inp = np.load(val_set)
    total_train = train_inp.shape[0]
    return train_inp[:tr_load_perc * total_train, :], val_inp
