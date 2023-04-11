from DataPrep import dataPrep
import numpy as np

# divide datasets from MATLAB and save as .npy
scenario = 'O1_60_64beam'
inputFile = './DataSet_matlab/DeepMIMO_Dataset/' + scenario + '.mat'
train_inp, val_inp = dataPrep(inputName=inputFile, save_shuffled_idx=False)
np.save('./DataSet_npy/' + scenario + '_tr', train_inp)
np.save('./DataSet_npy/' + scenario + '_val', val_inp)