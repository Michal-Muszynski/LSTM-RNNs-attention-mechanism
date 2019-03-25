# ICMI2018
# unimodal aesthestic emotion regression using GSR_1L_aud features

# import the required modules
from __future__ import print_function
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Merge, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop,Adamax
from keras import backend as K


# define variables
nb_features = 1100 # total number of features, also the number of neurons in the input layer of LSTM

time_step = 3  # the length of history (number of previous data instances) to include
batch_size = 32 # the model performs a weight update for every batch, smaller batch means faster learning but les stable weights
nb_epoch = 1000 # number of total epochs to train the model
H1 = 128 # number of neurons in the bottom hidden layer
H2 = 64 # number of neurons in the middle hidden layer
H3 = 32 # number of neurons in the top hidden layer
dropout_W1 = 0.5 # drop out weight (for preventing over-fitting) in H1
dropout_U1 = 0.5 # drop out weight (for preventing over-fitting) in H1
dropout_W2 = 0 # drop out weight (for preventing over-fitting) in H2
dropout_U2 = 0 # drop out weight (for preventing over-fitting) in H2
dropout_W3 = 0 # drop out weight (for preventing over-fitting) in H3
dropout_U3 = 0 # drop out weight (for preventing over-fitting) in H3

#opt_func = RMSprop(lr=0.0001) # training function
opt_func = Adamax(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# if the validation loss isn't decreasing for a number of epochs, stop training to prevent over-fitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

#label file format:[Awe,Boredom,Disgust,Touched,Wonder]
labelf = '/jmain01/home/JAD003/sxr01/zxh52-sxr01/leimin/ICMI2018/Labels/Aesthetic_em_win5_lag1_'
featf = '/jmain01/home/JAD003/sxr01/zxh52-sxr01/leimin/ICMI2018/Features/GSR/GSR_1L_aud/per_movie/GSR_filter_slid_win5_lag1_LIRIS_1L_AC_aud_'
logf =  '/jmain01/home/JAD003/sxr01/zxh52-sxr01/leimin/ICMI2018/Outputs/GSR/GSR_1L_aud_log_Awe.txt'
predf =  '/jmain01/home/JAD003/sxr01/zxh52-sxr01/leimin/ICMI2018/Predictions/GSR/GSR_1L_aud_pred_Awe.csv'

# averaged mse and cc over all cross validations
tst_mse_cv = []
tst_cc_cv = []

# function to reshape the panda.DataFrame format data to Keras style: (batch_size, time_step, nb_features)
def reshape_data(data, n_prev = time_step):
    docX = []
    for i in range(len(data)):
        if i < (len(data)-n_prev):
            docX.append(data.iloc[i:i+n_prev].as_matrix())
        else: # the frames in the last window use the same context
            docX.append(data.iloc[(len(data)-n_prev):len(data)].as_matrix())
    alsX = np.array(docX)
    return alsX

# custom evaluation metrics
def pearson_cc(y_true, y_pred):
    fsp = y_pred - K.mean(y_pred,axis=0)   
    fst = y_true - K.mean(y_true,axis=0) 
    devP = K.std(y_pred,axis=0)  
    devT = K.std(y_true,axis=0)

    return K.sum(K.mean(fsp*fst,axis=0)/(devP*devT))

# turn off the warnings, be careful when use this
import warnings
warnings.filterwarnings("ignore")

# save outputs to a log file in case there is a broken pipe
import sys
idlestdout = sys.stdout
logger = open(logf, "w")
sys.stdout = logger

# leave-one-movie-out cross validation
for i in range(30):
	# list of training files
	index_list = []
	index_list = range(30)
	index_list.remove(i)

	# read in annotation files
	# test set
	tst_label_file = r"%s%s.csv"%(labelf,str(i))
	tst_label_raw = pd.read_csv(tst_label_file, header=None, usecols=[0])
	y_test = tst_label_raw.values # shape (len(test),1)

	# train set
	trn_label_list = []
	trn_label = pd.DataFrame()
	for j in index_list:
		trn_label_inputf = pd.read_csv(r"%s%s.csv"%(labelf,str(j)),header=None, usecols=[0])
		trn_label_list.append(trn_label_inputf)
	trn_label = pd.concat(trn_label_list, axis=0, ignore_index=True)
	y_train = trn_label.values # shape (len(train),1)

	# read in feature files
	# test set
	tst_feat_file = r"%s%s.csv"%(featf,str(i))
	tst_feat = pd.read_csv(tst_feat_file, header=None) # shape (len(test),nb_features)
	X_test = reshape_data(tst_feat) # shape (len(test),time_step,nb_features)

	# train set
	trn_feat_list = []
	trn_feat = pd.DataFrame()
	for j in index_list:
		trn_feat_inputf = pd.read_csv(r"%s%s.csv"%(featf,str(j)),header=None)
		trn_feat_list.append(trn_feat_inputf)
	trn_feat = pd.concat(trn_feat_list, axis=0, ignore_index=True) # shape (len(train),nb_features)
	X_train = reshape_data(trn_feat) # shape (len(train),time_step,nb_features)

	# define model structure
	model = Sequential()
	model.add(LSTM(H1, input_shape=(time_step, nb_features),dropout_W=dropout_W1, dropout_U=dropout_U1, return_sequences=True)) # bottom hidden layer
	model.add(LSTM(H2, dropout_W=dropout_W2, dropout_U=dropout_U2, return_sequences=True)) # middle hidden layer
	model.add(LSTM(H3, dropout_W=dropout_W3, dropout_U=dropout_U3, return_sequences=False)) # top hidden layer
	model.add(Dense(1, activation='sigmoid')) # output layer, regression task, value range [-1,1]
	model.compile(loss=pearson_cc, optimizer=opt_func, metrics=[pearson_cc,'mse']) # define the optimizer for training, use cc and mse as the evaluation metric

	# carry out training
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

	# evaluation
	print('Test set is movie No.', i)
	tst_score, tst_cc, tst_mse = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
	print('Test cc:', tst_cc)
	print('Test mse:', tst_mse)
	print('\n')
	
	# save mse and cc of each fold when they are not NaN
	if not np.isnan(tst_mse):
		tst_mse_cv.append(tst_mse)

	if not np.isnan(tst_cc):
		tst_cc_cv.append(abs(tst_cc))

	# output predictions
	tst_pred = model.predict(X_test)
	tst_df = pd.DataFrame(tst_pred)
	tst_df.to_csv(predf, mode='a', index=False, header=False)

	# Flush outputs to log file
	logger.flush()


# compute average mse and cc over all folds
mse_mean = reduce(lambda x, y: x + y, tst_mse_cv) / float(len(tst_mse_cv))
cc_mean = reduce(lambda x, y: x + y, tst_cc_cv) / float(len(tst_cc_cv))

print('\n==============================')
print('Averaged test cc:', cc_mean)
print('Averaged test mse:', mse_mean)

# Flush outputs to log file
logger.flush()
logger.close()