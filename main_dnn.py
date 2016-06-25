'''
SUMMARY:  train DNN model from training clips
          6 dB F value: 27%, 0 dB F value: 20%, -6 dB F value: 14%
          Train time: 0.2 s/epoch on Tesla 2090
AUTHOR:   Qiuqiang Kong
Created:  2016.05.30
Modified: 2016.06.25
--------------------------------------
'''
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
import numpy as np
np.random.seed(1515)
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Dense, Dropout, Lambda, Flatten
from Hat.layers.cnn import Convolution2D, Convolution1D
from Hat.layers.pool import Pool2D, GlobalMaxPool
from Hat.callbacks import Validation, SaveModel
from Hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, reshape_3d_to_4d
from Hat.optimizers import Rmsprop
import prepareData as ppData
import config as cfg


# hyper-params
tr_fe_fd = cfg.tr_fe_mel_fd
max_len = 100
n_out = len( cfg.labels )

# prepare data
tr_X, tr_y = ppData.GetAllData( tr_fe_fd, max_len )
tr_y = sparse_to_categorical( tr_y, n_out )

print tr_X.shape, tr_y.shape
(_, n_time, n_freq) = tr_X.shape

# build model
md = Sequential()
md.add( InputLayer( (n_time, n_freq) ) )
md.add( Flatten() )
md.add( Dropout(0.1) )
md.add( Dense(500, 'relu') )
md.add( Dropout(0.1) )
md.add( Dense(500, 'relu') )
md.add( Dropout(0.1) )
md.add( Dense(500, 'relu') )
md.add( Dropout(0.1) )
md.add( Dense(n_out, 'sigmoid') )

md.summary()
md.plot_connection()

# optimizer
optimizer = Rmsprop(1e-4)

# callbacks
save_model = SaveModel( dump_fd='Md', call_freq=10 )
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, metric_types=['categorical_error', 'binary_crossentropy', 'confusion_matrix'], call_freq=1, dump_path='Results/validation.p' )
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=20, n_epoch=200, loss_type='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )