'''
SUMMARY:  train DNN model from training clips
          6 dB F value: 27%, 0 dB F value: 20%, -6 dB F value: 14%
          Train time: 0.2 s/epoch on Tesla 2090
AUTHOR:   Qiuqiang Kong
Created:  2016.05.30
Modified: 2016.06.25
          2016.10.11 modify variable name
--------------------------------------
'''
import numpy as np
import os
np.random.seed(1515)
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Dropout, Lambda, Flatten
from hat.layers.cnn import Convolution2D, Convolution1D
from hat.layers.pool import Pool2D, GlobalMaxPool
from hat.callbacks import Validation, SaveModel
from hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, reshape_3d_to_4d
from hat.optimizers import Rmsprop
import prepare_dev_data as pp_dev_data
import config as cfg


# hyper-params
tr_fe_fd = cfg.dev_tr_fe_mel_fd
max_len = 100
n_out = len( cfg.labels )

# prepare data
tr_X, tr_y = pp_dev_data.GetAllData( tr_fe_fd, max_len )
tr_y = sparse_to_categorical( tr_y, n_out )

print tr_X.shape, tr_y.shape
(_, n_time, n_freq) = tr_X.shape

# build model
seq = Sequential()
seq.add( InputLayer( (n_time, n_freq) ) )
seq.add( Flatten() )
seq.add( Dropout(0.1) )
seq.add( Dense(500, 'relu') )
seq.add( Dropout(0.1) )
seq.add( Dense(500, 'relu') )
seq.add( Dropout(0.1) )
seq.add( Dense(500, 'relu') )
seq.add( Dropout(0.1) )
seq.add( Dense(n_out, 'sigmoid') )
md = seq.combine()

md.summary()

# optimizer
optimizer = Rmsprop(1e-4)

# validation
validation = Validation( tr_x=tr_X, tr_y=tr_y, va_x=None, va_y=None, te_x=None, te_y=None, metrics=['categorical_error', 'binary_crossentropy'], call_freq=1, dump_path=None )

# save model
if not os.path.exists( cfg.dev_md_fd ): os.makedirs( cfg.dev_md_fd )
save_model = SaveModel( dump_fd=cfg.dev_md_fd, call_freq=10 )

# callbacks
callbacks = [validation, save_model]

### train model
md.fit( x=tr_X, y=tr_y, batch_size=20, n_epochs=200, loss_func='binary_crossentropy', optimizer=optimizer, callbacks=callbacks )