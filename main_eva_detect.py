'''
SUMMARY:  do detection on private data
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
Modified: 2016.10.11 Modify variable name
--------------------------------------
'''
import sys
sys.path.append('evaluation')
import numpy as np
np.random.seed(1515)
from hat.models import Sequential
from hat.layers.core import InputLayer, Dense, Dropout, Lambda, Flatten
from hat.layers.cnn import Convolution2D
from hat.layers.pool import Pool2D
from hat.callbacks import Validation, SaveModel
from hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, reshape_3d_to_4d, mat_2d_to_3d
from hat.optimizers import Rmsprop
from hat import serializations
import prepare_dev_data as pp_dev_data
import config as cfg
from evaluation import *
import pickle
import cPickle
import os

# hyper-params
agg_num = 100       # should be same as training phase
hop = 1
eva_fe_fd = cfg.eva_fe_mel_fd
thres = 0.2

# load model
md = serializations.load( cfg.dev_md_fd + '/md100.p' )

# evaluate for each test feature
names = os.listdir( cfg.eva_wav_fd )
names = sorted(names)
results = []
if not os.path.exists( cfg.eva_results_fd ): os.makedirs( cfg.eva_results_fd )

for na in names:
    print na
    # load data
    te_fe = eva_fe_fd + '/' + na[0:-4] + '.f'
    X = cPickle.load( open(te_fe, 'rb') )
    X3d = mat_2d_to_3d( X, agg_num, hop )

    # detect
    p_y_pred = md.predict( X3d )
    out_list = pp_dev_data.OutMatToList(p_y_pred, thres)
    out_path = cfg.eva_results_fd + '/' + na[0:-4] + '.txt'
    pp_dev_data.WriteOutToTxt( out_path, out_list )