'''
SUMMARY:  do detection on private data
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
--------------------------------------
'''
import sys
sys.path.append('/homes/qkong/my_code2015.5-/python/Hat')
sys.path.append('evaluation')
import numpy as np
np.random.seed(1515)
from Hat.models import Sequential
from Hat.layers.core import InputLayer, Dense, Dropout, Lambda, Flatten
from Hat.layers.cnn import Convolution2D
from Hat.layers.pool import Pool2D
from Hat.callbacks import Validation, SaveModel
from Hat.preprocessing import pad_trunc_seqs, sparse_to_categorical, reshape_3d_to_4d, mat_2d_to_3d
from Hat.optimizers import Rmsprop
import prepareData as ppData
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
md = pickle.load( open( 'Md/md100.p', 'rb' ) )

# evaluate for each test feature
names = os.listdir( cfg.eva_wav_fd )
names = sorted(names)
results = []

for na in names:
    print na
    # load data
    te_fe = eva_fe_fd + '/' + na[0:-4] + '.f'
    X = cPickle.load( open(te_fe, 'rb') )
    X3d = mat_2d_to_3d( X, agg_num, hop )

    # detect
    p_y_pred = md.predict( X3d )
    out_list = ppData.OutMatToList(p_y_pred, thres)
    out_path = 'Results_eva/' + na[0:-4] + '.txt'
    ppData.WriteOutToTxt( out_path, out_list )