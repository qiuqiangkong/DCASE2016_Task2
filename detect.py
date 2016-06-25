'''
SUMMARY:  detect from long segment
AUTHOR:   Qiuqiang Kong
Created:  2016.05.30
Modified: 2016.06.25
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
agg_num = 100
hop = 1
te_fe_fd = cfg.te_fe_mel_fd
test_noise = '-6'   # can be '0_', '6_', '-6'
thres = 0.2

# load model
md = pickle.load( open( 'Md/md100.p', 'rb' ) )

# evaluate for each test feature
names = os.listdir(cfg.ann_fd)
names = sorted(names)
results = []

for na in names:
    if na[10:12]==test_noise:
        print na
        # load data
        ann_path = cfg.ann_fd + '/' + na
        gt_list = ppData.ReadAnn( ann_path )        # ground truth list
        te_fe = te_fe_fd + '/' + na[0:-4] + '.f'
        X = cPickle.load( open(te_fe, 'rb') )
        X3d = mat_2d_to_3d( X, agg_num, hop )

        # detect
        p_y_pred = md.predict( X3d )
        out_list = ppData.OutMatToList(p_y_pred, thres)
        out_path = 'Results/' + na[0:-4] + '.txt'
        ppData.WriteOutToTxt( out_path, out_list )
        
        # evaluate
        eva = DCASE2016_EventDetection_SegmentBasedMetrics( cfg.labels )
        r = eva.evaluate( gt_list, out_list ).results()
        results.append( r )
    
# show average results of each file
ppData.ShowResults( results )