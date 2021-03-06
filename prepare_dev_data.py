'''
SUMMARY:  Extract features, copy from acoustic_classification_2013
AUTHOR:   Qiuqiang Kong
Created:  2016.05.23
Modified: 2016.06.25
          2016.10.11 Modify variable name
--------------------------------------
'''
import sys
sys.path.append('activity_detection')
import numpy as np
from scipy import signal
import wavio
import config as cfg
import os
import matplotlib.pyplot as plt
import cPickle
import librosa
from scipy import io
from hat.preprocessing import pad_trunc_seq
from activity_detection import activity_detection

### readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

# calculate mel feature
def GetMel( wav_fd, fe_fd, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert fs==cfg.fs
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=40, fmin=0., fmax=22100 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )
        X = X[:, n_delete:]
        
        # DEBUG. print mel-spectrogram
        #plt.matshow(np.log(X.T), origin='lower', aspect='auto')
        #plt.show()
        #pause
        
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
          
###
# return X: (n_sample,max_len,n_freq), y: (n_sample)
def GetAllData( fe_fd, max_len ):
    names = os.listdir( fe_fd )
    names = sorted(names)
    Xall, yall = [], []
    for na in names:
        # load data
        path = fe_fd + '/' + na
        lb = na[0:-5]
        X = cPickle.load( open(path, 'rb') )
        
        # pad or trunc data
        X = pad_trunc_seq( X, max_len )
        Xall.append( X )
        yall.append( cfg.lb_to_id[lb] )
            
    return np.array( Xall ), np.array( yall )
            

###
# read annotation file    
def ReadAnn( txt_file ):
    lists = []
    fr = open( txt_file, 'r')
    index = 0;
    for line in fr.readlines():
        line_list = line.split('\t')
        bgn, fin, lb = float(line_list[0]), float(line_list[1]), line_list[2].split('\n')[0]
        if lb=='keys': lb='keysDrop'
        lists.append( { 'event_label':lb, 'event_onset':bgn, 'event_offset':fin } )
        index += 1
        
    fr.close()
    return lists

###
# get out_list from scores
def OutMatToList( scores, thres ):
    n_smooth = 10
    N, n_class = scores.shape
    
    lists = []
    for i1 in xrange( n_class ):
        bgn_fin_pairs = activity_detection( scores[:,i1], thres, n_smooth )
        for i2 in xrange( len(bgn_fin_pairs) ): 
            lists.append( { 'event_label':cfg.id_to_lb[i1], 
                            'event_onset':bgn_fin_pairs[i2]['bgn'] / (44100./1024.), 
                            'event_offset':bgn_fin_pairs[i2]['fin'] / (44100./1024.) } )
    return lists

# show f value
def ShowResults( results ):
    F_ary = []
    for r in results:
        F_ary.append( r['overall']['F'] )
    print 'mean F value:', np.mean( F_ary )

def WriteOutToTxt( out_path, out_list ):
    f = open( out_path, 'w' )
    for li in out_list:
        f.write( str(li['event_onset']) + '\t' + str(li['event_offset']) + '\t' + li['event_label'] + '\n')
    f.close()
    print 'Write out detections to', out_path, 'successfully!'

###    
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
if __name__ == "__main__":
    CreateFolder( cfg.dev_fe_fd )
    CreateFolder( cfg.dev_tr_fe_mel_fd )
    CreateFolder( cfg.dev_te_fe_mel_fd )
    
    GetMel( cfg.dev_tr_wav_fd, cfg.dev_tr_fe_mel_fd, n_delete=1 )
    GetMel( cfg.dev_te_wav_fd, cfg.dev_te_fe_mel_fd, n_delete=1 )
