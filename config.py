'''
SUMMARY:  config files
AUTHOR:   Qiuqiang Kong
Created:  2016.05.30
Modified: -
--------------------------------------
'''
# develop data
tr_wav_fd = '/homes/qkong/datasets/DCASE2016/Event Detection (synthetic)/dcase2016_task2_train_dev/dcase2016_task2_train'
te_wav_fd = '/homes/qkong/datasets/DCASE2016/Event Detection (synthetic)/dcase2016_task2_train_dev/dcase2016_task2_dev/sound'
tr_fe_mel_fd = 'Fe/trMel'
te_fe_mel_fd = 'Fe/teMel'
tr_fe_cqt_fd = 'Fe/trCqt'
te_fe_cqt_fd = 'Fe/teCqt'

ann_fd = '/homes/qkong/datasets/DCASE2016/Event Detection (synthetic)/dcase2016_task2_train_dev/dcase2016_task2_dev/annotation'

# private evaluate data
eva_wav_fd = '/homes/qkong/datasets/DCASE2016/Event Detection (synthetic)/dcase2016_task2_test/dcase2016_task2_test'
eva_fe_mel_fd = 'Fe_eva/Mel'

labels = [ 'clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock' , 'laughter', 'pageturn', 'phone', 'speech' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 44100.
win = 1024.