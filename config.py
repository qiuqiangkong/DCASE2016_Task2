'''
SUMMARY:  config files
AUTHOR:   Qiuqiang Kong
Created:  2016.05.30
Modified: 2016.10.11 modify variable name
--------------------------------------
'''
# development data
dev_root = '/vol/vssp/AP_datasets/audio/dcase2016/task2/dcase2016_task2_train_dev'
dev_tr_wav_fd = dev_root + '/dcase2016_task2_train'
dev_te_wav_fd = dev_root + '/dcase2016_task2_dev/sound'
dev_ann_fd = dev_root + '/dcase2016_task2_dev/annotation'

# private evaluation data
eva_root = '/vol/vssp/AP_datasets/audio/dcase2016/task2/dcase2016_task2_test'
eva_wav_fd = eva_root + '/dcase2016_task2_test'

# your workspace
scrap_fd = '/vol/vssp/msos/qk/DCASE2016_task2_scrap'

dev_fe_fd = scrap_fd + '/Fe_dev'
dev_tr_fe_mel_fd = dev_fe_fd + '/trMel'
dev_te_fe_mel_fd = dev_fe_fd + '/teMel'
dev_md_fd = scrap_fd + '/Md_dev'
dev_results_fd = scrap_fd + '/results_dev'

eva_fe_fd = scrap_fd + '/Fe_eva'
eva_fe_mel_fd = eva_fe_fd + '/Mel'
eva_results_fd = scrap_fd + '/results_eva'


# global configurations
labels = [ 'clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock' , 'laughter', 'pageturn', 'phone', 'speech' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 44100.
win = 1024.