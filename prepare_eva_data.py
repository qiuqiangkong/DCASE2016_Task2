'''
SUMMARY:  calculate feature for private evaluate data
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
Modified: 2016.10.11 Modify variable name
--------------------------------------
'''
import prepare_dev_data as pp_dev_data
import config as cfg

if __name__ == "__main__":
    pp_dev_data.CreateFolder( cfg.eva_fe_fd )
    pp_dev_data.CreateFolder( cfg.eva_fe_mel_fd )
    
    pp_dev_data.GetMel( cfg.eva_wav_fd, cfg.eva_fe_mel_fd, n_delete=1 )
    pp_dev_data.GetMel( cfg.eva_wav_fd, cfg.eva_fe_mel_fd, n_delete=1 )