'''
SUMMARY:  calculate feature for private evaluate data
AUTHOR:   Qiuqiang Kong
Created:  2016.06.26
Modified: -
--------------------------------------
'''
import prepareData as ppData
import config as cfg

if __name__ == "__main__":
    ppData.CreateFolder('Fe_eva')
    ppData.CreateFolder('Fe_eva/Mel')
    ppData.CreateFolder('Results_eva')
    ppData.CreateFolder('Md')
    
    ppData.GetMel( cfg.eva_wav_fd, cfg.eva_fe_mel_fd, n_delete=1 )
    ppData.GetMel( cfg.eva_wav_fd, cfg.eva_fe_mel_fd, n_delete=1 )