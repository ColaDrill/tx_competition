import numpy as np
import pandas as pd
import tensorflow as tf
import utils
import nffm
import os
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]='0'


def create_hparams():
    return tf.contrib.training.HParams(
        k=8,
        batch_size=4096,
        optimizer="adam",
        learning_rate=0.0002,
        num_display_steps=100,
        num_eval_steps=1000,
        l2=0.000002,
        hidden_size=[128,128],
        evl_batch_size=5000,
        all_process=1,
        idx=0,
        epoch=int(44628906//4096),
        mode='train',
        data_path='../lookalike_final/ffm_data_logloss/ffm_data/',
        sub_name='sub',
        single_features=['aid','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType', 'age', 'gender','education', 'consumptionAbility', 'LBS', 'carrier', 'house'],
        mutil_features=['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall','marriageStatus','ct','os'],
        )






hparams=create_hparams()
hparams.path='./model/'
utils.print_hparams(hparams)


       
hparams.aid=['aid','advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']        
hparams.user=['age', 'gender','education', 'consumptionAbility', 'LBS', 'carrier', 'house','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall','marriageStatus','ct','os'] 
hparams.num_features=[]
preds=nffm.train(hparams)   
test_df=pd.read_csv('data/test.csv')
test_df['score']=preds
test_df['score']=test_df['score'].apply(lambda x:round(x,9))
test_df[['aid','uid','score']].to_csv('result_sub/submission_'+str(hparams.sub_name)+'.csv',index=False)     
    
    