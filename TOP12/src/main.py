import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from tqdm import tqdm
import gc, hashlib, os,pickle
from sklearn.metrics import log_loss, roc_auc_score
import sys
sys.path.append("../models")
from tf_DeepFM import DeepFM

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

train = pd.read_csv('../input_round2/train.csv')
train.loc[train['label']==-1,'label']=0
train_old = pd.read_csv('../input_round1/train.csv')
train_old.loc[train_old['label']==-1,'label']=0
train = train.append(train_old)
test2 = pd.read_csv('../input_round2/test2.csv')

train_y = train['label'].values
res = test2[['aid','uid']].copy()

static_feat = ['aid','LBS','age','carrier','consumptionAbility','ct','education','gender','house',
               'marriageStatus','os','advertiserId','campaignId','creativeId','adCategoryId','productId',
               'productType','creativeSize','uid_cnt','aid_cnt','uid_cvr',]
dynamic_feat = ['interest1','interest2','interest5','kw1','kw2','topic2','interest3','interest4','kw3','topic1','topic3']

dynamic_dict = pickle.load(open('../cache/dynamic_dict.pkl','rb'))
train_dynamic_index = np.fromfile('../cache/train_dynamic_index.bin', dtype=np.int64)
train_dynamic_lengths = pickle.load(open('../cache/train_dynamic_lengths.pkl','rb'))
test_dynamic_index = pickle.load(open('../cache/test2_dynamic_index.pkl','rb'))
test_dynamic_lengths = pickle.load(open('../cache/test2_dynamic_lengths.pkl','rb'))
train_static_index = pickle.load(open('../cache/train_static_index.pkl','rb'))
test_static_index = pickle.load(open('../cache/test2_static_index.pkl','rb'))
train_uid_cvr = pickle.load(open('../cache/train_uid_cvr.pkl','rb'))
test_uid_cvr = pickle.load(open('../cache/test2_uid_cvr.pkl','rb'))
train_uid_cvr += train_static_index.max()+1
test_uid_cvr += train_static_index.max()+1
train_uid_cvr = train_uid_cvr.reshape([54338514,-1])
test_uid_cvr = test_uid_cvr.reshape([11727304,-1])
train_static_index = np.concatenate((train_static_index,train_uid_cvr),axis=1)
test_static_index = np.concatenate((test_static_index,test_uid_cvr),axis=1)

dy_total_feature_size = len(dynamic_dict)+1
st_total_feature_size = train_static_index.max()+1
field_sizes = [len(static_feat), len(dynamic_feat)]
total_feature_sizes = [st_total_feature_size,dy_total_feature_size]
dynamic_max_len = 30
train_dynamic_index = train_dynamic_index.reshape([-1,len(dynamic_feat)*dynamic_max_len])
test_dynamic_index = test_dynamic_index.reshape([-1,len(dynamic_feat)*dynamic_max_len])

fold = 3
pre = 0.0
for i in range(fold):
    dfm = DeepFM(field_sizes=field_sizes,
             total_feature_sizes=total_feature_sizes,
             dynamic_max_len=dynamic_max_len,
             learning_rate=0.0002,
             epoch=1,
             batch_size=8192,
             embedding_size=8,
             deep_layers=[256,128],
             loss_type="logloss",
             random_seed = i*17+1011,
             dropout_deep=[1.0,1.0,1.0])
    dfm.fit(train_static_index, train_dynamic_index, train_dynamic_lengths, train_y,combine=False)
    pre += dfm.predict(test_static_index,test_dynamic_index,test_dynamic_lengths)/fold
    rng_state = np.random.get_state()
    np.random.shuffle(train_static_index)
    np.random.set_state(rng_state)
    np.random.shuffle(train_dynamic_index)
    np.random.set_state(rng_state)
    np.random.shuffle(train_dynamic_lengths)
    np.random.set_state(rng_state)
    np.random.shuffle(train_y)

res['score'] = pre
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv('../submission.csv',index=False)
