import pandas as pd
import numpy as np
from tqdm import tqdm
import hashlib,pickle,csv,gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
def binning(series, bin_num):
    bins = np.linspace(series.min(), series.max(), bin_num)
    labels = [i for i in range(bin_num-1)]
    out = pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(float)
    return out

f_in = open('../input_round1/userFeature.data', 'r')
f_out = open('../input_round1/userFeature.csv', 'w')
headers = ['LBS','age','appIdAction','appIdInstall','carrier','consumptionAbility','ct','education','gender','house',
           'interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','marriageStatus','os','topic1','topic2','topic3','uid']
writer = csv.DictWriter(f_out, headers)
writer.writeheader()
for i, line in enumerate(tqdm(f_in)):
    line = line.strip().split('|')
    userFeature_dict = {}
    for each in line:
        each_list = each.split(' ')
        userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
    writer.writerow(userFeature_dict)
f_out.close()
f_in.close()

f_in = open('../input_round2/userFeature.data', 'r')
f_out = open('../input_round2/userFeature.csv', 'w')
headers = ['LBS','age','appIdAction','appIdInstall','carrier','consumptionAbility','ct','education','gender','house',
           'interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','marriageStatus','os','topic1','topic2','topic3','uid']
writer = csv.DictWriter(f_out, headers)
writer.writeheader()
for i, line in enumerate(tqdm(f_in)):
    line = line.strip().split('|')
    userFeature_dict = {}
    for each in line:
        each_list = each.split(' ')
        userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
    writer.writerow(userFeature_dict)
f_out.close()
f_in.close()

train = pd.read_csv('../input_round2/train.csv')
train.loc[train['label']==-1,'label']=0
train_old = pd.read_csv('../input_round1/train.csv')
train_old.loc[train_old['label']==-1,'label']=0
test2 = pd.read_csv('../input_round2/test2.csv')
test2['label'] = -1
train = train.append(train_old)

userFeature = pd.read_csv('../input_round2/userFeature.csv')
userFeature_old = pd.read_csv('../input_round1/userFeature.csv')
userFeature = userFeature.append(userFeature_old)
userFeature = userFeature.drop_duplicates('uid')

adFeature = pd.read_csv('../input_round2/adFeature.csv')
adFeature_old = pd.read_csv('../input_round1/adFeature.csv')
adFeature = adFeature.append(adFeature_old)

train = train.merge(userFeature,'left','uid')
test2 = test2.merge(userFeature,'left','uid')
train = train.merge(adFeature,'left','uid')
test2 = test2.merge(adFeature,'left','uid')

static_feat = ['aid','LBS','age','carrier','consumptionAbility','ct','education','gender','house',
               'marriageStatus','os','advertiserId','campaignId','creativeId','adCategoryId','productId',
               'productType','creativeSize','uid_cnt','aid_cnt',]
dynamic_feat = ['interest1','interest2','interest5','kw1','kw2','topic2','interest3','interest4','kw3','topic1','topic3']

dynamic_dict = {}
ids = 1
for t in tqdm(userFeature[dynamic_feat].values):
    for i,dy in enumerate(t):
        feat = dynamic_feat[i]
        if isinstance(dy,str):
            for tt in dy.split(' '):
                key = feat+'_'+tt
                if key not in dynamic_dict:
                    dynamic_dict[key] = (ids,1)
                    ids += 1
                else:
                    dynamic_dict[key] = (dynamic_dict[key][0],dynamic_dict[key][1]+1)
pickle.dump(dynamic_dict,open('../cache/dynamic_dict.pkl','wb'))

dynamic_max_len = 30
train_dynamic_index = []
train_dynamic_lengths = []
for t in tqdm(train[dynamic_feat].values):
    tmp = []
    tmp2 = []
    for i,dy in enumerate(t):
        feat = dynamic_feat[i]
        if isinstance(dy,str):
            ttt = [dynamic_dict[feat+'_'+tt][0] for tt in dy.split(' ')]
            if len(ttt)> dynamic_max_len:
                ttt = ttt[:dynamic_max_len]
                tmp2.append(dynamic_max_len)
            else:
                ttt = ttt+[0]*(dynamic_max_len-len(ttt))
                tmp2.append(len(ttt))
            tmp.append(ttt)
        else:
            tmp.append([0]*dynamic_max_len)
            tmp2.append(1)
    train_dynamic_index.append(tmp)
    train_dynamic_lengths.append(tmp2)
train_dynamic_index = np.array(train_dynamic_index)
train_dynamic_lengths = np.array(train_dynamic_lengths)
train_dynamic_index.tofile('../cache/train_dynamic_index.bin')
pickle.dump(train_dynamic_lengths,open('../cache/train_dynamic_lengths.pkl','wb'),protocol=4)

test_dynamic_index = []
test_dynamic_lengths = []
for t in tqdm(test2[dynamic_feat].values):
    tmp = []
    tmp2 = []
    for i,dy in enumerate(t):
        feat = dynamic_feat[i]
        if isinstance(dy,str):
            ttt = [dynamic_dict[feat+'_'+tt][0] for tt in dy.split(' ')]
            if len(ttt)> dynamic_max_len:
                ttt = ttt[:dynamic_max_len]
                tmp2.append(dynamic_max_len)
            else:
                ttt = ttt+[0]*(dynamic_max_len-len(ttt))
                tmp2.append(len(ttt))
            tmp.append(ttt)
        else:
            tmp.append([0]*dynamic_max_len)
            tmp2.append(1)
    test_dynamic_index.append(tmp)
    test_dynamic_lengths.append(tmp2)
test_dynamic_index = np.array(test_dynamic_index)
test_dynamic_lengths = np.array(test_dynamic_lengths)
pickle.dump(test_dynamic_index,open('../cache/test2_dynamic_index.pkl','wb'),protocol=4)
pickle.dump(test_dynamic_lengths,open('../cache/test2_dynamic_lengths.pkl','wb'),protocol=4)

data = pd.concat([train,test2])
t = data.groupby('uid').size().reset_index().rename(columns={0:'uid_cnt'})
data = data.merge(t,'left','uid')
t = data.groupby('aid').size().reset_index().rename(columns={0:'aid_cnt'})
data = data.merge(t,'left','aid')
data['aid_cnt'] = np.round(np.log1p(data['aid_cnt'])/np.log(1.03))
data['LBS'] = data['LBS'].fillna(-1)
data['gender'] = data['gender'].fillna(-1)
data['house'] = data['house'].fillna(-1)
data['marriageStatus'] = data['marriageStatus'].fillna('-1')
st_total_feature_size = 0
for c in tqdm(static_feat):
    try:
        data[c] = LabelEncoder().fit_transform(data[c].apply(int))+st_total_feature_size
    except:
        data[c] = LabelEncoder().fit_transform(data[c])+st_total_feature_size
    st_total_feature_size += len(data[c].unique())
train_static_index = data[data.label!=-1]
train_static_index = train_static_index[static_feat].as_matrix()
test2_static_index = data[data.label==-1]
test2_static_index = test2_static_index[static_feat].as_matrix()
pickle.dump(train_static_index,open('../cache/train_static_index.pkl','wb'),protocol=4)
pickle.dump(test2_static_index,open('../cache/test2_static_index.pkl','wb','wb'),protocol=4)

kfold = 6
tmp = None
skf = KFold(n_splits=kfold, shuffle=True,random_state=1011)
for i, (train_index, test_index) in enumerate(skf.split(train)):
    print('kfold: {}  of  {} : '.format(i+1, kfold))
    aa = train.iloc[train_index]
    bb = train.iloc[test_index]
    t = aa.groupby('uid').size().reset_index().rename(columns={0:'count'})
    bb = bb.merge(t,'left','uid')
    t = aa[aa.label==1]
    t = t.groupby('uid').size().reset_index().rename(columns={0:'click'})
    bb = bb.merge(t,'left','uid')
    if tmp is None:
        tmp = bb
    else:
        tmp = pd.concat([tmp,bb])
tmp = tmp.fillna({'click':0})
del tmp['label']
train = train.merge(tmp,'left',['aid','uid'])
t = train.groupby('uid').size().reset_index().rename(columns={0:'count'})
test2 = test2.merge(t,'left','uid')
t = train[train.label==1]
t = t.groupby('uid').size().reset_index().rename(columns={0:'click'})
test2 = test2.merge(t,'left','uid')
test2 = test2.fillna({'click':0})
data = pd.concat([train,test2])
data['cvr'] = data['click']/ data['count']
bins = 50
data['cvr_bin'] = binning(data['cvr'],bins+1)
data = data.fillna({'cvr_bin':bins})
train_uid_cvr = data[data.label!=-1]
train_uid_cvr = train_uid_cvr['cvr_bin'].as_matrix()
test2_uid_cvr = data[data.label==-1]
test2_uid_cvr = test2_uid_cvr['cvr_bin'].as_matrix()
pickle.dump(train_uid_cvr,open('../cache/train_uid_cvr.pkl','wb'))
pickle.dump(test2_uid_cvr,open('../cache/test2_uid_cvr.pkl','wb'))
