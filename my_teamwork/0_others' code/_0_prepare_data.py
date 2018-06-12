
# coding=utf-8
# author:WL

"""
开源一个转CSV的代码。不需要大内存，只提取训练集和测试集中出现过得用户ID，分开存储，减小数据写入读取压力。。
做训练时直接全部 concat就行，或者分开训练
"""

import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import gc
import math
import numpy as np


def get_seed_group():
    if os.path.exitst('./data/aid_group.pkl'):
        uid_group = pickle.load(open('./data/aid_group.pkl','rb'))
        aid2uid = pickle.load(open('./data/aid2uid.pkl','rb'))
    else:
        train = pd.read_csv('./data/preliminary_contest_data/train.csv',header = 0)
        test = pd.read_csv('./data/preliminary_contest_data/test1.csv',header = 0)
        data = pd.concat([train,test],axis=0)
        del train
        del test
        ge.collect()

        seed = pd.DataFrame(data['aid'].value_counts())
        seed['aid'] = seed.index
        seed.columns = ['num','aid']
        seed.head()
        seed['num_cut'] = seed['num'].map(lambda x:np.log(0.5*x))
        seed['num_cut'] = pd.cut(seed['num_cut'],bins=12,labels=False)
        seed.loc[seed['num_cut']==3,'num_cut'] = 4
        seed.loc[seed['num_cut']==0,'num_cut'] = 2
        seed.loc[seed['num_cut']==1,'num_cut'] = 2

        aid_group = {}
        group = seed.num_cut.values
        aid = seed.aid.values

        for i in range(len(aid)):
            if group[i] not in aid_group.keys():
                aid_group[group[i]] = [aid[i]]
            else:
                aid_group[group[i]].append(aid[i])

        pickle.dump(aid_group,open('./data/aid_group.pkl','wb'))

        aid2uid = {}
        aid = data.aid.values
        uid = data.uid.values

        for i in range(len(uid)):
            if aid[i] not in aid2uid.keys():
                aid2uid[aid[i]] = [uid[i]]
            else:
                aid2uid[aid[i]].append(uid[i])

        pickle.dump(aid2uid,open('./data/aid2uid.pkl','wb'))

    return aid_group,aid2uid


def get_user_feature(uid_list,group_num):
    uid_list = {uid:i for i,uid in enumerate(uid_list)}
    userFeature_data = []
    with open('./data/preliminary_contest_data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            if int(line[0].split()[-1]) in uid_list.keys():
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)

        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('./data/preliminary_contest_data/userFeature_%d.csv' % group_num, index=False)
    gc.collect()


aid_group,aid2uid = get_seed_group()

for gn in uid_group.keys():
    if gn != 11:
        aid_list = uid_group[gn]
        uid_list = []
        for aid in aid_list:
            uid_list.extend(aid2uid[aid])

        get_user_feature(uid_list,gn)
        print(gn,"done!")






















