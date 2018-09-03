# -*- coding: utf-8 -*-
# @Time    : 07/06/2018 5:57 PM
# @Author  : Inf.Turing
# @Site    :
# @File    : add_feature.py
# @Software: PyCharm

import pandas as pd
import numpy as np

import os
import time


def log(x):
    os.system('echo ' + str(x))


import pickle


def run(path):
    full = pd.DataFrame()

    # 增加用户出现次数特征 统计计数
    def get_series(col='uid'):
        if col in full.keys():
            return full[col]
        else:
            train = pd.read_csv(path + '/train/' + col, header=None)
            train_length = len(train)
            test = pd.read_csv(path + '/test/' + col, header=None)
            predict = pd.read_csv(path + '/predict/' + col, header=None)
            train1 = pd.read_csv(path + '/train1/' + col, header=None)
            predict1 = pd.read_csv(path + '/predict1/' + col, header=None)
            full[col] = pd.concat([train, test, predict, train1, predict1]).reset_index(drop=True)[0]
        return train_length

    # 68996077
    train_length = get_series('label')
    len_full = len(full)
    log(len_full)
    log(train_length)

    def fix(feature, max=0):
        try:
            word2index = pickle.load(open(path + '/dic.pkl', 'rb'))
            if feature not in word2index.keys():
                log(feature)
                word2index[feature] = {}
                if max > 0:
                    for idx, val in enumerate(range(max)):
                        word2index[feature][val] = idx + 2
                else:
                    get_series(feature)
                    temp = full[feature].unique().tolist()
                    word2index[feature] = {}
                    for idx, val in enumerate(temp):
                        word2index[feature][val] = idx + 2
                print(word2index[feature])
                pickle.dump(word2index, open(path + '/dic.pkl', 'wb'))
        except:
            pass

    def save(full, feature='uid', type='num', max=15):
        full[full.label != -1][feature][:train_length].to_csv(path + '/train/' + feature, index=False, header=False)
        full[:68996077][full.label != -1][feature][train_length:].to_csv(path + '/test/' + feature, index=False,
                                                                         header=False)
        full[:68996077][full.label == -1][feature].to_csv(path + '/predict/' + feature, index=False, header=False)

        full[68996077:][full.label != -1][feature].to_csv(path + '/train1/' + feature, index=False, header=False)
        full[68996077:][full.label == -1][feature].to_csv(path + '/predict1/' + feature, index=False, header=False)
        if type == 'cate':
            fix(feature, max)

    def feature_count(full, features=[]):
        new_feature = 'new_count'
        for i in features:
            get_series(i)
            new_feature += '_' + i
        log(new_feature)
        try:
            del full[new_feature]
        except:
            pass
        temp = full.groupby(features).size().reset_index().rename(columns={0: new_feature})
        full = full.merge(temp, 'left', on=features)
        # save(full, new_feature)
        return full

    # 用户uid 计数以及组合广告特征的计数
    log('deal user counts features')
    full = feature_count(full, ['uid'])
    print(full['new_count_uid'].value_counts())
    log('save new_count_uid')
    save(full, 'new_count_uid', 'cate', -1)

    for i in ['advertiserId', 'campaignId', 'creativeSize', 'adCategoryId', 'productId', 'productType']:
        feature = 'new_count_' + i + '_uid'
        full = feature_count(full, [i, 'uid'])
        print(full[feature].value_counts())
        log('save:' + feature)
        save(full, feature, 'cate', -1)

    full = feature_count(full, ['creativeSize', 'productType', 'uid'])
    full['new_count_creativeSize_productType_uid'] = full['new_count_creativeSize_productType_uid'].apply(
        lambda x: min(x, 40))
    print(full['new_count_creativeSize_productType_uid'].value_counts())
    log('save')
    save(full, 'new_count_creativeSize_productType_uid', 'cate', -1)

    full = feature_count(full, ['adCategoryId', 'productType', 'uid'])
    full['new_count_adCategoryId_productType_uid'] = full['new_count_adCategoryId_productType_uid'].apply(
        lambda x: min(x, 40))
    print(full['new_count_adCategoryId_productType_uid'].value_counts())
    log('save')
    save(full, 'new_count_adCategoryId_productType_uid', 'cate', -1)

    # 分aid计算各特征分布信息熵
    for i in ['aid', 'age', 'gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house',
              'marriageStatus', 'ct', 'os']:
        full = feature_count(full, [i])

    for i in ['aid']:
        for j in ['age', 'gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house',
                  'marriageStatus', 'ct', 'os']:
            t = time.time()
            full = feature_count(full, [i, j])
            full['new_inf_' + i + '_' + j] = np.log1p(
                full['new_count_' + j] * full['new_count_' + i] / full['new_count_' + i + '_' + j] / len_full)

            min_v = full['new_inf_' + i + '_' + j].min()
            full['new_inf_' + i + '_' + j] = full['new_inf_' + i + '_' + j].apply(
                lambda x: int(float('%.1f' % min(x - min_v, 1.5)) * 10))
            print(full['new_inf_' + i + '_' + j].value_counts())
            log("Multiprocess cpu" + str((time.time() - t) / 60))
            save(full, 'new_inf_' + i + '_' + j, 'cate', max=15)

    for i in ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
              'topic2', 'topic3', 'appIdAction', 'appIdInstall']:
        get_series(i)
        full['len_' + i] = full[i].apply(lambda x: min(len(x.split()), 30))
        print(full['len_' + i].value_counts())
        save(full, 'len_' + i, 'cate', -1)

    # 多值特征的信息熵
    def get_inf(cond, keyword):
        get_series(cond)
        get_series(keyword)

        # 背景字典 每个ID出现的次数
        back_dict = {}
        # 不同条件下 每个id 出现的次数condi_dict
        condi_dict = {}
        # 不同条件下aid 每个id 出现的次数

        # 预先生成字典。省的判断慢
        for i in full[cond].unique():
            condi_dict[i] = {}

        for i, row in full[[cond, keyword]].iterrows():
            word_list = row[keyword].split()
            for word in word_list:
                # 对背景字典加1
                try:
                    back_dict[word] = back_dict[word] + 1
                except:
                    # 没有该词则设为0
                    back_dict[word] = 1
                try:
                    # 该条件下的该词的出现次数加1
                    condi_dict[row[cond]][word] = condi_dict[row[cond]][word] + 1
                except:
                    condi_dict[row[cond]][word] = 1

        # 先获取平均熵
        max_inf_list = []
        mean_inf_list = []
        condi_count = full.groupby(cond)[cond].count().to_dict()
        for i, row in full[[cond, keyword]].iterrows():
            word_list = row[keyword].split()

            count = len(word_list)
            prob = 1
            prob_list = []
            for word in word_list:
                temp_prob = condi_count[row[cond]] * back_dict[word] / condi_dict[row[cond]][word] / len_full
                prob = prob * temp_prob
                prob_list.append(temp_prob)
            mean_inf_list.append(np.log1p(prob) / count)
            max_inf_list.append(
                np.log1p(np.min(prob)))
        return max_inf_list, mean_inf_list

    # 计算maxpool 和meanpool
    for i in ['aid']:
        for j in ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                  'topic2', 'topic3', 'appIdAction', 'appIdInstall']:
            log('new_inf_' + i + '_' + j)
            full['new_inf_' + i + '_' + j + '_max'], full['new_inf_' + i + '_' + j + '_mean'] = get_inf(i, j)

            min_v = full['new_inf_' + i + '_' + j + '_max'].min()
            full['new_inf_' + i + '_' + j + '_max'] = full['new_inf_' + i + '_' + j + '_max'].apply(
                lambda x: int(float('%.1f' % min(x - min_v, 1.5)) * 10))
            save(full, 'new_inf_' + i + '_' + j + '_max', 'cate', 16)

            min_v = full['new_inf_' + i + '_' + j + '_mean'].min()
            full['new_inf_' + i + '_' + j + '_mean'] = full['new_inf_' + i + '_' + j + '_mean'].apply(
                lambda x: int(float('%.1f' % min(x - min_v, 1.5)) * 10))
            save(full, 'new_inf_' + i + '_' + j + '_mean', 'cate', 16)
