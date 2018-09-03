# 从认真跑完一个开源开始

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os


def log(x):
    os.system('echo ' + str(x))


import pickle

threshold = 1000
random.seed(2018)


def pre_data(path):
    if os.path.exists(path + '/userFeature.csv'):
        user_feature = pd.read_csv(path + '/userFeature.csv')
    else:
        userFeature_data = []
        with open(path + '/userFeature.data', 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                    print(i)
            user_feature = pd.DataFrame(userFeature_data)
            user_feature.to_csv(path + '/userFeature.csv', index=False)

    test1 = pd.read_csv(path + '/test1.csv')
    test2 = pd.read_csv(path + '/test2.csv')
    predict_df = pd.concat([test1, test2]).reset_index(drop=True)
    train_df = pd.read_csv(path + '/train.csv')
    train_df = pd.merge(train_df, pd.read_csv(path + '/adFeature.csv'), on='aid', how='left')
    predict_df = pd.merge(predict_df, pd.read_csv(path + '/adFeature.csv'), on='aid', how='left')
    train_df = pd.merge(train_df, user_feature, on='uid', how='left')
    predict_df = pd.merge(predict_df, user_feature, on='uid', how='left')
    train_df = train_df.fillna('-1')
    predict_df = predict_df.fillna('-1')
    train_df.loc[train_df['label'] == -1, 'label'] = 0
    predict_df['label'] = -1

    train_df, test_df, _, _ = train_test_split(train_df, train_df, test_size=0.02, random_state=2018)

    print(len(train_df), len(test_df), len(predict_df))
    return train_df, test_df, predict_df


def output_label(train_df, test_df, predict_df, path):
    with open(path + '/test/label', 'w') as f:
        for i in list(test_df['label']):
            f.write(str(i) + '\n')
    with open(path + '/predict/label', 'w') as f:
        for i in list(predict_df['label']):
            f.write(str(i) + '\n')
    with open(path + '/train/label', 'w') as f:
        for i in list(train_df['label']):
            f.write(str(i) + '\n')


def single_features(train_df, test_df, predict_df, word2index, path):
    single_ids_features = ['aid', 'uid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId',
                           'productId', 'productType', 'age', 'gender', 'education', 'consumptionAbility', 'LBS',
                           'carrier', 'house']

    for s in single_ids_features:
        cont = {}

        with open(path + '/train/' + str(s), 'w') as f:
            for line in list(train_df[s].values):
                f.write(str(line) + '\n')
                if str(line) not in cont:
                    cont[str(line)] = 0
                cont[str(line)] += 1

        with open(path + '/test/' + str(s), 'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line) + '\n')

        with open(path + '/predict/' + str(s), 'w') as f:
            for line in list(predict_df[s].values):
                f.write(str(line) + '\n')
        index = []
        for k in cont:
            if cont[k] >= threshold:
                index.append(k)
        word2index[s] = {}
        for idx, val in enumerate(index):
            word2index[s][val] = idx + 2
        log(s + ' done!')


def mutil_ids(train_df, test_df, predict_df, word2index, path):
    features_mutil = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                      'topic2', 'topic3', 'appIdAction', 'appIdInstall', 'marriageStatus', 'ct', 'os']
    for s in features_mutil:
        cont = {}
        with open(path + '/train/' + str(s), 'w') as f:
            for lines in list(train_df[s].values):
                f.write(str(lines) + '\n')
                for line in lines.split():
                    if str(line) not in cont:
                        cont[str(line)] = 0
                    cont[str(line)] += 1

        with open(path + '/test/' + str(s), 'w') as f:
            for line in list(test_df[s].values):
                f.write(str(line) + '\n')

        with open(path + '/predict/' + str(s), 'w') as f:
            for line in list(predict_df[s].values):
                f.write(str(line) + '\n')
        index = []
        for k in cont:
            if cont[k] >= threshold:
                index.append(k)
        word2index[s] = {}
        for idx, val in enumerate(index):
            word2index[s][val] = idx + 2
        log(s + ' done!')


def run(file_path, feature_path):
    if os.path.exists(feature_path + '/dic.pkl'):
        word2index = pickle.load(open(feature_path + '/dic.pkl', 'rb'))
    else:
        word2index = {}
    log('loading data')
    train_df, test_df, predict_df = pre_data(file_path)
    log('output label')
    output_label(train_df, test_df, predict_df, feature_path)
    log('output single_features')
    single_features(train_df, test_df, predict_df, word2index, feature_path)
    log('output mutil_ids')
    mutil_ids(train_df, test_df, predict_df, word2index, feature_path)
    pickle.dump(word2index, open(feature_path + '/dic.pkl', 'wb'))
