import os
import pandas as pd
import random

threshold = 1000
random.seed(2018)


def log(x):
    os.system('echo ' + str(x))


def pre_data(path):
    if os.path.exists(path + '/userFeature2.csv'):
        user_feature = pd.read_csv(path + '/userFeature2.csv')
    else:
        userFeature_data = []
        with open(path + '/userFeature2.data', 'r') as f:
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
            user_feature.to_csv(path + '/userFeature2.csv', index=False)

    test1 = pd.read_csv(path + '/test11.csv')
    test2 = pd.read_csv(path + '/test12.csv')
    train_df = pd.read_csv(path + '/train1.csv')
    predict_df = pd.concat([test1, test2]).reset_index(drop=True)
    train_df = pd.merge(train_df, pd.read_csv(path + '/adFeature1.csv'), on='aid', how='left')
    predict_df = pd.merge(predict_df, pd.read_csv(path + '/adFeature1.csv'), on='aid', how='left')

    train_df = pd.merge(train_df, user_feature, on='uid', how='left')
    predict_df = pd.merge(predict_df, user_feature, on='uid', how='left')
    train_df = train_df.fillna('-1')
    predict_df = predict_df.fillna('-1')
    train_df.loc[train_df['label'] == -1, 'label'] = 0
    predict_df['label'] = -1
    print(len(train_df), len(predict_df))
    return train_df, predict_df


def output_label(train_df, predict_df, path):
    with open(path + '/predict1/label', 'w') as f:
        for i in list(predict_df['label']):
            f.write(str(i) + '\n')
    with open(path + '/train1/label', 'w') as f:
        for i in list(train_df['label']):
            f.write(str(i) + '\n')


def single_features(train_df, predict_df, path):
    single_ids_features = ['aid', 'uid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId',
                           'productId', 'productType', 'age', 'gender', 'education', 'consumptionAbility', 'LBS',
                           'carrier', 'house']
    for s in single_ids_features:
        with open(path + '/train1/' + str(s), 'w') as f:
            for line in list(train_df[s].values):
                f.write(str(line) + '\n')

        with open(path + '/predict1/' + str(s), 'w') as f:
            for line in list(predict_df[s].values):
                f.write(str(line) + '\n')


def mutil_ids(train_df, predict_df, path):
    features_mutil = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                      'topic2', 'topic3', 'appIdAction', 'appIdInstall', 'marriageStatus', 'ct', 'os']
    for s in features_mutil:
        with open(path + '/train1/' + str(s), 'w') as f:
            for lines in list(train_df[s].values):
                f.write(str(lines) + '\n')

        with open(path + '/predict1/' + str(s), 'w') as f:
            for line in list(predict_df[s].values):
                f.write(str(line) + '\n')


def run(file_path, feature_path):
    log('loading data')
    train_df, predict_df = pre_data(file_path)
    log('output_label')
    output_label(train_df, predict_df, feature_path)
    log('output single_features')
    single_features(train_df, predict_df, feature_path)
    log('output mutil_ids')
    mutil_ids(train_df, predict_df, feature_path)
