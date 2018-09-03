# 预处理复赛数据
import os
import _thread
import time
import pandas as pd

os.system('free -h')
os.system('cat /proc/cpuinfo| grep "processor"| wc -l')
os.system('mkdir /dev/cos_cache')
os.system('rm -rf /cos_cache')
os.system('ln -s /dev/cos_cache /cos_cache')
os.system('pip install pickle')


# 定时拷贝log
def cp_log():
    while True:
        os.system('cp -f /gaiastack/log/stderr.log /cos_person/log/lgb_test_stderr.log')
        os.system('cp -f /gaiastack/log/stdout.log /cos_person/log/lgb_test_stdout.log')
        time.sleep(60)


def log(x):
    os.system('echo ' + str(x))


# _thread.start_new_thread(cp_log, ())

# 原始文件目录
file_path = '/cos_person/data'
# 程序主目录
root_path = '/cos_person/tadv2'
# 特征文件生成目录
feature_path = '/cos_person/tadv2/data'

import sys

sys.path.append(root_path)

from src import base_data, base_data_1, add_feature_with_new
from models import nffm


def run():
    # 预处理数据
    log('base_data')
    base_data.run(file_path, feature_path)
    # 预处理初赛数据
    log('base_data_1')
    base_data_1.run(file_path, feature_path)
    # 增加特征
    log('add_feature_with_new')
    add_feature_with_new.run(feature_path)
    # 训练模型
    log('train mean')
    nffm.run(file_path, feature_path, root_path, 'mean')
    nffm.run(file_path, feature_path, root_path, 'max')
    log('train max')
    # concat
    weight = {
        'mean': 0.5,
        'max': 0.5,
    }

    result = pd.DataFrame()
    for i in weight:
        temp = pd.read_csv(file_path + '/' + i + '.csv')
        print(i, temp.head(), temp['score'].mean())
        if len(result) == 0:
            result['aid'] = temp.aid
            result['uid'] = temp.uid
            result['score'] = weight[i] * temp['score']
        else:
            result['score'] = weight[i] * temp['score'] + result['score']

    result['score'] = result['score'].apply(lambda x: float('%.6f' % x))
    print(result.head())
    print(result['score'].mean())
    result.to_csv(root_path + '/submission.csv', index=False)


run()
