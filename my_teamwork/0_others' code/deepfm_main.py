# -*- coding:utf-8 -*-

import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import sys


from tf_DeepFM import DeepFM

"""
static_index的维度: 行数*特征数量
dynamic_index的维度: 行数*[特征数量 * max_len]
dynamic_lengths的维度: 行数*特征数量
"""

sys.path.append('../')
from utils.args import args
from utils import tencent_data_functions


#param
field_sizes = [len(args.static_features), len(args.dynamic_features)]
total_feature_sizes = [args.st_total_feature_size, args.dy_total_feature_size]
dynamic_max_len = args.dynamic_max_len
learning_rate = args.lr
weight_decay = args.weight_decay
epochs = args.epochs
batch_size = args.batch_size



y, static_index, dynamic_index, dynamic_lengths, extern_lr_index = \
    tencent_data_functions.load_concatenate_tencent_data(args.train_train_data_path, 7)

valid_y, valid_static_index, valid_dynamic_index, valid_dynamic_lengths, valid_exctern_lr_index = \
    tencent_data_functions.load_concatenate_tencent_data(args.train_test_data_path, 2)

test_y, test_static_index, test_dynamic_index, test_dynamic_lengths, test_extern_lr_index = \
    tencent_data_functions.load_concatenate_tencent_data(args.test_data_path, 3)


y_pred = np.array([0.0] * len(test_y))

for i in range(1):
    dfm = DeepFM(field_sizes=field_sizes, total_feature_sizes=total_feature_sizes,
              dynamic_max_len=dynamic_max_len, learning_rate=learning_rate,
              l2_reg=weight_decay, epoch=1, batch_size=batch_size)
    dfm.fit(static_index, dynamic_index, dynamic_lengths, y,
            valid_static_index, valid_dynamic_index, valid_dynamic_lengths, valid_y, combine=False)
    y_pred += dfm.predict(test_static_index, test_dynamic_index, test_dynamic_lengths)

# y_pred /= 4.0

f = open('res.csv', 'wb')
for y in y_pred:
    f.write('%.6f' % (y) + '\n')
f.close()







