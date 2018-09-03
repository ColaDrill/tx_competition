import os

import tensorflow as tf
from models import utils
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
from sklearn.metrics import log_loss
import time
import random

from tensorflow.contrib.layers.python.layers import batch_norm
import numpy as np

os.system('pip install pickle')
import pickle
import pandas as pd
from sklearn import metrics
import random

import time


def log(x):
    os.system('echo ' + str(x))


gloab_auc = {}


def print_step_info(prefix, global_step, info):
    utils.print_out(
        "%sstep %d lr %g logloss %.6f gN %.2f, %s" %
        (prefix, global_step, info["learning_rate"],
         info["train_ppl"], info["avg_grad_norm"], time.ctime()))


class TextIterator:
    def __init__(self, hparams, mode, batch_size=None):
        self.single_file = {}
        self.multi_file = {}
        self.num_file = {}

        for s in hparams.single_features:
            self.single_file[s] = open(hparams.data_path + '/' + mode + '/' + s, 'r')
        for s in hparams.multi_features:
            self.multi_file[s] = open(hparams.data_path + '/' + mode + '/' + s, 'r')
        for s in hparams.num_features:
            self.num_file[s] = open(hparams.data_path + '/' + mode + '/' + s, 'r')

        self.label_file = open(hparams.data_path + '/' + mode + '/label', 'r')
        self.word2index = pickle.load(open(hparams.data_path + '/dic.pkl', 'rb'))
        hparams.dict = self.word2index

        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = hparams.batch_size

        if 'train' in mode:
            self.idx = hparams.idx
            self.all_process = int(hparams.all_process)
        else:
            self.idx = 0
            self.all_process = 1
        self.hparams = hparams
        random.seed(2018)

    def reset(self):
        self.idx = (self.idx + 1) % self.all_process
        for s in self.single_file:
            self.single_file[s].seek(0)
        for s in self.multi_file:
            self.multi_file[s].seek(0)
        for s in self.num_file:
            self.num_file[s].seek(0)
        self.label_file.seek(0)

    def next(self):
        single_ids = {}
        multi_ids = {}
        labels = []
        multi_length = {}
        num_ids = {}
        for s in self.hparams.single_features:
            temp = []
            for i in range(self.batch_size * self.all_process):
                try:
                    ss = self.single_file[s].readline().strip()
                except:
                    pass
                if ss == "":
                    break
                if i % self.all_process != self.idx:
                    continue
                try:
                    if int(ss) in self.hparams.dict[s]:
                        ss = int(ss)
                except:
                    pass

                try:
                    temp.append(self.hparams.dict[s][ss])
                except:
                    temp.append(0)
            single_ids[s] = temp

        for s in self.hparams.num_features:
            temp = []
            for i in range(self.batch_size * self.all_process):
                try:
                    ss = self.num_file[s].readline().strip()
                except:
                    pass
                if ss == "":
                    break
                if i % self.all_process != self.idx:
                    continue
                temp.append(float(ss))
            num_ids[s] = temp

        max_len = 0
        for s in self.hparams.multi_features:
            temp = []
            temp_len = []
            for i in range(self.batch_size * self.all_process):
                try:
                    mm = self.multi_file[s].readline().strip()
                except:
                    pass
                if mm == "":
                    break
                if i % self.all_process != self.idx:
                    continue
                t = []
                t_len = []
                for m in mm.split():
                    try:
                        t.append(self.hparams.dict[s][m])
                    except:
                        t.append(0)
                temp.append(t)
                temp_len.append(len(t))
                max_len = max(len(t), max_len)
            multi_length[s] = temp_len
            multi_ids[s] = temp
        max_len = 100
        for t in multi_ids:
            for i in range(len(multi_ids[t])):
                if len(multi_ids[t][i]) <= max_len:
                    multi_ids[t][i] += (max_len - len(multi_ids[t][i])) * [1]
                else:
                    multi_ids[t][i] = random.sample(multi_ids[t][i], max_len)

        for i in range(self.batch_size * self.all_process):
            try:
                ss = self.label_file.readline().strip()
            except:
                pass
            if ss == "":
                break
            if i % self.all_process != self.idx:
                continue
            labels.append(int(float(ss)))

        if len(single_ids['aid']) == 0:
            self.reset()
            raise StopIteration
        return (labels, single_ids, multi_ids, num_ids, multi_length)


temp = []


class Model(object):
    def __init__(self, hparams):
        self.dropout_keep_deep = [1, 1, 1, 1, 1, 1]
        self.f1 = hparams.aid.copy()
        self.f2 = hparams.user.copy()
        self.batch_norm_decay = 0.9
        self.single_ids = {}
        self.num_ids = {}
        self.multi_ids = {}
        self.multi_mask = {}
        self.emb_v1 = {}
        self.emb_v2 = {}
        self.emb_combine_aid_v2 = {}
        self.norm_num = {}
        self.cross_params = []
        self.layer_params = []
        self.length = {}
        self.bias = tf.Variable(tf.truncated_normal(shape=[1], mean=0.0, stddev=0.0001), name='bias')
        self.use_dropout = tf.placeholder(tf.bool)
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.feature_all_length = len(hparams.single_features) + len(hparams.multi_features)
        feature_all_length = self.feature_all_length
        self.label = tf.placeholder(shape=(None), dtype=tf.float32)
        norm = hparams.norm
        for s in hparams.num_features:
            self.num_ids[s] = tf.placeholder(shape=(None,), dtype=tf.float32)
            if s in norm:
                self.norm_num[s] = self.batch_norm_layer(tf.reshape(self.num_ids[s], [-1, 1]), self.use_dropout, s)
            else:
                self.norm_num[s] = self.num_ids[s][:, None]

        for s in hparams.single_features:
            self.single_ids[s] = tf.placeholder(shape=(None,), dtype=tf.int32)
            # 保留一个 输入层
            self.emb_v1[s] = tf.Variable(
                tf.truncated_normal(shape=[len(hparams.dict[s]) + 2, 1], mean=0.0, stddev=0.0001), name='emb_v1_' + s)
            # 如果是广告特征
            if s in self.f1:
                # 每个广告生成一个对应用户的field的参数。最后 投射到k个维度
                self.emb_v2[s] = tf.Variable(
                    tf.truncated_normal(shape=[len(hparams.dict[s]) + 2, len(self.f2), hparams.k], mean=0.0,
                                        stddev=0.0001), name='emb_v2_' + s)
            elif s in self.f2:
                # 为每个用户 生成 对应广告field的参数 。最后投射到k个维度
                self.emb_v2[s] = tf.Variable(
                    tf.truncated_normal(shape=[len(hparams.dict[s]) + 2, len(self.f1), hparams.k], mean=0.0,
                                        stddev=0.0001), name='emb_v2_' + s)

        for s in hparams.multi_features:
            if len(hparams.dict[s]) + 2 >= 100000:
                with tf.device('/cpu:0'):
                    self.multi_ids[s] = tf.placeholder(shape=(None, None), dtype=tf.int32)
                    self.length[s] = tf.placeholder(shape=(None,), dtype=tf.int32)
                    self.multi_mask[s] = tf.sequence_mask(self.length[s], 100, dtype=tf.float32)
                    self.emb_v1[s] = tf.get_variable(shape=[len(hparams.dict[s]) + 2, 1], initializer=initializer,
                                                     name='emb_v1_' + s)
                    if s in self.f1:
                        # 估计是打印不出来了
                        print(666666666)
                        self.emb_v2[s] = tf.Variable(
                            tf.truncated_normal(shape=[len(hparams.dict[s]) + 2, len(self.f2), hparams.k], mean=0.0,
                                                stddev=0.0001), name='emb_v2_' + s)
                    elif s in self.f2:
                        self.emb_v2[s] = tf.Variable(
                            tf.truncated_normal(shape=[len(hparams.dict[s]) + 2, len(self.f1), hparams.k], mean=0.0,
                                                stddev=0.0001), name='emb_v2_' + s)
            else:
                self.multi_ids[s] = tf.placeholder(shape=(None, None), dtype=tf.int32)
                self.length[s] = tf.placeholder(shape=(None,), dtype=tf.int32)
                self.multi_mask[s] = tf.sequence_mask(self.length[s], 100, dtype=tf.float32)
                self.emb_v1[s] = tf.get_variable(shape=[len(hparams.dict[s]) + 2, 1], initializer=initializer,
                                                 name='emb_v1_' + s)
                if s in self.f1:
                    self.emb_v2[s] = tf.Variable(
                        tf.truncated_normal(shape=[len(hparams.dict[s]) + 2, len(self.f2), hparams.k], mean=0.0,
                                            stddev=0.0001), name='emb_v2_' + s)
                elif s in self.f2:
                    self.emb_v2[s] = tf.Variable(
                        tf.truncated_normal(shape=[len(hparams.dict[s]) + 2, len(self.f1), hparams.k], mean=0.0,
                                            stddev=0.0001), name='emb_v2_' + s)

        self.build_graph(hparams)
        self.optimizer(hparams)
        params = tf.trainable_variables()

    def build_graph(self, hparams):
        # 生成权重的方法
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        # lr wide 部分
        emb_inp_v1 = {}
        # 输入层，每个输入分配一个权重
        for s in hparams.single_features:
            emb_inp_v1[s] = tf.gather(self.emb_v1[s], self.single_ids[s])
        # 每个输入取一个avgpool 池化
        for s in hparams.multi_features:
            emb_inp_v1[s] = tf.reduce_sum(
                tf.gather(self.emb_v1[s], self.multi_ids[s]) * self.multi_mask[s][:, :, None],
                axis=1) / tf.cast(self.length[s], tf.float32)[:, None]

        # 线性层合并
        emb_inp_v1 = tf.concat([emb_inp_v1[s] for s in emb_inp_v1], -1)
        # 线性求和
        w1 = tf.reduce_sum(emb_inp_v1, [-1])

        # poly
        # 二阶交叉  ffm层
        emb_inp_v2 = {}
        for s in hparams.single_features:
            if s in hparams.aid + hparams.user:
                emb_inp_v2[s] = tf.gather(self.emb_v2[s], self.single_ids[s])

        # embedding
        for s in hparams.multi_features:
            emb_inp_v2[s] = tf.reduce_sum(
                tf.gather(self.emb_v2[s], self.multi_ids[s]) * self.multi_mask[s][:, :, None, None],
                axis=1) / tf.cast(self.length[s], tf.float32)[:, None, None]
        # 使用maxpool
        # for s in hparams.multi_features:
        #     emb_inp_v2[s] = tf.reduce_sum(
        #         tf.gather(self.emb_v2[s], self.multi_ids[s]) * self.multi_mask[s][:, :, None, None],
        #         axis=1) / tf.cast(self.length[s], tf.float32)[:, None, None]
        x = [[], []]

        for s in self.f1:
            x[0].append(emb_inp_v2[s][:, None, :, :])
        for s in self.f2:
            x[1].append(emb_inp_v2[s][:, :, None, :])
        x[0] = tf.concat(x[0], 1)
        x[1] = tf.concat(x[1], 2)

        emb_rep_v = x[0] * x[1]
        emb_rep_v = tf.reshape(emb_rep_v, [-1, len(self.f1) * len(self.f2) * hparams.k])
        emb_rep = emb_rep_v
        temp = []
        temp.append(emb_rep)
        for s in hparams.num_features:
            temp.append(self.norm_num[s])
        emb_rep = tf.concat(temp, -1)

        # num_layer = len(hparams.hidden_size)
        #
        # # 输入层
        # input_size = len(self.f1) * len(self.f2) * hparams.k + len(hparams.num_features)
        # glorot = np.sqrt(2.0 / (input_size + hparams.hidden_size[0]))
        # weights = {}
        # weights['w_0'] = tf.Variable(
        #     np.random.normal(loc=0, scale=glorot, size=(input_size, hparams.hidden_size[0])), dtype=np.float32)
        # # 隐藏层
        # for i in range(1, num_layer):
        #     glorot = np.sqrt(2.0 / (hparams.hidden_size[i - 1] + hparams.hidden_size[i]))
        #     weights['w_' + str(i)] = tf.Variable(
        #         np.random.normal(loc=0, scale=glorot, size=(hparams.hidden_size[i - 1], hparams.hidden_size[i])),
        #         dtype=np.float32)
        #
        # weights['w_' + str(num_layer)] = tf.get_variable("w_" + str(num_layer), [hparams.hidden_size[-1], 1],
        #                                                  initializer=initializer)
        #
        # weights['b_' + str(num_layer)] = tf.Variable(tf.constant(-3.5), dtype=np.float32)
        # hidden_outputs = emb_rep
        # for i in range(num_layer):
        #     print('add layer')
        #     hidden_outputs = tf.matmul(hidden_outputs, weights['w_' + str(i)])
        #     hidden_outputs = self.batch_norm_layer(hidden_outputs, self.use_dropout, 'hidden_' + str(i))
        #     hidden_outputs = tf.nn.dropout(tf.nn.relu(hidden_outputs), 0.99)
        #
        # out = tf.matmul(hidden_outputs, weights['w_' + str(num_layer)]) + weights['b_' + str(num_layer)]
        #
        # score = w1 + out[:, 0]

        W_1 = tf.get_variable("W_1", [emb_rep.get_shape()[-1], hparams.hidden_size[0]], initializer=initializer)

        hidden_outputs_1 = tf.tensordot(emb_rep, W_1, [[-1], [0]])
        hidden_outputs_1 = self.batch_norm_layer(hidden_outputs_1, self.use_dropout, 'train_1')
        hidden_outputs_1 = tf.nn.relu(hidden_outputs_1)

        W_2 = tf.get_variable("W_2", [hparams.hidden_size[0], hparams.hidden_size[1]], initializer=initializer)

        hidden_outputs_2 = tf.tensordot(hidden_outputs_1, W_2, [[-1], [0]])
        hidden_outputs_2 = self.batch_norm_layer(hidden_outputs_2, self.use_dropout, 'train_2')
        hidden_outputs_2 = tf.nn.relu(hidden_outputs_2)

        W_3 = tf.get_variable("W_3", [hparams.hidden_size[1], 1], initializer=initializer)
        b_3 = tf.Variable(tf.constant(-3.5), dtype=np.float32)

        w_3 = tf.tensordot(hidden_outputs_2, W_3, [[-1], [0]]) + b_3

        score = w1 + w_3[:, 0]

        self.prob = tf.sigmoid(score)
        logit_1 = tf.log(self.prob)
        logit_0 = tf.log(1 - self.prob)
        self.loss = -tf.reduce_mean(self.label * logit_1 + (1 - self.label) * logit_0)
        self.cost = -tf.reduce_mean(self.label * logit_1 + (1 - self.label) * logit_0)
        self.saver_ffm = tf.train.Saver()

    def optimizer(self, hparams):
        self.lrate = tf.Variable(hparams.learning_rate, trainable=False)
        if hparams.optimizer == "sgd":
            opt = tf.train.GradientDescentOptimizer(self.lrate)
        elif hparams.optimizer == "adam":
            opt = tf.train.AdamOptimizer(self.lrate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif hparams.optimizer == "ada":
            opt = tf.train.AdagradOptimizer(learning_rate=self.lrate, initial_accumulator_value=1e-8)
        params = tf.trainable_variables()

        gradients = tf.gradients(self.cost, params, colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
        self.grad_norm = gradient_norm
        self.update = opt.apply_gradients(zip(clipped_grads, params))

    def dey_lrate(self, sess, lrate):
        sess.run(tf.assign(self.lrate, lrate))

    def train(self, sess, iterator):
        data = iterator.next()
        dic = {}
        for s in self.single_ids:
            dic[self.single_ids[s]] = data[1][s]

        for s in self.multi_ids:
            dic[self.multi_ids[s]] = data[2][s]
            dic[self.length[s]] = data[4][s]

        for s in self.num_ids:
            dic[self.num_ids[s]] = data[3][s]

        dic[self.use_dropout] = True
        dic[self.label] = data[0]
        return sess.run([self.cost, self.update, self.grad_norm], feed_dict=dic)

    def infer(self, sess, iterator, label, aid):
        data = iterator.next()
        label.extend(data[0])
        aid.extend(data[1]['aid'])
        dic = {}
        for s in self.single_ids:
            dic[self.single_ids[s]] = data[1][s]

        for s in self.multi_ids:
            dic[self.multi_ids[s]] = data[2][s]
            dic[self.length[s]] = data[4][s]
        for s in self.num_ids:
            dic[self.num_ids[s]] = data[3][s]

        dic[self.use_dropout] = False
        return sess.run(self.prob, feed_dict=dic)

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


def train(hparams):
    tf.set_random_seed(2018)
    random.seed(2018)
    train_iterator = TextIterator(hparams, mode="train")
    # 加速验证
    test_iterator = TextIterator(hparams, mode="test", batch_size=4096)
    predict_iterator = TextIterator(hparams, mode="predict", batch_size=4096)
    model = Model(hparams)
    config_proto = tf.ConfigProto(log_device_placement=0, allow_soft_placement=0)
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    global_step = 0
    train_loss = 0
    train_norm = 0
    best_loss = 0
    dey_cont = 0
    pay_cont = 0
    epoch = False
    epoch_cont = 0
    start_time = time.time()
    if hparams.mode != 'train':
        model.saver_ffm.restore(sess, os.path.join(hparams.path, 'model_' + str(hparams.idx)))
    else:
        while True:
            try:
                cost, _, norm = model.train(sess, train_iterator)
            except StopIteration:
                continue
            global_step += 1
            train_loss += cost
            train_norm += norm
            if global_step % hparams.num_display_steps == 0 or global_step >= hparams.epoch:
                info = {}
                info['learning_rate'] = hparams.learning_rate
                info["train_ppl"] = train_loss / hparams.num_display_steps
                info["avg_grad_norm"] = train_norm / hparams.num_display_steps
                train_loss = 0
                train_norm = 0
                print_step_info("  ", global_step, info)
                # 只验证一次
                if global_step >= hparams.epoch:
                    epoch = False
                    preds = []
                    label = []
                    aid = []
                    while True:
                        try:
                            pred = model.infer(sess, test_iterator, label, aid)
                            preds += list(pred)
                        except StopIteration:
                            break
                    res = {}
                    for i in range(len(aid)):
                        if aid[i] not in res:
                            res[aid[i]] = {}
                            res[aid[i]]['label'] = []
                            res[aid[i]]['pred'] = []
                        res[aid[i]]['label'].append(label[i] + 1)
                        res[aid[i]]['pred'].append(preds[i])
                    auc = []
                    for u in res:
                        fpr, tpr, thresholds = metrics.roc_curve(res[u]['label'], res[u]['pred'], pos_label=2)
                        loss_ = metrics.auc(fpr, tpr)
                        if np.isnan(loss_):
                            continue
                        gloab_auc[u] = loss_
                        auc.append(loss_)
                    loss_ = np.mean(auc)
                    if best_loss <= loss_:
                        model.saver_ffm.save(sess, os.path.join(hparams.path, 'model_' + str(hparams.sub_name)))
                        best_loss = loss_
                        T = (time.time() - start_time)
                        start_time = time.time()
                        utils.print_out("# Epcho-time %.2fs Eval AUC %.6f. Best AUC %.6f." % (T, loss_, best_loss))
                    else:
                        utils.print_out("# Epcho-time %.2fs Eval AUC %.6f. Best AUC %.6f." % (T, loss_, best_loss))
                        model.saver_ffm.restore(sess, os.path.join(hparams.path, 'model_' + str(hparams.sub_name)))

                    if global_step >= hparams.epoch:
                        model.saver_ffm.restore(sess, os.path.join(hparams.path, 'model_' + str(hparams.sub_name)))
                        break

    test_iterator = TextIterator(hparams, mode="test", batch_size=hparams.batch_size)
    # 拿测试集 训练
    print('train with test !')
    while True:
        try:
            cost, _, norm = model.train(sess, test_iterator)
        except StopIteration:
            break
    print('train with test done!')
    print("predict inference ...")
    preds = []
    label = []
    aid = []
    while True:
        try:
            pred = model.infer(sess, predict_iterator, label, aid)
            preds += list(pred)
        except StopIteration:
            break
    print('predict inference done!')
    return preds


def run(file_path, feature_path, root_path, type='mean'):
    hparams = tf.contrib.training.HParams(
        k=8,
        batch_size=4096,
        optimizer="adam",
        learning_rate=0.0002,
        num_display_steps=100,
        num_eval_steps=6000,
        l2=0.000002,
        hidden_size=[256, 128],
        all_process=1,
        idx=0,
        epoch=int(44628906 // 4096),
        mode='train',
        data_path=feature_path,
        sub_name='final',
        single_features=['aid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId',
                         'productType', 'age', 'gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house'],
        multi_features=['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                        'topic2', 'topic3', 'appIdAction', 'appIdInstall', 'marriageStatus', 'ct', 'os'],
    )

    # 信息类
    inf_list = []
    for i in ['aid']:
        for j in ['age', 'gender', 'education', 'LBS', 'consumptionAbility', 'house',
                  'marriageStatus', 'ct', 'os']:
            inf_list.append('new_inf_' + i + '_' + j)

    for i in ['aid']:
        for j in ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                  'topic2', 'topic3', 'appIdInstall', 'appIdAction']:
            inf_list.append('new_inf_' + i + '_' + j + '_' + type)

    # length
    for i in ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
              'topic2', 'topic3', 'appIdAction', 'appIdInstall']:
        inf_list.append('len_' + i)

    # uid count
    hparams.single_features = hparams.single_features + inf_list + ['new_count_campaignId_uid', 'new_count_uid',
                                                                    'new_count_productId_uid',
                                                                    'new_count_adCategoryId_uid',
                                                                    'new_count_productType_uid',
                                                                    'new_count_creativeSize_uid',
                                                                    'new_count_advertiserId_uid',
                                                                    'new_count_creativeSize_productType_uid',
                                                                    'new_count_adCategoryId_productType_uid']

    print(inf_list)

    hparams.path = root_path + '/models'

    utils.print_hparams(hparams)

    all = hparams.single_features + hparams.multi_features

    hparams.aid = all
    hparams.user = all

    hparams.norm = []
    hparams.num_features = hparams.norm + []

    preds = train(hparams)

    predict_df = pd.read_csv(file_path + '/test2.csv')
    predict_df['score'] = preds[-len(predict_df):]
    predict_df['score'] = predict_df['score'].apply(lambda x: float('%.6f' % x))
    predict_df[['aid', 'uid', 'score']].to_csv(file_path + '/' + type + '.csv', index=False)
