"""
Created on Dec 10, 2017
@author: jachin,Nie

A pytorch implementation of NFM

Reference:
[1] Neural Factorization Machines for Sparse Predictive Analytics
    Xiangnan He,School of Computing,National University of Singapore,Singapore 117417,dcshex@nus.edu.sg
    Tat-Seng Chua,School of Computing,National University of Singapore,Singapore 117417,dcscts@nus.edu.sg
"""

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, field_sizes,
                 total_feature_sizes, dynamic_max_len = 30, extern_lr_size = 0, extern_lr_feature_size = 0,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[128], dropout_deep=[1.0, 1.0],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=1, batch_norm_decay=0.995,
                 verbose=True, random_seed=950104,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):

        self.field_sizes = field_sizes
        self.total_field_size = field_sizes[0] + field_sizes[1]
        self.total_feature_sizes = total_feature_sizes
        self.embedding_size = embedding_size
        self.dynamic_max_len = dynamic_max_len
        self.extern_lr_size = extern_lr_size
        self.extern_lr_feature_size = extern_lr_feature_size

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers

        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        #self.train_result, self.valid_result = [], []

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            self.static_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * static_feature_size
            self.dynamic_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_value")  # None * [dynamic_feature_size * max_len]
            self.dynamic_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_value") # None * dynamic_feature_size
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # lr part
            self.static_lr_embs = tf.nn.embedding_lookup(self.weights["static_lr_embeddings"],
                                                         self.static_index) # None * static_feature_size * 1
            self.static_lr_embs = tf.reshape(self.static_lr_embs,[-1, self.field_sizes[0]])
            self.dynamic_lr_embs = tf.nn.embedding_lookup(self.weights["dynamic_lr_embeddings"],
                                                          self.dynamic_index) # None * [dynamic_feature_size * max_len] * 1
            self.dynamic_lr_embs = tf.reshape(self.dynamic_lr_embs,[-1, self.field_sizes[1],
                                                                    self.dynamic_max_len]) # None * dynamic_feature_size * max_len
            self.dynamic_lr_embs = tf.reduce_sum(self.dynamic_lr_embs,axis=2) # None * dynamic_feature_size

            self.dynamic_lr_embs = tf.div(self.dynamic_lr_embs, tf.to_float(self.dynamic_lengths)) # None * dynamic_feature_size

            # ffm part
            self.static_ffm_embs = tf.nn.embedding_lookup(self.weights["static_ffm_embeddings"],
                                                          self.static_index) # None * static_feature_size * [k * F]
            self.dynamic_ffm_embs = tf.nn.embedding_lookup(self.weights["dynamic_ffm_embeddings"],
                                                          self.dynamic_index) # None * [dynamic_feature_size * max_len] * [k * F]
            self.dynamic_ffm_embs = tf.reshape(self.dynamic_ffm_embs, [-1, self.field_sizes[1],
                                                                       self.dynamic_max_len, self.embedding_size * self.total_field_size]) # None * [dynamic_feature_size * max_len] * [k * F]
            self.ffm_mask = tf.sequence_mask(tf.reshape(self.dynamic_lengths,[-1]), maxlen= self.dynamic_max_len) # [None * dynamic_feature] * max_len
            self.ffm_mask = tf.expand_dims(self.ffm_mask, axis=-1) # [None * dynamic_feature] * max_len * 1
            self.ffm_mask = tf.concat([self.ffm_mask for i in range(self.embedding_size * self.total_field_size)], axis = -1) # [None * dynamic_feature] * max_len * [k * F]
            self.dynamic_ffm_embs = tf.reshape(self.dynamic_ffm_embs,[-1, self.dynamic_max_len, self.embedding_size * self.total_field_size]) # [None * dynamic_feature] * max_len * [k * F]
            self.dynamic_ffm_embs = tf.multiply(self.dynamic_ffm_embs, tf.to_float(self.ffm_mask)) # [None * dynamic_feature] * max_len * [k * F]
            self.dynamic_ffm_embs = tf.reshape(tf.reduce_sum(self.dynamic_ffm_embs, axis=1),[-1, self.field_sizes[1],
                                                                                             self.embedding_size * self.total_field_size]) # None * dynamic_feature_size * [k * F]
            self.padding_lengths = tf.concat([tf.expand_dims(self.dynamic_lengths, axis=-1)
                                              for i in range(self.embedding_size * self.total_field_size)],axis=-1) # None * dynamic_feature_size * [k * F]
            self.dynamic_ffm_embs = tf.div(self.dynamic_ffm_embs, tf.to_float(self.padding_lengths)) # None * dynamic_feature_size * [k * F]


            self.ffm_embs_col = tf.reshape(tf.concat([self.static_ffm_embs, self.dynamic_ffm_embs], axis=1),
                                           [-1, self.total_field_size, self.total_field_size, self.embedding_size]) # None * F * F * k
            self.ffm_embs_row = tf.transpose(self.ffm_embs_col, [0, 2, 1, 3]) # None * F * F * k
            self.ffm_embs_out = tf.multiply(self.ffm_embs_col, self.ffm_embs_row) # None *F * F * k
            #self.ffm_embs_out = tf.reshape(self.ffm_embs_out, [-1, self.total_field_size * self.total_field_size * self.embedding_size])

            self.ones = tf.ones_like(self.ffm_embs_out)
            self.op = tf.contrib.linalg.LinearOperatorTriL(tf.transpose(self.ones,[0,3,1,2])) # None *k * F *F
            self.upper_tri_mask = tf.less(tf.transpose(self.op.to_dense(), [0,2,3,1]), self.ones) # None *F * F * k

            self.ffm_embs_out = tf.boolean_mask(self.ffm_embs_out, self.upper_tri_mask) # [None * F * (F-1) * k]
            self.ffm_embs_out = tf.reshape(self.ffm_embs_out, [-1, self.total_field_size * (self.total_field_size-1) // 2
                                                              * self.embedding_size]) # None * [F * (F-1) / 2 * k]


            # ---------- Deep component ----------
            self.y_deep = self.ffm_embs_out #tf.reshape(self.ffm_embs_col,[-1, self.total_field_size * self.total_field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.matmul(self.y_deep, self.weights["layer_%d" % i])
                #self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer

            # ---------- DEEPFFM ----------
            #concat_input = tf.concat([self.static_lr_embs, self.dynamic_lr_embs, self.y_deep], axis=1)
            self.out = tf.add(tf.matmul(self.y_deep, self.weights["concat_projection"]), self.weights["concat_bias"])
            #self.out = tf.add(tf.reshape(tf.reduce_sum(self.out,axis=1),[-1,1]), self.weights['concat_bias'])
            self.out = tf.add(self.out, tf.reshape(tf.reduce_sum(self.static_lr_embs,axis=1),[-1,1]))
            self.out = tf.add(self.out, tf.reshape(tf.reduce_sum(self.dynamic_lr_embs,axis=1),[-1,1]))
            self.out = tf.add(self.out, tf.reshape(tf.reduce_sum(self.ffm_embs_out,axis=1),[-1,1]))

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            elif self.loss_type == "aucloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = self.auc_loss(self.label, self.out)
            elif self.loss_type == "rank":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = self.binary_crossentropy_with_ranking(self.label, self.out)  

            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        self.l2_reg)(self.weights["layer_%d"%i])

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)


            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        #config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def _initialize_weights(self):
        weights = dict()
        # lr part
        weights["static_lr_embeddings"] = tf.Variable(
            tf.random_normal([self.total_feature_sizes[0], 1], 0.0, 0.0001, seed=self.random_seed),
            name="static_lr_embeddings")
        weights["dynamic_lr_embeddings"] = tf.Variable(
            tf.random_normal([self.total_feature_sizes[1], 1], 0.0, 0.0001, seed=self.random_seed),
            name="dynamic_lr_embeddings")
        if self.extern_lr_size:
            weights["extern_lr_embeddings"] = tf.Variable(
            tf.random_normal([self.extern_lr_size, 1], 0.0, 0.001, seed=self.random_seed),
            name="extern_lr_embeddings")

        # embeddings
        weights["static_ffm_embeddings"] = tf.Variable(
            tf.random_normal([self.total_feature_sizes[0], self.embedding_size * self.total_field_size], 0.0, 0.0001, seed=self.random_seed),
            name="static_ffm_embeddings")  # static_feature_size * [K * F]
        weights["dynamic_ffm_embeddings"] = tf.Variable(
            tf.random_normal([self.total_feature_sizes[1], self.embedding_size * self.total_field_size], 0.0, 0.0001, seed=self.random_seed),
            name="dynamic_ffm_embeddings")  # dynamic_feature_size * [K * F]

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.total_field_size * (self.total_field_size -1) // 2 * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            # weights["bias_%d" % i] = tf.Variable(
            #     np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
            #     dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        input_size = self.deep_layers[-1]
        if self.extern_lr_size:
            input_size += self.extern_lr_feature_size
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(-3.5), dtype=np.float32)

        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    def get_batch(self, static_index, dynamic_index, dynamic_lengths, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return static_index[start:end], dynamic_index[start:end], dynamic_lengths[start:end],\
               [[y_] for y_ in y[start:end]]


    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c,d ):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)


    def fit_on_batch(self, static_index, dynamic_index, dynamic_lengths, y):
        feed_dict = {self.static_index: static_index,
                     self.dynamic_index: dynamic_index,
                     self.dynamic_lengths: dynamic_lengths,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def fit(self, train_static_index, train_dynamic_index, train_dynamic_lengths, train_y,
            valid_static_index=None, valid_dynamic_index=None, valid_dynamic_lengths=None, valid_y=None, combine=False,evaluate_train=False):
        """
        :param train_static_index:
        :param train_dynamic_index:
        :param train_dynamic_lengths:
        :param train_y:
        :param valid_static_index:
        :param valid_dynamic_index:
        :param valid_dynamic_lengths:
        :param valid_y:
        :return:
        """
        print("fit begin")
        print(train_static_index.shape, train_dynamic_index.shape, train_dynamic_lengths.shape, train_y.shape)
        has_valid = valid_static_index is not None
        if has_valid:
            print(valid_static_index.shape , valid_dynamic_index.shape, valid_dynamic_lengths.shape, valid_y.shape)
        if has_valid and combine:
            train_static_index = np.concatenate([train_static_index, valid_static_index], axis=0)
            train_dynamic_index = np.concatenate([train_dynamic_index, valid_dynamic_index], axis=0)
            train_dynamic_lengths = np.concatenate([train_dynamic_lengths, valid_dynamic_lengths], axis=0)
            train_y = np.concatenate([train_y, valid_y], axis=0)

        for epoch in range(self.epoch):
            total_loss = 0.0
            total_size = 0.0
            batch_begin_time = time()
            t1 = time()
            # self.shuffle_in_unison_scary(train_static_index, train_dynamic_index,
            #                              train_dynamic_lengths, train_y)
            total_batch = int(len(train_y) / self.batch_size)
            for i in range(total_batch):
                offset = i * self.batch_size
                end = (i+1) * self.batch_size
                end = end if end < len(train_y) else len(train_y)
                static_index_batch, dynamic_index_batch, dynamic_lengths_batch, y_batch\
                    = self.get_batch(train_static_index, train_dynamic_index, train_dynamic_lengths,
                                     train_y, self.batch_size, i)
                batch_loss = self.fit_on_batch(static_index_batch, dynamic_index_batch, dynamic_lengths_batch, y_batch)
                total_loss += batch_loss * (end - offset)
                total_size += end - offset
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.6f time: %.1f s' %
                          (epoch + 1, i + 1, total_loss / total_size, time() - batch_begin_time))
                    total_loss = 0.0
                    total_size = 0.0
                    batch_begin_time = time()

            # evaluate training and validation datasets
            if evaluate_train:
                train_result = self.evaluate(train_static_index, train_dynamic_index,
                                            train_dynamic_lengths, train_y)
            #self.train_result.append(train_result)
            if has_valid and not combine:
                valid_result = self.evaluate(valid_static_index, valid_dynamic_index,
                                             valid_dynamic_lengths, valid_y)
            #    self.valid_result.append(valid_result)
            if self.verbose > 0:
                if has_valid and not combine:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, valid_result, time() - t1))
                if evaluate_train:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, time() - t1))

        print("fit end")


    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False


    def predict(self, static_index, dynamic_index, dynamic_lengths, y = []):
        """
        :param static_index:
        :param dynamic_index:
        :param dynamic_lengths:
        :return:
        """
        print("predict begin")
        # dummy y
        if len(y) == 0:
            dummy_y = [1] * len(static_index)
        else:
            dummy_y = y
        batch_index = 0
        batch_size = 4096
        static_index_batch, dynamic_index_batch, dynamic_lengths_batch, y_batch\
            = self.get_batch(static_index, dynamic_index, dynamic_lengths, dummy_y, batch_size, batch_index)
        y_pred = None
        total_loss = 0.0
        total_size = 0.0
        while len(static_index_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.static_index: static_index_batch,
                         self.dynamic_index: dynamic_index_batch,
                         self.dynamic_lengths: dynamic_lengths_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out, batch_loss = self.sess.run((self.out, self.loss), feed_dict=feed_dict)
            total_loss += batch_loss * num_batch
            total_size += num_batch
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            static_index_batch, dynamic_index_batch, dynamic_lengths_batch, y_batch \
                = self.get_batch(static_index, dynamic_index, dynamic_lengths, dummy_y, batch_size, batch_index)
        print("valid logloss is %.6f" % (total_loss / total_size))
        print("predict end")
        return y_pred


    def evaluate(self, static_index, dynamic_index, dynamic_lengths, y):
        """
        :param static_index:
        :param dynamic_index:
        :param dynamic_lengths:
        :param y:
        :return:
        """
        print("evaluate begin")
        print("predicting ing")
        b_time = time()
        y_pred = self.predict(static_index, dynamic_index, dynamic_lengths, y)
        print("predicting costs %.1f" %(time()- b_time))
        print("counting eval ing")
        b_time = time()
        res =  self.eval_metric(y, y_pred)
        print("counting eval cost %.1f" %(time()- b_time))
        print("evaluate end")
        return res
