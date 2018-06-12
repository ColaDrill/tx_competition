from scipy import sparse
from keras.models import Sequential
from keras.layers import Dense
import scipy.io as sio
from sklearn.metrics import roc_curve, auc
import os
from keras import backend as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"  


def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):  
    y_pred = K.cast(y_pred >= threshold, 'float32')  
    # N = total number of negative labels  
    N = K.sum(1 - y_true)  
    # FP = total number of false alerts, alerts from the negative class labels  
    FP = K.sum(y_pred - y_pred * y_true)  
    return FP/N  

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):  
    y_pred = K.cast(y_pred >= threshold, 'float32')  
    # P = total number of positive labels  
    P = K.sum(y_true)  
    # TP = total number of correct alerts, alerts from the positive class labels  
    TP = K.sum(y_pred * y_true)  
    return TP/P 

def auc(y_true, y_pred):  
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)  
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)  
    binSizes = -(pfas[1:]-pfas[:-1])  
    s = ptas*binSizes  
    return K.sum(s, axis=0)  

def dnn(hidden_layers,feat_dim,output_dim):
    model = Sequential()
    model.add(Dense(hidden_layers[0],input_dim=feat_dim,
                    init='uniform',activation='relu'))
    for l in hidden_layers:
        model.add(Dense(l,init='uniform',activation='relu'))
    model.add(Dense(output_dim,init='uniform',activation='sigmoid'))    
    return model


if __name__ == '__main__':
    train_x = sparse.load_npz('nntest.npz').tocsc()
    train_y = sio.loadmat('nny.mat')['y'][0]
    model = dnn([100,30,10],10,1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])  
    checkpointer = ModelCheckpoint(filepath='weights4.h5', verbose=1, save_best_only=True)
    model.fit(train_x,train_y,nb_epoch=150,callbacks=[checkpointer],validation_split=0.1, batch_size=10)
