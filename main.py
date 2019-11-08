# -*- coding: utf-8 -*-

from algorithm.hps import vad_file,get_vad_segments
import keras
import os
from audio import feature
from sklearn import preprocessing
from sklearn.model_selection  import train_test_split
import numpy as np
from model import lstm
from tools import file_tools
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from time import time,strftime,localtime
from numba import jit



"""
一些配置
"""
model_path =  './data/model/your_model.h5'#训练好的模型（预测用）
short_wav_path = './data/short_audio/'#短音频存放地址
train = True#训练或预测


def do_train():
    """
    cnn-lstm 训练模型
    三分类：男,女，其他噪音
    """
    #获取特征和标签
    path, dirs, files = os.walk(short_wav_path).__next__()
    if os.path.exists(path+'x.npy') and os.path.exists(path+'y.npy'):
        print('读取特征文件！')
        X = np.load(path+'x.npy')
        Y = np.load(path+'y.npy')
        X = np.asarray(X).astype(np.float32)
        Y = np.asarray(Y).astype(np.float32)
    else:
        print('生成特征文件')
        X = []
        strY = []
        for dir in dirs:
            _feature = feature.get_dir_feature34_from_audio(path+dir+'\\*.wav')
            _truncated = keras.preprocessing.sequence.pad_sequences(_feature, maxlen=300, truncating='post',padding = 'post')
            _label =  [dir for i in range(0,len(_feature))]
            X.extend(_truncated)
            strY.extend(_label)
        lb = preprocessing.LabelBinarizer()
        Y = lb.fit_transform(strY)
        _save = ''
        for s in set(strY):
            _save=_save+str(s)+ str(lb.transform([s]))
        file_tools.save_to_file(path+'label.txt',_save)
        X = np.asarray(X).astype(np.float32)
        Y = np.asarray(Y).astype(np.float32)
        np.save(path+'x.npy', X)
        np.save(path+'y.npy', Y)
    #print(X.shape, Y.shape)
    #拆分测试集和训练集
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)


    model = lstm.cnn_lstm((300, 34),3)

    filepath =  path+strftime('%Y.%m.%d',localtime(time()))+".h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    #model = cnn.cnn_multi_attention()
    #model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adamax, metrics=[categorical_accuracy])
    #model.compile(loss='mean_squared_error', optimizer="adam")
    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=200,shuffle=True,class_weight='auto',callbacks=callbacks_list)#callbacks=[early_stopping],
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores)


def model_predict(wavfile, model, storeVad=False):
    """
    先切分，然後每段vad做分類，取最終結果
    """
    indexs = np.zeros((3))
    if storeVad is True:
        vad_paths = vad_file(wavfile, 3)
        for vad in vad_paths:
            f = feature.feature34_from_wav(vad)
            f = np.asarray([f])
            _truncated = keras.preprocessing.sequence.pad_sequences(f, maxlen=300, truncating='post',
                                                                       padding='post')
            p = model.predict(_truncated)
            index = np.argmax(p[0])
            indexs[index] +=1
    else:
        rate,pcms = get_vad_segments(wavfile, 3)
        for pcm in pcms:
            f = feature.feature34_from_pcm(rate,pcm)
            f = np.asarray([f])
            _truncated = keras.preprocessing.sequence.pad_sequences(f, maxlen=300, truncating='post',
                                                                       padding='post')
            p = model.predict(_truncated)
            index = np.argmax(p[0])
            indexs[index] +=1
    return indexs



label = {0: "Blank",
         1:"LittleNoise",
         2:"Noise",
         3:"FemaleWithNoise",
         4:"MaleWithNoise",
         5:"Female(LowProbability)",
         6:"Male(LowProbability)",
         7:"Female(HighProbability)",
         8:"Male(HighProbability)"}

@jit(nopython=True)
def handle_result(weight):
    """权重相加后处理"""
    if weight[0] > weight[2]:
        return  5
    elif weight[0] < weight[2]:
        return 6
    else:
        return 2




def predict(model,long_audio_path):
    """
    功能：性别预测
    参数：model 加载的模型
         long_audio_path 长音频文本
    返回 权重，最终标签
    """
    weight = model_predict(long_audio_path, model)
    info_index = handle_result(weight)
    return weight,label[info_index]

if __name__ ==  '__main__':
    if train == True:
        do_train();
    else:
        model = keras.models.load_model(model_path)
        predict(model,"xxxxx.wav")#




