import glob
from pyAudioAnalysis import audioFeatureExtraction
from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
import os

def feature34_from_wav(wavfile):
    """
    读取wav音频
    返回34维度特征 shape（n,34）
    """
    rate,date = get_audio_data(wavfile)
    F = audioFeatureExtraction.stFeatureExtraction(date, rate, 0.05*rate, 0.025*rate)#0.05,0.025   #0.025*rate, 0.01*rate
    Z = np.transpose(F[0])
    return Z


def feature34_from_pcm(rate,pcm):
    """
    读取wav音频
    返回34维度特征 shape（n,34）
    """
    data = np.fromstring(pcm,np.short)
    F = audioFeatureExtraction.stFeatureExtraction(data, rate, 0.05*rate, 0.025*rate)#0.05,0.025   #0.025*rate, 0.01*rate
    Z = np.transpose(F[0])
    return Z



def get_audio_data(_file):
    """
    获取音频rate,array，支持wav和mp3
    """
    file_extension = os.path.splitext(_file)[1]#获取文件后缀名
    if file_extension.__contains__("mp3"):
        sound = AudioSegment.from_mp3(_file)
        rate = sound.frame_rate
        array = sound.get_array_of_samples()
        array = np.array(array)
        return  rate,array
    if file_extension.__contains__("wav"):
        rate, array = wavfile.read(_file)
        return  rate,array



def get_dir_feature34_from_audio(dirpath):
    """
    获取一个文件夹下所有的音频文件的特征
    返回 list<feature34>
    """
    features = []
    files = glob.glob(dirpath)
    for audio_file in files:
        feature = feature34_from_wav(audio_file)
        features.append(feature);
    return features


def get_dir_feature34_from_audio_with_file(dirpath):
    """
    获取一个文件夹下所有的音频文件的特征
    返回 list<feature34>
    """
    #features = []
    #audiofiles = []
    files = glob.glob(dirpath)
    for audio_file in files:
        feature = feature34_from_wav(audio_file)
        yield feature,audio_file
        #features.append(feature);
        #audiofiles.append(audio_file)
    #return features,audiofiles



