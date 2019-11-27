# -*- coding: utf-8 -*-
import contextlib
import wave
import subprocess
from scipy import *
import sys
import webrtcvad
from pydub import AudioSegment
import numpy as np
import collections
import os
import matplotlib.pyplot as plt
import glob


maleFemaleFreq = [120, 232]
TS=3 #time for simple method

humanVoiceMinMAx = [80,255]
maleMinMax=[60,160]
femaleMinMax=[180,225]
HPSLoop=5


def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        #assert sample_rate in (8000, 16000, 32000)

        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        #sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                #sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    """
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    """
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def main_vad(audio,sample_rate,vad = 2,dirName = ""):
    vad = webrtcvad.Vad(vad)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    #rates = []
    datas = []
    for i, segment in enumerate(segments):
        #"""
        path = str(i)+ '.wav'
        #print(segment)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        write_wave(dirName+path, segment, sample_rate)
        from scipy.io import wavfile
        sample_rate, segment = wavfile.read(dirName+path)
        #print(segment)
        #"""
        datas.append( segment)
    return datas


def vad_file(file,vad = 2,dirName = ""):
    """
    切分到独立的文件夹,返回路径，如果之前已经切分，将不会执行Vad操作，需要先刪除vad文件夾
    file：wav文件
    vad：等级 1 2 3  越高越严格 音频纯度越好 但也会存在丢失情况
    dirName：存放的
    """
    if len(dirName) == 0:
        dirName = os.path.split(file)[0]
    dir = dirName + '\\' + os.path.basename(file).split('.')[0]
    vad_paths = glob.glob(os.path.join(dir,"*.wav"))
    if len(vad_paths)>0:
        return vad_paths
    sample_rate, segments = get_vad_segments(file, vad)
    vad_paths = []

    for i, segment in enumerate(segments):
        path = str(i)+ '.wav'

        if not os.path.exists(dir):
            os.makedirs(dir)
        write_wave(os.path.join(dir,path), segment, sample_rate)
        vad_paths.append(os.path.join(dir,path))
    return vad_paths


def get_vad_segments(file,vad=2):
    audio, sample_rate = read_wave(file)
    vad = webrtcvad.Vad(vad)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    return sample_rate,segments


def vad_one_dir(file,vad = 2,dirName = ""):
    """
    切分到同一个的文件夹
    """
    sample_rate,segments = get_vad_segments(file,vad)
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    for i, segment in enumerate(segments):
        path = dirName+os.path.basename(file)+'_'+str(i)+ '.wav'
        write_wave(path, segment, sample_rate)


def convert(remote_filename):
    infile = remote_filename
    # get filename without subfix
    fulllist = remote_filename.split(".")
    namelist = fulllist[-2].split("/")
    if len(namelist) == 1:
        outname = namelist[0]
    else:
        outname = namelist[-1]
    outname = outname + '.wav'
    outfile = ''.join(outname.split('?'))
    subprocess.call(['ffmpeg', '-y', '-i', infile, '-vn', '-acodec', 'pcm_s16le', '-f', 'wav', outfile])
    return outfile

def splitWAV(fileName,interval,dirName = ""):
    wav = AudioSegment.from_wav(fileName)
    splitedWavs =  [wav[i*1000:(i+interval)*1000] for i in range(0,int(wav.__len__()*0.001),interval)]
    for j, _wav in enumerate(splitedWavs):
        _wav.export(dirName+str(j)+".wav", format="wav")


def splitsToOneSecondData(rate,data):
    totalTime = len(data) / rate
    partLen = int(rate)
    parts = [data[i * partLen:(i + 1) * partLen] for i in range(int(totalTime))]
    if len(parts) == 0:
        parts = [data]
    return parts


def avg_hps(sample_rate, datas, maleMinMax=[60, 110], femaleMinMax=[180, 270]):
    """
    每段的统计类型
    """
    total_statistics = 0
    for pcm_data in datas:
        totalTime = len(pcm_data) / sample_rate
        #HPSLoopVad = int((totalTime) * 1.666)
        # HPSLoopVad = max(HPSLoopVad,5)#0.73
        HPSLoopVad = 5  #
        #t1  = time.time()

        #_resoult, _sumMale, _sumFemale = HPS(sample_rate, pcm_data, hpsLoop=HPSLoopVad, maleMinMax=maleMinMax,femaleMinMax=femaleMinMax)

        if totalTime < 1:
            continue
        _resoult = hps_orgin(sample_rate, pcm_data, totalTime)
        total_statistics += _resoult;
    print("total_statistics",total_statistics)
    return total_statistics



def hps(sample_rate, pcm_data, hpsLoop=5, maleMinMax=[60, 160], femaleMinMax=[180, 270]):
    """

    :param sample_rate:
    :param pcm_data:
    :param hpsLoop:
    :param maleMinMax:
    :param femaleMinMax:
    :return:     -1 女  1 男  0 无
    """
    resultParts = []

    totalTime = len(pcm_data) / sample_rate
    partLen = int(sample_rate / 2)
    parts = [pcm_data[i * partLen:(i + 1) * partLen] for i in range(int(totalTime * 2))]
    # 总共分割了T份数据
    for data in parts:
        if (len(data) == 0): continue
        # 汉明窗
        window = np.hamming(len(data))
        data = data * window
        # 变换后的振幅
        fftV = abs(fft(data)) / sample_rate
        fftR = copy(fftV)
        for i in range(2, hpsLoop):
            # 按i个步长去取特征振幅
            tab = copy(fftV[::i])
            for j, dx in enumerate(tab):
                tab[j] = max(fftV[int(j * i):int((j + 1) * i)])
            fftR = fftR[:len(tab)] * tab
        resultParts.append(fftR)
    if len(resultParts) == 0:
        return 0, 0, 0
    # 取中间的处理结果
    r_len = len(resultParts[int(len(resultParts) / 2)])
    result = [0] * r_len
    # resultParts长度大小和Parts一样都是T个
    for res in resultParts:
        if (len(res) != r_len): continue
        result += res
    # 应该是一个频率 分贝图\
    """
    x = [n for n in range(0, 300)]
    plt.plot(x, result[0:300])
    plt.show()
    """
    _sumMale = sum(result[maleMinMax[0]:maleMinMax[1]])
    _sumFemale = sum(result[femaleMinMax[0]:femaleMinMax[1]])
    if _sumMale > _sumFemale:
        return 1, _sumMale, _sumFemale
    return -1, _sumMale, _sumFemale

def download_from_sql(fileName, _recongnize, cursor):
    _path, _name = os.path.split(fileName)
    newPath = "/static/dataForGender\\\\"  +_name
    sql = "update local_gender_log_copy set gender_recongnize ="+str(_recongnize)+"  WHERE outpath = \""+newPath+"\""
    cursor.execute(sql)
    results = cursor.fetchall()
    return results

def get_all_flg_data(cursor):
    sql = "select outpath from local_gender_log_copy WHERE flg = 1"
    cursor.execute(sql)
    results = cursor.fetchall()
    return results


def hps_orgin(rate, dataVoice, T=3):
    """
    原版hps 仅仅更改了返回值
    :param rate:
    :param dataVoice:
    :param T:
    :return:
    """

    if( T >len(dataVoice)/rate): T = len(dataVoice)/rate
    dataVoice = dataVoice[max(0, int(len(dataVoice) / 2) - int(T / 2 * rate)):min(len(dataVoice) - 1, int(len(dataVoice) / 2) + int(T / 2 * rate))]
    partLen=int(rate)
    parts = [ dataVoice[i*partLen:(i+1)*partLen] for i in range(int(T))]
    resultParts = []
    for data in parts:
        if(len(data)==0): continue
        window = np.hamming(len(data))
        data = data*window
        fftV = abs(fft(data))/rate
        fftR = copy(fftV)
        for i in range(2,HPSLoop):
            tab = copy(fftV[::i])
            fftR = fftR[:len(tab)]
            fftR *= tab
        resultParts.append(fftR)
    if len(resultParts) is 0:
        return 0
    result = [0]*len(resultParts[int(len(resultParts)/2)])
    for res in resultParts:
        if(len(res)!=len(result)): continue
        result += res

    if(sum(result[maleMinMax[0]:maleMinMax[1]]) > sum(result[femaleMinMax[0]:femaleMinMax[1]])): return 1
    return -1

def simpleRecognition(rate, data):
    if(checkBaseFreq(maleFemaleFreq[0], rate, data) < checkBaseFreq(maleFemaleFreq[1], rate, data)): return 1
    return -1
def checkBaseFreq(freq, ratio, data):
    box = int(1/freq*ratio)
    return sum([listVariation(data[int(i*box):int((i+1)*box-1)],data[int((i+1)*box):int((i+2)*box-1)]) for i in range( max(0,int(len(data)/box/2-(TS/2*freq))), min(int(len(data)/box)-2,int(len(data)/box/2+(TS/2*freq))),1)])
def listVariation(list1, list2): return sum([ abs(int(x)-int(y)) for x,y in zip(list1, list2)])


def vad_avg_hps(file,vad = 2):
    array, rate = read_wave(file)
    vadDatas = main_vad(array,rate,vad)
    return avg_hps(rate, vadDatas)

def vad(file,vad = 2,dir_name = ''):
    vad_one_dir(file,vad,dir_name)


def spectrogram(rate, dataVoice):
    """
    得到音频的频谱图
    """
    T = len(dataVoice)/rate
    partLen=int(rate)
    parts = [ dataVoice[i*partLen:(i+1)*partLen] for i in range(int(T))]
    resultParts = []
    for data in parts:
        if(len(data)==0): continue
        window = np.hamming(len(data))
        data = data*window
        _fft = fft(data)
        resultParts.append(_fft)
    return resultParts