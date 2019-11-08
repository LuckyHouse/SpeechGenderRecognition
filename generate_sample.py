import glob
from algorithm import hps

def vad(dir,aim_dir):
    files = glob.glob(dir)
    for file in files:
        hps.vad(file,3,aim_dir)

if __name__ ==  '__main__':
    vad('./data/long_audio/*.wav','./data/label_data/')