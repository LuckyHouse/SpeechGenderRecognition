# SpeechGenderRecognition

This model recognizes gender by analyzing real call recording. It is a Keras implementation of a CNN&LSTM which predict long audio with short audio.


#  Dependencies

-   Python3.6+
-   Keras2.3
-   scipy, numpy, Pandas, pyAudioAnalysis, pydub, h5py
-   Webrtcvad2.0.10
-   Sklearn

## Data

Generate short audio by:
```sh
generate_sample.py
```
It will generate short audio from long audio by VAD((Voice Activity Detection),then you need  to label them and put them in three folders.

Train data files:

    ├── ...
    ├── data
    │   ├── long_audio          #wav files before VAD
    │   ├── model                  #save model
    │   └── short_audio         #wav files after VAD
    │───├── female          # wav files with label female
    │───├── male          #  wav files with label male
    │───└── noise          # wav files with label noise
    └── ...


## Train

-  set train=true in:
```sh
	main.py
```
-   It will create feature and label data at first time:
```sh
	x.npy y.npy label.txt
```
-  If you use your own data, please **delete them** first.

## Predict

-  set train=false and model_path in:
```sh
	main.py
```

![](/img/info.png)


##  Accuracy

|       gender         |precision                          |recall                         |
|----------------|-------------------------------|-----------------------------|
|female			 |0.896         |0.89    |
|male            |0.909           |0.871            |




## Thanks

-  [https://github.com/tyiannak/pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
-  [https://github.com/wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad)
