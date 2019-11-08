from keras.models import Sequential
from keras.layers import LSTM,Dense,Activation,SimpleRNN,Conv1D,MaxPool1D,Flatten,Reshape,Dropout
from keras.callbacks import EarlyStopping
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop
from .keras_self_attention import SeqSelfAttention
from .keras_multi_head import MultiHeadAttention
from tensorflow import keras
import keras.backend as K
from keras_pos_embd import TrigPosEmbedding

def cnn():
    model = Sequential()
    model.add(Conv1D(32, (5, ),input_shape=(( 300,34))))
    model.add(Dropout(0.5))
    model.add(MaxPool1D())
    model.add(Conv1D(64, (5, )))
    model.add(MaxPool1D())
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer="adam")
    return model

def cnn_self_attention():
    model = Sequential()
    '''
    model.add(TrigPosEmbedding(
        input_shape=((300, 34)),
        mode=TrigPosEmbedding.MODE_ADD,  # Use `add` mode (default)
        name='Pos-Embd'))
    '''
    model.add(Conv1D(32, 3,input_shape=(( 300,34))))
    model.add(Dropout(0.5))
    model.add(MaxPool1D())
    model.add(Conv1D(64, 2))
    model.add(MaxPool1D())
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    #model.add(keras.layers.Lambda(lambda x: K.batch_flatten(x)))
    model.add(Flatten())
    #model.add(Dense(64,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[categorical_accuracy],
    )
    return model

def cnn_multi_attention():
    model = Sequential()
    model.add(TrigPosEmbedding(
        input_shape=(300,34),
        mode=TrigPosEmbedding.MODE_CONCAT,  # MODE_ADD  MODE_EXPAND
        output_dim=2,
        name='Pos-Embd'))

    model.add(Conv1D(64, (2,),input_shape=(( 300,35))))
    model.add(Dropout(0.5))
    model.add(MaxPool1D())
    model.add(Conv1D(32, 2))
    model.add(MaxPool1D())
    model.add(Dropout(0.5))
    model.add(MultiHeadAttention(head_num=16,name='Multi-Head'))

    model.add(Flatten())
    #model.add(Dropout(0.5))
    #model.add(Dense(64,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.summary()
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[categorical_accuracy],
    )
    return model

def attention():
    input_layer = keras.layers.Input(
        shape=(300, 34),
        name='Input',
    )
    att_layer = MultiHeadAttention(
        head_num=3,
        name='Multi-Head',
    )(input_layer)
    model = keras.models.Model(inputs=input_layer, outputs=att_layer)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics={},
    )
    return model