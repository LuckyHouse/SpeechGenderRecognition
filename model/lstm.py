import keras
# import tensorflow.keras as keras
# from keras.models import Sequential
from keras.layers import LSTM,Dense,Activation,SimpleRNN,Conv1D,MaxPool1D,Flatten,Reshape,Dropout,MaxPooling1D,Masking
from keras.metrics import categorical_accuracy
# from keras.callbacks import EarlyStopping
# from keras.metrics import categorical_accuracy
# from keras.optimizers import RMSprop


def cnn_lstm(shape,numclasses):
    Dense = keras.layers.Dense
    lstmmodel_1 = keras.models.Sequential()
    #lstmmodel_1.add(Masking(mask_value=0, input_shape=(300,34)))

    lstmmodel_1.add(Conv1D(32, 3, input_shape=shape))#卷积层降噪
    lstmmodel_1.add(Dropout(0.2))
    lstmmodel_1.add(MaxPooling1D())


    #lstmmodel_1.add(keras.layers.Bidirectional(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2,input_shape=(150,34),return_sequences=True),input_shape=(150,34)))
    lstmmodel_1.add(keras.layers.Bidirectional(keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences=True)))
    lstmmodel_1.add(Dropout(0.2))
    lstmmodel_1.add(keras.layers.Flatten())
    lstmmodel_1.add(Dropout(0.2))
    #lstmmodel_1.add(Dense(32))
    lstmmodel_1.add(Dense(numclasses,
                                    activation='softmax',
                                    bias_initializer=keras.initializers.Constant(0.1),
                                    kernel_regularizer=keras.regularizers.l2(0.01),
                                    bias_regularizer=keras.regularizers.l2(0.01)
                             ))
    lstmmodel_1.compile(loss="categorical_crossentropy", optimizer='adamax', #optimizer=keras.optimizers.RMSprop(lr=0.001),
                  metrics=[categorical_accuracy])
    return lstmmodel_1;




def bi_lstm(vocab_size, embed_size, feature_size, num_classes, regularizers_lambda, dropout_rate):
    inputs = keras.Input(shape=(feature_size,), name='input_data')
    embed = keras.layers.Embedding(vocab_size + 1, embed_size,
                                   embeddings_initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
                                   mask_zero=True,
                                   input_length=feature_size,
                                   input_shape=(feature_size,),
                                   name='embedding')(inputs)
    mask = keras.layers.Masking(mask_value=0, name='masking')(embed)
    bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(100,
                                                           bias_initializer=keras.initializers.Constant(0.1),
                                                           dropout=dropout_rate,
                                                           recurrent_dropout=dropout_rate,
                                                           implementation=2), name='Bi-LSTM')(mask)
    outputs = keras.layers.Dense(num_classes,
                                 activation='softmax',
                                 bias_initializer=keras.initializers.Constant(0.1),
                                 kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 name='dense')(bi_lstm)
    model = keras.Model(inputs, outputs)
    return model

