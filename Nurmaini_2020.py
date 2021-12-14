
import os 
## choose GPU=0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import numpy as np
import pandas as pd
import sys
import csv
import random as rn
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Input, Dense, Conv1D, AveragePooling1D, MaxPooling1D, Dropout, Activation, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.models
import tensorflow.keras as K
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import random

# tf.debugging.set_log_device_placement(True)

seed_value= 66
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)



############ Read data ##############
## Read data files
print('Read data')
train_data = pd.read_csv('/projects/energics/work/data/Train_Data_filtered.csv',sep=',')
test_data = pd.read_csv('/projects/energics/work/data/Test_Data_filtered.csv',sep=',')


## Read labels and replace sinus with 0 and arrhythmia with 1
train_labels = pd.read_csv('/projects/energics/work/data/Train_Lab.csv',sep='\t',index_col=0).replace({'sinus': 0, 'arrhythmia': 1})
test_labels = pd.read_csv('/projects/energics/work/data/Test_Lab.csv',sep='\t',index_col=0).replace({'sinus': 0, 'arrhythmia': 1})

print("Data loaded")
print("----------------------------------------")




window_Size = 9000

## Adjust labels to segments.---> each segment has one label ---> here we have 2 segments 
## for each ECG. 

test_labels = test_labels.to_numpy()
train_labels = train_labels.to_numpy()


### Expand the dimension to apply the convolutions
train_data = np.expand_dims(train_data,2)
test_data = np.expand_dims(test_data,2)
print("----------------------------------------")
print("Define model architecture")

### define CNN architecture

input_size = 14336

input_layer = Input(shape =(input_size,1))
x = Conv1D(filters=64, kernel_size=3,activation='relu', padding="valid", name='Conv_1')(input_layer)
x = Conv1D(filters=64, kernel_size=3, padding="valid",activation='relu', name='Conv_2')(x)
x = MaxPooling1D(pool_size=2, name='Pooling_1')(x)
x = Conv1D(filters=128, kernel_size=3, padding="valid",activation='relu', name='Conv_3')(x)
x = Conv1D(filters=128, kernel_size=3, padding="valid",activation='relu', name='Conv_4')(x)
x = MaxPooling1D(pool_size=2, name='Pooling_2')(x)
x = Conv1D(filters=256, kernel_size=3, padding="valid",activation='relu', name='Conv_5')(x)
x = Conv1D(filters=256, kernel_size=3, padding="valid",activation='relu', name='Conv_6')(x)
x = Conv1D(filters=256, kernel_size=3, padding="valid",activation='relu', name='Conv_7')(x)
x = MaxPooling1D(pool_size=2, name='Pooling_3')(x)
x = Conv1D(filters=512, kernel_size=3, padding="valid",activation='relu', name='Conv_8')(x)
x = Conv1D(filters=512, kernel_size=3, padding="valid",activation='relu', name='Conv_9')(x)
x = Conv1D(filters=512, kernel_size=3, padding="valid",activation='relu', name='Conv_10')(x)
x = MaxPooling1D(pool_size=2, name='Pooling_4')(x)
x = Conv1D(filters=512, kernel_size=3, padding="valid",activation='relu', name='Conv_11')(x)
x = Conv1D(filters=512, kernel_size=3, padding="valid",activation='relu', name='Conv_12')(x)
x = Conv1D(filters=512, kernel_size=3, padding="valid",activation='relu', name='Conv_13')(x)
x = MaxPooling1D(pool_size=2, name='Pooling_5')(x)
x = Flatten(name='Flatten1')(x)
x = Dense(1000,activation='relu')(x)
x = Dense(1000,activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs = input_layer, outputs = output)
## Compile model
# Adam_optim = K.optimizers.Adam(learning_rate=0.0001)
Adam_optim = K.optimizers.Adam(learning_rate=0.001)


model.compile(loss='binary_crossentropy',
               optimizer=Adam_optim,
               metrics=['accuracy'])
model.summary()
print("----------------------------------------")
print("Fit model on training data")
history = model.fit(train_data, train_labels, epochs = 100, batch_size=16, validation_split=0.10, shuffle=True, verbose=2)


score_tr = model.evaluate(train_data,train_labels,verbose=2)
score_ts = model.evaluate(test_data,test_labels,verbose=2)
print('---------------------------------------------------')
print('Training evaluation')
print(score_tr)
print('Test evaluation')
print(score_ts)
print('---------------------------------------------------')


model.save_weights('nurmani_2020.h5')
print('Weights saved to nurmani_2020.h5 file')




