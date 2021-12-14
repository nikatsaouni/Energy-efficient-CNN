
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



# ### Delete the column with the indeces
# train_data_.drop(train_data_.columns[[0]], axis=1, inplace=True)
# test_data_.drop(test_data_.columns[[0]], axis=1, inplace=True)



## Read labels and replace sinus with 0 and arrhythmia with 1
train_labels = pd.read_csv('/projects/energics/work/data/Train_Lab.csv',sep='\t',index_col=0).replace({'sinus': 0, 'arrhythmia': 1})
test_labels = pd.read_csv('/projects/energics/work/data/Test_Lab.csv',sep='\t',index_col=0).replace({'sinus': 0, 'arrhythmia': 1})

print("Data loaded")
print("----------------------------------------")
## Split data to segments of 30 secs = 128data points* 30 = 3840 with overlap 50% = 1920 data points
## In total 6 segments for each ECG = 90 seconds
def splitDataFrame(df, window_Size):
    def splitToRows(df, window_Size):
    	counter = 0
    	step = int(window_Size/2)
    	for sample in range(df.shape[0]):
            for segments in range(int(np.floor(df.shape[1]/(step)))-1):
                new_df[counter] = np.array(df.iloc[sample,segments*step:(segments*step+window_Size)])     
                counter =counter + 1
    
    new_df = pd.DataFrame()
    splitToRows(df, window_Size)
    return(new_df.T)



# def encode_labels(df):
#     enc = OneHotEncoder()
#     enc.fit(df)
#     out = enc.transform(df).toarray()
#     return out

# print("Create 9000 data points segments")

# ## Create segments
# sampl_freq = 128 ## sampling frequenct
# sec_per_segm = 30 ## seconds per window
# window_Size = sampl_freq*sec_per_segm ## window size in data points

# window_Size = 9000

##uncomment for segments
# train_data = splitDataFrame(train_data, window_Size)
# test_data = splitDataFrame(test_data, window_Size)


## Adjust labels to segments.---> each segment has one label ---> here we have 2 segments 
## for each ECG. 


test_labels['1'] = test_labels['0']
test_labels['1'][test_labels['0'] == 1] = 0
test_labels['1'][test_labels['0'] == 0] = 1
test_labels = test_labels.to_numpy()

train_labels['1'] = train_labels['0']
train_labels['1'][train_labels['0'] == 1] = 0
train_labels['1'][train_labels['0'] == 0] = 1
train_labels = train_labels.to_numpy()

# train_labels = encode_labels(train_labels).to_numpy()
# test_labels = encode_labels(test_labels).to_numpy()



# train_labels = pd.DataFrame(np.repeat(train_labels.values,2,axis=0)).to_numpy()
# test_labels = pd.DataFrame(np.repeat(test_labels.values,2,axis=0)).to_numpy()


### Expand the dimension to apply the convolutions
train_data = np.expand_dims(train_data,2)
test_data = np.expand_dims(test_data,2)
print("----------------------------------------")
print("Define model architecture")

### define CNN architecture
# input_size = window_Size

input_size = 14336

input_layer = Input(shape =(input_size,1))
x = Conv1D(filters=128, kernel_size=50,activation='relu', padding="valid", strides =3 ,name='Conv_1')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2, strides=3, name='Pooling_1')(x)

x = Conv1D(filters=32, kernel_size=7, padding="valid",activation='relu', name='Conv_2')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2 , name='Pooling_2')(x)

x = Conv1D(filters=32, kernel_size=10, padding="valid",activation='relu', name='Conv_3')(x)
x = Conv1D(filters=128, kernel_size=5, padding="valid",activation='relu', name='Conv_4')(x)
x = MaxPooling1D(pool_size=2, name='Pooling_3')(x)

x = Conv1D(filters=256, kernel_size=15, padding="valid",activation='relu', name='Conv_5')(x)
x = MaxPooling1D(pool_size=2, name='Pooling_4')(x)

x = Conv1D(filters=512, kernel_size=5, padding="valid",activation='relu', name='Conv_6')(x)
x = Conv1D(filters=128, kernel_size=3, padding="valid",activation='relu', name='Conv_7')(x)

x = Flatten(name='Flatten1')(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.1)(x)
output = Dense(2, activation='softmax')(x)





model = Model(inputs = input_layer, outputs = output)
## Compile model

model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
model.summary()
print("----------------------------------------")
print("Fit model on training data")
history = model.fit(train_data, train_labels, epochs = 30, batch_size=50, validation_split=0.10, shuffle=True, verbose=2)


score_tr = model.evaluate(train_data,train_labels,verbose=2)
score_ts = model.evaluate(test_data,test_labels,verbose=2)
print('---------------------------------------------------')
print('Training evaluation')
print(score_tr)
print('Test evaluation')
print(score_ts)
print('---------------------------------------------------')


model.save_weights('Yidirima.h5')
print('Weights saved to Yidirima.h5 file')




