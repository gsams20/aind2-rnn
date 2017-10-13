import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import re


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    length = len(series)
    range_value = length-window_size
    for i in range(range_value):
        X.append(series[i:i+window_size])
        
    y=series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:window_size])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model= Sequential()
    model.add(LSTM(5,input_shape=(window_size, 1)))
    model.add(Dense(1, input_dim=5))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = '!,.:;?'
    text = text.replace('\n',' ')    # replacing '\n' with '' simply removes the sequence
    text = text.replace('\r',' ')
    text = text.replace('\xa0',' ')
    return re.sub(r'[^A-Za-z0-9\s'+punctuation+']', '',text)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    length = len(text)
    range_value = length-window_size
    i=0
    while i <= range_value:
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
        i=i+step_size
    
    # reshape each 
    inputs = np.asarray(inputs)
    inputs.shape = (np.shape(inputs)[0:window_size])
    #outputs = np.asarray(outputs)
    #outputs.shape = (len(outputs),1)
    
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model= Sequential()
    model.add(LSTM(200,input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    
    return model
