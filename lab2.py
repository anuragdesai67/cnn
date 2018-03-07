# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 18:06:04 2018

@author: anurag
"""
# Importing libraries

import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten, Input, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import losses
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



# Importing training data
def import_train_data(filename):
    train_file = pd.read_csv(filename, header=-1)
    train_x = train_file.iloc[:, 0:64].values
    train_y = train_file.iloc[:, 64].values
    return train_x,train_y


# Importing testing data
def import_test_data(filename):
    test_file = pd.read_csv(filename, header=-1)
    test_x = test_file.iloc[:,0:64].values
    test_y = test_file.iloc[:,64].values
    return test_x,test_y

# 1-of-c encoding output encoding
def encoding_train_test(train_y,test_y):
    encoding_train_y = np_utils.to_categorical(train_y) 
    encoding_test_y = np_utils.to_categorical(test_y)
    return encoding_train_y,encoding_test_y

#  Create model
def create_model(embedding_dim,act_fun):
    model = Sequential()
    model.add(Dense(16, input_dim=embedding_dim, activation=act_fun))
    model.add(Dense(10, activation=act_fun))
    model.add(Dense(10, activation='softmax'))
    return model

# Compile model
def compile_model(model, error_function):
    
    model.compile(loss=error_function, optimizer='adam', metrics=['accuracy'])
    return model

# Training the model
def train_model(model,num_epochs,b_size,train_x,encoding_train_y):
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model.fit(train_x, encoding_train_y, validation_split = 0.2,epochs=num_epochs, batch_size=b_size, callbacks=[early_stopping])

# Evaluate the model
def evaluate_model(model,test_x,encoding_test_y):
    scores = model.evaluate(test_x, encoding_test_y)
    return scores

def build_cnn(train_x,encoding_train_y,test_x,encoding_test_y):
    train_x = train_x.reshape(-1,8,8,1)
    test_x = test_x.reshape(-1,8,8,1)
    
    # Model Hyperparameters
    # following parameters are just examples, you can use different values according to their meaning
    sequence_length = 50 # you need to change this value to adapt to your trainning data set because you have different trainning set after you do data augmentation
    num_epochs = 25       
    filter_sizes = (2, 3, 4)
    num_filters = 100
    dropout_prob = (0.5, 0.5)
    hidden_dims = 100
    print (train_x.shape)
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(8,8,1), activation='relu',border_mode = 'valid')) # use this border_mode b/c not using Theano
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th')) # channel dimension/depth is dim 1 to match 32 x 3 x 3
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x,encoding_train_y, batch_size=128, nb_epoch=num_epochs,validation_split=0.2)
    scores = model.evaluate(test_x, encoding_test_y, verbose=0)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))
    print("\n")
    y_pred = model.predict_classes(train_x)
    print(classification_report(train_y, y_pred))
    print(confusion_matrix(train_y, y_pred))
    
    y_pred = model.predict_classes(test_x)
    print(classification_report(test_y, y_pred))
    print(confusion_matrix(test_y, y_pred))

    
# 1-of-c encoding output encoding
train_x,train_y = import_train_data("optdigits.tra")
test_x,test_y = import_test_data("optdigits.tes")
encoding_train_y,encoding_test_y = encoding_train_test(train_y,test_y)


embedding_dim = 64
# Training parameters
batch_size = 32
num_epochs = 30      # you need to change this value to set how many epochs you want model fit your trainning data before evaluate performance on test data
model = create_model(embedding_dim,K.relu)

print("Sum of Squares error---------------------------------------------------------------------------")
# Sum of Squares Error
model = compile_model(model,losses.mean_squared_error)
train_model(model,num_epochs,batch_size,train_x,encoding_train_y)
scores = evaluate_model(model,test_x,encoding_test_y)
print("\nAccuracy: %.2f%%" % (scores[1]*100))
print("\n")
y_pred = model.predict_classes(train_x)
print(classification_report(train_y, y_pred))
print(confusion_matrix(train_y, y_pred))
y_pred = model.predict_classes(test_x)
print(classification_report(test_y, y_pred))
print(confusion_matrix(test_y, y_pred))

print("Cross Entropy Error Function---------------------------------------------------------------------------")
# Cross Entropy Error Function
model = compile_model(model,losses.categorical_crossentropy)
train_model(model,num_epochs,batch_size,train_x,encoding_train_y)
scores = evaluate_model(model,test_x,encoding_test_y)
print("\nAccuracy: %.2f%%" % (scores[1]*100))
print("\n")

y_pred = model.predict_classes(train_x)
print(classification_report(train_y, y_pred))
print(confusion_matrix(train_y, y_pred))

y_pred = model.predict_classes(test_x)

print(classification_report(test_y, y_pred))
print(confusion_matrix(test_y, y_pred))

print("tanh hidden unit---------------------------------------------------------------------------")

#tanh hidden unit
model = create_model(embedding_dim,K.tanh)
model = compile_model(model,losses.categorical_crossentropy)
train_model(model,num_epochs,batch_size,train_x,encoding_train_y)
scores = evaluate_model(model,test_x,encoding_test_y)
print("\nAccuracy: %.2f%%" % (scores[1]*100))
print("\n")

y_pred = model.predict_classes(train_x)
print(classification_report(train_y, y_pred))
print(confusion_matrix(train_y, y_pred))
y_pred = model.predict_classes(test_x)
print(classification_report(test_y, y_pred))
print(confusion_matrix(test_y, y_pred))


print("CNN---------------------------------------------------------------------------")

# CNN
build_cnn(train_x,encoding_train_y,test_x,encoding_test_y)
            