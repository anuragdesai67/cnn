# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 18:06:04 2018
@author: anurag
"""
# Importing libraries

import numpy as np
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
from sklearn.preprocessing import StandardScaler



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
    #model.add(Dense(10, activation=act_fun))
    model.add(Dense(10, activation='softmax'))
    return model

# Compile model
def compile_model(model, error_function):
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=error_function, optimizer=adam, metrics=['accuracy'])
    return model

# Training the model
def train_model(model,num_epochs,b_size,train_x,encoding_train_y):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    import time
    start = time.clock()
    model.fit(train_x, encoding_train_y, validation_split = 0.2,epochs=num_epochs, batch_size=b_size, callbacks=[early_stopping])
    print (time.clock() - start)
# Evaluate the model
def evaluate_model(model,test_x,encoding_test_y):
    scores = model.evaluate(test_x, encoding_test_y)
    return scores

def build_cnn(train_x,encoding_train_y,test_x,encoding_test_y,dropout_prob1,dropout_prob2,num_epochs,number_of_filters,filter_size):
    train_x = train_x.reshape(-1,8,8,1)
    test_x = test_x.reshape(-1,8,8,1)
      
    
    print (train_x.shape)
    model = Sequential()
    model.add(Convolution2D(number_of_filters, filter_size,filter_size, input_shape=(8,8,1), activation='relu',border_mode = 'valid')) # use this border_mode b/c not using Theano
    model.add(Convolution2D(number_of_filters, filter_size, filter_size, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th')) # channel dimension/depth is dim 1 to match 32 x 3 x 3
    model.add(Dropout(dropout_prob1))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_prob2))
    model.add(Dense(10, activation='softmax'))
    

    
    
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    import time
    start = time.clock()
    model.fit(train_x,encoding_train_y, batch_size=128, nb_epoch=num_epochs,validation_split=0.2)
    print (time.clock() - start)
    scores = model.evaluate(test_x, encoding_test_y, verbose=0)
    print("\nAccuracy: %.2f%%" % (scores[1]*100))
    print("\n")
    y_pred = model.predict_classes(train_x)
    print(classification_report(train_y, y_pred))
    print(confusion_matrix(train_y, y_pred))
    
    y_pred = model.predict_classes(test_x)
    print(classification_report(test_y, y_pred))
    print(confusion_matrix(test_y, y_pred))
    
def get_normed_mean_cov(X):
    X_std = StandardScaler().fit_transform(X)
    X_mean = np.mean(X_std, axis=0)
    
    ## Automatic:
    #X_cov = np.cov(X_std.T)
    
    # Manual:
    X_cov = (X_std - X_mean).T.dot((X_std - X_mean)) / (X_std.shape[0]-1)
    
    return X_std, X_mean, X_cov

    
# 1-of-c encoding output encoding
train_x,train_y = import_train_data("optdigits.tra")
test_x,test_y = import_test_data("optdigits.tes")
encoding_train_y,encoding_test_y = encoding_train_test(train_y,test_y)
#X_std, X_mean, X_cov = get_normed_mean_cov(X_training)
# =============================================================================
# train_x, _, _ = get_normed_mean_cov(train_x)
# test_x, _, _ = get_normed_mean_cov(test_x)
# =============================================================================

embedding_dim = 64
# Training parameters
batch_size = 32
num_epochs = 30     
model = create_model(embedding_dim,K.relu)

# =============================================================================
# print("Sum of Squares error---------------------------------------------------------------------------")
# # Sum of Squares Error
# model = compile_model(model,losses.mean_squared_error)
# train_model(model,num_epochs,batch_size,train_x,encoding_train_y)
# scores = evaluate_model(model,test_x,encoding_test_y)
# print("\nOverall ClassificationAccuracy: %.2f%%" % (scores[1]*100))
# print("\n")
# print("TRAINING DATA---------------------------------------------------------------------------")
# y_pred = model.predict_classes(train_x)
# print(classification_report(train_y, y_pred))
# print(confusion_matrix(train_y, y_pred))
# print("TESTING DATA---------------------------------------------------------------------------")
# y_pred = model.predict_classes(test_x)
# print(classification_report(test_y, y_pred))
# print(confusion_matrix(test_y, y_pred))
# 
# print("Cross Entropy Error Function---------------------------------------------------------------------------")
# # Cross Entropy Error Function
# model = compile_model(model,losses.categorical_crossentropy)
# train_model(model,num_epochs,batch_size,train_x,encoding_train_y)
# scores = evaluate_model(model,test_x,encoding_test_y)
# print("\nAccuracy: %.2f%%" % (scores[1]*100))
# print("\n")
# print("TRAINING DATA---------------------------------------------------------------------------")
# y_pred = model.predict_classes(train_x)
# print(classification_report(train_y, y_pred))
# print("\n")
# print("\n")
# print(confusion_matrix(train_y, y_pred))
# print("TESTING DATA---------------------------------------------------------------------------")
# y_pred = model.predict_classes(test_x)
# print(classification_report(test_y, y_pred))
# print(confusion_matrix(test_y, y_pred))
# 
# 
# 
# print("tanh hidden unit---------------------------------------------------------------------------")
# 
# #tanh hidden unit
# model = create_model(embedding_dim,K.tanh)
# model = compile_model(model,losses.categorical_crossentropy)
# train_model(model,num_epochs,batch_size,train_x,encoding_train_y)
# scores = evaluate_model(model,test_x,encoding_test_y)
# print("\nAccuracy: %.2f%%" % (scores[1]*100))
# print("\n")
# 
# y_pred = model.predict_classes(train_x)
# print(classification_report(train_y, y_pred))
# print(confusion_matrix(train_y, y_pred))
# y_pred = model.predict_classes(test_x)
# print(classification_report(test_y, y_pred))
# print(confusion_matrix(test_y, y_pred))
# =============================================================================


print("CNN---------------------------------------------------------------------------")
# CNN
# Model Hyperparameters
num_epochs = 25       
filter_sizes = (2, 3, 4)
number_of_filters = 32
dropout_prob = (0.5, 0.5, 0.3)
build_cnn(train_x,encoding_train_y,test_x,encoding_test_y,dropout_prob[2],dropout_prob[1],num_epochs,number_of_filters,filter_sizes[2])
