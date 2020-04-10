# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:15:15 2020

@author: acer
"""

import pandas as pd
import numpy as np

df_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = df_train.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_sc = sc.fit_transform(training_set)

X_train = []
y_train = []

for i in range(60,len(df_train)):
    X_train.append(training_set_sc[i-60:i,0])
    y_train.append(training_set_sc[i,0])
    
#Converting X_train&y_train from list to arrays
X_train,y_train = np.array(X_train),np.array(y_train)

#Reshaping the training array batch_size, timestamps,attribute
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))


# Importing the Test Set

df_test = pd.read_csv("Google_Stock_Price_Test.csv")
testing_set = df_test.iloc[:,1:2].values

#Feature Scaling 
testing_set_sc = sc.transform(testing_set)

df_all = pd.concat((df_train['Open'],df_test['Open']),axis = 0)




# #Building RNN
# from keras.models import Sequential
# from keras.layers import Dense,LSTM,Dropout

# reg_model = Sequential()

# reg_model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
# reg_model.add(Dropout(0.2))

# #Adding Second Layer
# reg_model.add(LSTM(units = 50, return_sequences = True))

# # #Adding Third Layer
# reg_model.add(LSTM(units = 50, return_sequences = True))
# # reg_model.add(Dropout(0.2))

# reg_model.add(LSTM(units = 50 ))
# # reg_model.add(Dropout(0.2))

# reg_model.add(Dense(units = 1))

# reg_model.compile(optimizer = "adam",loss = "mean_squared_error")

# reg_model.summary()

# reg_model.fit(X_train,y_train,batch_size=21,epochs=100)


