##############################################################
######          Written by Wang CHEN                     #####
######     E-mail: wchen22@connect.hku.hk                #####
######  Revised on https://github.com/yangwohenmai/LSTM  #####
##############################################################



# LSTM for international airline passengers problem with window regression framing
'''
You may need to install some packages:
pip install keras
pip install tensorflow
pip install pandas
pip isntall matplotlib
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


"""
Predice the next data using the 3 previous data:
X1	X2	X3	Y
112	118	132	129
118	132	129	121
132	129	121	135
129	121	135	148
121	135	148	148

data format:
x -> y = [93,1,3] -> [93,1]
x = [[[x1,x2,x3]], 
     [[x1,x2,x3]], 
     [[x1,x2,x3]], 
     ...]

y = [[y1],
     [y2],
     [y3], 
     ...]
   
input_shape=(1,3)

You can change the parameter to change the prediction step:
look_back = 3
"""


# Define the dataloder class
class DataLoader():
    def __init__(self, look_back = 1) -> None:
          self.look_back = look_back
          self.scaler = None
    
    # Dataloder
    def Dataloader(self):
        # load the dataset
        dataframe = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        
        # Normalization
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = self.scaler.fit_transform(dataset)
        
        # Split dataset into training and test (2 : 1)
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train_data, test_data = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        
        # prediction step = 3
        look_back = 3
        trainX, trainY = self.create_dataset(train_data, look_back)
        testX, testY = self.create_dataset(test_data, look_back)
        
        # Reshape: [samples, time steps, features] = [93,1,3]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        return trainX, trainY, testX, testY
    
    # Generate the dataset format
    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)


# Define the model class
class Model():
    def __init__(self, look_back = 1) -> None:
        self.look_back = look_back
        self.model = self.Init_Model()
    
    # define the model
    def Init_Model(self):
        # 构建 LSTM 网络
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, self.look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    # train the model
    def train(self, trainX, trainY, epochs = 100, batch_size = 1):
        self.model.fit(trainX, trainY, epochs = epochs, batch_size = batch_size, verbose=2)


def main():
    look_back = 3
    Data_Loader = DataLoader(look_back = look_back)
    MODEL = Model(look_back = look_back)
    
    # Load data
    trainX, trainY, testX, testY = Data_Loader.Dataloader()
    # train the model
    MODEL.train(trainX, trainY)

    # Show the results
    # train
    trainPredict = MODEL.model.predict(trainX)
    # test
    testPredict = MODEL.model.predict(testX)
    # Denormalization
    trainPredict = Data_Loader.scaler.inverse_transform(trainPredict)
    trainY = Data_Loader.scaler.inverse_transform([trainY])
    testPredict = Data_Loader.scaler.inverse_transform(testPredict)
    testY = Data_Loader.scaler.inverse_transform([testY])
    
    # Calculate the errors
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    # Plot the results
    plt.plot(trainY[0], label = 'Real passengers (train)')
    plt.plot(trainPredict[:,0], label = 'Predicted passengers (train)')
    plt.legend()
    plt.savefig('Result_train.png')
    plt.close('all')
    plt.plot(testY[0], label = 'Real passengers (test)')
    plt.plot(testPredict[:,0], label = 'Predicted passengers (test)')
    plt.legend()
    plt.savefig('Result_test.png')
    


if __name__ == '__main__':
    main()