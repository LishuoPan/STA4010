# utf - 8
# Author: Lishuo Pan; Data: Dec 12 2018
# Load packages
import pandas as pd
import numpy as np
import math
from TradingModel import TradingModel
import matplotlib.pyplot as plt

def CompleteData(Matrix_array):
    # This module is used to complete the data
    # input: Matrix_array np.array
    # output: Matrix_array np.array completed

    n = Matrix_array.shape[0]
    d = Matrix_array.shape[1]
    for i in range(n):
        for j in np.arange(1,d):
            if math.isnan(Matrix_array[i,j]):
                Matrix_array[i,j] = Matrix_array[i-1,j]
    return Matrix_array


def min_dividor(ny_i_comp):
    # This module is used to
    # Input: ny_i_comp ny.array
    # output: ny_i_minuite_cut

    index = list()
    index.append(0)
    n = ny_i_comp.shape[0]
    # first_row = Data[0,:]
    for i in np.arange(1,n):
#         print(Data_with_date[i,:][0][15])
        if ny_i_comp[i,:][0][15] != ny_i_comp[i-1,:][0][15]:
            index.append(i)
    Data_minute = ny_i_comp[index,:]
    return Data_minute



def StrucData(ls, cut = 1):
    # This module is to structralize the df to matrix
    # input: ls list contain str of training_data
    #       cut indicator 1 to cut data into minute; 0 o.w.
    # output: ny.array

    data_frame = list()
    # read in csv file to pandas data frame
    for i in ls:
        # use value function to transform df to nyarray
        ny_i = pd.read_csv(i,header = 0,sep = ',').values
        ny_i_comp = CompleteData(ny_i)
        if cut == 1:
            ny_i_comp = min_dividor(ny_i_comp)
        else:
            pass
        data_frame.append(ny_i_comp)
    # stack array into matrix
    data_matrix = np.vstack(data_frame)
    data_matrix = np.delete(data_matrix,0,axis=1)

    train_time = len(data_matrix)
    time = np.linspace(1,train_time,train_time)
    data_matrix = np.hstack((time.reshape(-1,1),data_matrix))
    feature = data_matrix[:,(0,2,4)]
    # feature = data_matrix[:, 0].reshape(-1,1)
    y_bid = data_matrix[:,1]
    y_ask = data_matrix[:, 3]
    # print(data_matrix)
    return feature,y_bid,y_ask
# training phase
    # report model

# testing phase
    # print result



if __name__ == '__main__':
    cut = 1
    # read and structure training data

    [X_tr, y_bid_tr, y_ask_tr] = StrucData(["./training_data/Day1.csv",
                                      "./training_data/Day2.csv",
                                      "./training_data/Day3.csv"], cut)
    time_tr = X_tr.shape[0]
    # read and structure testing data
    [X_te, y_bid_te, y_ask_te] = StrucData(["./testing_data/Day4.csv"],cut)
    # add the past time to the test data
    X_te[:,0] = X_te[:,0]+time_tr
    Model = TradingModel(X_tr,y_bid_tr,y_ask_tr,100)
    Model.process(X_te,y_bid_te,y_ask_te)

    # model = TradingModel()