import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import joblib


""""
The function returns time series data.
Params:

      file_path: file path of the data file
      
      l_seq: time lag in the input and output sequence
      
      scale: flag for scaling

Returns: Augmented input and output data

       X_aug: Input data of size(N, l_seq, M) 
       Y_aug: Output data of size(N, l_seq, L)
       
       X: Original Input data of size(N, M)
       Y: Original Output data of size(N, L)
       
"""
def get_lstm_data(file_path, l_seq, scale=True):
    
    
    # Read data from csv file
    canada_train_pd = pd.read_csv(file_path, index_col=1)
    
    # Drop the rows with NAN values
    canada_train_pd = canada_train_pd.dropna() 
    
    # Exclude the non-numeric predictors
    input_columns = ['Den', 'SD', 'Elev', 'Lat', 'Long', 'Day of year',
       'days without snow', 'number frost-defrost', 'accum pos degrees',
       'average age SC', 'number layer', 'accum solid precip',
       'accum solid precip in last 10 days', 'total precip last 10 days',
       'average temp last 6 days']
    
    # Predictant column
    output_columns = ['SWE']
    
    # Filtered Data
    canada_train_pd = canada_train_pd[input_columns + output_columns]
    
    # Scaling the data
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        canada_train_pd.loc[:] = scaler.fit_transform(canada_train_pd.values)
        
    # Splitting the input and output data
    input_pd = canada_train_pd[input_columns]
    output_pd = canada_train_pd[output_columns]
    
    # Converting input data into time series data
    input_values = []
    for column in input_columns:
        input_values.append(input_pd[column].values.reshape(-1,1))
    input_scaled = np.concatenate(input_values, axis=1)
    
    n_sample = input_scaled.shape[0]
    m_feature = input_scaled.shape[1]
    
    input_scaled_aug = np.zeros((l_seq-1, m_feature))
    for i in range(np.shape(input_scaled_aug)[0]):
        for j in range(m_feature):
            input_scaled_aug[i,j] = input_scaled[0,j]
    
    input_scaled_aug = np.concatenate((input_scaled_aug, input_scaled), axis=0)
    
    # Augmented input data
    X = np.zeros((n_sample, l_seq, m_feature))
    for i in range(n_sample):
        tmp = 0
        X[i,:,:] = input_scaled_aug[i+tmp:i+tmp+l_seq,:]
        tmp = tmp+1
      
    # Converting output data into time series data
    output_scaled = output_pd.values.reshape(-1,1)
    n_sample = output_scaled.shape[0]
    m_feature = 1
    output_scaled_aug = np.zeros((l_seq-1, m_feature))
    for i in range(np.shape(output_scaled_aug)[0]):
        output_scaled_aug[i,:] = output_scaled[0,:]

    output_scaled_aug = np.concatenate((output_scaled_aug, output_scaled), axis=0)
    
    # Augmented output data
    Y = np.zeros((n_sample, l_seq, m_feature))

    for i in range(n_sample):
        tmp = 0
        Y[i,:,:] = output_scaled_aug[i+tmp:i+tmp+l_seq,:]
        tmp = tmp+1
    return X, Y