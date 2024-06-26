import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')


user_input = st.text_input('Enter Stock Ticker','AAPL').upper()
df = yf.download(user_input,start,end)

#Describing data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())
st.write(df.head())

# VISUALIATIONS

st.subheader('Closing Price vs Time chart')
fig =  plt.figure(figsize= (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig =  plt.figure(figsize= (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig =  plt.figure(figsize= (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
data_testing_array = scaler.fit_transform(data_testing)

# Splitting data into X_train and y_train
def create_data_array(data_array,time_step=1):
    dataX = []
    datay = []
    for i in range(time_step,data_array.shape[0]):
        a = data_array[i-time_step:i,0]
        dataX.append(a)
        datay.append(data_array[i,0])
    return np.array(dataX),np.array(datay)

time_step = 100
X_test,y_test = create_data_array(data_testing_array,time_step)

X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

# Loading model
model = load_model('stockpredict.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing],ignore_index=True)

X_test,y_test = create_data_array(data_testing_array,time_step)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

y_predicted = model.predict(X_test)
y_predicted = scaler.inverse_transform(y_predicted)
y_test = y_test.reshape(-1, 1)
y_test = scaler.inverse_transform(y_test)

# Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'y',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)