import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dropout,Dense,LSTM,Flatten
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error,mean_absolute_error

bitcoindataset= pd .read_csv("bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv")
#coin base = pd.read_csv("coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")
print(bitcoindataset.head(5))
bitcoindataset["date"] = pd.to_datetime(bitcoindataset["Timestamp"],unit="s").dt.date
group=bitcoindataset.groupby("date")
data=group["Close"].mean()

train=data.iloc[:len(data)-50]
test=data.iloc[len(train):]
#print(test.shape)
#print(train.head(5))
#print(test.head(5))
#test=data.iloc[:len(test_d)-10]
#print(test.shape)
#val=test_d.iloc[len(test):]
#print(val.shape)
#print(train.shape)
#print(train.isnull().sum())
#print(train.head(5))
#rint(test.shape)
#print(test.isnull().sum())
#print(test.head(5))
#print(val.shape)
#print(val.isnull().sum())
#print(val.head(5))

train=np.array(train)
test=np.array(test)
#val=np.array(val)

scale=MinMaxScaler()

train=train.reshape(train.shape[0],1)
test=test.reshape(test.shape[0],1)

#val=val.reshape(val.shape[0],1)

#print(train)
#print(train.shape)
scaled_train= scale.fit_transform(train)
scaled_test= scale.fit_transform(test)
#scaled_val= scale.fit_transform(val)
timestep=50
x_train=[]
y_train=[]

for i in range(scaled_train.shape[0]):
    #print(i)
    #print(scaled_train[i-timestep:i,0])
    #print(scaled_train[i,0])
    x_train.append(scaled_train[i-timestep:i,0])
    y_train.append(scaled_train[i,0])

#for i in range(val)
scaled_train=np.array(scaled_train)
y_train=np.array(y_train)

print(y_train.shape)
scaled_train=scaled_train.reshape(scaled_train.shape[0],scaled_train.shape[1],1) #reshaped for RNN
print(scaled_train.shape)
#print("x_train shape= ",x_train.shape)
#print("y_train shape= ",y_train.shape)
model = Sequential()

model.add(LSTM(10,input_shape=(None,1),activation='relu'))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam",metrics=['mse'])
history =model.fit(scaled_train,y_train,batch_size=64,epochs=100)
print(history.history['loss'])
#print(history.history['mse'])
pyplot.plot(history.history['loss'])
#pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()



inputs=data[len(data)-len(scaled_test)-timestep:]
print(type(inputs))
inputs=inputs.values.reshape(-1,1)
inputs=scale.transform(inputs)
x_test=[]
y_test=[]
for i in range(timestep,inputs.shape[0]):
    x_test.append(inputs[i-timestep:i,0])
    y_test.append(inputs[i,0])
x_test=np.array(x_test)
y_test = np.array(y_test)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)


predicted_data=model.predict(x_test)
predicted_data=scale.inverse_transform(predicted_data)
y_test=y_test.reshape(-1,1)
y_test=np.array(y_test)
y_test = scale.inverse_transform(y_test)

print(predicted_data,y_test)

print(mean_squared_error(y_test,predicted_data))
print(mean_absolute_error(y_test,predicted_data))

