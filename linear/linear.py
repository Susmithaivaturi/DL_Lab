# import libraries
from keras.models import Sequential 
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# data with noise
X=np.linspace(0,5,1000)
y=3*X+7+3*np.random.randn(1000)

# architecture
model=Sequential()
model.add(Dense(units=1,input_dim=1,activation='linear'))

# compile
model.compile(optimizer='sgd',loss='mse',metrics=['mae'])

# train
model.fit(X,y,epochs=10,batch_size=16)

# evaluate
model.evaluate(X,y)
result=model.predict(X)

# visualization
plt.scatter(X,y,label="original data")
plt.plot(X,result,label="predicted data",color="red")
plt.legend()
plt.savefig("linear1.png")
plt.show()
