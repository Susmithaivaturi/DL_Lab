# import libraries
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical
from keras.datasets import cifar10

# load data
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# architecture
model=Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(units=10,activation='softmax'))

# compile
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

# train
history=model.fit(x_train,y_train,epochs=10,batch_size=64)

# evaluate
loss, accuracy = model.evaluate(x_test,y_test)
print(f"accuracy: {accuracy}")
print(f"loss: {loss}")
