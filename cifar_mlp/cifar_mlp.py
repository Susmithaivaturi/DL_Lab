# import libraries
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# load data
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# architecture
model=Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(units=1024,activation='relu'))
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

# compile
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

# train
history = model.fit(x_train,y_train,epochs=100,batch_size=64, validation_data=(x_test,y_test))
print(history.history.items())
print(history.history.keys())

# evaluate
loss, accuracy = model.evaluate(x_test,y_test)
print(f"accuracy: {accuracy}")
print(f"loss: {loss}")

# visualization
plt.figure(figsize=(12,5))
# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label="train accuracy",color='blue')
plt.plot(history.history['val_accuracy'],label="validation accuracy",color='red')
plt.legend()
plt.title("Epoch vs Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'],label="train loss",color='blue')
plt.plot(history.history['val_loss'],label="validation loss",color='red')
plt.legend()
plt.title("Epoch vs Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("epoch_100_acc_loss.png")
plt.show()
