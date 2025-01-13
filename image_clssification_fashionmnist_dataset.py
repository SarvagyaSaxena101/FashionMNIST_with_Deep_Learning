import keras 
from keras.layers import Dense
from keras.layers import Flatten
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential

(x_train,y_train),(x_test,y_test) = keras.datasets.fashion_mnist.load_data()

x_train_scaled = x_train/255
x_test_scaled = x_test/255


classes = ['top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','boots']

y_train_categorical = keras.utils.to_categorical(y_train,num_classes=10)
y_test_categorical = keras.utils.to_categorical(y_test,num_classes=10)

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(900,activation='sigmoid'))
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_scaled,y_train_categorical,epochs=50)

predictions = model.predict(x_test_scaled)
print(np.argmax(predictions[0]))
print(classes[np.argmax(predictions[0])])
plt.imshow(x_test_scaled[0])
plt.show()