import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

#importing the data
#unpacks images to x_train/x_test and labels to y_train/y_test
(X_train,y_train),(X_test,y_test) = mnist.load_data()
#Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

#Initializing the classifier Network
classifier = Sequential()

classifier.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
classifier.add(Dropout(0.2))

#Adding a second LSTM network layer
classifier.add(LSTM(128))

#Adding dense hidden layer
#Adding a dense hidden layer
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.2))

#Adding the output layer
classifier.add(Dense(10, activation='softmax'))

#Compiling the network
classifier.compile( loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-6),
              metrics=['accuracy'] )

#Fitting the data to the model
classifier.fit(X_train,
         y_train,
          epochs=3,
          validation_data=(X_test, y_test))
        
test_loss, test_acc = classifier.evaluate(X_test, y_test)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))
