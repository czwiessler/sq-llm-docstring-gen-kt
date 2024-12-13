from keras.layers import Convolution1D
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Dropout
from data import process_data
from keras import backend as K
import numpy  as np
import os

if not os.listdir('datasets/processed'):
    process_data()

arrhy_data = np.loadtxt(open('datasets/processed/arrhythmia.csv', 'r'), skiprows=1)
malignant_data = np.loadtxt(open('datasets/processed/malignant-ventricular-ectopy.csv', 'r'), skiprows=1)
arrhy_data = arrhy_data[:len(malignant_data)]
arrhy_len = len(arrhy_data)/500

i = 0
X_train = []
inter_X_train = []
inter_y_train = []
y_train = []
nb_filters = 32
nb_epoch = 10
batch_size = 8
counter = 0

for _ in range(arrhy_len):
    counter += 1
    if not (counter % batch_size):
        X_train.append(inter_X_train)
        y_train.append(inter_y_train)
        inter_X_train = []
        inter_y_train = []

    inter_X_train.append(np.asarray(arrhy_data[i:i+500]))
    inter_y_train.append(0)
    inter_X_train.append(np.asarray(malignant_data[i:i+500]))
    inter_y_train.append(1)
    i += 500

validation_size = int(0.1  * len(X_train))

# remove the bugged batch
X_train.pop(0)
y_train.pop(0)

# split training and testing sets
X_train, X_test = np.split(X_train, [len(X_train)-validation_size])
y_train, y_test = np.split(y_train, [len(y_train)-validation_size])

# checking batch lengths
for batch in X_train:
    if len(batch) != 16:
        print("uneven batch with len: {}".format(len(batch)))
    for example in batch:
        if len(example) != 500:
            print("uneven example with len: {}".format(len(example)))


# shape = (X_train.shape[0], 16, 500)
shape = X_train.shape[1:]

# in numpy if arrays are not the same shape they will not appear in the .shape method
print("shape: {}".format(shape))

model = Sequential()
model.add(Convolution1D(nb_filters, 3, input_shape=shape, activation='relu'))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filters, 3, activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("ok")
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, validation_data=(X_test, y_test))