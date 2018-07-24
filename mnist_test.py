'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from BoundedImageClassifier import *
from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

batch_size = 128
num_classes = 10
epochs = 10

beta = 125

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

idx = np.random.choice(len(x_train), 5000)
x_train = x_train[idx]
y_train = y_train[idx]
idx = np.random.choice(len(x_train), 1000)
y_test = y_test[idx]
x_test = x_test[idx]

model = create_image_classifier(input_shape=input_shape, beta=beta, num_classes=num_classes)

hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_data=(x_test, y_test))

plt.plot(hist.history['dense_2_acc'], label='Training Accuracy')
plt.plot(hist.history['val_dense_2_acc'], label='Validation Accuracy')
plt.title(r"Training and Validation Loss for \beta = %.3f" % beta)
plt.legend()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score)
print('Test accuracy:', score)
plt.show()
