import scipy.misc
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import np_utils
from sklearn.model_selection import train_test_split
from PIL import Image

# 71 characters
nb_classes = 71
# input image dimensions
img_rows, img_cols = 32, 32

# ary = np.load("hiragana.npz")['arr_0']
# print(ary.shape)

ary = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15
X_train = np.zeros([nb_classes * 160, img_rows, img_cols], dtype=np.float32)
for i in range(nb_classes * 160):
    # X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
    X_train[i] = np.array(Image.fromarray(ary[i]).resize((img_rows, img_cols), resample = 3)).astype(np.float32)
 
y_train = np.repeat(np.arange(nb_classes), 160)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# convert class vectors to categorical matrices
y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

# data augmentation
datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)