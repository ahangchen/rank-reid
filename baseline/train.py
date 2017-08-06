from __future__ import division, print_function, absolute_import

import os
from random import shuffle

import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.layers import Input
from keras.layers import Dense, Flatten, Dropout
from keras.initializers import RandomNormal
from keras.models import Model
from keras import backend as K
from keras.models import load_model

'''
DATASET = '../dataset/Duke'
LIST = os.path.join(DATASET, 'train.list')
TRAIN = os.path.join(DATASET, 'bounding_box_train')
class_count = 702
'''

DATASET = '../dataset/Market'
LIST = os.path.join(DATASET, 'train.list')
TRAIN = os.path.join(DATASET, 'bounding_box_train')
class_count = 751
'''
DATASET = '../dataset/CUHK03'
LIST = os.path.join(DATASET, 'train.list')
TRAIN = os.path.join(DATASET, 'bbox_train')
'''

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
set_session(sess)

# load pre-trained resnet50
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

x = base_model.output
x = Flatten(name='flatten')(x)
x = Dropout(0.5)(x)
x = Dense(class_count, activation='softmax', name='fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
net = Model(inputs=[base_model.input], outputs=[x])

for layer in net.layers:
    layer.trainable = True

# load data
images, labels = [], []
with open(LIST, 'r') as f:
    for line in f:
        line = line.strip()
        img, lbl = line.split()
        img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        images.append(img[0])
        labels.append(int(lbl))

img_cnt = len(labels)
shuffle_idxes = range(img_cnt)
shuffle(shuffle_idxes)
shuffle_imgs = list()
shuffle_labels = list()
for idx in shuffle_idxes:
    shuffle_imgs.append(images[idx])
    shuffle_labels.append(labels[idx])
images = np.array(shuffle_imgs)
labels = to_categorical(shuffle_labels)

# train
batch_size = 64
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    width_shift_range=0.2,  # 0.
    height_shift_range=0.2)

val_datagen = ImageDataGenerator()

net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
net.fit_generator(train_datagen.flow(images[: int(0.9 * img_cnt)], labels[: int(0.9 * img_cnt)], batch_size=batch_size),
                  steps_per_epoch=len(images) / batch_size + 1, epochs=20,
                  validation_data=val_datagen.flow(images[int(0.9 * img_cnt):], labels[int(0.9 * img_cnt):],
                                                   batch_size=batch_size),
                  validation_steps=img_cnt / 10 / batch_size + 1)
net.save('0.ckpt')
