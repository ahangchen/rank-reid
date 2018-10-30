from __future__ import division, print_function, absolute_import

import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

import utils.cuda_util0
from random import shuffle

import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Dropout, Softmax, Conv2D, Reshape, BatchNormalization
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD, Adagrad, Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


def load_data(LIST, TRAIN, camera_cnt, class_cnt):
    images_in_cameras, labels_in_cameras = [[] for _ in range(camera_cnt)], [[] for _ in range(camera_cnt)]
    last_labels, label_cnts  = [-1 for _ in range(camera_cnt)], [-1 for _ in range(camera_cnt)]
    with open(LIST, 'r') as f:
        for line in f:
            line = line.strip()
            img = line
            lbl = line.split('_')[0]
            camera = int(line.split('_')[1][1]) - 1
            if last_labels[camera] != lbl:
                label_cnts[camera] += 1
            last_labels[camera] = lbl
            img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            images_in_cameras[camera].append(img[0])
            labels_in_cameras[camera].append(label_cnts[camera])
    for i in range(camera_cnt):
        img_cnt = len(labels_in_cameras[i])
        shuffle_idxes = range(img_cnt)
        shuffle(shuffle_idxes)
        shuffle_imgs = list()
        shuffle_labels = list()
        for idx in shuffle_idxes:
            shuffle_imgs.append(images_in_cameras[i][idx])
            shuffle_labels.append(labels_in_cameras[i][idx])
        images_in_cameras[i] = np.array(shuffle_imgs)
        labels_in_cameras[i] = to_categorical(shuffle_labels, class_cnt)
    return images_in_cameras, labels_in_cameras


def multi_generator(images_in_cameras, labels_in_cameras, batch_size, train=True):
    camera_cnt = len(images_in_cameras)
    if train:
        img_aug = ImageDataGenerator(
            shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
            horizontal_flip=0.5)
    else:
        img_aug = ImageDataGenerator()
    generators = [img_aug.flow(images_in_cameras[i], labels_in_cameras[i], batch_size=batch_size) for i in range(camera_cnt)]
    while True:
        feed_images = []
        feed_labels = []
        for i in range(camera_cnt):
            images, labels = generators[i].next()
            if len(labels) != batch_size:
                images, labels = generators[i].next()
            feed_images.append(images)
            feed_labels.append(labels)
        yield feed_images, feed_labels


def multi_branch_model(class_cnt, camera_cnt):
    # load pre-trained resnet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    for layer in base_model.layers:
        layer.trainable = True
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
    for layer in base_model.layers[: len(base_model.layers)//3*2]:
        layer.trainable = False

    img_inputs = []
    softmax_outputs = []
    for i in range(camera_cnt):
        img_inputs.append(Input(shape=(224,224,3), name='img_%d' % i))
        x = base_model(img_inputs[i])
        x = Dropout(0.5)(Flatten()(x))
        sm_output = Dense(class_cnt, name='sm_out_%d' % i, activation='softmax', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
        softmax_outputs.append(sm_output)
    net = Model(inputs=img_inputs, outputs=softmax_outputs)
    loss_dict = {}
    loss_weights = {}
    for i in range(camera_cnt):
        loss_dict['sm_out_%d' % i] = 'categorical_crossentropy'
        loss_weights['sm_out_%d' % i] = 0.15
    net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss=loss_dict, metrics=['accuracy'], loss_weights=loss_weights)
    plot_model(net, to_file='multi_branch.png')
    return net


def multi_branch_train(train_list, train_dir, class_count, camera_cnt, target_model_path):
    images_in_cameras, labels_in_cameras = load_data(train_list, train_dir, camera_cnt, class_count)

    train_images = [[] for _ in range(camera_cnt)]
    val_images = [[] for _ in range(camera_cnt)]
    train_labels = [[] for _ in range(camera_cnt)]
    val_labels = [[] for _ in range(camera_cnt)]
    for i in range(camera_cnt):
        data_cnt = len(images_in_cameras[i])
        train_images[i] = images_in_cameras[i][:data_cnt//10 * 9]
        train_labels[i] = labels_in_cameras[i][:data_cnt//10 * 9]
        val_images[i] = images_in_cameras[i][data_cnt//10 * 9:]
        val_labels[i] = labels_in_cameras[i][data_cnt//10 * 9:]
    max_train_images_cnt = max([len(train_images[i]) for i in range(camera_cnt)])
    max_val_images_cnt = max([len(val_images[i]) for i in range(camera_cnt)])
    print('max_train_images_cnt:%d' % max_train_images_cnt)
    print('max_val_images_cnt:%d' % max_val_images_cnt)
    net = multi_branch_model(class_count, camera_cnt)
    batch_size = 14
    # early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    # save_best = ModelCheckpoint(target_model_path, monitor='val_loss', save_best_only=True)
    net.fit_generator(multi_generator(train_images, train_labels, batch_size),
                        steps_per_epoch=max_train_images_cnt / batch_size + 1,
                        epochs=12,
                        validation_data=multi_generator(val_images, val_labels, batch_size, train=False),
                        validation_steps=max_val_images_cnt / batch_size + 1,
                        # callbacks=[save_best]
                      )
    net.save(target_model_path)


def multi_softmax_pretrain_on_dataset(source, project_path='/home/cwh/coding/rank-reid', dataset_parent='/home/cwh/coding'):
    camera_cnt = 6
    if source == 'market':
        train_list = project_path + '/dataset/market_train.list'
        train_dir = dataset_parent + '/Market-1501/train'
        class_count = 751
    elif source == 'grid':
        train_list = project_path + '/dataset/grid_train.list'
        train_dir = dataset_parent + '/grid_label'
        class_count = 250
    elif source == 'cuhk':
        train_list = project_path + '/dataset/cuhk_train.list'
        train_dir = dataset_parent + '/cuhk01'
        class_count = 971
    elif source == 'viper':
        train_list = project_path + '/dataset/viper_train.list'
        train_dir = dataset_parent + '/viper'
        class_count = 630
    elif source == 'duke':
        train_list = project_path + '/dataset/duke_train.list'
        train_dir = dataset_parent + '/DukeMTMC-reID/train'
        class_count = 702
        camera_cnt = 8
    elif 'grid-cv' in source:
        cv_idx = int(source.split('-')[-1])
        train_list = project_path + '/dataset/grid-cv/%d.list' % cv_idx
        train_dir = dataset_parent + '/underground_reid/cross%d/train' % cv_idx
        class_count = 125
    elif 'mix' in source:
        train_list = project_path + '/dataset/mix.list'
        train_dir = dataset_parent + '/cuhk_grid_viper_mix'
        class_count = 250 + 971 + 630
    else:
        train_list = 'unknown'
        train_dir = 'unknown'
        class_count = -1
    multi_branch_train(train_list, train_dir, class_count, camera_cnt, '../pretrain/' + source + '_multi_pretrain.h5')


if __name__ == '__main__':
    # sources = ['market', 'grid', 'cuhk', 'viper']
    sources = ['market']
    for source in sources:
        multi_softmax_pretrain_on_dataset(source)
