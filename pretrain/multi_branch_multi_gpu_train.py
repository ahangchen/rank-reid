from __future__ import division, print_function, absolute_import

import math
import os
import tensorflow as tf
import utils.cuda_util_test
from random import shuffle

import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.utils.training_utils import multi_gpu_model

from utils.file_helper import safe_remove, safe_rmdir


def load_data(LIST, TRAIN, camera_cnt, class_cnt):
    images_in_cameras, labels_in_cameras = [[] for _ in range(camera_cnt)], [[] for _ in range(camera_cnt)]
    last_labels, label_cnts = [-1 for _ in range(camera_cnt)], [-1 for _ in range(camera_cnt)]
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
            rotation_range=30, brightness_range=[0.8, 1.0], zoom_range=0.1,
            shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
            horizontal_flip=0.5)
    else:
        img_aug = ImageDataGenerator()
    generators = [img_aug.flow(images_in_cameras[i], labels_in_cameras[i], batch_size=batch_size) for i in
                  range(camera_cnt)]
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
    for layer in base_model.layers[: len(base_model.layers) // 3]:
        layer.trainable = False
    img_inputs = []
    softmax_outputs = []
    for i in range(camera_cnt):
        img_inputs.append(Input(shape=(224, 224, 3), name='img_%d' % i))
        x = base_model(img_inputs[i])
        x = Dropout(0.5)(Flatten()(x))
        sm_output = Dense(class_cnt, name='sm_out_%d' % i, activation='softmax')(x)
        softmax_outputs.append(sm_output)
    net = Model(inputs=img_inputs, outputs=softmax_outputs)
    # plot_model(net, to_file='multi_branch.png')
    return net


def multi_branch_train(train_list, train_dir, class_count, camera_cnt, target_model_path):
    images_in_cameras, labels_in_cameras = load_data(train_list, train_dir, camera_cnt, class_count)
    train_images = [[] for _ in range(camera_cnt)]
    val_images = [[] for _ in range(camera_cnt)]
    train_labels = [[] for _ in range(camera_cnt)]
    val_labels = [[] for _ in range(camera_cnt)]
    for i in range(camera_cnt):
        data_cnt = len(images_in_cameras[i])
        train_images[i] = images_in_cameras[i][:data_cnt // 10 * 9]
        train_labels[i] = labels_in_cameras[i][:data_cnt // 10 * 9]
        val_images[i] = images_in_cameras[i][data_cnt // 10 * 9:]
        val_labels[i] = labels_in_cameras[i][data_cnt // 10 * 9:]
    train_images_cnt = [len(train_images[i]) for i in range(camera_cnt)]
    sum_train_images_cnt = sum(train_images_cnt)
    loss_weights = [train_images_cnt[i] / sum_train_images_cnt for i in range(camera_cnt)]
    print('loss weights')
    print(loss_weights)
    max_train_images_cnt = max(train_images_cnt)
    max_val_images_cnt = max([len(val_images[i]) for i in range(camera_cnt)])
    print('max_train_images_cnt:%d' % max_train_images_cnt)
    print('max_val_images_cnt:%d' % max_val_images_cnt)

    loss_dict = {}
    loss_weights_dict = {}
    for i in range(camera_cnt):
        loss_dict['sm_out_%d' % i] = 'categorical_crossentropy'
        loss_weights_dict['sm_out_%d' % i] = loss_weights[i]
    with tf.device("/cpu:0"):
        net = multi_branch_model(class_count, camera_cnt)
    batch_size = 64

    net.get_layer('resnet50').trainable = False
    multi_model = multi_gpu_model(net, 4)
    multi_model.compile(optimizer=Adam(lr=0.001), loss=loss_dict,
                metrics=['accuracy'], loss_weights=loss_weights_dict)

    multi_model.fit_generator(multi_generator(train_images, train_labels, batch_size),
                      steps_per_epoch=max_train_images_cnt / batch_size + 1,
                      epochs=5,
                      validation_data=multi_generator(val_images, val_labels, batch_size, train=False),
                      validation_steps=max_val_images_cnt / batch_size + 1,
                      verbose=2
                      )

    net.get_layer('resnet50').trainable = True
    multi_model = multi_gpu_model(net, 4)

    multi_model.compile(optimizer=SGD(lr=3.5e-4, momentum=0.9, decay=0.01), loss=loss_dict,
                metrics=['accuracy'], loss_weights=loss_weights_dict)
    log_path = target_model_path.replace('.h5', '_logs')
    safe_rmdir(log_path)
    tb = TensorBoard(log_dir=log_path, histogram_freq=1, write_graph=False)
    # save_best = ModelCheckpoint(target_model_path, save_best_only=True)

    multi_model.fit_generator(multi_generator(train_images, train_labels, batch_size),
                      steps_per_epoch=max_train_images_cnt / batch_size  + 1,
                      epochs=25,
                      validation_data=next(multi_generator(val_images, val_labels, 90, train=False)),
                      # validat ion_data=multi_generator(val_images, val_labels, 90, train=False),
                      validation_steps=max_val_images_cnt / 90 + 1,
                      verbose=2,
                      # callbacks=[tb]
                      )
    net.save(target_model_path)


def multi_softmax_pretrain_on_dataset(source, project_path='/home/cwh/coding/rank-reid',
                                      dataset_parent='/home/cwh/coding'):
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
