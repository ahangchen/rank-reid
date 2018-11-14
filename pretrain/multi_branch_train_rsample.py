from __future__ import division, print_function, absolute_import

import os
import utils.cuda_util_test
from random import shuffle
import keras.backend as K

import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import TensorBoard
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from utils.file_helper import safe_rmdir


def load_data(LIST, TRAIN, camera_cnt, class_cnt):
    images, labels = [], []
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
            images.append(img)
            camera_labels = [-1 for _ in range(camera_cnt)]
            camera_labels[camera] = label_cnts[camera]
            labels.append(camera_labels)
    print('labels cnt')
    print(label_cnts)
    img_cnt = len(images)
    shuffle_idxes = range(img_cnt)
    shuffle(shuffle_idxes)
    shuffle_imgs = list()
    labels_in_cameras = [[] for _ in range(camera_cnt)]
    for idx in shuffle_idxes:
        shuffle_imgs.append(images[idx].reshape(224, 224, 3))
        for ci in range(camera_cnt):
            if labels[idx][ci] == -1:
                labels_in_cameras[ci].append(np.zeros([label_cnts[ci] + 1]))
            else:
                labels_in_cameras[ci].append(to_categorical(labels[idx][ci], label_cnts[ci] + 1))
    images_in_cameras = np.array(shuffle_imgs)
    print(images_in_cameras.shape)
    for ci in range(camera_cnt):
        labels_in_cameras[ci] = np.array(labels_in_cameras[ci])
    return images_in_cameras, labels_in_cameras


def multi_generator(images_in_cameras, labels_in_cameras, batch_size, train=True):
    camera_cnt = len(labels_in_cameras)
    if train:
        img_aug = ImageDataGenerator(
            rotation_range=30, brightness_range=[0.8, 1.0], zoom_range=0.1,
            shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
            horizontal_flip=0.5)
    else:
        img_aug = ImageDataGenerator()
    generator = img_aug.flow(images_in_cameras, batch_size=batch_size)
    img_cnt = len(images_in_cameras)
    cur_cnt = 0
    while True:
        feed_labels = []
        images = generator.next()
        if len(images) != batch_size:
            images = generator.next()
            cur_cnt = cur_cnt % img_cnt
        for i in range(camera_cnt):
            feed_labels.append(labels_in_cameras[i][cur_cnt: cur_cnt + batch_size])
        yield images, feed_labels


def selective_categorical_crossentropy(y_true, y_pred):
    # print('y_pred.shape')
    # print(y_pred.shape)
    # print('y_true.shape')
    # print(y_true.shape)
    return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true, axis=1)

def selective_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())* K.sum(y_true, axis=1)


# def selective_accuracy(y_true, y_pred):



def multi_branch_model(label_sizes, camera_cnt):
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    softmax_outputs = []
    img_input = Input(shape=(224, 224, 3))
    avp = base_model(img_input)
    for i in range(camera_cnt):
        x = Dropout(0.5)(Flatten()(avp))
        x = Dense(label_sizes[i])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        sm_output = Activation('softmax', name='sm_out_%d' % i)(x)
        softmax_outputs.append(sm_output)
    net = Model(inputs=[img_input], outputs=softmax_outputs)
    # plot_model(net, to_file='multi_branch.png')
    return net


def multi_branch_train(train_list, train_dir, class_count, camera_cnt, target_model_path):
    images_in_cameras, labels_in_cameras = load_data(train_list, train_dir, camera_cnt, class_count)
    data_cnt = len(images_in_cameras)
    train_images = images_in_cameras[:data_cnt // 10 * 9]
    train_labels = [labels_in_cameras[i][:data_cnt // 10 * 9] for i in range(camera_cnt)]
    val_images = images_in_cameras[data_cnt // 10 * 9:]
    val_labels = [labels_in_cameras[i][data_cnt // 10 * 9:] for i in range(camera_cnt)]

    train_images_cnt = len(train_images)
    val_images_cnt = len(val_images)

    loss_dict = {}
    for i in range(camera_cnt):
        loss_dict['sm_out_%d' % i] = selective_categorical_crossentropy
    label_sizes = [len(labels_in_cameras[ci][0]) for ci in range(camera_cnt)]
    print(label_sizes)
    net = multi_branch_model(label_sizes, camera_cnt)
    batch_size = 12
    net.compile(optimizer=Adam(lr=3.5e-4), loss=loss_dict,
                    metrics=[selective_accuracy])
    log_path = target_model_path.replace('.h5', '_logs')
    safe_rmdir(log_path)
    net.fit_generator(multi_generator(train_images, train_labels, batch_size),
                      steps_per_epoch=train_images_cnt / batch_size  + 1,
                      epochs=30,
                      validation_data=multi_generator(val_images, val_labels, 90, train=False),
                      # validation_data=multi_generator(val_images, val_labels, 90, train=False),
                      validation_steps=val_images_cnt / 90 + 1,
                      verbose=2,
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
    multi_branch_train(train_list, train_dir, class_count, camera_cnt, '../pretrain/' + source + '_rs_multi_pretrain.h5')


if __name__ == '__main__':
    # sources = ['market', 'grid', 'cuhk', 'viper']
    sources = ['market']
    for source in sources:
        multi_softmax_pretrain_on_dataset(source)
