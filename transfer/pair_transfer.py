import os

from pretrain.eval import test_pair_predict

import cuda_util
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.engine import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras.utils import plot_model, to_categorical
from numpy.random import randint


def reid_img_prepare(LIST, TRAIN):
    images = []
    with open(LIST, 'r') as f:
        for line in f:
            if 'jp' not in line:
                continue
            line = line.strip()
            img = line.split()[0]
            img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            images.append(img[0])
    images = np.array(images)
    return images


def gen_neg_right_img_ids(left_similar_persons, img_cnt, batch_size):
    right_img_ids = list()
    right_img_idxes = randint(img_cnt * 9 / 10, img_cnt - 1, size=batch_size)
    for i in range(batch_size):
        right_img_ids.append(left_similar_persons[i][right_img_idxes[i]])
    right_img_ids = np.array(right_img_ids)
    binary_labels = np.zeros(batch_size)
    return right_img_ids, binary_labels


def gen_right_img_ids(cur_epoch, mid_score, similar_matrix, similar_persons, left_img_ids, img_cnt, batch_size):
    pos_prop = 4
    if cur_epoch % pos_prop == 0:
        # select from rank1 and similarity > mid_score as pos samples
        pos_right_idxes = np.where(similar_matrix[left_img_ids, :1] > mid_score)
        neg_left_img_ids = left_img_ids[np.setdiff1d(np.arange(batch_size), np.array(pos_right_idxes).reshape(-1))]

        pos_right_img_ids = similar_persons[left_img_ids][pos_right_idxes]
        if len(neg_left_img_ids) > 0:
            # other turn to negative samples
            neg_left_similar_persons = similar_persons[neg_left_img_ids]
            neg_right_img_ids, neg_binary_labels = gen_neg_right_img_ids(neg_left_similar_persons, img_cnt,
                                                                         len(neg_left_img_ids))
            right_img_ids = np.concatenate((pos_right_img_ids, neg_right_img_ids))
            binary_labels = np.concatenate([np.ones(len(pos_right_img_ids)), neg_binary_labels])
        else:
            right_img_ids = np.array(pos_right_img_ids).reshape(-1)
            binary_labels = np.ones(batch_size)

    else:
        # select from last match for negative
        left_similar_persons = similar_persons[left_img_ids]
        right_img_ids, binary_labels = gen_neg_right_img_ids(left_similar_persons, img_cnt, batch_size)
    right_img_ids = right_img_ids.astype(int)
    return right_img_ids, binary_labels


def pair_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=False):
    cur_epoch = 0

    img_cnt = len(similar_persons)
    rank1_scores = similar_matrix[:, :1].reshape(img_cnt)
    sorted_score = np.sort(-rank1_scores)
    mid_score = sorted_score[img_cnt / 2]
    while True:
        left_img_ids = randint(img_cnt, size=batch_size)
        right_img_ids, binary_labels = gen_right_img_ids(cur_epoch, mid_score,
                                                         similar_matrix, similar_persons,
                                                         left_img_ids,
                                                         img_cnt, batch_size)
        left_images = train_images[left_img_ids]
        right_images = train_images[right_img_ids]
        cur_epoch += 1
        yield [left_images, right_images], [to_categorical(binary_labels, 2)]


def eucl_dist(inputs):
    x, y = inputs
    return (x - y) ** 2


def pair_transfer_model(pair_model_path):
    base_model = load_model(pair_model_path)
    model = Model(inputs=base_model.inputs, outputs=[base_model.get_layer('bin_out').output], name='binary_model')
    plot_model(model, to_file='model_transfer.png')
    return model


def pair_transfer(train_generator, val_generator, source_model_path, batch_size=48):
    model = pair_transfer_model(source_model_path)
    model.compile(optimizer='nadam',
                  loss={'bin_out': 'categorical_crossentropy'},
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001,
                                cooldown=0, min_lr=0)
    model.fit_generator(train_generator,
                        steps_per_epoch=16500 / batch_size + 1,
                        epochs=30,
                        validation_data=val_generator,
                        validation_steps=1800 / batch_size + 1,
                        callbacks=[early_stopping, auto_lr])
    model.save('pair_transfer.h5')


def pair_transfer_2market():
    DATASET = '../dataset/Market'
    LIST = os.path.join(DATASET, 'pretrain.list')
    TRAIN = os.path.join(DATASET, 'bounding_box_train')
    train_images = reid_img_prepare(LIST, TRAIN)
    batch_size = 64
    similar_persons = np.genfromtxt('../pretrain/train_renew_pid.log', delimiter=' ')
    similar_matrix = np.genfromtxt('../pretrain/train_renew_ac.log', delimiter=' ')
    pair_transfer(
        pair_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=True),
        pair_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=False),
        '../pretrain/pair_pretrain.h5',
        batch_size=batch_size
    )


def pair_transfer_2grid():
    DATASET = '/home/cwh/coding/grid_train_probe_gallery/cross0'
    LIST = os.path.join(DATASET, 'pretrain/test_track.txt')
    TRAIN = os.path.join(DATASET, 'pretrain')
    train_images = reid_img_prepare(LIST, TRAIN)
    batch_size = 64
    # similar_persons = np.genfromtxt('../pretrain/grid_cross0/train_renew_pid.log', delimiter=' ')
    # similar_matrix = np.genfromtxt('../pretrain/grid_cross0/train_renew_ac.log', delimiter=' ')
    similar_persons = np.genfromtxt('../pretrain/grid_cross0/cross_filter_pid.log', delimiter=' ') - 1
    similar_matrix = np.genfromtxt('../pretrain/grid_cross0/cross_filter_score.log', delimiter=' ')

    pair_transfer(
        pair_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=True),
        pair_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=False),
        '../pretrain/pair_pretrain.h5',
        batch_size=batch_size
    )


if __name__ == '__main__':
    pair_transfer_2grid()
    test_pair_predict('../transfer/pair_transfer.h5',
                      '/home/cwh/coding/grid_train_probe_gallery/cross0/probe',
                      '/home/cwh/coding/grid_train_probe_gallery/cross0/gallery',
                      'pid_path', 'score_path'
                      )
