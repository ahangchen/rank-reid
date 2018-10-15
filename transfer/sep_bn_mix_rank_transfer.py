import os

from keras.regularizers import l1_l2

import utils.cuda_util
import numpy as np
from keras import Input
from keras import backend as K
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.engine import Model
from keras.layers import Flatten, Lambda, Dense, Conv2D, Dropout, BatchNormalization
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
from keras.utils import plot_model, to_categorical
from numpy.random import randint, choice, shuffle

from transfer.triplet_resnet import Triplet_ResNet50
from utils.file_helper import safe_remove


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


def reid_data_prepare(data_list_path, train_dir_path):
    class_img_labels = dict()
    class_cnt = -1
    last_label = -2
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img = line
            lbl = int(line.split('_')[0])
            if lbl != last_label:
                class_cnt = class_cnt + 1
                cur_list = list()
                class_img_labels[str(class_cnt)] = cur_list
            last_label = lbl

            img = image.load_img(os.path.join(train_dir_path, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            class_img_labels[str(class_cnt)].append(img[0])

    return class_img_labels


def gen_neg_right_img_ids(left_similar_persons, left_similar_matrix, batch_size):
    right_img_ids = list()
    # right_img_idxes1 = randint(0, 25, size=batch_size/2)
    # right_img_idxes2 = randint(50, 12500, size=batch_size/2)
    # right_img_idxes = np.concatenate([right_img_idxes2, right_img_idxes1])
    right_img_idxes = randint(50, len(left_similar_persons[0]), size=batch_size)
    # right_img_idxes = randint(25, 50, size=batch_size)
    right_img_scores = list()
    for i in range(batch_size):
        right_img_ids.append(left_similar_persons[i][right_img_idxes[i]])
        right_img_scores.append(left_similar_matrix[i][right_img_idxes[i]])
    right_img_ids = np.array(right_img_ids)
    return right_img_ids, np.array(right_img_scores)


def gen_pos_right_img_ids(left_similar_persons, left_similar_matrix, batch_size):
    right_img_ids = list()
    # right_img_idxes1 = randint(0, 25, size=batch_size / 2)
    # right_img_idxes2 = randint(50, 12500, size=batch_size / 2)
    # right_img_idxes = np.concatenate([right_img_idxes1, right_img_idxes2])
    right_img_idxes = randint(0, 25, size=batch_size)
    right_img_scores = list()
    for i in range(batch_size):
        right_img_ids.append(left_similar_persons[i][right_img_idxes[i]])
        right_img_scores.append(left_similar_matrix[i][right_img_idxes[i]])
    right_img_ids = np.array(right_img_ids)
    return right_img_ids, np.array(right_img_scores)


def gen_right_img_infos(cur_epoch, similar_matrix, similar_persons, left_img_ids, img_cnt, batch_size,
                        vision_matrix=None):
    pos_prop = 2
    if cur_epoch % pos_prop != 0:
        print '\ngen_pos_right_img_ids, epoch: %d' % cur_epoch
        # select from last match for negative
        left_similar_persons = similar_persons[left_img_ids]
        left_similar_matrix = similar_matrix[left_img_ids]
        right_img_ids, right_img_scores = gen_pos_right_img_ids(left_similar_persons, left_similar_matrix, batch_size)
    else:
        # select from last match for negative
        print 'gen_neg_right_img_ids, epoch: %d' % cur_epoch
        left_similar_persons = similar_persons[left_img_ids]
        left_similar_matrix = similar_matrix[left_img_ids]
        right_img_ids, right_img_scores = gen_neg_right_img_ids(left_similar_persons, left_similar_matrix, batch_size)
    right_img_ids = right_img_ids.astype(int)
    return right_img_ids, right_img_scores


def pair_generator(left_label, class_img_labels, batch_size, cur_step, train=False):
    pos_prop = 5
    if cur_step % pos_prop == 0:
        right_label = left_label
    else:
        right_label = np.copy(left_label)
        shuffle(right_label)
    # select by label
    left_images = list()
    right_images = list()
    if train:
        slice_start = 0
    else:
        # val
        slice_start = 0.9
    for i in range(batch_size):
        len_left_label_i = len(class_img_labels[str(left_label[i])])
        left_images.append(class_img_labels[str(left_label[i])][int(slice_start * len_left_label_i):][
                               choice(len_left_label_i - int(len_left_label_i * slice_start))])
        len_right_label_i = len(class_img_labels[str(right_label[i])])
        right_images.append(class_img_labels[str(right_label[i])][int(slice_start * len_right_label_i):][
                                choice(len_right_label_i - int(len_right_label_i * slice_start))])

    left_images = np.array(left_images)
    right_images = np.array(right_images)
    binary_label = (left_label == right_label).astype(int)
    left_label = to_categorical(left_label, num_classes=len(class_img_labels))
    right_label = to_categorical(right_label, num_classes=len(class_img_labels))
    return left_images, right_images, left_label, right_label, binary_label


def sub(inputs):
    x, y = inputs
    return (x - y)


def triplet_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=False,
                                   source_datas=None):
    cur_step = 0
    img_cnt = len(similar_persons)
    while True:
        if cur_step % 3 == 0:
            left_ids = randint(len(source_datas), size=batch_size)
            left_images, right_images1, left_label, right_label1, right_img_scores1 = pair_generator(left_ids,
                                                                                                     source_datas,
                                                                                                     batch_size,
                                                                                                     cur_step, train)
            left_images, right_images2, left_label, right_label2, right_img_scores2 = pair_generator(left_ids,
                                                                                                     source_datas,
                                                                                                     batch_size,
                                                                                                     cur_step + 1,
                                                                                                     train)
            sub_scores = right_img_scores1 - right_img_scores2
            cur_step += 1
            print right_img_scores1
            print right_img_scores2
            yield [left_images, right_images1, right_images2], \
                  [sub_scores, right_img_scores1, right_img_scores2]
        else:
            if train:
                left_img_ids = randint(img_cnt * 9 / 10, size=batch_size)
            else:
                left_img_ids = randint(img_cnt * 9 / 10, img_cnt, size=batch_size)

            right_img_ids1, right_img_scores1 = gen_right_img_infos(cur_step,
                                                                    similar_matrix, similar_persons,
                                                                    left_img_ids,
                                                                    img_cnt, batch_size)
            right_img_ids2, right_img_scores2 = gen_right_img_infos(cur_step + 1,
                                                                    similar_matrix, similar_persons,
                                                                    left_img_ids,
                                                                    img_cnt, batch_size)
            left_images = train_images[left_img_ids]
            right_images1 = train_images[right_img_ids1]
            right_images2 = train_images[right_img_ids2]
            sub_scores = right_img_scores1 - right_img_scores2
            print right_img_scores1
            print right_img_scores2
            cur_step += 1

            yield [left_images, right_images1, right_images2], \
                  [sub_scores, right_img_scores1, right_img_scores2]


def eucl_dist(inputs):
    x, y = inputs
    # return K.mean(K.square((x - y)), axis=1)
    # return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True) + K.epsilon())
    return K.square(x - y)


def cross_entropy_loss(real_score, predict_score):
    predict_prob = 1 / (1 + K.exp(-predict_score))
    real_prob = 1 / (1 + K.exp(-real_score))
    cross_entropy = -real_prob * K.log(predict_prob) - (1 - real_prob) * K.log(1 - predict_prob)
    return cross_entropy


def rank_transfer_model(pair_model_path):
    pair_model = load_model(pair_model_path, custom_objects={'cross_entropy_loss': cross_entropy_loss})
    img0 = Input(shape=(224, 224, 3), name='img_0')
    img1 = Input(shape=(224, 224, 3), name='img_1')
    img2 = Input(shape=(224, 224, 3), name='img_2')


    if len(pair_model.layers) == 17:
        model = pair_model
    else:
        resnet = pair_model.layers[2]
        base_model = Triplet_ResNet50(resnet)
        avg_pool0, avg_pool1, avg_pool2 = base_model([img0, img1, img2])
        flatten0 = Flatten()(avg_pool0)
        flatten1 = Flatten()(avg_pool1)
        flatten2 = Flatten()(avg_pool2)
        feature0 = Lambda(lambda x: K.l2_normalize(x, axis=1))(flatten0)
        feature1 = Lambda(lambda x: K.l2_normalize(x, axis=1))(flatten1)
        feature2 = Lambda(lambda x: K.l2_normalize(x, axis=1))(flatten2)

        dis1 = Lambda(eucl_dist, name='square1')([feature0, feature1])
        dis2 = Lambda(eucl_dist, name='square2')([feature0, feature2])

        score1 = Dense(1, activation='sigmoid', name='score1')(Dropout(0.9)(dis1))
        score2 = Dense(1, activation='sigmoid', name='score2')(Dropout(0.9)(dis2))
        sub_score = Lambda(sub, name='sub_score')([score1, score2])

        model = Model(inputs=[img0, img1, img2], outputs=[sub_score, score1, score2])

        model.get_layer('score1').set_weights(pair_model.get_layer('bin_out').get_weights())
        model.get_layer('score2').set_weights(pair_model.get_layer('bin_out').get_weights())




    plot_model(model, to_file='rank_model.png')

    print(model.summary())
    return model


def rank_transfer(train_generator, val_generator, source_model_path, target_model_path, batch_size=48):
    model = rank_transfer_model(source_model_path)
    model.compile(
        optimizer=SGD(lr=0.0005, momentum=0.9),  # 'adam',
        loss={
            'sub_score': cross_entropy_loss,
            'score1': 'binary_crossentropy',
            'score2': 'binary_crossentropy',
        },
        loss_weights={
            'sub_score': 1.,
            'score1': 0.5,
            'score2': 0.5,
        },
        # metrics=['accuracy']
    )

    # early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001,
                                cooldown=0, min_lr=0)
    if 'market-' in target_model_path:
        train_data_cnt = 16500
        val_data_cnt = 1800
    else:
        train_data_cnt = 1600
        val_data_cnt = 180
    safe_remove(target_model_path)
    model.fit_generator(train_generator,
                        steps_per_epoch=train_data_cnt / batch_size + 1,
                        epochs=5,
                        validation_data=val_generator,
                        validation_steps=val_data_cnt / batch_size + 1,
                        callbacks=[
                            auto_lr,  # early_stopping,  # monitor_bn_nan,
                            ModelCheckpoint(target_model_path, monitor='val_loss', verbose=0,
                                            save_best_only=True, save_weights_only=False,
                                            mode='auto', period=1)
                        ]
                        )


def delta_sure_pos_idxs(delta_matrix_path):
    delta_matrix = np.genfromtxt(delta_matrix_path, delimiter=' ')
    delta_matrix_top = delta_matrix[:, 0].reshape(-1)
    delta_sure_pos_idx = np.where(delta_matrix_top > 0.8)
    delta_sure_neg_idx = np.where(delta_matrix_top < 0.1)
    return delta_sure_pos_idx, delta_sure_neg_idx


def rank_transfer_2dataset(source_pair_model_path, target_train_list, target_model_path, target_train_path,
                           rank_pid_path, rank_score_path, source_train_list, source_train_dir):
    train_images = reid_img_prepare(target_train_list, target_train_path)
    class_img_labels = reid_data_prepare(source_train_list, source_train_dir)
    batch_size = 24
    similar_persons = np.genfromtxt(rank_pid_path, delimiter=' ')
    similar_matrix = np.genfromtxt(rank_score_path, delimiter=' ')

    rank_transfer(
        triplet_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=True,
                                       source_datas=class_img_labels),
        triplet_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=False,
                                       source_datas=class_img_labels),
        source_pair_model_path,
        target_model_path,
        batch_size=batch_size
    )


if __name__ == '__main__':
    pair_model = load_model('../pretrain/cuhk_pair_pretrain.h5')
    # pair_model = load_model('../cuhk_market-rank_transfer.h5')
    base_model = pair_model.layers[3]
    base_model = Model(inputs=base_model.get_input_at(0), outputs=[base_model.get_output_at(0)], name='resnet50')
    print isinstance(base_model.layers[-20], Conv2D)
    print isinstance(base_model.layers[-20], BatchNormalization)

    rank_transfer_2dataset('../pretrain/cuhk_pair_pretrain.h5', '../dataset/market_train.list',
                           'rank_transfer_test.h5',
                           '/home/cwh/coding/Market-1501/train',
                           '/home/cwh/coding/rank-reid/data_clean/cross_filter_pid.log',
                           '/home/cwh/coding/rank-reid/data_clean/cross_filter_score.log')
