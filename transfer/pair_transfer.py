import os

import numpy as np
from keras import Input
from keras import backend as K
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.engine import Model
from keras.layers import Lambda, Dense, Dropout, Flatten
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.utils import plot_model, to_categorical
from numpy.random import randint, shuffle, choice

from baseline.evaluate import market_result_eval
from pretrain.eval import test_pair_predict
from utils.file_helper import safe_remove

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def mix_data_prepare(data_list_path, train_dir_path):
    class_img_labels = dict()
    class_cnt = -1
    last_label = -2
    last_type = ''
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img = line
            lbl = int(line.split('_')[0])
            img_type = line.split('.')[-1]
            if lbl != last_label or img_type != last_type:
                class_cnt = class_cnt + 1
                cur_list = list()
                class_img_labels[str(class_cnt)] = cur_list
            last_label = lbl
            last_type = img_type

            img = image.load_img(os.path.join(train_dir_path, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            class_img_labels[str(class_cnt)].append(img[0])
    return class_img_labels


def reid_data_prepare(data_list_path, train_dir_path):
    if 'mix' in data_list_path:
        return mix_data_prepare(data_list_path, train_dir_path)
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


def pair_generator(class_img_labels, batch_size, train=False):
    cur_epoch = 0
    pos_prop = 5
    while True:
        left_label = randint(len(class_img_labels), size=batch_size)
        if cur_epoch % pos_prop == 0:
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
        cur_epoch += 1
        yield [left_images, right_images], [binary_label]


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


def gen_hard_neg_right_img_ids(left_similar_persons, left_similar_matrix, batch_size):
    right_img_ids = list()
    right_img_scores = list()
    for i in range(batch_size):
        hard_right_img_idxes = np.where(left_similar_matrix[i][70: 12500] > 0.5)[0] + 70
        if len(hard_right_img_idxes) < batch_size:
            right_img_idxes = np.concatenate(
                hard_right_img_idxes, randint(70, 12500, batch_size - len(hard_right_img_idxes)))
        else:
            right_img_idxes = hard_right_img_idxes[randint(len(hard_right_img_idxes), size=batch_size)]
        right_img_ids.append(left_similar_persons[i][right_img_idxes[i]])
        right_img_scores.append(left_similar_matrix[i][right_img_idxes[i]])
    right_img_ids = np.array(right_img_ids)
    return right_img_ids, np.array(right_img_scores)/10


def gen_neg_right_img_ids(left_similar_persons, left_similar_matrix, batch_size):
    right_img_ids = list()
    right_img_idxes = randint(100, 12500, size=batch_size)
    right_img_scores = list()
    for i in range(batch_size):
        right_img_ids.append(left_similar_persons[i][right_img_idxes[i]])
        right_img_scores.append(left_similar_matrix[i][right_img_idxes[i]])
    right_img_ids = np.array(right_img_ids)
    return right_img_ids, np.array(right_img_scores)/10


def gen_pos_right_img_ids(left_similar_persons, left_similar_matrix, batch_size):
    right_img_ids = list()
    right_img_idxes = randint(0, 25, size=batch_size)
    right_img_scores = list()
    for i in range(batch_size):
        right_img_ids.append(left_similar_persons[i][right_img_idxes[i]])
        right_img_scores.append(left_similar_matrix[i][right_img_idxes[i]])
    right_img_ids = np.array(right_img_ids)
    return right_img_ids, np.array(right_img_scores)


def gen_right_img_infos(cur_epoch, similar_matrix, similar_persons, left_img_ids, img_cnt, batch_size):
    pos_prop = 2
    if cur_epoch % pos_prop == 0:
        print 'gen_pos_right_img_ids: %d' % cur_epoch
        # select from last match for negative
        left_similar_persons = similar_persons[left_img_ids]
        left_similar_matrix = similar_matrix[left_img_ids]
        right_img_ids, right_img_scores = gen_pos_right_img_ids(left_similar_persons, left_similar_matrix, batch_size)
    else:
        # select from last match for negative
        print 'gen_neg_right_img_ids: %d' % cur_epoch
        left_similar_persons = similar_persons[left_img_ids]
        left_similar_matrix = similar_matrix[left_img_ids]
        right_img_ids, right_img_scores = gen_neg_right_img_ids(left_similar_persons, left_similar_matrix, batch_size)
    right_img_ids = right_img_ids.astype(int)
    return right_img_ids, right_img_scores


def pair_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=False):
    cur_epoch = 0
    img_cnt = len(similar_persons)
    while True:
        if train:
            left_img_ids = randint(img_cnt / 10 * 9, size=batch_size)
        else:
            left_img_ids = randint(img_cnt / 10 * 9, img_cnt, size=batch_size)
        right_img_ids, right_img_scores = gen_right_img_infos(cur_epoch,
                                                              similar_matrix, similar_persons,
                                                              left_img_ids,
                                                              img_cnt, batch_size)

        left_images = train_images[left_img_ids]
        right_images1 = train_images[right_img_ids]
        print right_img_scores
        cur_epoch += 1
        yield [left_images, right_images1], [right_img_scores]


def eucl_dist(inputs):
    x, y = inputs
    # return K.mean(K.square((x - y)), axis=1)
    return K.square((x - y))


def cos_dist(inputs):
    x, y = inputs
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    return K.dot(x, K.transpose(y))


def dis_sigmoid(dis):
    return K.expand_dims(2 / (1 + K.exp(dis)))


def focal_loss_fixed(y_true, y_pred):
    gamma = 2.
    alpha = .25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def pair_model(source_model_path, num_classes):
    pair_model = load_model(source_model_path)
    # base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    base_model = pair_model.layers[2]
    base_model = Model(inputs=base_model.get_input_at(0), outputs=[base_model.get_output_at(0)], name='resnet50')
    img1 = Input(shape=(224, 224, 3), name='img_1')
    img2 = Input(shape=(224, 224, 3), name='img_2')
    feature1 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(Flatten()(base_model(img1)))
    feature2 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(Flatten()(base_model(img2)))
    dis = Lambda(eucl_dist, name='square')([feature1, feature2])
    # judge = Lambda(dis_sigmoid, name='bin_out')(dis)
    judge = Dense(1, activation='sigmoid', name='bin_out')(Dropout(0.8)(dis))

    model = Model(inputs=[img1, img2], outputs=[judge])
    model.get_layer('bin_out').set_weights(pair_model.get_layer('bin_out').get_weights())
    plot_model(model, to_file='model_combined.png')
    # for layer in base_model.layers[:-10]:
    #     layer.trainable = False
    # for layer in base_model.layers:
    #     layer.trainable = True
    return model


def common_lr(epoch):
    if epoch < 20:
        return 0.01
    else:
        return 0.001


def pair_tune(source_model_path, train_generator, val_generator, tune_dataset, batch_size=48, num_classes=751):
    model = pair_model(source_model_path, num_classes)
    model.compile(
        optimizer=SGD(lr=0.001, momentum=0.9),
        # optimizer='adadelta',
        loss={'bin_out': 'binary_crossentropy'},
        loss_weights={
            'bin_out': 1.
        },
        metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001,
                                cooldown=0, min_lr=0)
    # save_model = ModelCheckpoint('resnet50-{epoch:02d}-{val_ctg_out_1_acc:.2f}.h5', period=2)
    model.fit_generator(train_generator,
                        steps_per_epoch=16500 / batch_size + 1,
                        epochs=10,
                        validation_data=val_generator,
                        validation_steps=1800 / batch_size + 1,
                        callbacks=[auto_lr, early_stopping,
                                   ModelCheckpoint(tune_dataset + '_pair_pretrain.h5', monitor='val_loss', verbose=0,
                                                   save_best_only=True, save_weights_only=False,
                                                   mode='auto', period=1)
                                   ])
    # model.save(tune_dataset + '_pair_pretrain.h5')


def pair_pretrain_on_dataset(source, target, project_path='/home/cwh/coding/rank-reid',
                             dataset_parent='/home/cwh/coding'):
    if target == 'market':
        train_list = project_path + '/dataset/market_train.list'
        train_dir = dataset_parent + '/Market-1501/train'
        class_count = 751
    elif target == 'markets1':
        train_list = project_path + '/dataset/markets1_train.list'
        train_dir = dataset_parent + '/markets1'
        class_count = 751
    elif target == 'grid':
        train_list = project_path + '/dataset/grid_train.list'
        train_dir = dataset_parent + '/grid_label'
        class_count = 250
    elif target == 'cuhk':
        train_list = project_path + '/dataset/cuhk_train.list'
        train_dir = dataset_parent + '/cuhk01'
        class_count = 971
    elif target == 'viper':
        train_list = project_path + '/dataset/viper_train.list'
        train_dir = dataset_parent + '/viper'
        class_count = 630
    elif target == 'duke':
        train_list = project_path + '/dataset/duke_train.list'
        train_dir = dataset_parent + '/DukeMTMC-reID/train'
        class_count = 702
    elif 'grid-cv' in target:
        cv_idx = int(target.split('-')[-1])
        train_list = project_path + '/dataset/grid-cv/%d.list' % cv_idx
        train_dir = dataset_parent + '/grid_train_probe_gallery/cross%d/train' % cv_idx
        class_count = 125
    elif 'mix' in target:
        train_list = project_path + '/dataset/mix.list'
        train_dir = dataset_parent + '/cuhk_grid_viper_mix'
        class_count = 250 + 971 + 630
    else:
        train_list = 'unknown'
        train_dir = 'unknown'
        class_count = -1

    train_images = reid_img_prepare(train_list, train_dir)
    # similar_persons = np.genfromtxt('/home/cwh/coding/rank-reid/data_clean/cross_filter_pid.log', delimiter=' ')
    similar_persons = np.genfromtxt('/home/cwh/coding/TrackViz/data/cuhk_market-train/cross_filter_pid.log', delimiter=' ')
    # if 'cross' in rank_pid_path:
    #     similar_persons = similar_persons - 1
    # similar_matrix = np.genfromtxt('/home/cwh/coding/rank-reid/data_clean/cross_filter_score.log', delimiter=' ')
    # similar_matrix = np.genfromtxt('/home/cwh/coding/TrackViz/sorted_vision_score.txt', delimiter=' ')
    similar_matrix = np.genfromtxt('/home/cwh/coding/TrackViz/data/cuhk_market-train/cross_filter_score.log', delimiter=' ')
    batch_size = 16
    pair_tune(
        '../pretrain/' + source + '_pair_pretrain.h5',
        pair_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=True),
        pair_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=False),
        target,
        batch_size=batch_size, num_classes=class_count
    )


if __name__ == '__main__':
    # sources = ['cuhk_grid_viper_mix']
    sources = ['cuhk']
    target = 'market'
    pair_model('../pretrain/cuhk_pair_pretrain.h5', 751)
    for source in sources:
        pair_pretrain_on_dataset(source, target)

    transform_dir = '/home/cwh/coding/Market-1501'
    safe_remove('pair_transfer_pid.log')
    test_pair_predict('market_pair_pretrain.h5',
                      transform_dir + '/probe', transform_dir + '/test',
                      'pair_transfer_pid.log', 'pair_transfer_score.log')
    market_result_eval('pair_transfer_pid.log', TEST='/home/cwh/coding/Market-1501/test',
                       QUERY='/home/cwh/coding/Market-1501/probe')

    # sources = ['grid-cv-%d' % i for i in range(10)]
    # for source in sources:
    #     softmax_pretrain_on_dataset(source,
    #                                 project_path='/home/cwh/coding/rank-reid',
    #                                 dataset_parent='/home/cwh/coding')
    #     pair_pretrain_on_dataset(source,
    #                              project_path='/home/cwh/coding/rank-reid',
    #                              dataset_parent='/home/cwh/coding')
