import os
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
from numpy.random import randint

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


def gen_neg_right_img_ids(left_similar_persons, left_similar_matrix, batch_size):
    right_img_ids = list()
    # right_img_idxes1 = randint(0, 25, size=batch_size/2)
    # right_img_idxes2 = randint(50, 12500, size=batch_size/2)
    # right_img_idxes = np.concatenate([right_img_idxes2, right_img_idxes1])
    right_img_idxes = randint(50, len(left_similar_persons[0]), size=batch_size)
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


def gen_right_img_infos(cur_epoch, similar_matrix, similar_persons, left_img_ids, img_cnt, batch_size):
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


def triplet_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=False, sure_pos_idxes=None):
    cur_epoch = 0

    img_cnt = len(similar_persons)

    while True:
        if train:
            left_img_ids = randint(img_cnt*9/10, size=batch_size)

        else:
            left_img_ids = randint(img_cnt*9/10, img_cnt, size=batch_size)
        right_img_ids1, right_img_scores1 = gen_right_img_infos(cur_epoch,
                                                                similar_matrix, similar_persons,
                                                                left_img_ids,
                                                                img_cnt, batch_size)
        right_img_ids2, right_img_scores2 = gen_right_img_infos(cur_epoch + 1,
                                                                similar_matrix, similar_persons,
                                                                left_img_ids,
                                                                img_cnt, batch_size)
        left_images = train_images[left_img_ids]
        right_images1 = train_images[right_img_ids1]
        right_images2 = train_images[right_img_ids2]
        cur_epoch += 1
        # print cur_epoch
        print right_img_scores1
        print right_img_scores2
        print 'left pos, right neg, epoch: %d' % cur_epoch
        # yield [left_images, right_images1, right_images2], [sub_scores, right_img_scores1, right_img_scores2]
        yield [left_images, right_images1, right_images2], [right_img_scores1, right_img_scores2]


def sub(inputs):
    x, y = inputs
    return (x - y) # *10


def cos_dist(inputs):
    x, y = inputs
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)
    return (1 + K.dot(x,K.transpose(y)))/2


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
    pair_model = load_model(pair_model_path)
    # pair_model1 = load_model(pair_model_path)
    # pair_model2 = load_model(pair_model_path)
    # pair_model3 = load_model(pair_model_path)
    resnet = pair_model.layers[2]
    # base_model1 = pair_model1.layers[2]
    # base_model2 = pair_model2.layers[2]
    # base_model3 = pair_model3.layers[2]

    base_model = Triplet_ResNet50(resnet)

    # base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    # base_model1 = Model(inputs=[base_model1.get_input_at(0)], outputs=[base_model1.get_output_at(0)], name='resnet50')
    # base_model2 = Model(inputs=[base_model2.get_input_at(0)], outputs=[base_model2.get_output_at(0)], name='resnet50_1')
    # base_model3 = Model(inputs=[base_model3.get_input_at(0)], outputs=[base_model3.get_output_at(0)], name='resnet50_2')

    img0 = Input(shape=(224, 224, 3), name='img_0')
    img1 = Input(shape=(224, 224, 3), name='img_1')
    img2 = Input(shape=(224, 224, 3), name='img_2')

    avg_pool0, avg_pool1, avg_pool2 = base_model([img0, img1, img2])
    feature0 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(Flatten()(Dropout(0.5)(avg_pool0)))
    feature1 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(Flatten()(Dropout(0.5)(avg_pool1)))
    feature2 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(Flatten()(Dropout(0.5)(avg_pool2)))
    # feature0 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(Flatten()(base_model1(img0)))
    # feature1 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(Flatten()(base_model2(img1)))
    # feature2 = Lambda(lambda  x: K.l2_normalize(x,axis=1))(Flatten()(base_model3(img2)))

    # score1 = Lambda(eucl_dist, name='score1')([feature0, feature1])
    # score2 = Lambda(eucl_dist, name='score2')([feature0, feature2])

    dis1 = Lambda(eucl_dist, name='square1')([feature0, feature1])
    dis2 = Lambda(eucl_dist, name='square2')([feature0, feature2])



    # dis = Input(shape=(2048,), name='dis_input')
    # dis_score = Dense(1, activation='sigmoid', name='score')(dis)

    # dis_model = Model(inputs=[dis], outputs=[dis_score])
    # dis_model.get_layer('score').set_weights(pair_model.get_layer('bin_out').get_weights())
    # dis_model.get_layer('score').trainable = False

    # score1 = dis_model(dis1)
    # score1 = Lambda(lambda x:x, name='score1')(score1)

    # score2 = dis_model(dis2)
    # score2 = Lambda(lambda x: x, name='score2')(score2)
    score1 = Dense(1, activation='sigmoid', name='score1')(Dropout(0.9)(dis1))
    score2 = Dense(1, activation='sigmoid', name='score2')(Dropout(0.9)(dis2))
    # score1 = Lambda(cos_dist, name='score1')([feature0, feature1])
    # score2 = Lambda(cos_dist, name='score2')([feature0, feature2])
    # sub_score = Lambda(sub, name='sub_score')([score1, score2])

    # model = Model(inputs=[img0, img1, img2], outputs=[sub_score, score1, score2])
    model = Model(inputs=[img0, img1, img2], outputs=[score1, score2])

    # model = Model(inputs=[img0, img1, img2], outputs=[sub_score])
    # model.get_layer('score1').set_weights(pair_model.get_layer('bin_out').get_weights())
    # model.get_layer('score1').trainable = False
    # model.get_layer('score2').set_weights(pair_model.get_layer('bin_out').get_weights())
    # model.get_layer('score2').trainable = False
    plot_model(model, to_file='rank_model.png')


    print(model.summary())
    return model


class MonitorNanOnBN(Callback):
    """Callback that rollback nan BN weights
    """

    def __init__(self):
        super(MonitorNanOnBN, self).__init__()
        self.last_weights = None

    def on_batch_begin(self, batch, logs=None):
        self.last_weights = list()
        for layer in self.model.layers[3].layers:
            self.last_weights.append(layer.get_weights())

    def on_batch_end(self, batch, logs=None):
        # base_model1 = self.model.layers[3]
        # base_model2 = self.model.layers[4]
        # base_model3 = self.model.layers[5]
        # avg_weights = list()
        # for i in range(base_model1.layers):
            # if isinstance(base_model1.layers[i], Conv2D):
            # elif isinstance(base_model1.layers[i], BatchNormalization):
                # avg_weights = np.average()

        for i, layer in enumerate(self.model.layers[3].layers):
            if isinstance(layer, BatchNormalization):
                moving_averages = layer.get_weights()[2]
                for moving_average in moving_averages:
                    if np.isnan(moving_average):
                        print('Batch Normalization moving average nan on Batch %d' % (batch))
                        layer.set_weights(self.last_weights[i])
                        # self.model.stop_training = True
                        break
                    else:
                        # print 'moving average'
                        # print moving_average
                        # print 'before max'
                        # before_layer_weights = self.model.layers[3].layers[i-1].get_weights()
                        # print max(before_layer_weights[0].max(), before_layer_weights[1].max())
                        break

def rank_transfer(train_generator, val_generator, source_model_path, target_model_path, batch_size=48):
    model = rank_transfer_model(source_model_path)
    model.compile(
        optimizer=SGD(lr=0.001, momentum=0.9), # 'adam',
        # optimizer='adadelta',
                  loss={
                      # 'sub_score': 'mse',
                      'score1': 'binary_crossentropy',
                      'score2': 'binary_crossentropy',
                      # 'sub_score': 'mse'
                  },
                  loss_weights={
                      # 'sub_score': 0.5,
                      'score1':0.5,
                      'score2': 0.5
                  },
                  metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001,
                                cooldown=0, min_lr=0)
    monitor_bn_nan = MonitorNanOnBN()
    if 'market-' in target_model_path:
        train_data_cnt = 3000
        val_data_cnt = 300
    else:
        train_data_cnt = 160
        val_data_cnt = 18
    safe_remove(target_model_path)
    model.fit_generator(train_generator,
                            steps_per_epoch=train_data_cnt / batch_size + 1,
                            epochs=50,
                            validation_data=val_generator,
                            validation_steps=val_data_cnt / batch_size + 1,
                            callbacks=[
                               auto_lr, # early_stopping,  # monitor_bn_nan,
                               ModelCheckpoint(target_model_path, monitor='val_loss', verbose=0,
                                                                save_best_only=True, save_weights_only=False,
                                                                mode='auto', period=1)
                            ]
                        )
    #safe_remove(target_model_path)
    # model.save('simple_rank_transfer.h5')
    # model.save(target_model_path)




def delta_sure_pos_idxs(delta_matrix_path):
    delta_matrix = np.genfromtxt(delta_matrix_path, delimiter=' ')
    delta_matrix_top = delta_matrix[:, 0].reshape(-1)
    delta_sure_pos_idx = np.where(delta_matrix_top > 0.8)
    delta_sure_neg_idx = np.where(delta_matrix_top < 0.1)
    return delta_sure_pos_idx, delta_sure_neg_idx


def rank_transfer_2dataset(source_pair_model_path, target_train_list, target_model_path, target_train_path,
                           rank_pid_path, rank_score_path):

    # rank_transfer_model(source_pair_model_path)
    train_images = reid_img_prepare(target_train_list, target_train_path)
    batch_size = 24
    similar_persons = np.genfromtxt(rank_pid_path, delimiter=' ')
    # if 'cross' in rank_pid_path:
    #     similar_persons = similar_persons - 1
    similar_matrix = np.genfromtxt(rank_score_path, delimiter=' ')
    track_score_path = '/home/cwh/coding/TrackViz/viper_market_deltas_score.txt'
    # delta_sure_pos_idx, _ = delta_sure_pos_idxs(track_score_path)
    # print delta_sure_pos_idx

    rank_transfer(
        triplet_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=True),
        triplet_generator_by_rank_list(train_images, batch_size, similar_persons, similar_matrix, train=False),
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

