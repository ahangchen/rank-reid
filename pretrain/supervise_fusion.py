from keras import Input
from keras.engine import Model
from keras.layers import Dense
from keras.optimizers import SGD

from utils.file_helper import read_lines
from numpy.random import randint, shuffle, choice
import numpy as np
import cuda_util

def train_tracks():
    answer_path = '/home/cwh/coding/TrackViz/data/market/train.txt'
    answer_lines = read_lines(answer_path)
    real_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            real_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            real_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])
    return real_tracks


def predict_img_scores():
    predict_score_path = '/home/cwh/coding/TrackViz/data/market_market-train/renew_ac.log'
    vision_persons_scores = np.genfromtxt(predict_score_path, delimiter=' ').astype(float)
    return vision_persons_scores


def predict_pids():
    predict_person_path = '/home/cwh/coding/TrackViz/data/market_market-train/renew_pid.log'
    predict_persons = np.genfromtxt(predict_person_path, delimiter=' ').astype(int)
    return predict_persons


def load_market_infos():
    market_train_tracks = np.array(train_tracks()).astype(int)
    vision_persons_scores = predict_img_scores()
    vision_ranking_pids = predict_pids()
    return market_train_tracks, vision_persons_scores, vision_ranking_pids


def fusion_generator(market_train_tracks, vision_persons_scores, vision_ranking_pids, batch_size=16, val=False):

    while True:
        left_v_idx = randint(len(market_train_tracks), size=batch_size)
        right_v_idx = randint(len(market_train_tracks), size=batch_size)
        left_im_id = left_v_idx
        right_im_id = vision_ranking_pids[left_v_idx][:, right_v_idx]
        vision_score = vision_persons_scores[left_v_idx][:, right_v_idx]
        left_tracks = market_train_tracks[left_im_id]
        right_tracks = market_train_tracks[right_im_id]
        left_pid, left_c, left_t, left_s = left_tracks[:, 0], left_tracks[:, 1], left_tracks[:, 2], left_tracks[:, 3]
        right_pid, right_c, right_t, right_s = right_tracks[:, 0], right_tracks[:, 1], right_tracks[:, 2], right_tracks[:, 3]

        feature = np.array([[vision_score, left_c, right_c,  left_t - right_t], [right_v_idx,left_c - right_c, left_s - right_s]])
        label = np.logical_and(left_pid == right_pid, left_c != left_c).astype(int)
        yield [feature], [label]


def supervised_model():
    # vision_score = Input(shape=[1, 1])
    # track_info = Input(shape=[3, 1])
    # [vison score, c1, c2,  t1-t2,
    # vision ranking index, c1-c2,s1-s2]
    feature = Input(shape=[2, 3])
    fusion_rst = Dense(2)(feature)
    fusion_score = Dense(1)(fusion_rst)
    net = Model(inputs=[feature], outputs=[fusion_score])
    net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return net

def fusion(net, batch_size, target_model_path):
    market_train_tracks, vision_persons_scores, vision_ranking_pids = load_market_infos()
    train_datagen = fusion_generator(market_train_tracks, vision_persons_scores, vision_ranking_pids, batch_size)
    val_datagen = fusion_generator(market_train_tracks, vision_persons_scores, vision_ranking_pids, batch_size, val=True)
    net.fit_generator(
        generator=train_datagen, validation_data=val_datagen,
        validation_steps= 1200/batch_size + 1,
        steps_per_epoch=12936 / batch_size + 1, epochs=10,
    )

    net.save(target_model_path)


def fusion_test():
    print ''

if __name__ == '__main__':
    model = supervised_model()
    fusion(model, 16, 'fusion.h5')