# coding=utf-8
import os

from keras.layers import BatchNormalization

import utils.cuda_util

from keras import backend as K
import tensorflow as tf
from keras.engine import Model
from keras.models import load_model
from keras.preprocessing import image

from baseline.evaluate import train_predict, test_predict, grid_result_eval, market_result_eval
from transfer.simple_rank_transfer import cross_entropy_loss


#
def focal_loss_fixed(y_true, y_pred):
    gamma = 2.
    alpha = .25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))



def train_pair_predict(pair_model_path, target_train_path, pid_path, score_path):
    model = load_model(pair_model_path)
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    train_predict(model, target_train_path, pid_path, score_path)


def test_pair_predict(pair_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    # todo
    model = load_model(pair_model_path, custom_objects={'focal_loss_fixed': focal_loss_fixed})
    # model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    # model = Model(inputs=[model.input], outputs=[model.get_layer('avg_pool').output])
    test_predict(model, target_probe_path, target_gallery_path, pid_path, score_path)


def extract_imgs(dir_path):
    imgs = []
    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name:
            continue
        if 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        elif 's' in image_name:
            # market
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        else:
            continue
        image_path = os.path.join(dir_path, image_name)
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        imgs.append(x)
    return imgs


def tf_eucl_dist(inputs):
    x, y = inputs
    return K.square((x - y))

def avg_eucl_dist(inputs):
    x, y = inputs
    return K.mean(K.square((x - y)), axis=1)


def train_rank_predict(rank_model_path, target_train_path, pid_path, score_path):
    model = load_model(rank_model_path, custom_objects={'cross_entropy_loss': cross_entropy_loss})
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    train_predict(model, target_train_path, pid_path, score_path)


def test_rank_predict(rank_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    model = load_model(rank_model_path, custom_objects={'cross_entropy_loss': cross_entropy_loss})
    for layer in model.get_layer('resnet50').layers:
        if isinstance(layer, BatchNormalization):
            print(layer.get_weights())
            break
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    test_predict(model, target_probe_path, target_gallery_path, pid_path, score_path)


def grid_eval(source, transform_dir):
    target = 'grid'
    for i in range(10):
        test_pair_predict(source + '_pair_pretrain.h5',
                          transform_dir + 'cross%d' % i + '/probe', transform_dir + 'cross%d' % i + '/test',
                          source + '_' + target + '_pid.log', source + '_' + target + '_score.log')
        grid_result_eval(source + '_' + target + '_pid.log', 'gan.log')


def market_eval(source, transform_dir):
    target = 'market'
    test_pair_predict(source + '_pair_pretrain.h5',
                          transform_dir + '/probe', transform_dir + '/test',
                          source + '_' + target + '_pid.log', source + '_' + target + '_score.log')


if __name__ == '__main__':
    target = 'market'
    target_path = '/home/cwh/coding/Market-1501'
    probe_path = target_path + '/probe'
    gallery_path = target_path + '/test'
    pid_path = 'ret_pid.txt'
    score_path = 'ret_score.txt'
    test_rank_predict('transfer/gt_rank_transfer.h5', probe_path, gallery_path, pid_path, score_path)
    market_result_eval(pid_path, 'market_eval.txt', gallery_path, probe_path)