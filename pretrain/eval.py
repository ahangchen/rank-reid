# coding=utf-8
import os
import cuda_util
import numpy as np

from keras import Input
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Flatten, Lambda, Dense
from keras.preprocessing import image
from keras import backend as K
#
from pretrain.pair_train import eucl_dist, dis_sigmoid

from keras.engine import Model
from keras.models import load_model

from baseline.evaluate import train_predict, test_predict, grid_result_eval, market_result_eval, extract_feature, \
    extract_info
from transfer.simple_rank_transfer import cross_entropy_loss


def train_pair_predict(pair_model_path, target_train_path, pid_path, score_path):
    model = load_model(pair_model_path)
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    # model = Model(inputs=[model.input], outputs=[model.get_layer('avg_pool').output])
    train_predict(model, target_train_path, pid_path, score_path)


def test_pair_predict(pair_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    # todo
    model = load_model(pair_model_path)
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
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        imgs.append(x)
    return imgs


def test_pair_acc_predict(pair_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    pair_model = load_model(pair_model_path)
    # feature_net = Model(inputs=[pair_model.get_layer('resnet50').get_input_at(0)],
    #               outputs=[pair_model.get_layer('resnet50').get_output_at(0)])
    # feature_net = Model(inputs=[feature_net.input], outputs=[feature_net.get_layer('avg_pool').output])
    # test_f_s, test_info = extract_feature(target_gallery_path, feature_net)
    # query_f_s, query_info = extract_feature(target_probe_path, feature_net)
    # eval_net = Model(inputs=pair_model.get_layer('square').input, outputs=pair_model.get_layer('bin_out').output)
    eval_net = Model(inputs=pair_model.inputs, outputs=pair_model.get_layer('bin_out').output)
    query_info = extract_info(target_probe_path)
    test_info = extract_info(target_gallery_path)
    probe_imgs = extract_imgs(target_probe_path)
    gallery_imgs = extract_imgs(target_gallery_path)
    similarity_matrix = []
    for i, probe_img in enumerate(probe_imgs):
        if i % 1 == 0:
            print 'acc for probe %d' % i
        eval_batch_size = 256
        probe_img_cp = np.array([probe_img for i in range(len(test_info))])
        similarity_line = eval_net.predict([probe_img_cp, np.array(gallery_imgs)], batch_size=eval_batch_size)
        similarity_matrix.append(similarity_line[:, 0])

    result = np.array(similarity_matrix)
    result_argsort = np.argsort(-result, axis=1)
    np.savetxt(pid_path, result_argsort, fmt='%d')
    np.savetxt(score_path, result, fmt='%.4f')


def tf_eucl_dist(inputs):
    x, y = inputs
    return K.square((x - y))

def avg_eucl_dist(inputs):
    x, y = inputs
    return K.mean(K.square((x - y)), axis=1)


# def test_generator(query_f_s, test_f_s):

def faster_test_pair_acc_predict(pair_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    pair_model = load_model(pair_model_path)
    feature_net = Model(inputs=[pair_model.get_layer('resnet50').get_input_at(0)],
                  outputs=[pair_model.get_layer('resnet50').get_output_at(0)])
    feature_net = Model(inputs=[feature_net.input], outputs=[feature_net.get_layer('avg_pool').output])
    # test_f_s, test_info = [1, 2]
    # query_f_s, query_info = [3, 4]
    test_f_s, test_info = extract_feature(target_gallery_path, feature_net)
    query_f_s, query_info = extract_feature(target_probe_path, feature_net)

    input1 = Input(shape=(2048,), name='input1')
    input2 = Input(shape=(2048,), name='input2')
    dis = Lambda(eucl_dist, name='square')([input1, input2])
    # judge = Lambda(dis_sigmoid, name='bin_out')(dis)
    judge = Dense(1, activation='sigmoid', name='bin_out')(dis)
    # dis = Lambda(avg_eucl_dist, name='square')([input1, input2])
    eval_net = Model(inputs=[input1, input2], outputs=[judge])
    # eval_net = Model(inputs=[input1, input2], outputs=[dis])
    eval_net.get_layer('bin_out').set_weights(pair_model.get_layer('bin_out').get_weights())

    similarity_matrix = []
    for i, query_f in enumerate(query_f_s):
        if i % 100 == 0:
            print 'acc for probe %d' % i
        eval_batch_size = 1024
        query_f_cp = np.array([query_f for i in range(len(test_info))]).reshape([len(test_info), 2048])
        # similarity_line = np.array([eval_net.predict([query_f.reshape(1, 2048), test_f_s[i].reshape(1, 2048)],
        #                                    batch_size=eval_batch_size)[0] for i in range(len(test_info))])
        similarity_line = eval_net.predict([query_f_cp, np.array(test_f_s)],
                                                     batch_size=eval_batch_size).reshape(-1)
        # similarity_matrix.append(similarity_line[:, 1])
        similarity_matrix.append(similarity_line)

    result = np.array(similarity_matrix)
    result_argsort = np.argsort(-result, axis=1)
    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    result = np.array(result)
    np.savetxt(pid_path, result_argsort, fmt='%d')
    np.savetxt(score_path, result, fmt='%.4f')


def train_rank_predict(rank_model_path, target_train_path, pid_path, score_path):
    model = load_model(rank_model_path, custom_objects={'cross_entropy_loss': cross_entropy_loss})
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    train_predict(model, target_train_path, pid_path, score_path)


def test_rank_predict(rank_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    model = load_model(rank_model_path, custom_objects={'cross_entropy_loss': cross_entropy_loss})
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
    # train_pair_predict('market_pair_pretrain.h5', '/home/cwh/coding/grid_train_probe_gallery/cross0/train', 'market_grid_pid.txt', 'market_grid_score.txt')
    # grid_test_base_eval('../baseline/0.ckpt', 'grid_cross0_single')
    # cross0: [0.072, 0.144, 0.184, 0.232, 0.408]
    # cross0_gan: [0.136, 0.344, 0.416, 0.544, 0.648]
    # grid_test_pair_eval('../transfer/pair_pretrain.h5', 'grid_cross0_pair_pretrain')
    # [0.088, 0.16, 0.2, 0.296, 0.456]
    # grid_test_pair_eval('../transfer/pair_transfer.h5', 'grid_cross0_transfer')
    # [0.192, 0.312, 0.376, 0.496, 0.624] epoch7
    # [0.192, 0.312, 0.392, 0.48, 0.648] st epoch3
    # grid_test_rank_eval('../transfer/rank_transfer.h5', 'grid_cross0_rank_transfer')
    # [0.184, 0.304, 0.344, 0.456, 0.656]
    # test_pair_predict('grid_softmax_pretrain.h5', '/home/cwh/coding/Market-1501/probe', '/home/cwh/coding/Market-1501/test', 'market_market_pid_test.txt', 'market_market_score_test.txt')
    # test_pair_predict('market_softmax_pretrain.h5', '/home/wxt/reid_data-gan/cross0_gan/probe', '/home/wxt/reid_data-gan/cross0_gan/test', 'market_market_pid_test_gan.txt', 'market_market_score_test_gan.txt')
    # test_pair_predict('market_pair_pretrain.h5', '/home/cwh/coding/grid_train_probe_gallery/cross0/probe', '/home/cwh/coding/grid_train_probe_gallery/cross0/test', 'market_market_pid_test_softmax.txt', 'market_market_score_test_softmax.txt')
    # grid_result_eval('market_market_pid_test_softmax.txt', './gan.log')
    # [0.104, 0.176, 0.264, 0.312, 0.416]
    # grid_eval('market', '/home/wxt/ReidGAN/transformgrid2marketstyle')
    # market_eval('grid', '/home/wxt/ReidGAN/market2grid_style')
    market_eval('market', '/home/cwh/coding/Market-1501')
    market_result_eval('market_market_pid.log')
    # market_result_eval('/home/cwh/coding/TrackViz/data/market_market-test/cross_filter_pid.log')
    # test_rank_predict('../transfer/rank_transfer_test.h5',
    #                   '/home/cwh/coding/Market-1501/probe', '/home/cwh/coding/Market-1501/test',
    #                   'rank_pid.log', 'rank_ac.log')
    # market_result_eval('rank_pid.log')
    # grid_result_eval('/home/cwh/coding/TrackViz/data/market_grid-cv0-test/cross_filter_pid.log')
    # market_result_eval('/home/cwh/coding/TrackViz/data/market_market-test/cross_filter_pid.log')

