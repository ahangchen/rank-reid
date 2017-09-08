import os
from keras.engine import Model
from keras.models import load_model

from baseline.evaluate import train_predict, test_predict, grid_result_eval
from transfer.simple_rank_transfer import cross_entropy_loss
from util import safe_mkdir


# def market_train_pair_eval():
#     model = load_model('pair_pretrain.h5')
#     model = Model(inputs=[model.get_layer('model_1').get_input_at(0)],
#                   outputs=[model.get_layer('model_1').get_output_at(0)])
#     DATASET = '../dataset/Market'
#     TRAIN = os.path.join(DATASET, 'bounding_box_train')
#     train_predict(TRAIN, model, '.')
#
#
# def market_test_pair_eval():
#     model = load_model('pair_pretrain.h5')
#     model = Model(inputs=[model.get_layer('model_1').get_input_at(0)],
#                   outputs=[model.get_layer('model_1').get_output_at(0)])
#     DATASET = '../dataset/Market'
#     TEST = os.path.join(DATASET, 'bounding_box_test')
#     QUERY = os.path.join(DATASET, 'query')
#     test_predict(model, QUERY, TEST, '.')


def grid_test_base_eval(model_path, log_dir_path):
    model = load_model(model_path)
    # DATASET = '/home/cwh/coding/grid_train_probe_gallery/cross0_gan'
    DATASET = '/home/cwh/coding/grid_train_probe_gallery/cross0'
    probe = os.path.join(DATASET, 'probe')
    gallery = os.path.join(DATASET, 'test')
    safe_mkdir(log_dir_path)
    test_predict(model, probe, gallery, log_dir_path)
    grid_result_eval(log_dir_path+'/test_renew_pid.log')


def train_pair_predict(pair_model_path, target_train_path, pid_path, score_path):
    model = load_model(pair_model_path)
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    train_predict(model, target_train_path, pid_path, score_path)


def test_pair_predict(pair_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    model = load_model(pair_model_path)
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    test_predict(model, target_probe_path, target_gallery_path, pid_path, score_path)


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
    test_pair_predict('market_pair_pretrain.h5', '/home/cwh/coding/Market-1501/probe', '/home/cwh/coding/Market-1501/test', 'market_market_pid_test.txt', 'market_market_score_test.txt')