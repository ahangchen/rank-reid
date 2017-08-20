import os
from keras.engine import Model
from keras.models import load_model

from baseline.evaluate import train_predict, test_predict, grid_result_eval
from transfer.rank_transfer import cross_entropy_loss
from util import safe_mkdir


def market_train_pair_eval():
    model = load_model('pair_pretrain.h5')
    model = Model(inputs=[model.get_layer('model_1').get_input_at(0)],
                  outputs=[model.get_layer('model_1').get_output_at(0)])
    DATASET = '../dataset/Market'
    TRAIN = os.path.join(DATASET, 'bounding_box_train')
    train_predict(TRAIN, model, '.')


def market_test_pair_eval():
    model = load_model('pair_pretrain.h5')
    model = Model(inputs=[model.get_layer('model_1').get_input_at(0)],
                  outputs=[model.get_layer('model_1').get_output_at(0)])
    DATASET = '../dataset/Market'
    TEST = os.path.join(DATASET, 'bounding_box_test')
    QUERY = os.path.join(DATASET, 'query')
    test_predict(model, QUERY, TEST, '.')


def grid_train_pair_eval(model_path, log_dir_path):
    model = load_model(model_path)
    model = Model(inputs=[model.get_layer('model_1').get_input_at(0)],
                  outputs=[model.get_layer('model_1').get_output_at(0)])
    DATASET = '/home/cwh/coding/underground_reid/cross0'
    TRAIN = os.path.join(DATASET, 'pretrain')
    safe_mkdir(log_dir_path)
    train_predict(TRAIN, model, log_dir_path)


def grid_test_base_eval(model_path, log_dir_path):
    model = load_model(model_path)
    # DATASET = '/home/cwh/coding/grid_train_probe_gallery/cross0_gan'
    DATASET = '/home/cwh/coding/grid_train_probe_gallery/cross0'
    probe = os.path.join(DATASET, 'probe')
    gallery = os.path.join(DATASET, 'test')
    safe_mkdir(log_dir_path)
    test_predict(model, probe, gallery, log_dir_path)
    grid_result_eval(log_dir_path+'/test_renew_pid.log')


def grid_test_pair_eval(model_path, log_dir_path):
    model = load_model(model_path)
    model = Model(inputs=[model.get_layer('model_1').get_input_at(0)],
                  outputs=[model.get_layer('model_1').get_output_at(0)])
    DATASET = '/home/cwh/coding/grid_train_probe_gallery/cross0'
    probe = os.path.join(DATASET, 'probe')
    gallery = os.path.join(DATASET, 'test')
    safe_mkdir(log_dir_path)
    test_predict(model, probe, gallery, log_dir_path)
    grid_result_eval(log_dir_path+'/test_renew_pid.log')


def grid_test_rank_eval(model_path, log_dir_path):
    model = load_model(model_path, custom_objects={'cross_entropy_loss': cross_entropy_loss})
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    DATASET = '/home/cwh/coding/grid_train_probe_gallery/cross0'
    probe = os.path.join(DATASET, 'probe')
    gallery = os.path.join(DATASET, 'test')
    safe_mkdir(log_dir_path)
    test_predict(model, probe, gallery, log_dir_path)
    grid_result_eval(log_dir_path + '/test_renew_pid.log')


if __name__ == '__main__':
    # grid_test_base_eval('../baseline/0.ckpt', 'grid_cross0_single')
    # cross0: [0.072, 0.144, 0.184, 0.232, 0.408]
    # cross0_gan: [0.136, 0.344, 0.416, 0.544, 0.648]
    # grid_test_pair_eval('../transfer/pair_transfer.h5', 'grid_cross0_transfer')
    grid_test_pair_eval('../transfer/pair_pretrain.h5', 'grid_cross0_pair_pretrain')
    # [0.088, 0.16, 0.2, 0.296, 0.456]
    # [0.192, 0.312, 0.376, 0.496, 0.624] epoch7
    # grid_test_rank_eval('../transfer/rank_transfer.h5', 'grid_cross0_rank_transfer')
    # [0.184, 0.304, 0.344, 0.456, 0.656]