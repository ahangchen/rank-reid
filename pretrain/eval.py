import os

from keras import Input
from keras.applications import ResNet50

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.engine import Model
from keras.models import load_model

from baseline.evaluate import train_predict, test_predict, grid_result_eval
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
                     transform_dir +  '/probe', transform_dir +  '/test',
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
    market_eval('grid', '/home/cwh/coding/Market-1501')