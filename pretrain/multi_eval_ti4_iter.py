import utils.cuda_util_test
from keras import Model
from keras.models import load_model
import numpy as np

from baseline.evaluate import market_result_eval, extract_feature, sort_similarity
from utils.file_helper import safe_remove



def test_predict(net, probe_path, gallery_path, pid_path, score_path):
    net = Model(inputs=[net.get_layer('resnet50').get_input_at(0)], outputs=[net.get_layer('resnet50').get_output_at(0)])
    # net = Model(inputs=[net.input], outputs=[net.get_layer('avg_pool').output])
    test_f, test_info = extract_feature(gallery_path, net)
    query_f, query_info = extract_feature(probe_path, net)
    result, result_argsort = sort_similarity(query_f, test_f)
    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    result = np.array(result)
    safe_remove(pid_path)
    safe_remove(score_path)
    np.savetxt(pid_path, result_argsort, fmt='%d')
    np.savetxt(score_path, result, fmt='%.4f')


if __name__ == '__main__':
    source = 'market'
    target = 'market'
    for i in range(10):
        net = load_model('../pretrain/' + source + '_multi_sl_pretrain_%d.h5' % i)
        target_path = '/home/cwh/coding/Market-1501'
        probe_path = target_path + '/probe'
        gallery_path = target_path + '/test'
        train_path = target_path + '/train'
        pid_path = 'ret_train_pid.txt'
        score_path = 'ret_train_score.txt'
        # train_predict(net, train_path, pid_path, score_path)
        test_predict(net, probe_path, gallery_path,  pid_path, score_path)
        market_result_eval(pid_path, 'market_eval.txt', gallery_path, probe_path)
    net = load_model('../pretrain/' + source + '_multi_sl_pretrain.h5')
    target_path = '/home/cwh/coding/Market-1501'
    probe_path = target_path + '/probe'
    gallery_path = target_path + '/test'
    train_path = target_path + '/train'
    pid_path = 'ret_train_pid.txt'
    score_path = 'ret_train_score.txt'
    # train_predict(net, train_path, pid_path, score_path)
    test_predict(net, probe_path, gallery_path, pid_path, score_path)
    market_result_eval(pid_path, 'market_eval.txt', gallery_path, probe_path)