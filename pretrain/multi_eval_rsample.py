import utils.cuda_util_test
from keras import Model
from keras.models import load_model
import numpy as np
import keras.backend as K

from baseline.evaluate import market_result_eval, extract_feature, sort_similarity
from utils.file_helper import safe_remove



def selective_categorical_crossentropy(y_true, y_pred):
    # print('y_pred.shape')
    # print(y_pred.shape)
    # print('y_true.shape')
    # print(y_true.shape)
    return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true, axis=1)

def selective_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())* K.sum(y_true, axis=1)


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
    net = load_model('../pretrain/' + source + '_rs_multi_pretrain.h5', custom_objects={
        'selective_categorical_crossentropy': selective_categorical_crossentropy,
        'selective_accuracy':selective_accuracy})
    target_path = '/home/cwh/coding/Market-1501'
    probe_path = target_path + '/probe'
    gallery_path = target_path + '/test'
    train_path = target_path + '/train'
    pid_path = 'ret_train_pid.txt'
    score_path = 'ret_train_score.txt'
    # train_predict(net, train_path, pid_path, score_path)
    test_predict(net, probe_path, gallery_path, pid_path, score_path)
    market_result_eval(pid_path, 'market_eval.txt', gallery_path, probe_path)