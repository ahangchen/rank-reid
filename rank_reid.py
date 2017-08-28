import sys
from pretrain.eval import train_pair_eval,test_pair_eval, train_rank_eval, test_rank_eval


def origin_eval(source, target):
    source_model_path = ''
    if source == 'market':
        source_model_path = '/home/cwh/Mission/coding/python/pretrain/market_pair_pretrain.h5'
    elif source == 'grid':
        source_model_path = '/home/cwh/Mission/coding/python/pretrain/grid_pair_pretrain.h5'
    elif source == 'cuhk':
        source_model_path = '/home/cwh/Mission/coding/python/pretrain/cuhk_pair_pretrain.h5'
    target_dataset_path = ''
    if target == 'market':
        target_dataset_path = '/home/cwh/coding/Market-1501'
    elif target == 'grid':
        target_dataset_path = '/home/cwh/coding/grid_train_probe_gallery'
    target_probe_path = target_dataset_path + '/probe'
    target_train_path = target_dataset_path + '/train'
    target_gallery_path = target_dataset_path + '/gallery'
    train_pair_eval(source_model_path, source + '_' + target + '_log', target_train_path)
    test_pair_eval(source_model_path, source + '_' + target + '_log', target_probe_path, target_gallery_path)


if __name__ == '__main__':
    opt = sys.argv[0]
    if opt == '0':
        source = sys.argv[1]
        target = sys.argv[2]
        origin_eval(source, target)
    else:
        pass
