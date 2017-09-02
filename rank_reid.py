import sys
from pretrain.eval import train_pair_predict,test_pair_predict, train_rank_predict, test_rank_eval
from transfer.simple_rank_transfer import rank_transfer_2dataset


def get_source_target_info(source, target):
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
    return source_model_path, target_dataset_path


def vision_predict(source, target, train_pid_path, train_score_path, test_pid_path, test_score_path):
    source_model_path, target_dataset_path = get_source_target_info(source, target)
    target_probe_path = target_dataset_path + '/probe'
    target_train_path = target_dataset_path + '/train'
    target_gallery_path = target_dataset_path + '/gallery'
    train_pair_predict(source_model_path, target_train_path, train_pid_path, train_score_path)
    test_pair_predict(source_model_path, target_probe_path, target_gallery_path, test_pid_path, test_score_path)


def rank_transfer(source, target, fusion_train_rank_pids_path, fusion_train_rank_scores_path):
    source_model_path, target_dataset_path = get_source_target_info(source, target)
    target_train_path = target_dataset_path + '/train'
    target_model_path = source + '_' + target
    rank_transfer_2dataset(source_model_path, target_model_path, target_train_path,
                           fusion_train_rank_pids_path, fusion_train_rank_scores_path)
    return target_model_path


def rank_predict(transfer_train_rank_pids_path, transfer_train_rank_scores_path,
                    transfer_test_rank_pids_path, transfer_test_rank_scores_path):
    return

if __name__ == '__main__':
    opt = sys.argv[1]
    if opt == '0':
        source = sys.argv[2]
        target = sys.argv[3]
        vision_train_rank_pids_path = sys.argv[4]
        vision_train_rank_scores_path = sys.argv[5]
        vision_test_rank_pids_path = sys.argv[6]
        vision_test_rank_scores_path = sys.argv[7]
        vision_predict(source, target,
                       vision_train_rank_pids_path, vision_train_rank_scores_path,
                       vision_test_rank_pids_path, vision_test_rank_scores_path)
    elif opt == 1:
        source = sys.argv[2]
        target = sys.argv[3]
        fusion_train_rank_pids_path = sys.argv[4]
        fusion_train_rank_scores_path = sys.argv[5]
        transfer_train_rank_pids_path = sys.argv[6]
        transfer_train_rank_scores_path = sys.argv[7]
        transfer_test_rank_pids_path = sys.argv[8]
        transfer_test_rank_scores_path = sys.argv[9]
        rank_transfer(source, target, fusion_train_rank_pids_path, fusion_train_rank_scores_path)
        rank_predict(transfer_train_rank_pids_path, transfer_train_rank_scores_path,
                    transfer_test_rank_pids_path, transfer_test_rank_scores_path)
    else:
        pass
