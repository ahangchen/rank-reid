import numpy as np

def delta_sure_pos_idxs(delta_matrix_path):
    delta_matrix = np.genfromtxt(delta_matrix_path, delimiter=' ')
    delta_matrix_top = delta_matrix[:, 0].reshape(-1)
    delta_sure_pos_idx = np.where(delta_matrix_top > 0.8)
    delta_sure_neg_idx = np.where(delta_matrix_top < 0.1)

    return delta_sure_pos_idx, delta_sure_neg_idx


# def rand_id_in(batch_size, delta_sure_pos_idx):
#     return delta_sure_pos_idx[]

if __name__ == '__main__':
    delta_matrix_path = '/home/cwh/coding/TrackViz/test_deltas_score.txt'
    delta_matrix = np.genfromtxt(delta_matrix_path, delimiter=' ')
    delta_matrix_top = delta_matrix[:, 0].reshape(-1)
    pid_matrix_path = '/home/cwh/coding/TrackViz/data/market_grid-cv3-test/cross_filter_pid.log'
    pid_matrix = np.genfromtxt(pid_matrix_path, delimiter=' ')
    pid_matrix_top = pid_matrix[:, 0].reshape(-1)
    sort_score_idx_s = sorted(range(len(delta_matrix_top)), key=lambda k: -delta_matrix_top[k])
    sort_deltas = delta_matrix_top[sort_score_idx_s]
    sort_pids = pid_matrix_top[sort_score_idx_s]
    for i in range(50):
        print '%d vs %d = %d, delta score: %4f' % (sort_score_idx_s[i], sort_pids[i], sort_pids[i] - sort_score_idx_s[i] == 775, sort_deltas[i])









