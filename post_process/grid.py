import numpy as np


def train_res_transform(renew_pid_path):
    predict_pids = np.genfromtxt(renew_pid_path, delimiter=' ')
    np.savetxt(renew_pid_path.replace('tran_', ''), predict_pids + 1, fmt='%d')


def test_res_transform(renew_pid_path):
    predict_pids = np.genfromtxt(renew_pid_path, delimiter=' ') + 1
    for i, pids in enumerate(predict_pids):
        for j, pid in enumerate(pids):
            if pid > 774:
                predict_pids[i][j] += 125
    np.savetxt(renew_pid_path.replace('test_', ''), predict_pids + 1, fmt='%d')

if __name__ == '__main__':
    test_res_transform('../transfer/grid_cross0_simple_rank_transfer/test_renew_pid.log')