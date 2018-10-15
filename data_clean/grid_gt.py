import numpy as np

def grid_gt_matrix():
    matrix = np.zeros([250, 249])
    matrix[:, 0] = np.ones(250)
    pids = list()

    for i in range(250):
        pid = np.arange(0, 250)
        pid = np.delete(pid, i)
        if i % 2 == 0:
            pid = np.delete(pid, i)
            pid = np.concatenate([[i + 1], pid])
        else:
            pid = np.delete(pid, i - 1)
            pid = np.concatenate([[i - 1], pid])
        pids.append(pid)

    np.savetxt('grid_cross_filter_pid.log', pids, '%d')
    np.savetxt('grid_cross_filter_score.log', matrix, '%4f')

if __name__ == '__main__':
    grid_gt_matrix()

