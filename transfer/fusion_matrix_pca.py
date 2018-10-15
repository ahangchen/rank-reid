import numpy as np
from sklearn.decomposition import PCA


def fusion_pca(fusion_pid_path, fusion_score_path):
    fusion_pids = np.genfromtxt(fusion_pid_path, delimiter=' ')
    fusion_matrix = np.genfromtxt(fusion_score_path, delimiter=' ')
    sorted_fusion_pid_idxes = np.argsort(fusion_pids, axis=1)
    for i in range(len(fusion_matrix)):
        fusion_matrix[i] = fusion_matrix[i][sorted_fusion_pid_idxes[i]]
    pca_matrix = PCA(249).fit_transform(fusion_matrix)
    np.savetxt('pca_market.log', pca_matrix, fmt='%4f')
    return pca_matrix

if __name__ == '__main__':
    source = 'grid'
    target = 'market'
    fusion_train_rank_pids_path = '/home/cwh/coding/TrackViz/data/%s_%s-train/cross_filter_pid.log' % (source, target)
    fusion_train_rank_scores_path = '/home/cwh/coding/TrackViz/data/%s_%s-train/cross_filter_score.log' % (
    source, target)

    fusion_pca(fusion_train_rank_pids_path, fusion_train_rank_scores_path)