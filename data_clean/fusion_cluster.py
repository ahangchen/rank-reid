import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics

from utils.file_helper import read_lines


class DataProvider(object):
    def __init__(self):
        self.data = None
    def load_data(self):
        return self
    def provide(self):
        return self.data

class AffinityProvider(DataProvider):
    def __init__(self, pid_path, score_path):
        super(AffinityProvider, self).__init__()
        self.pid_path = pid_path
        self.score_path = score_path
    def load_data(self):
        print('loading matrix')
        sorted_pid = np.genfromtxt(self.pid_path, delimiter=' ').astype(int)
        print('pid shape')
        print(sorted_pid.shape)
        sorted_w = np.genfromtxt(self.score_path, delimiter=' ')
        print('w shape')
        print(sorted_w.shape)
        print('loading done, fixing diag')
        w = np.zeros((sorted_w.shape[0], sorted_w.shape[1] + 1))
        w[:, 0] = 1
        w[:, 1:50] = sorted_w[:, :49]

        pid = np.ones((sorted_pid.shape[0], sorted_pid.shape[1] + 1), dtype=np.int32)
        pid[:, 0] = np.arange(sorted_pid.shape[0]).T
        pid[:, 1:] = sorted_pid
        print('resorting')
        w = np.exp(w)
        for i in range(w.shape[0]):
            w[i] = w[i][pid[i]] / sum(w[i])
        print('make symmetric')
        w = (w.T + w) / 2
        self.data = w
        return self


class FeatureProvider(DataProvider):
    def __init__(self, feature_path):
        super(DataProvider, self).__init__()
        self.feature_path = feature_path
    def load_data(self):
        self.data = np.genfromtxt(self.feature_path, delimiter=' ')
        return self


class Cluster(object):
    def __init__(self, data_provider, cluster):
        self.data_provider = data_provider
        self.score = 0
        self.result = None
        self.cluster = cluster
        self.predict_path = self.data_provider.__class__.__name__ + '_' + self.__class__.__name__ + '.txt'
    def merge(self, result_path=None):
        data = self.data_provider.provide()
        self.result = self.cluster.fit_predict(data)
        if result_path:
            np.savetxt(self.predict_path, self.result, '%d')
        return self
    def eval(self, label_path):
        label = np.genfromtxt(label_path, delimiter=' ')
        self.score = metrics.adjusted_rand_score(label, self.result)
        print(self.score)


class KMeansCluster(Cluster):
    def __init__(self, data_provider, class_cnt):
        super(KMeansCluster, self).__init__(data_provider, KMeans(n_clusters=class_cnt))


class SpectralCluster(Cluster):
    def __init__(self, data_provider, class_cnt):
        if isinstance(data_provider, FeatureProvider):
            cluster = SpectralClustering(n_clusters=class_cnt)
        else:
            cluster = SpectralClustering(n_clusters=class_cnt, affinity='precomputed')

        super(SpectralCluster, self).__init__(data_provider, cluster)
        self.predict_path = self.predict_path


def gen_market_gt_class(train_path):
    file_paths = read_lines(train_path)
    ids = []
    for file_path in file_paths:
        ids.append(int(file_path.split('_')[0]))
    np.savetxt('market_gt.txt', np.array(ids),'%d')



def do_cluster():
    pid_path = '/home/cwh/coding/TrackViz/data/cuhk_market-train/renew_pid.log'
    score_path = '/home/cwh/coding/TrackViz/data/cuhk_market-train/renew_score.log'
    feature_path = '../baseline/feature.txt'
    label_path = 'market_gt.txt'
    class_cnt = 700
    fp = FeatureProvider(feature_path).load_data()
    vap = AffinityProvider(pid_path, score_path).load_data()
    fap = AffinityProvider(pid_path, score_path).load_data()
    KMeansCluster(fp, class_cnt).merge().eval(label_path)
    SpectralCluster(fp, class_cnt).merge().eval(label_path)
    SpectralCluster(vap, class_cnt).merge('visual_affinity_spectral.txt').eval(label_path)
    SpectralCluster(fap, class_cnt).merge('fusion_affinity_spectral.txt').eval(label_path)



if __name__ == '__main__':
    do_cluster()