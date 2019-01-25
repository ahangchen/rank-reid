from __future__ import division, print_function, absolute_import

import argparse
import os
from datetime import datetime

import numpy as np
import torch

from baseline.eval_util import compute_mAP
from baseline.eval_util import get_id
from utils.file_helper import write


def evaluate(index, ql, qc, gl, gc):
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
    return ap_tmp, CMC_tmp


def result_eval(predict_path, log_path='market_result_eval.log', TEST='Market-1501/test',
                       QUERY='Market-1501/probe'):
    res = np.genfromtxt(predict_path, delimiter=' ')
    print('predict info get, extract gallery info start')
    gallery_cam, gallery_label = get_id(TEST)
    gallery_cam = np.array(gallery_cam)
    gallery_label = np.array(gallery_label)
    query_cam, query_label = get_id(QUERY)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(res[i], query_label[i], query_cam[i], gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print(datetime.now().strftime("%Y.%m.%d-%H:%M:%S\t") + predict_path + '\nRank@1:%f Rank@5:%f Rank@10:%f mAP:%f\n' % (
          CMC[0], CMC[4], CMC[9], ap / len(query_label)))
    write(log_path,
          datetime.now().strftime("%Y.%m.%d-%H:%M:%S\t") + predict_path + '\tRank@1 Rank@5 Rank@10 mAP:\n%f\t%f\t%f\t%f\n' % (
          CMC[0], CMC[4], CMC[9], ap / len(query_label)))


def arg_parse():
    parser = argparse.ArgumentParser(description='eval on txt')
    # parser.add_argument('--target_dataset_path', default='/home/cwh/coding/dataset/grid-cv-1', type=str, help='')
    parser.add_argument('--target_dataset_path', default='/home/cwh/coding/dataset/market', type=str, help='')
    # parser.add_argument('--target_dataset_path', default='/home/cwh/coding/dataset/duke', type=str, help='')
    # parser.add_argument('--pid_path', default='/home/cwh/coding/TrackViz/data/market_duke-test/cross_filter_pid.log', type=str)
    # parser.add_argument('--pid_path', default='/home/cwh/coding/TrackViz/data/duke_market-r-test/cross_filter_pid.log', type=str)
    parser.add_argument('--pid_path', default='/home/cwh/coding/TrackViz/data/market_market-test/cross_filter_pid.log', type=str)
    # parser.add_argument('--pid_path', default='/home/cwh/coding/taudl_pyt/baseline/eval/market_market-test/pid.txt', type=str)
    # parser.add_argument('--pid_path', default='/home/cwh/coding/taudl_pyt/baseline/eval/market_grid-cv-1-test/pid.txt', type=str)
    # parser.add_argument('--result_path', default='/home/cwh/coding/taudl_pyt/market_eval_result.txt', type=str)
    # parser.add_argument('--result_path', default='/home/cwh/coding/taudl_pyt/duke_eval_result.txt', type=str)
    parser.add_argument('--result_path', default='/home/cwh/coding/taudl_pyt/market_eval_result.txt', type=str)
    opt = parser.parse_args()
    return opt


def main(opt):
    probe_path = opt.target_dataset_path + '/probe'
    gallery_path = opt.target_dataset_path + '/test'
    result_eval(opt.pid_path, opt.result_path, gallery_path, probe_path)

if __name__ == '__main__':
    opt = arg_parse()
    main(opt)
