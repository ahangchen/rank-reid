import os
import numpy as np

def extract_info(dir_path):
    infos = []
    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name:
            continue
        if 's' in image_name or 'f' in image_name:
            # market && duke
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        elif 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        else:
            continue
        infos.append((person, camera))

    return infos


def gt_ranking(infos):
    gt_answer_ids = list()
    i = 0
    last_pid = -1
    cur_gt_person_ids = list()
    while i < len(infos):
        person = infos[i][0]
        if person != last_pid:
            if last_pid != -1:
                gt_answer_ids.append(cur_gt_person_ids)
            cur_gt_person_ids = list()
        cur_gt_person_ids.append(i)
        i += 1
        last_pid = person
    gt_answer_ids.append(cur_gt_person_ids)
    gt_persons_scores = [list() for i in range(len(infos))]
    gt_persons_ids = [list() for i in range(len(infos))]


    cur_answer_index = 0
    for i in range(len(infos)):
        cur_answers = gt_answer_ids[cur_answer_index]
        if i > cur_answers[-1]:
            cur_answer_index += 1
            if cur_answer_index >= 750:
                print cur_answer_index
            cur_answers = gt_answer_ids[cur_answer_index]
        gt_persons_ids[i].extend(cur_answers)

    for i in range(len(infos)):
        gt_persons_scores[i].extend(np.ones(len(gt_persons_ids[i])))
        gt_persons_scores[i].extend(np.zeros(len(infos)- len(gt_persons_ids[i])))

    for i in range(len(infos)):
        gt_persons_id = gt_persons_ids[i]
        gt_range_start = gt_persons_id[0]
        gt_range_end = gt_persons_id[-1]
        if gt_range_start != 0:
            gt_persons_id.extend(range(0, gt_range_start))
        if gt_range_end != len(infos) - 1:
            gt_persons_id.extend(range(gt_range_end + 1, len(infos)))
    gt_persons_ids = np.array(gt_persons_ids)
    gt_persons_scores = np.array(gt_persons_scores)
    np.savetxt('cross_filter_pid.log', gt_persons_ids, '%d')
    np.savetxt('cross_filter_score.log', gt_persons_scores, '%4f')




if __name__ == '__main__':
    infos = extract_info('/home/cwh/coding/Market-1501/train')
    scores = gt_ranking(infos)