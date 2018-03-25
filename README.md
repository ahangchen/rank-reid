## Rank Re-identification

### Introduction
- Using RankNet to regress ranking probability

### Model

![](img/rank_model.png)

- Base Network：ResNet50
- Input：Given a ranking list, choose a probe image A, two gallery image B, C
- Output：Compute the probability that rank AB > rank AC

### Hardware
- NVIDIA TITANX 11G
- Memory: >=16G

### Code
- baseline：ResNet52 base network
  - [evaluate.py](baseline/evaluate.py)
    - `extract_feature`: compute ranking result by base network and evaluate rank accuracy + mAP
    - `similarity_matrix`: Cosine similarity of CNN features(accelerated by GPU)
    - On test set, compute ranking table with `test_predict`
    - On training set，compute ranking table with `train_predict`
    - Compute rank accuracy and mAP with `map_rank_quick_eval` on Market1501(can be extended to DukeMTMC4ReID)
    - Compute rank accuracy with `grid_result_eval` on GRID
  - [train.py](baseline/train.py)
    - Use source dataset to pretrain ResNet52 base network
- pair: pretrain siamese network
  - [pair_train.py](pair/pair_train.py)：pretrain with two input images
    - pair_generator: data generator, selecting positive and negative samples according to person id
    - pair_model: build a Keras based Siamese network
  - [eval](pretrian/eval.py)：evaluate on Siamese Network and ranknet
    - load corresponding model
    - call function in baseline/evaluate.py for test

- transfer: incremental training with ranking table
  - [simple_rank_transfer.py](transfer/pair_transfer.py): learning to rank with three input images
    - triplet_generator_by_rank_list：image generator
    - rank_transfer_model：three input image, one ranking loss


#### Reference
- [RankNet](https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf)