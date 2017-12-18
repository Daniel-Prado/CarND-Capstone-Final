WORKSPACE=`dirname $(realpath $0)`/../../

DATA_DIR=${WORKSPACE}/data/image_classification/1/
TRAIN_DIR=${WORKSPACE}/logs/image_classification/1/train
EVAL_DIR=${WORKSPACE}/logs/image_classification/1/eval

# Run the training
alias train_image_classification='CUDA_VISIBLE_DEVICES=0 python patent_od/image_classification/train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR}'

# Run the evaluation
alias eval_image_classification='CUDA_VISIBLE_DEVICES=1 python patent_od/image_classification/eval.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --wait_for_checkpoints=True'

alias board_image_classification='tensorboard --logdir=${WORKSPACE}/logs/image_classification/1/'
