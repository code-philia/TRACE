LANG=all
MODEL_TAG=Salesforce/codet5-base
GPU=0
DATA_NUM=-1
BS=8
LR=5e-5
SRC_LEN=512
TRG_LEN=128
PATIENCE=5
EPOCH=1
WARMUP=100
label_num=3
paper_name=CoEdPilot
MODEL_DIR=./model_${label_num}_${paper_name}
OUTPUT_DIR=${MODEL_DIR}/${LANG}
SUMMARY_DIR=.
DATA_DIR=dataset_fine_grain/${LANG}
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
SELECT_METHOD=bm25
beam_size=10
ckpt_path=${OUTPUT_DIR}/checkpoint-last/pytorch_model.bin

mkdir -p ${OUTPUT_DIR}
mkdir -p ${RES_DIR}

CUDA_VISIBLE_DEVICES=1
python run.py --do_test --load_generator_path $ckpt_path\
    --model_type codet5 --data_num $DATA_NUM   \
    --num_train_epochs $EPOCH --warmup_steps $WARMUP --learning_rate $LR --patience $PATIENCE   \
    --tokenizer_name $MODEL_TAG --model_name_or_path $MODEL_TAG --data_dir $DATA_DIR  \
    --output_dir $OUTPUT_DIR  --summary_dir $SUMMARY_DIR \
    --save_last_checkpoints --always_save_model --res_dir $RES_DIR \
    --estimator_batch_size $BS \
    --train_batch_size $BS --eval_batch_size $BS --max_source_length $SRC_LEN --max_target_length $TRG_LEN \
    --select_method $SELECT_METHOD \
    --label_num $label_num --paper_name $paper_name\
    --beam_size $beam_size