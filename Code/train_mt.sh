# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-MT_EN_HI}
MODEL=${2:-facebook/mbart-large-cc25}
DATA_DIR=${3:-"$REPO/Data/Processed_Data"}
OUT_DIR=${4:-"$REPO/Results"}

EPOCH=5
TRAIN_BATCH_SIZE=8 # Adjust depending on GPU memory
EVAL_BATCH_SIZE=4 # Adjust depending on GPU memory
NUM_BEAMS=5
MODEL_OUTPUT_DIR=/tmp/mt_model # Adjust this as needed

python $PWD/Code/run_seq2seq.py \
    --model_name_or_path $MODEL \
    --data_dir $DATA_DIR/$TASK/ \
    --source_id en_XX --target_id en_XX \
    --output_dir $MODEL_OUTPUT_DIR \
    --save_path $OUT_DIR/$TASK/translations.txt  \
    --overwrite_output_dir  \
    --do_train --do_eval  \
    --fp16 \
    --evaluation_strategy steps \
    --eval_steps 1500 \
    --logging_steps 1500 \
    --save_steps 6000 \
    --predict_with_generate \
    --no_use_fast_tokenizer \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --pad_to_max_length \
    --num_train_epochs $EPOCH \
    --eval_beams $NUM_BEAMS \
    --early_stopping