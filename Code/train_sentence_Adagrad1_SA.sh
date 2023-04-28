# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-Sentiment_EN_ES}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
OUT_DIR=${5:-"$REPO/Results"}

# EPOCH=5
# BATCH_SIZE=16
# MAX_SEQ=256

EPOCH=8
BATCH_SIZE=64
MAX_SEQ=256
SCALE_PARAMETER=0.5
DECAY_RATE=0.01
BETA1=0.0
dir=`basename "$TASK"`
if [ $dir == "Devanagari" ] || [ $dir == "Romanized" ]; then
  OUT=`dirname "$TASK"`
else
  OUT=$TASK
fi

python $PWD/Code/BertSequence.py \
  --data_dir $DATA_DIR/$TASK \
  --output_dir $OUT_DIR/$OUT \
  --model_type $MODEL_TYPE \
  --model_name $MODEL \
  --num_train_epochs $EPOCH \
  --train_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ \
  --scale_parameter $SCALE_PARAMETER \
  --decay_rate $DECAY_RATE \
  --beta1 $BETA1 \
  --save_steps -1