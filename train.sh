#!/usr/bin/bash
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
MODEL_TYPE=${2:-bert}
TASK=${3:-POS_EN_ES}
DATA_DIR=${4:-"$REPO/Data/Processed_Data"}
CODE_DIR=${5:-"$REPO/Code"}
OUTPUT_DIR=${6:-"$REPO/Results"}
mkdir -p $OUTPUT_DIR
echo "Fine-tuning $MODEL on $TASK"

if [ $TASK == 'LID_EN_ES' ]; then 
  bash $CODE_DIR/train_token.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'LID_EN_HI' ]; then
  bash $CODE_DIR/train_token.sh "$TASK/Devanagari" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'POS_EN_ES' ]; then
  bash $CODE_DIR/train_token.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'POS_EN_HI_UD' ]; then
  bash $CODE_DIR/train_token.sh "$TASK/Devanagari" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'POS_EN_HI_FG' ]; then
  bash $CODE_DIR/train_token.sh "$TASK/Devanagari" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'NER_EN_ES' ]; then
  bash $CODE_DIR/train_token.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'NER_EN_HI' ]; then
  bash $CODE_DIR/train_token.sh "$TASK/Devanagari" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'Sentiment_EN_ES' ]; then
  bash $CODE_DIR/train_sentence.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'Sentiment_EN_HI' ]; then
  bash $CODE_DIR/train_sentence.sh "$TASK/Devanagari" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'QA_EN_HI' ]; then
  bash $CODE_DIR/train_qa.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'NLI_EN_HI' ]; then
  bash $CODE_DIR/train_nli.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
elif [ $TASK == 'ALL' ]; then
  bash $CODE_DIR/train_token.sh "LID_EN_ES" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
  bash $CODE_DIR/train_token.sh "LID_EN_HI/Devanagari" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
  bash $CODE_DIR/train_token.sh "NER_EN_ES" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
  bash $CODE_DIR/train_token.sh "NER_EN_HI/Devanagari" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
  bash $CODE_DIR/train_token.sh "POS_EN_ES" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
  bash $CODE_DIR/train_token.sh "POS_EN_HI_UD/Devanagari" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
  bash $CODE_DIR/train_token.sh "POS_EN_HI_FG/Devanagari" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
  bash $CODE_DIR/train_sentence.sh "Sentiment_EN_ES" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
  bash $CODE_DIR/train_sentence.sh "Sentiment_EN_HI/Devanagari" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
  bash $CODE_DIR/train_nli.sh "NLI_EN_HI" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
  bash $CODE_DIR/train_qa.sh "QA_EN_HI" $MODEL $MODEL_TYPE $DATA_DIR $OUTPUT_DIR
fi

