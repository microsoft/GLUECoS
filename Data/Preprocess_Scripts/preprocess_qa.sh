# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

ORIGINAL_DATA_DIR=$PWD/Data/Original_Data
PROCESSED_DIR=$PWD/Data/Processed_Data
PREPROCESS_DIR=$PWD/Data/Preprocess_Scripts
OUTPUT_DIR=$PWD
INP_FILE=$PWD/Data/Preprocess_Scripts/temp_drqa.txt
PART1=`dirname "$INP_FILE"`
PART2=`basename "$INP_FILE"`

#preprocesss for DrQA
python3 $PREPROCESS_DIR/preprocess_drqa.py --data_dir $ORIGINAL_DATA_DIR

#run DrQA
git clone https://github.com/facebookresearch/DrQA.git
cd DrQA
git checkout 96f343c
pip install elasticsearch==7.8.0 nltk==3.5 scipy==1.5.0 prettytable==0.7.2 tqdm==4.46.1 regex==2020.6.8 termcolor==1.1.0 scikit-learn==0.23.1 numpy==1.18.5 torch==1.4.0
python3 setup.py develop
pip install spacy==2.3.0
python3 -m spacy download xx_ent_wiki_sm
python3 -c "import nltk;nltk.download(['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])"
./download.sh
sed -i 's/np.load(filename)/np.load(filename, allow_pickle=True)/g' drqa/retriever/utils.py
sed -i 's/\[\x27tokenizer_class\x27\], {},/\[\x27tokenizer_class\x27\], {\x27model\x27: \x27xx_ent_wiki_sm\x27},/g' scripts/distant/generate.py
sed -i 's/{\x27annotators\x27: {\x27ner\x27}}/{\x27annotators\x27: {\x27ner\x27}, \x27model\x27: \x27xx_ent_wiki_sm\x27}/g' scripts/distant/generate.py
sed -i 's/if any/#if any/g' drqa/tokenizers/spacy_tokenizer.py
sed -i 's/self.nlp.tagger/#self.nlp.tagger/g' drqa/tokenizers/spacy_tokenizer.py
patch scripts/distant/generate.py <<EOF
263a264
>     random.seed(0)
EOF
python3 scripts/distant/generate.py $PART1 $PART2 $PREPROCESS_DIR --tokenizer spacy --dev-split 0.2 --n-docs 1 --workers 1

cd ./..
# Squad format processor
python3 $PREPROCESS_DIR/preprocess_qa_en_hi.py --output_dir $PROCESSED_DIR
