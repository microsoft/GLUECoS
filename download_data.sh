#!/bin/bash
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
SUBSCRIPTION_KEY=${1:-""}
ORIGINAL_DIR="$REPO/Data/Original_Data"
PREPROCESS_DIR="$REPO/Data/Preprocess_Scripts"
PROCESSED_DIR="$REPO/Data/Processed_Data"
mkdir -p $PROCESSED_DIR

# get transliterations
function get_transliterations {
    if [ -z $SUBSCRIPTION_KEY ]; then
        python3 transliterator.py --input_file all_roman.txt
    else
        python3 transliterator.py --input_file all_roman.txt --subscription_key $SUBSCRIPTION_KEY
    fi
}

# download LID EN ES dataset
function download_lid_en_es {
    OUTPATH=$ORIGINAL_DIR/LID_EN_ES/temp
    mkdir -p $OUTPATH
    if [ ! -f $OUTPATH/en_es_test_data.tsv ]; then
        wget -c http://mirror.aclweb.org/emnlp2014/workshops/CodeSwitch/data/Spanish_English/test/en_es_test_data.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/en_es_training_offsets.tsv ]; then
        wget -c http://mirror.aclweb.org/emnlp2014/workshops/CodeSwitch/data/Spanish_English/training/en_es_training_offsets.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -d $OUTPATH/Release ]; then
      if [ ! -f $OUTPATH/Release.zip ]; then
        wget -c https://code-switching.github.io/2018/files/spa-eng/Release.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/Release.zip -d $OUTPATH
    fi

    python3 $PREPROCESS_DIR/preprocess_lid_en_es.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR
 
    rm -rf $OUTPATH
    echo "Downloaded LID EN ES"
}

# download LID EN HI dataset
function download_lid_en_hi {
    OUTPATH=$ORIGINAL_DIR/LID_EN_HI/temp
    mkdir -p $OUTPATH
    
    if [ ! -f $OUTPATH/HindiEnglish_FIRE2013_AnnotatedDev.txt ]; then
        wget -c https://cse.iitkgp.ac.in/resgrp/cnerg/qa/fire13translit/FIRE_Data/HindiEnglish_FIRE2013_AnnotatedDev.txt -P $OUTPATH -q --show-progress
    fi

    if [ ! -f $OUTPATH/HindiEnglish_FIRE2013_Test_GT.txt ]; then
        wget -c https://cse.iitkgp.ac.in/resgrp/cnerg/qa/fire13translit/FIRE_Data/HindiEnglish_FIRE2013_Test_GT.txt -P $OUTPATH -q --show-progress
    fi

    if [ ! -d $OUTPATH/ICON_POS ]; then
      if [ ! -f $OUTPATH/ICON_POS.zip ]; then
        wget -c http://www.amitavadas.com/ICON2016/ICON_POS.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/ICON_POS.zip -d $OUTPATH
    fi

    python3 $PREPROCESS_DIR/preprocess_lid_en_hi.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR
    
    rm -rf $OUTPATH
    echo "Downloaded LID EN HI"
}

# download NER EN ES dataset
function download_ner_en_es {
    OUTPATH=$ORIGINAL_DIR/NER_EN_ES/temp
    mkdir -p $OUTPATH
    if [ ! -f $OUTPATH/train_offset.tsv ]; then
        wget -c https://code-switching.github.io/2018/files/spa-eng/train_offset.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/dev_offset.tsv ]; then
        wget -c https://code-switching.github.io/2018/files/spa-eng/dev_offset.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -d $OUTPATH/Release ]; then
      if [ ! -f $OUTPATH/Release.zip ]; then
        wget -c https://code-switching.github.io/2018/files/spa-eng/Release.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/Release.zip -d $OUTPATH
    fi

    python3 $PREPROCESS_DIR/preprocess_ner_en_es.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR
    
    rm -rf $OUTPATH 
    echo "Downloaded NER EN ES"
}

# download NER EN HI dataset
function download_ner_en_hi {
    OUTPATH=$ORIGINAL_DIR/NER_EN_HI/temp
    mkdir -p $OUTPATH
    if [ ! -f $OUTPATH/annotatedData.csv ]; then
        wget -c https://github.com/SilentFlame/Named-Entity-Recognition/raw/master/Twitterdata/annotatedData.csv -P $OUTPATH -q --show-progress
    fi
    
    python3 $PREPROCESS_DIR/preprocess_ner_en_hi.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR

    rm -rf $OUTPATH 
    echo "Downloaded NER EN HI"
}

# download POS EN HI UD dataset
function download_pos_en_hi_ud {
    OUTPATH=$ORIGINAL_DIR/POS_EN_HI_UD/temp
    mkdir -p $OUTPATH
    if [ ! -d $OUTPATH/master ]; then
      if [ ! -f $OUTPATH/master.zip ]; then
        wget -c https://github.com/CodeMixedUniversalDependencies/UD_Hindi_English/archive/master.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/master.zip -d $OUTPATH
    fi
    

    # insert patch command here
    patch $OUTPATH/UD_Hindi_English-master/crawl_tweets.py <<EOF
    478c478,479
    <         dep_tweet = zip(annot['ids'], annot['tweet'], norm, annot['pos'], annot['cpos'],
    ---
    >         tweet_id = [tid]*len(annot['ids'])
    >         dep_tweet = zip(tweet_id,annot['ids'], annot['tweet'], norm, annot['pos'], annot['cpos'],
EOF

    python3 $PREPROCESS_DIR/preprocess_pos_en_hi_ud.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR

    rm -rf $OUTPATH 
    echo "Downloaded POS EN HI UD"
}

# download POS EN HI FG dataset
function download_pos_en_hi_fg {
    OUTPATH=$ORIGINAL_DIR/POS_EN_HI_FG/temp
    mkdir -p $OUTPATH
    if [ ! -d $OUTPATH/ICON_POS ]; then
      if [ ! -f $OUTPATH/ICON_POS.zip ]; then
        wget -c http://www.amitavadas.com/ICON2016/ICON_POS.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/ICON_POS.zip -d $OUTPATH
    fi

    python3 $PREPROCESS_DIR/preprocess_pos_en_hi_fg.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR

    rm -rf $OUTPATH 
    echo "Downloaded POS EN HI FG"
}

# download Sentiment EN ES dataset
function download_sentiment_en_es {
    OUTPATH=$ORIGINAL_DIR/Sentiment_EN_ES/temp
    mkdir -p $OUTPATH
    if [ ! -f $OUTPATH/cs-en-es-corpus-wassa2015.txt ]; then
        wget -c http://www.grupolys.org/software/CS-CORPORA/cs-en-es-corpus-wassa2015.txt -P $OUTPATH -q --show-progress
    fi

    python3 $PREPROCESS_DIR/preprocess_sent_en_es.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR

    rm -rf $OUTPATH 
    echo "Downloaded Sentiment EN ES"
}

# download Sentiment EN HI dataset
function download_sentiment_en_hi {
    OUTPATH=$ORIGINAL_DIR/Sentiment_EN_HI/temp
    mkdir -p $OUTPATH
    if [ ! -d $OUTPATH/SAIL_2017 ]; then
      if [ ! -f $OUTPATH/SAIL_2017.zip ]; then
        wget -c http://amitavadas.com/SAIL/Data/SAIL_2017.zip -P $OUTPATH -q --show-progress
      fi
      unzip -qq $OUTPATH/SAIL_2017.zip -d $OUTPATH
    fi

    python3 $PREPROCESS_DIR/preprocess_sent_en_hi.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR

    rm -rf $OUTPATH 
    echo "Downloaded Sentiment EN HI"
}

# download QA EN HI dataset
function download_qa_en_hi {
    OUTPATH=$ORIGINAL_DIR/QA_EN_HI/
    mkdir -p $OUTPATH
    if [ ! -f $OUTPATH/code_mixed_qa_train.json ]; then
      wget -c https://raw.githubusercontent.com/khyathiraghavi/code_switched_QA/master/code_mixed_qa_train.json -P $OUTPATH -q --show-progress
    fi

    # rm -rf $OUTPATH 
    echo "Downloaded QA EN HI"
}

# download NLI EN HI dataset
function download_nli_en_hi {
    OUTPATH=$ORIGINAL_DIR/NLI_EN_HI/temp
    mkdir -p $OUTPATH
    if [ ! -d $OUTPATH/all_keys_json ]; then
        if [ ! -f $OUTPATH/all_keys_json.zip ]; then
            wget -c https://www.cse.iitb.ac.in/~pjyothi/indiccorpora/all_keys_json.zip -P $OUTPATH -q --show-progress
        fi
        unzip -qq $OUTPATH/all_keys_json.zip -d $OUTPATH
    fi

    if [ ! -f $OUTPATH/all_only_id.json ]; then
        url=$'https://api.onedrive.com/v1.0/drives/85FEAFEE8D8062F3/items/85FEAFEE8D8062F3!28569?select=id%2C%40content.downloadUrl&authkey=!ADungCV7vUzIE_g'
        wget $url -q -O - | python3 -c "import json, sys; j=json.load(sys.stdin); print(j['@content.downloadUrl'])" | wget -i - -O $OUTPATH/all_only_id.json -q --show-progress
    fi

    python3 $PREPROCESS_DIR/preprocess_nli_en_hi.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR

    rm -rf $OUTPATH 
    echo "Downloaded NLI EN HI"
}

# download POS EN ES dataset
function download_pos_en_es {
    OUTPATH=$ORIGINAL_DIR/POS_EN_ES/temp
    mkdir -p $OUTPATH
    if [ ! -f $OUTPATH/herring1_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring1_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring2_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring2_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring3_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring3_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring5_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring5_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring6_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring6_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring7_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring7_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring8_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring8_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring9_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring9_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring10_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring10_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring11_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring11_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring12_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring12_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring13_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring13_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring14_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring14_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring15_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring15_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring16_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring16_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/herring17_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/herring17_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria1_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria1_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria2_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria2_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria4_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria4_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria7_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria7_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria10_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria10_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria16_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria16_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria18_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria18_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria19_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria19_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria20_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria20_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria21_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria21_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria24_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria24_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria27_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria27_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria30_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria30_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria31_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria31_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/maria40_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/maria40_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre1_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre1_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre2_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre2_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre3_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre3_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre4_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre4_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre5_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre5_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre6_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre6_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre7_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre7_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre8_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre8_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre9_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre9_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre10_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre10_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre11_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre11_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre12_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre12_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/sastre13_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/sastre13_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon1_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon1_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon2_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon2_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon3_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon3_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon4_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon4_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon5_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon5_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon6_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon6_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon7_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon7_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon8_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon8_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon9_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon9_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon11_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon11_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon13_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon13_cgwords.tsv -P $OUTPATH -q --show-progress
    fi
    if [ ! -f $OUTPATH/zeledon14_cgwords.tsv ]; then
        wget -c http://bangortalk.org.uk/tsvs/miami/zeledon14_cgwords.tsv -P $OUTPATH -q --show-progress
    fi

    python3 $PREPROCESS_DIR/preprocess_pos_en_es.py --data_dir $ORIGINAL_DIR --output_dir $PROCESSED_DIR

    rm -rf $OUTPATH 
    echo "Downloaded POS EN ES"
}

# download MT EN HI dataset
function download_mt_en_hi {
    OUTPATH=$ORIGINAL_DIR/MT_EN_HI/temp
    mkdir -p $OUTPATH
    if [ ! -f $OUTPATH/CMUHinglishDoG.zip ]; then
        wget -c http://festvox.org/cedar/data/notyet/CMUHinglishDoG.zip -P $OUTPATH -q --show-progress
        unzip -qq $OUTPATH/CMUHinglishDoG.zip -d $OUTPATH
    fi
    if [ ! -f $OUTPATH/618a14f.zip ]; then
        wget -c https://github.com/festvox/datasets-CMU_DoG/archive/618a14f.zip -P $OUTPATH -q --show-progress
        unzip -qq $OUTPATH/618a14f.zip -d $OUTPATH
    fi

    python3 $PREPROCESS_DIR/preprocess_mt_en_hi.py $OUTPATH $ORIGINAL_DIR/MT_EN_HI/ $PROCESSED_DIR/MT_EN_HI

    rm -rf $OUTPATH 
    echo "Downloaded MT EN HI"
}

get_transliterations
download_lid_en_hi
download_ner_en_hi
download_pos_en_es
download_pos_en_hi_fg
download_sentiment_en_hi
download_qa_en_hi
download_pos_en_hi_ud
download_nli_en_hi
download_sentiment_en_es
download_lid_en_es
download_ner_en_es
# download_mt_en_hi
