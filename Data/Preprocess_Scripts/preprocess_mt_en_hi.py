# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import json
import os
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("tmp_data_dir", type=str)
parser.add_argument("id_dir", type=str)
parser.add_argument("output_dir", type=str)
args = parser.parse_args()

en_file_root = os.path.join(args.tmp_data_dir, 'datasets-CMU_DoG-618a14f27546165859305649aa84e6ac8710bb63', 'Conversations')
hien_file_root = os.path.join(args.tmp_data_dir, 'CMUHinglishDoG', 'Conversations_Hinglish')
os.makedirs(args.output_dir, exist_ok=True)

for split in ['train', 'valid', 'test']:
    english_files = os.listdir(os.path.join(en_file_root, 'train'))
    hinglish_files = os.listdir(os.path.join(hien_file_root, split))

    sentences = []
    count = 0
    for f in hinglish_files:
        en_file_path = f.split('.json')[0] + '.json'
        if en_file_path in english_files:
            count += 1
            en = json.load(open(os.path.join(en_file_root, 'train', en_file_path)))
            hien = json.load(open(os.path.join(hien_file_root, split, f)))
            assert len(en['history']) == len(hien['history'])
            for x, y in zip(en['history'], hien['history']):
                assert x['docIdx'] == y['docIdx']
                assert x['uid'] == y['uid']
                assert x['utcTimestamp'] == y['utcTimestamp']

                # Preprocessing
                x['text'] = re.sub('\t|\n', ' ', x['text'])
                y['text'] = re.sub('\t|\n', ' ', y['text'])
                to_append = (y['text'], x['text'])
                to_append = to_append[::-1]
                sentences.append(to_append)

    if split == "test":
        sentences = [[t[0], " "] for t in sentences]

    print(count, len(sentences))
    with open(os.path.join(args.output_dir, split + '.txt'), 'w') as f:
        f.write('\n'.join(['\t'.join(x) for x in sentences]))
