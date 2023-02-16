# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import os
import json
import argparse

def convert_to_squad(infile,list_all):
    with open(infile) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        answer_dict={}
        data = json.loads(line)
        data['id'] = 200 + i
        question = ' '.join(data['question'])
        start, end = data['answers'][0]
        doc = data['document']
        ans = ' '.join(doc[start: end + 1])
        answer_dict["title"]=data['id']
        answer_dict["paragraphs"] = []
        answer = ans
        context = ' '.join(doc[0:])
        answer_indices = [i for i in range(len(context))  if context.startswith(answer, i)] 
        if len(answer_indices)>0:
            all_answers=[]
            for i in answer_indices:
                all_answers.append({"text":answer,"answer_start":i})
            qas=[{"question":question,"id":data['id'],"answers":all_answers,"is_impossible":False}]
            answer_dict["paragraphs"].append({"qas":qas,"context":context})
        list_all.append(answer_dict)
    return list_all


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="Processed data directory")
    args = parser.parse_args()

    # setting paths
    new_path = args.output_dir +'/QA_EN_HI/'

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    print(os.getcwd())

    with open('./Data/Preprocess_Scripts/temp_dev.txt','r') as infile:
        final_dev = json.load(infile)
    with open('./Data/Preprocess_Scripts/temp_train.txt','r') as infile:
        final_train = json.load(infile)
    
    all_dev = convert_to_squad('./Data/Preprocess_Scripts/temp_drqa.dsdev',final_dev)
    all_train = convert_to_squad('./Data/Preprocess_Scripts/temp_drqa.dstrain',final_train)

    dev={}
    dev["version"] =  "v2.0"
    dev["data"] = all_dev

    train={}
    train["version"] =  "v2.0"
    train["data"] = all_train

    with open(new_path+'dev-v2.0.json','w') as outfile:
        json.dump(dev,outfile)
    
    with open(new_path+'train-v2.0.json','w') as outfile:
        json.dump(train,outfile)

    os.unlink('Data/Preprocess_Scripts/temp_dev.txt')
    os.unlink('Data/Preprocess_Scripts/temp_train.txt')
    os.unlink('Data/Preprocess_Scripts/temp_drqa.txt')
    os.unlink('Data/Preprocess_Scripts/temp_drqa.dsdev')
    os.unlink('Data/Preprocess_Scripts/temp_drqa.dstrain')

if __name__=='__main__':
    main()