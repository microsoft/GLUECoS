# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import os
import json
import argparse

def convert_to_squad(x):
	answer_dict={}
	answer_dict['title']=x['id']
	answer_dict['paragraphs'] = []
	answer = x['answer']
	context = x['context']
	answer_indices = [i for i in range(len(context))  if context.startswith(answer, i)] 
	if len(answer_indices)>0:
		all_answers=[]
		for i in answer_indices:
			all_answers.append({'text':answer,'answer_start':i})
		qas=[{'question':x['query'],'id':x['id'],'answers':all_answers,'is_impossible':False}]
		answer_dict['paragraphs'].append({'qas':qas,'context':context})
	return answer_dict

def make_temp_file(original_path):

	with open(original_path+'code_mixed_qa_train.json','r') as infile:
			data = json.load(infile)
		
	with open(original_path+'/ID_Files/dev_ids.txt','r') as infile:
		con=infile.readlines()
	dev_ids = [int(x.strip('\n')) for x in con]

	with open(original_path+'/ID_Files/train_ids.txt','r') as infile:
		con=infile.readlines()
	train_ids = [int(x.strip('\n')) for x in con]

	all_questions=data['questions']
	hindi_questions=[]
	for i in all_questions:
		if i['language']=='Hindi':
			hindi_questions.append(i)


	final_dev=[]
	final_train=[]
	without_context=[]
	
	for x in hindi_questions:
		answer_dict=convert_to_squad(x)
		if len(answer_dict['paragraphs'])>0:
			if answer_dict['title'] in dev_ids:
				final_dev.append(answer_dict)
			elif answer_dict['title'] in train_ids:
				final_train.append(answer_dict) 
		else:
			without_context.append(x)
	
	with open('Data/Preprocess_Scripts/temp_dev.txt','w+') as df, open('Data/Preprocess_Scripts/temp_train.txt','w+') as tf:
		json.dump(final_dev,df)
		json.dump(final_train,tf)

	return without_context

def convert_to_drqa(without_context):
	with open('Data/Preprocess_Scripts/temp_drqa.txt','w') as f:
		for x in without_context:
			question = x['query']
			answer = [x['answer']]
			f.write(json.dumps({'question': question, 'answer': answer}))
			f.write('\n')

def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	args = parser.parse_args()

	# setting paths
	original_path = args.data_dir +'/QA_EN_HI/'

	without_context = make_temp_file(original_path)
	convert_to_drqa(without_context)

if __name__=='__main__':
	main()
	