# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import os
import shutil
import argparse
import random
import json 

# make processed file from movie IDs
def process_files(final_key_path,all_id_path):
	
	with open(final_key_path,'r') as infile:
		final_key = json.load(infile)
	with open(all_id_path,'r') as infile:
		all_id = json.load(infile)
	
	movie_key={}
	for i in final_key:
		movie_key.update({i["ID"]:i["Conversation"]})
	
	all=[]
	for i in all_id:
		premise = movie_key.get(i["Premise ID"])
		all.append({"ID":i["ID"],"Premise ID":i["Premise ID"],"Premise":premise,"Hypothesis":i["Hypothesis"],"Label":i["Label"]})

	with open('temp.json','w') as outfile:
		json.dump(all,outfile)

# make processed file from ID and input files
def make_split_file(id_file,input_file,output_file,mode):
	
	with open(id_file,'r') as f:
		con=f.readlines()
	ids=[x.strip('\n') for x in con]
	
	with open(input_file,'r') as infile:
		all_sentences=json.load(infile)

	all_json=[]
	for j in ids:
		for i in all_sentences:
			if (str(i["ID"]) == j):
				if mode=='test':
					if i["Hypothesis"].strip()!='\'':
						all_json.append({"Premise":i["Premise"],"Hypothesis":i["Hypothesis"],"Label": 'entailment'})
				else:
					if i["Label"] == 'entailed':
						label='entailment' 
					else:
						label='contradictory'   
					all_json.append({"Premise":i["Premise"],"Hypothesis":i["Hypothesis"],"Label": label})

	with open(output_file,'w') as outfile:
		json.dump(all_json,outfile)

# convert to XNLI format
def convert_xnli_form(new_path):
	
	if not os.path.exists(new_path+'/XNLI-1.0'):
		os.mkdir(new_path+'/XNLI-1.0')
	if not os.path.exists(new_path+'/XNLI-MT-1.0'):
		os.mkdir(new_path+'/XNLI-MT-1.0')
	if not os.path.exists(new_path+'/XNLI-MT-1.0/multinli/'):
		os.mkdir(new_path+'/XNLI-MT-1.0/multinli')

	train_file = new_path+'/XNLI-MT-1.0/multinli/multinli.train.en.tsv'
	test_file = new_path+'/XNLI-1.0/xnli.test.tsv'

	with open(new_path+'/train.json','r') as infile:
		train_data = json.load(infile)

	with open(new_path+'/test.json','r') as infile:
		test_data = json.load(infile)

	with open(train_file,'w') as outfile:
		outfile.write('Premise'+'\t'+'Hypothesis'+'\t'+'Label'+'\n')
		for i in train_data:
			temp_premise=''
			j=i["Premise"].split('\n')
			for y,x in enumerate(j[:-1:2]):
				temp_premise+=j[y]+' : ' +j[y+1] +' ## '
			outfile.write(temp_premise+'\t'+i["Hypothesis"]+'\t'+i["Label"]+'\n')

	with open(test_file,'w') as outfile:
		outfile.write('en'+'\t'+'Label'+'\t\t\t\t\t'+'Premise'+'\t'+'Hypothesis'+'\t\t\t\t\t\t\t\t\t'+'Premise'+'\t'+'Hypothesis'+'\n')
		for i in test_data:
			temp_premise=''
			j=i["Premise"].split('\n')
			for y,x in enumerate(j[:-1:2]):
				temp_premise+=j[y]+' : ' +j[y+1] +' ## '
			outfile.write('en'+'\t'+i["Label"]+'\t\t\t\t\t'+temp_premise+'\t'+i["Hypothesis"]+'\t\t\t\t\t\t\t\t\t'+temp_premise+'\t'+i["Hypothesis"]+'\n')

	os.unlink(new_path+'/train.json')
	os.unlink(new_path+'/test.json')

def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	parser.add_argument("--output_dir", default=None, type=str, help="Processed data directory")
	args = parser.parse_args()

	# setting paths
	original_path = args.data_dir+'/NLI_EN_HI/temp/'
	id_dir = args.data_dir+'/NLI_EN_HI/ID_Files'
	new_path = args.output_dir +'/NLI_EN_HI/'  

	if not os.path.exists(new_path):
		os.mkdir(new_path)
	
	# call required functions to process
	process_files(original_path+'all_keys_json/Final_Key.json',args.data_dir+'/NLI_EN_HI/temp/all_only_id.json')

	# make train, test and validation files
	make_split_file(id_dir+'/train_ids.txt','temp.json',new_path+'/train.json',mode='train')
	make_split_file(id_dir+'/test_ids.txt','temp.json',new_path+'/test.json',mode='test')

	# convert to xnli format
	convert_xnli_form(new_path)

	os.unlink('temp.json')
	
if __name__=='__main__':
	main()