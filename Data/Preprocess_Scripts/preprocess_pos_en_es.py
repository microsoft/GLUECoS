# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import os
import argparse

# read original files
def make_sentence_file(input_file):
	with open(input_file,'r') as f:
		con=f.readlines()
	lines = [x.strip('\n') for x in con]

	prev_id = lines[1].split('\t')[1]
	new_sent=''
	with open('temp_word.txt','a') as wf, open('temp_sentence.txt','a') as sf:
		for i in lines[1:len(lines)-1] :
			if i!='': 
				j=i.split('\t')
				word=j[3]
				lang=j[-4]
				curr_id=j[1]
				conversation_id=j[-3]
				if j[4]!='':
					if len(j[4].split('.'))>1:
						tag_first=j[4].split('.')[1].strip('\n')
						tag=tag_first.split()[0]
					else:
						tag=j[4].strip('\n')
					if(prev_id==curr_id):
						wf.write(conversation_id + '__' + curr_id+ '\t' + word+'\t'+lang+'\t'+tag+'\n')
						new_sent = new_sent + word + '__' + lang + '__' + tag + ' '
					else:
						wf.write('\n'+conversation_id + '__' + curr_id+ '\t'+word+'\t'+lang+'\t'+tag+'\n')
						sf.write(conversation_id + '__' + curr_id+ '\t'+new_sent+'\n')
						new_sent=''
						new_sent = new_sent + word + '__' + lang + '__' + tag + ' '
						prev_id=curr_id

# make processed file from ID and input files
def make_split_file(id_file,input_file,output_file,mode):
	
	with open(id_file,'r') as f:
		con=f.readlines()
	ids=[x.strip('\n') for x in con]

	with open(input_file,'r') as infile:
		con=infile.readlines()
	all_sentences=[x.strip('\n') for x in con]

	id_flag=False
	with open(output_file,'w') as outfile:
		for i in all_sentences:
			if i!='':
				j=i.split('\t')
				if j[0] in ids:
					if mode=='test':
						if j[1]!='':
							outfile.write(j[1]+'\t'+j[2]+'\t'+'N'+'\n')
					else:
						outfile.write(j[1]+'\t'+j[2]+'\t'+j[3]+'\n')
					id_flag=True

			else:
				if id_flag:
					outfile.write('\n')
					id_flag=False
		
def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	parser.add_argument("--output_dir", default=None, type=str, required=True, help="Processed data directory")
	args = parser.parse_args()

	# setting paths
	original_path = args.data_dir+'/POS_EN_ES/'
	id_dir = args.data_dir + '/POS_EN_ES/ID_Files'
	new_path = args.output_dir + '/POS_EN_ES'

	if not os.path.exists(new_path):
		os.mkdir(new_path)

	# clean original filees
	input_files = os.listdir(original_path+'temp')
	input_files = sorted(input_files)
	for i in input_files:
		make_sentence_file(original_path + 'temp/' +i)
	
	# make train, test and validation files
	make_split_file(id_dir+'/train_ids.txt','temp_word.txt',new_path+'/train.txt',mode='train')
	make_split_file(id_dir+'/test_ids.txt','temp_word.txt',new_path+'/test.txt',mode='test')
	make_split_file(id_dir+'/validation_ids.txt','temp_word.txt',new_path+'/validation.txt',mode='valid')

	# append all data in one file
	open(new_path+'/all.txt', 'w+').writelines([l for l in open(new_path+'/train.txt').readlines()])
	open(new_path+'/all.txt', 'a').writelines([l for l in open(new_path+'/test.txt').readlines()])
	open(new_path+'/all.txt', 'a').writelines([l for l in open(new_path+'/validation.txt').readlines()])

	# delete temp files
	os.unlink('temp_sentence.txt')
	os.unlink('temp_word.txt')

if __name__=='__main__':
	main()