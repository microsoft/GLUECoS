# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import os
import subprocess
import shlex
import shutil
import argparse

# download tweets
def download_tweets(original_path):

	# copy twitter credentials in required file
	shutil.copy('twitter_authentication.txt',original_path+'/Release/twitter_auth.txt')

	os.chdir(original_path+'/Release')

	#run scripts for scraping validation tweets
	os.system('chmod +x runScripts.sh')
	subprocess.call(shlex.split('./runScripts.sh ./../dev_offset.tsv gold'))
	original_validation_path = './Files/data.tsv'
	new_validation_path = 'validation.tsv'
	shutil.copyfile(original_validation_path,new_validation_path)

	#run scripts for scraping train tweets
	subprocess.call(shlex.split('./runScripts.sh ./../train_offset.tsv gold'))
	original_train_path = './Files/data.tsv'
	new_train_path = 'train.tsv'
	shutil.copyfile(original_train_path,new_train_path)


# final format from above obtained files
def make_temp_file(original_path):

	with open(original_path + 'Release/validation.tsv','r+', encoding='utf-8') as infile:
		con=infile.readlines()
	validation_content=[x.strip('\n') for x in con]

	with open(original_path + 'Release/train.tsv','r+', encoding='utf-8') as infile:
		con=infile.readlines()
	train_content=[x.strip('\n') for x in con]

	prev_id = validation_content[0].split('\t')[0]
	with open('temp.txt','w+', encoding='utf-8') as outfile:
		for i in validation_content:
			if i!='':
				j=i.split('\t')
				curr_id=j[0]
				word=j[4].replace(chr(65039),'')
				tag=j[5]
				if curr_id==prev_id:
					if word!='':
						outfile.write(curr_id + '\t'+ word+'\t'+tag+'\n')
				else:
					if word!='':
						outfile.write('\n' + curr_id + '\t' + word+'\t'+tag+'\n')
					prev_id=curr_id

	prev_id = train_content[0].split('\t')[0]
	with open('temp.txt','a', encoding='utf-8') as outfile:
		for i in train_content:
			if i!='':
				j=i.split('\t')
				curr_id=j[0]
				word=j[4].replace(chr(65039),'')
				tag=j[5]
				if curr_id==prev_id:
					if word!='':
						outfile.write(curr_id + '\t'+ word+'\t'+tag+'\n')
				else:
					if word!='':
						outfile.write('\n' + curr_id + '\t' + word+'\t'+tag+'\n')
					prev_id=curr_id

# make processed file from ID and input files
def make_split_file(id_file,input_file,output_file,mode):
	
	with open(id_file,'r',encoding='utf-8') as f:
		con=f.readlines()
	ids=[x.strip('\n') for x in con]

	with open(input_file,'r',encoding='utf-8') as infile:
		con=infile.readlines()
	all_sentences=[x.strip('\n') for x in con]

	temp_dict={}
	for i in all_sentences:
		if i!='':
			j=i.split('\t')
			if j[0] in ids:
				if mode=='test':
					if j[0] in temp_dict.keys():
						res=temp_dict.get(j[0])
						if j[1]!='':
							res += j[1]+ '\t' + 'O' + '\n'
							temp_dict.update({j[0]:res})
					else:
						if j[1]!='':
							temp_dict.update({j[0]:j[1]+ '\t' + 'O' + '\n'})
				else:   
					if j[0] in temp_dict.keys():
						res=temp_dict.get(j[0])
						res += j[1]+ '\t' + j[2] + '\n'
						temp_dict.update({j[0]:res})
					else:
						temp_dict.update({j[0]:j[1]+ '\t' + j[2] + '\n'})

	
	with open(output_file,'w',encoding='utf-8') as outfile:
		for i in ids:
			if i in temp_dict.keys():
				outfile.write(temp_dict.get(i)+'\n')
			else:
				if mode=='test':
					outfile.write('not found' + '\t' + 'O' + '\n\n')


def main():
	parser = argparse.ArgumentParser()

  # Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	parser.add_argument("--output_dir", default="en", type=str, help="Processed data directory")
	args = parser.parse_args()

	original_path = args.data_dir+'/NER_EN_ES/temp/'
	new_path = args.output_dir +'/NER_EN_ES/' 
	id_dir = args.data_dir + '/NER_EN_ES/ID_Files' 
	owd = os.getcwd()  

	if not os.path.exists(new_path):
		os.mkdir(new_path)
	
	# call required functions to process
	download_tweets(original_path)
	os.chdir(owd)
	make_temp_file(original_path)

	# make train, test and validation files
	make_split_file(id_dir+'/train_ids.txt','temp.txt',new_path+'/train.txt',mode='train')
	make_split_file(id_dir+'/test_ids.txt','temp.txt',new_path+'/test.txt',mode='test')
	make_split_file(id_dir+'/validation_ids.txt','temp.txt',new_path+'/validation.txt',mode='valid')
	
	# append all data in one file
	open(new_path+'/all.txt', 'w+', encoding='utf-8').writelines([l for l in open(new_path+'/train.txt', 'r', encoding='utf-8').readlines()])
	open(new_path+'/all.txt', 'a',encoding='utf-8').writelines([l for l in open(new_path+'/test.txt','r',encoding='utf-8').readlines()])
	open(new_path+'/all.txt', 'a',encoding='utf-8').writelines([l for l in open(new_path+'/validation.txt','r',encoding='utf-8').readlines()])

	# delete temp files
	os.unlink('temp.txt')

if __name__=='__main__':
	main()

