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

	#run scripts for scraping train tweets
	os.system("chmod +x runScripts.sh")
	subprocess.call(shlex.split('./runScripts.sh ./../en_es_training_offsets.tsv gold'))
	original_train_path = './Files/data.tsv'
	new_train_path = 'train.tsv'
	shutil.copyfile(original_train_path,new_train_path)

	#run scripts for scraping test tweets
	subprocess.call(shlex.split('./runScripts.sh ./../en_es_test_data.tsv reg'))
	original_test_path = './Files/data.tsv'
	new_test_path = 'test.tsv'
	shutil.copyfile(original_test_path,new_test_path)


# final format from the above obtained files
def make_temp_file(original_path):

	with open(original_path + 'Release/train.tsv','r+') as infile:
		con=infile.readlines()
	train_content=[x.strip('\n') for x in con]

	with open(original_path + 'Release/test.tsv','r+') as infile:
		con=infile.readlines()
	test_content=[x.strip('\n') for x in con]

	prev_id = train_content[0].split('\t')[0]
	with open('temp.txt','w+') as outfile:
		for i in train_content:
			if i!='':
				j=i.split('\t')
				curr_id=j[0]
				word=j[4]
				tag=j[5]
				if curr_id==prev_id:
					outfile.write(curr_id + '\t'+ word+'\t'+tag+'\n')
				else:
					outfile.write('\n' + curr_id + '\t' + word+'\t'+tag+'\n')
					prev_id=curr_id

	prev_id = test_content[0].split('\t')[0]
	with open('temp.txt','a') as outfile:
		for i in test_content:
			if i!='':
				j=i.split('\t')
				curr_id=j[0]
				word=j[4]
				#tag=j[5]
				if curr_id==prev_id:
					outfile.write(curr_id + '\t'+ word+'\n')
				else:
					outfile.write('\n' + curr_id + '\t' + word+'\n')
					prev_id=curr_id

# make processed file from ID and input files
def make_split_file(id_file,input_file,output_file,mode):
	
	with open(id_file,'r') as f:
		con=f.readlines()
	ids=[x.strip('\n') for x in con]

	with open(input_file,'r') as infile:
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
							res += j[1]+ '\t' + 'OTHER' + '\n'
							temp_dict.update({j[0]:res})
					else:
						if j[1]!='':
							temp_dict.update({j[0]:j[1]+ '\t' + 'OTHER' + '\n'})
				else:   

					if j[2]=='lang1':
						tag='EN'
					elif j[2]=='lang2':
						tag='ES'
					else:
						tag='OTHER'

					if j[0] in temp_dict.keys():
						res=temp_dict.get(j[0])
						res += j[1]+ '\t' + tag + '\n'
						temp_dict.update({j[0]:res})
					else:
						temp_dict.update({j[0]:j[1]+ '\t' + tag + '\n'})

	
	with open(output_file,'w') as outfile:
		for i in ids:
			if i in temp_dict.keys():
				outfile.write(temp_dict.get(i)+'\n')
			else:
				if mode=='test':
					outfile.write('not found' + '\t' + 'OTHER' + '\n\n')


def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	parser.add_argument("--output_dir", default=None, type=str, required=True, help="Processed data directory")
	args = parser.parse_args()

	# setting paths
	original_path = args.data_dir+'/LID_EN_ES/temp/'
	new_path = args.output_dir +'/LID_EN_ES/'  
	id_dir = args.data_dir + '/LID_EN_ES/ID_Files'
	owd = os.getcwd()  

	if not os.path.exists(new_path):
		os.mkdir(new_path)
	
	# call required functions to process tweets
	download_tweets(original_path)
	os.chdir(owd)
	make_temp_file(original_path)

	# make train, test and validation files
	make_split_file(id_dir+'/train_ids.txt','temp.txt',new_path+'/train.txt',mode='train')
	make_split_file(id_dir+'/test_ids.txt','temp.txt',new_path+'/test.txt',mode='test')

	to_remove = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 721, 722, 723, 724, 781, 794, 805, 1259, 1260, 1261, 1262, 1263, 1264, 1387, 1524, 1532]
	with open(new_path + '/test.txt') as f:
		lines = f.read().strip().split('\n\n')
	lines_out = []
	for i in range(len(lines)):
		if i not in to_remove:
			lines_out.append(lines[i])
	with open(new_path + '/test.txt', "w") as f:
		f.write("\n\n".join(lines_out) + "\n")

	make_split_file(id_dir+'/validation_ids.txt','temp.txt',new_path+'/validation.txt',mode='valid')
	
	# append all data in one file
	open(new_path+'/all.txt', 'w+').writelines([l for l in open(new_path+'/train.txt').readlines()])
	open(new_path+'/all.txt', 'a').writelines([l for l in open(new_path+'/test.txt').readlines()])
	open(new_path+'/all.txt', 'a').writelines([l for l in open(new_path+'/validation.txt').readlines()])

	# delete temp files
	os.unlink('temp.txt')

if __name__=='__main__':
	main()