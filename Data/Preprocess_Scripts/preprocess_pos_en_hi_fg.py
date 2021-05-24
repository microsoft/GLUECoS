# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import os
import shutil
import argparse

# process test files
def process_test(input_file,output_file):
	
	with open(input_file,'r',encoding='utf-8') as infile:
		con=infile.readlines()
	sentences=[x.strip('\n') for x in con]

	with open(output_file,'w',encoding='utf-8') as outfile:
		for i in sentences:
			if i!='':
				j=i.split('\t')
				if j[0]!='':
					outfile.write(j[0]+'\t'+j[1]+'\t'+ 'N_NN' +'\n')
			else:
				outfile.write('\n')

def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	parser.add_argument("--output_dir", default=None, type=str, required=True, help="Processed data directory")
	args = parser.parse_args()

	# setting paths
	original_path = args.data_dir+ '/POS_EN_HI_FG/temp/ICON_POS/Processed Data/'
	new_path = args.output_dir +'/POS_EN_HI_FG/'  

	if not os.path.exists(new_path):
		os.mkdir(new_path)
	if not os.path.exists(new_path+'Romanized'):
		os.mkdir(new_path+'Romanized')
	if not os.path.exists(new_path+'Devanagari'):
		os.mkdir(new_path+'Devanagari')

	# copy train and validtaion files in processed folder
	shutil.copy(original_path+'Romanized/train.txt',new_path+'Romanized/train.txt')
	shutil.copy(original_path+'Romanized/validation.txt',new_path+'Romanized/validation.txt')
	shutil.copy(original_path+'Devanagari/train.txt',new_path+'Devanagari/train.txt')
	shutil.copy(original_path+'Devanagari/validation.txt',new_path+'Devanagari/validation.txt')

	# process test files
	process_test(original_path+'Romanized/test.txt',new_path+'Romanized/test.txt')
	process_test(original_path+'Devanagari/test.txt',new_path+'Devanagari/test.txt')
	
	# append all data in one file
	open(new_path+'Romanized/all.txt', 'w+',encoding='utf-8').writelines([l for l in open(new_path+'Romanized/train.txt','r',encoding='utf-8').readlines() ])
	open(new_path+'Romanized/all.txt', 'a',encoding='utf-8').writelines([l for l in open(new_path+'Romanized/test.txt','r',encoding='utf-8').readlines() ])
	open(new_path+'Romanized/all.txt', 'a',encoding='utf-8').writelines([l for l in open(new_path+'Romanized/validation.txt','r',encoding='utf-8').readlines() ])

	# append all data in one file
	open(new_path+'Devanagari/all.txt', 'w+',encoding='utf-8').writelines([l for l in open(new_path+'Devanagari/train.txt','r',encoding='utf-8').readlines() ])
	open(new_path+'Devanagari/all.txt', 'a',encoding='utf-8').writelines([l for l in open(new_path+'Devanagari/test.txt','r',encoding='utf-8').readlines() ])
	open(new_path+'Devanagari/all.txt', 'a',encoding='utf-8').writelines([l for l in open(new_path+'Devanagari/validation.txt','r',encoding='utf-8').readlines() ])

if __name__=='__main__':
	main()