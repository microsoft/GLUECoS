# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import os
import shutil
import argparse

# process test files
def process_test(input_file,output_file):
	
	with open(input_file,'r') as infile:
		con=infile.readlines()
	sentences=[x.strip('\n') for x in con]

	with open(output_file,'w') as outfile:
		for i in sentences:
			if i!='':
				j=i.split('\t')
				if j[0]!='':
					outfile.write(j[0]+'\t'+'neutral'+'\n')
			else:
				outfile.write('\n')

def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	parser.add_argument("--output_dir", default=None, type=str, required=True, help="Processed data directory")
	args = parser.parse_args()

	# setting paths
	original_path = args.data_dir+'/Sentiment_EN_HI/temp/SAIL_2017/Processed Data/'
	new_path = args.output_dir +'/Sentiment_EN_HI/'  

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
	open(new_path+'Romanized/all.txt', 'w+').writelines([l for l in open(new_path+'Romanized/train.txt').readlines() ])
	open(new_path+'Romanized/all.txt', 'a').writelines([l for l in open(new_path+'Romanized/test.txt').readlines() ])
	open(new_path+'Romanized/all.txt', 'a').writelines([l for l in open(new_path+'Romanized/validation.txt').readlines() ])

	# append all data in one file
	open(new_path+'Devanagari/all.txt', 'w+').writelines([l for l in open(new_path+'Devanagari/train.txt').readlines() ])
	open(new_path+'Devanagari/all.txt', 'a').writelines([l for l in open(new_path+'Devanagari/test.txt').readlines() ])
	open(new_path+'Devanagari/all.txt', 'a').writelines([l for l in open(new_path+'Devanagari/validation.txt').readlines() ])

if __name__=='__main__':
	main()