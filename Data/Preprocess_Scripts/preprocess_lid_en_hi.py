# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import os
import shutil
import argparse

#Copy original files here temporarily to process
def make_temp_file(original_path):
	original_path_validation = original_path + '/HindiEnglish_FIRE2013_AnnotatedDev.txt'
	new_path_validation = './temp_validation.txt'
	shutil.copy(original_path_validation,new_path_validation)

	original_path_test = original_path + '/HindiEnglish_FIRE2013_Test_GT.txt'
	new_path_test = './temp_test.txt'
	shutil.copy(original_path_test,new_path_test)

#process validation file
def process_validation(new_path):
	with open('validation_roman.txt','w') as outfile_roman,open('validation_deva.txt','w') as outfile_deva:
		with open('temp_validation.txt','r') as infile:
			con=infile.readlines()
			sentences=[x.strip('\n') for x in con]
			for i in sentences:
				if i!='':
					words=i.split()
					for j in words:
						word_tag=j.split('\\')
						word_roman=word_tag[0]
						word_deva=word_tag[0]
						tag=word_tag[1][0]
						if tag=='E':
							tag='EN'
						elif tag=='H':
							tag='HI'
							word_deva=word_tag[1][2:]
						else:
							tag='OTHER'
						outfile_roman.write(word_roman+'\t'+tag+'\n')
						outfile_deva.write(word_deva+'\t'+tag+'\n')
					outfile_roman.write('\n')
					outfile_deva.write('\n')

	shutil.copy('validation_roman.txt',new_path +'/Romanized/validation.txt')
	shutil.copy('validation_deva.txt',new_path + '/Devanagari/validation.txt')
	os.unlink('validation_roman.txt')
	os.unlink('validation_deva.txt')
	os.unlink('temp_validation.txt')

#process test file
def process_test(new_path,trans_pairs):
	with open('test_roman.txt','w') as outfile:
		with open('temp_test.txt','r') as infile:
			con=infile.readlines()
			sentences=[x.strip('\n') for x in con]
			for i in sentences:
				if i!='':
					words=i.split()
					for j in words:
						word_tag=j.split('\\')
						word=word_tag[0]
						tag=word_tag[1]
						if tag=='en':
							tag='EN'
						elif tag=='hi':
							tag='HI'
						else:
							tag='OTHER'
						if word!='':
							outfile.write(word+'\t'+'OTHER'+'\n')
					outfile.write('\n')

	shutil.copy('test_roman.txt',new_path +'/Romanized/test.txt')
	# do_transliterate('test_roman.txt',new_path +'/Devanagari/test.txt')
	os.unlink('test_roman.txt')

	if len(trans_pairs.keys())>0:
		with open('test_deva.txt','w') as outfile:
			with open('temp_test.txt','r') as infile:
				con=infile.readlines()
				sentences=[x.strip('\n') for x in con]
				for i in sentences:
					if i!='':
						words=i.split()
						for j in words:
							word_tag=j.split('\\')
							word=word_tag[0]
							tag=word_tag[1]
							if tag=='en':
								tag='EN'
								new_word=word
							elif tag=='hi':
								new_word=trans_pairs.get(word)
								tag='HI'
							else:
								tag='OTHER'
								new_word=word
							outfile.write(new_word+'\t'+'OTHER'+'\n')
						outfile.write('\n')

		shutil.copy('test_deva.txt',new_path +'/Devanagari/test.txt')
		os.unlink('test_deva.txt')
	os.unlink('temp_test.txt')

# process train file
def process_train(original_path,new_path):
	
	with open(original_path+'/ICON_POS/Processed Data/Romanized/train.txt','r') as infile:
		con=infile.readlines()
	sentences=[x.strip('\n') for x in con]

	with open(new_path+'/Romanized/train.txt','w') as outfile:
		for i in sentences:
			if i!='':
				j=i.split('\t')
				if j[1]=='hi':
					tag='HI'
				elif j[1]=='en':
					tag='EN'
				else:
					tag='OTHER'
				outfile.write(j[0]+'\t'+tag+'\n')
			else:
				outfile.write('\n')
	
	with open(original_path+'/ICON_POS/Processed Data/Devanagari/train.txt','r') as infile:
		con=infile.readlines()
	sentences=[x.strip('\n') for x in con]

	with open(new_path+'/Devanagari/train.txt','w') as outfile:
		for i in sentences:
			if i!='':
				j=i.split('\t')
				if j[1]=='hi':
					tag='HI'
				elif j[1]=='en':
					tag='EN'
				else:
					tag='OTHER'
				outfile.write(j[0]+'\t'+tag+'\n')
			else:
				outfile.write('\n')

def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	parser.add_argument("--output_dir", default="en", type=str, help="Processed data directory")
	args = parser.parse_args()

	# setting paths
	original_path = args.data_dir+'/LID_EN_HI/temp/'
	new_path = args.output_dir +'/LID_EN_HI/'  

	if not os.path.exists(new_path):
		os.mkdir(new_path)
	if not os.path.exists(new_path+'Romanized'):
		os.mkdir(new_path+'Romanized')
	if not os.path.exists(new_path+'Devanagari'):
		os.mkdir(new_path+'Devanagari')

	trans_pairs={}
	# downloading transliterations
	if os.path.exists('transliterations.txt'):
		with open('transliterations.txt','r') as infile:
			con=infile.readlines()
		sent=[x.strip('\n') for x in con]

		for i in sent:
			if i!='':
				j=i.split('\t')
				trans_pairs.update({j[0]:j[1]})

	# call required functions to process
	make_temp_file(original_path)
	process_validation(new_path)
	process_test(new_path,trans_pairs)
	process_train(original_path,new_path)

	# append all data in one file
	open(new_path+'Romanized/all.txt', 'w+').writelines([l for l in open(new_path+'Romanized/train.txt').readlines() ])
	open(new_path+'Romanized/all.txt', 'a').writelines([l for l in open(new_path+'Romanized/test.txt').readlines() ])
	open(new_path+'Romanized/all.txt', 'a').writelines([l for l in open(new_path+'Romanized/validation.txt').readlines() ])

	# append all data in one file
	if len(trans_pairs.keys())>0:
		open(new_path+'Devanagari/all.txt', 'w+').writelines([l for l in open(new_path+'Devanagari/train.txt').readlines()])
		open(new_path+'Devanagari/all.txt', 'a').writelines([l for l in open(new_path+'Devanagari/test.txt').readlines()])
		open(new_path+'Devanagari/all.txt', 'a').writelines([l for l in open(new_path+'Devanagari/validation.txt').readlines()])

if __name__=='__main__':
	main()