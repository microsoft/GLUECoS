# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import sys
import os
from subprocess import call
import shlex
import argparse

def scrape_tweets(original_path):
	
	# get twitter authentication keys
	with open('twitter_authentication.txt','r') as infile:
		con=infile.readlines()
	twitter_keys=[x.strip('\n') for x in con]
	
	os.chdir(original_path)
	with open('crawl_tweets.py','r') as infile:
		con=infile.readlines()
	
	with open('crawl_tweets_copy.py','w') as outfile:
		for i in con:
			if i.startswith('consumer_key'):
				temp = 'consumer_key = \'{0}\' '.format(twitter_keys[0])
				outfile.write(temp+'\n')
			elif i.startswith('consumer_secret'):
				temp = 'consumer_secret = \'{0}\' '.format(twitter_keys[1])
				outfile.write(temp+'\n')
			elif i.startswith('access_key'):
				temp = 'access_key = \'{0}\' '.format(twitter_keys[2])
				outfile.write(temp+'\n')
			elif i.startswith('access_secret'):
				temp = 'access_secret = \'{0}\' '.format(twitter_keys[3])
				outfile.write(temp+'\n')
			else:
				outfile.write(i)

	#scraping tweets
	call(shlex.split('python crawl_tweets_copy.py -i tweet_ids_train.txt -a train-annot.json -o tweets_train.conll'))
	call(shlex.split('python crawl_tweets_copy.py -i tweet_ids_dev.txt -a dev-annot.json -o tweets_dev.conll'))
	call(shlex.split('python crawl_tweets_copy.py -i tweet_ids_test.txt -a test-annot.json -o tweets_test.conll'))

def make_files(original_path,new_path):

	#processing each file to get in desired format for evaluation
	with open(new_path+'/Devanagari/validation.txt','w') as f1,open(new_path+'/Romanized/validation.txt','w') as f2:
		with open(original_path+'/tweets_dev.conll','r') as infile:
			con=infile.readlines()
		sentences=[x.strip('\n') for x in con]
		for i in sentences:
			if i!='':
				j=i.split('\t')
				word_roman = j[2]
				word_deva = j[3]
				pos=j[4]
				lang=j[-2]
				if lang=='hi':
					lang='HI'
				elif lang=='en':
					lang='EN'
				else:
					lang='OTHER'
				f1.write(word_deva+'\t'+lang+'\t'+pos+'\n')
				f2.write(word_roman+'\t'+lang+'\t'+pos+'\n')
			else:
				f1.write('\n')
				f2.write('\n')

	with open(original_path+'/tweet_ids_test.txt','r') as f:
		con=f.readlines()
	test_ids=[x.strip('\n') for x in con]
	
	temp_dict_roman={}
	temp_dict_deva={}

	with open(original_path+'/tweets_test.conll','r') as infile:
		con=infile.readlines()
	sentences=[x.strip('\n') for x in con]
	for i in sentences:
		if i!='':
			j=i.split('\t')
			tweet_id=j[0]
			word_roman = j[2]
			word_deva = j[3]
			pos=j[4]
			lang=j[-2]
			if lang=='hi':
				lang='HI'
			elif lang=='en':
				lang='EN'
			else:
				lang='OTHER'
			if j[0] in temp_dict_deva.keys():
				res=temp_dict_deva.get(j[0])
				if word_deva!='':
					res += word_deva+'\t'+lang+'\t'+'NOUN'+'\n'
					temp_dict_deva.update({j[0]:res})
			else:
				if word_deva!='':
					temp_dict_deva.update({j[0]:word_deva+'\t'+lang+'\t'+'NOUN'+'\n'})
			
			if j[0] in temp_dict_roman.keys():
				res=temp_dict_roman.get(j[0])
				if word_roman!='':
					res += word_roman+'\t'+lang+'\t'+'NOUN'+'\n'
					temp_dict_roman.update({j[0]:res})
			else:
				if word_roman!='':
					temp_dict_roman.update({j[0]:word_roman+'\t'+lang+'\t'+'NOUN'+'\n'})

	with open(new_path+'/Devanagari/test.txt','w') as f1,open(new_path+'/Romanized/test.txt','w') as f2:
		for i in test_ids:
			if i in temp_dict_deva.keys():
				f1.write(temp_dict_deva.get(i)+'\n')
			else:
				f1.write('not found' + '\t' + 'OTHER' + '\t' + 'NOUN' + '\n\n')
			
			if i in temp_dict_roman.keys():
				f2.write(temp_dict_roman.get(i)+'\n')
			else:
				f2.write('not found' + '\t' + 'OTHER' + '\t' + 'NOUN' + '\n\n')

	with open(new_path+'/Devanagari/train.txt','w') as f1,open(new_path+'/Romanized/train.txt','w') as f2:
		with open(original_path+'/tweets_train.conll','r') as infile:
			con=infile.readlines()
		sentences=[x.strip('\n') for x in con]
		for i in sentences:
			if i!='':
				j=i.split('\t')
				word_roman = j[2]
				word_deva = j[3]
				pos=j[4]
				lang=j[-2]
				if lang=='hi':
					lang='HI'
				elif lang=='en':
					lang='EN'
				else:
					lang='OTHER'
				f1.write(word_deva+'\t'+lang+'\t'+pos+'\n')
				f2.write(word_roman+'\t'+lang+'\t'+pos+'\n')
			else:
				f1.write('\n')
				f2.write('\n')

def main():
	parser = argparse.ArgumentParser()

  # Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	parser.add_argument("--output_dir", default=None, type=str, required=True, help="Processed data directory")
	args = parser.parse_args()

	owd=os.getcwd()
	original_path = args.data_dir+'/POS_EN_HI_UD/temp/UD_Hindi_English-master'
	new_path = args.output_dir+'/POS_EN_HI_UD/'

	if not os.path.exists(new_path):
		os.mkdir(new_path)
	if not os.path.exists(new_path+'Romanized'):
		os.mkdir(new_path+'Romanized')
	if not os.path.exists(new_path+'Devanagari'):
		os.mkdir(new_path+'Devanagari')

	scrape_tweets(original_path)
	os.chdir(owd)
	make_files(original_path,new_path)

	# append all data in one file
	open(new_path+'Romanized/all.txt', 'w+').writelines([l for l in open(new_path+'Romanized/train.txt').readlines() ])
	open(new_path+'Romanized/all.txt', 'a').writelines([l for l in open(new_path+'Romanized/test.txt').readlines() ])
	open(new_path+'Romanized/all.txt', 'a').writelines([l for l in open(new_path+'Romanized/validation.txt').readlines() ])

	# append all data in one file
	open(new_path+'Devanagari/all.txt', 'w+').writelines([l for l in open(new_path+'Devanagari/train.txt').readlines() ])
	open(new_path+'Devanagari/all.txt', 'a').writelines([l for l in open(new_path+'Devanagari/test.txt').readlines() ])
	open(new_path+'Devanagari/all.txt', 'a').writelines([l for l in open(new_path+'Devanagari/validation.txt').readlines() ])

if __name__=="__main__":
	main()