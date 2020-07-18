# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import tweepy, sys, os, re, time, math
import operator
import unidecode
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
import argparse

def download_tweets(tweet_keys,original_path):
	original_path_text = original_path +'/cs-en-es-corpus-wassa2015.txt'
	lines = [line.strip() for line in open(original_path_text,'r').readlines()]

	tweet_ids = []
	with open('tweet_ids.txt','w') as f:
		for line in lines:
			if (not line.startswith('#')):
				f.write(line.split('\t')[0]+'\n')


	# downloading tweets from tweet IDs file
	auth = tweepy.OAuthHandler(tweet_keys.get("consumer_key"),tweet_keys.get("consumer_secret"))
	auth.set_access_token(tweet_keys.get("access_token"),tweet_keys.get("access_secret"))
	api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

	inpfile = 'tweet_ids.txt'
	outfile = 'downloaded_tweets.txt'
	tweet_ids = [line.rstrip() for line in open(inpfile,'r').readlines()]
	count = 0
	f = open(outfile,'w')
	start = time.time()

	for i in range(math.ceil(float(len(tweet_ids))/100)):
		tweets = api.statuses_lookup(id_=tweet_ids[100*i:min(100*(i+1),len(tweet_ids))])
		for tweet in tweets:
			f.write('TID: '+str(tweet.id)+'\n')
			try:
				f.write(tweet.retweeted_tweet.extended_tweet['full_text'].replace('\n', ' ')+'\n')
			except AttributeError:
				try:
					f.write(tweet.retweeted_tweet.text.replace('\n', ' ')+'\n')
				except AttributeError:
					try:
						f.write(tweet.extended_tweet['full_text'].replace('\n', ' ')+'\n')
					except AttributeError:
						f.write(tweet.text.replace('\n', ' ')+'\n')
		count += 100
		print('%.2f' %(float(count)/len(tweet_ids)))
		print('%.1f' % (time.time()-start))

	f.close()

# prepare sentiment annotated file
def annotate_sentiment(original_path):
	tweets = {}
	sentiments = {}
	tweet_ids = {}

	original_path_text = original_path+'/cs-en-es-corpus-wassa2015.txt'

	with open('downloaded_tweets.txt', 'r') as rf:
		lines = [line.strip() for line in rf.readlines()]
		for i in range(0,len(lines),2):
			tweets[lines[i].split(' ')[1]] = lines[i+1]
			tweet_ids[lines[i].split(' ')[1]] = (lines[i].split())[1]
	with open(original_path_text, 'r') as rf:
		lines = [line.strip() for line in rf.readlines()]
		for line in lines:
			if not line.startswith('#'):
				sentiments[line.split('\t')[0]] = line.split('\t')[1]

	with open('sentiment_annotated.txt','w') as of:
		for tid in tweets:
			of.write(tweet_ids[tid]+'\t'+tweets[tid]+'\t'+sentiments[tid]+'\n')

# clean tweets
def expand_word_forms(sent):
	csent = ''
	for word in sent.split(' '):
		if (word.endswith("'s")):
			csent += ' '
			csent += word.replace("'s", "s")	
		elif (word.endswith("'re")):
			csent += ' '
			csent += word.replace("'re", ' are')
		elif (word.endswith("'m")):
			csent += ' '
			csent += word.replace("'m", ' am')
		elif (word.endswith("n't")):
			csent += ' '
			csent += word.replace("n't", ' not')
		elif (word.endswith("'ll")):
			csent += ' '
			csent += word.replace("'ll", ' will')
		elif (word.endswith("'ve")):
			csent += ' '
			csent += word.replace("'ve", ' have')
		else:
			csent += ' '
			csent += word
	return csent.strip()

def clean_tweet(tweet):
	# cleaning the tweet after spliting the tweet using Tweet Tokenizer
	tknzr = TweetTokenizer()
	return expand_word_forms(unidecode.unidecode(' '.join([token.lstrip('#') for token in tknzr.tokenize(tweet) if (not token.startswith('http')) and (not token.startswith('@')) and (token.lower() != 'rt')])))

def process_tweets():
	sent = ""
	sentences=[]
	sent_tagged={}
	sent_ids={}
	new_sent = ""
	input_file = 'sentiment_annotated.txt'
	with open(input_file, 'r') as rf:
		content = rf.readline()
		con = content.strip().split('\t')
		while content:
			if len(con)>1:
				sent = clean_tweet(con[1])
				for token in sent.split(' '):
					if (token != ''):
						new_sent = new_sent + token.lower() + " "
				label = con[2]
				if label == 'P':
					new_label = 'positive'
				elif label =='N':
					new_label = 'negative'
				else:
					new_label = 'neutral'
				sent_tagged.update({new_sent.strip():new_label})
				sent_ids.update({new_sent.strip():con[0]})
				sentences.append(new_sent.strip())
			new_sent=""
			content = rf.readline()
			con = content.strip().split('\t')

	with open('cleanTweets.txt','w+') as outfile:
		for i in sentences:
			outfile.write(sent_ids.get(i) + "\t" + i +"\t" + sent_tagged.get(i)+"\n")

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
					temp_dict.update({j[0]:j[1]+ '\t' + 'neutral' + '\n'})
				else:   
					temp_dict.update({j[0]:j[1]+ '\t' + j[2] + '\n'})

	
	with open(output_file,'w') as outfile:
		for i in ids:
			if i in temp_dict.keys():
				outfile.write(temp_dict.get(i))
			else:
				if mode=='test':
					outfile.write('not found' + '\t' + 'neutral' + '\n')


def main():
	parser = argparse.ArgumentParser()

   # Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	parser.add_argument("--output_dir", default=None, type=str, required=True, help="Processed data directory")
	args = parser.parse_args()

	# setting paths
	original_path = args.data_dir + '/Sentiment_EN_ES/temp/'
	new_path = args.output_dir +'/Sentiment_EN_ES/'  
	id_dir = args.data_dir + '/Sentiment_EN_ES/ID_Files'  

	if not os.path.exists(new_path):
		os.mkdir(new_path)

	# get twitter keys
	with open('twitter_authentication.txt','r') as infile:
		con=infile.readlines()
	sent=[x.strip('\n') for x in con]

	tweet_keys={"consumer_key":sent[0],"consumer_secret":sent[1],"access_token":sent[2],"access_secret":sent[3]}
	
	# process tweets
	download_tweets(tweet_keys,original_path)
	annotate_sentiment(original_path)
	process_tweets()

	# make train, test and validation files
	make_split_file(id_dir+'/train_ids.txt','cleanTweets.txt',new_path+'/train.txt',mode='train')
	make_split_file(id_dir+'/test_ids.txt','cleanTweets.txt',new_path+'/test.txt',mode='test')
	make_split_file(id_dir+'/validation_ids.txt','cleanTweets.txt',new_path+'/validation.txt',mode='valid')
	
	# append all data in one file
	open(new_path+'/all.txt', 'w+').writelines([l for l in open(new_path+'/train.txt').readlines()])
	open(new_path+'/all.txt', 'a').writelines([l for l in open(new_path+'/test.txt').readlines()])
	open(new_path+'/all.txt', 'a').writelines([l for l in open(new_path+'/validation.txt').readlines()])
	
	# delete temp files
	os.unlink('cleanTweets.txt')
	os.unlink('tweet_ids.txt')
	os.unlink('downloaded_tweets.txt')
	os.unlink('sentiment_annotated.txt')

if __name__=='__main__':
	main()