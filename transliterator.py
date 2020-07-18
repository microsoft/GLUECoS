# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import requests
import sys
import os
import glob
import operator
import argparse

# subscription_key = ''

def get_token(subscription_key):
	fetch_token_url = 'https://southeastasia.api.cognitive.microsoft.com/sts/v1.0/issueToken'
	headers = {
		'Ocp-Apim-Subscription-Key': subscription_key
	}
	response = requests.post(fetch_token_url, headers=headers)
	access_token = str(response.text)
	return access_token

def get_transliteration(vocab, headers):
	base_url = 'https://api.cognitive.microsofttranslator.com'
	path = '/transliterate?api-version=3.0&language=hi&fromScript=Latn&toScript=Deva'
	trans={}
	count=0
	body=[]
	constructed_url = base_url + path
	query=''
	while(count<=6500):
		for i in range(count,(count+500),50):
			for j in range(i,i+50):
				query += vocab[j] + ' '
			body.append({'text' : query.strip()})
			query=''
		response = requests.post(constructed_url, headers=headers, json=body)
		result = response.json()
		for j,i in enumerate(result):
			trans.update({body[j]['text']:i['text']})
		body=[]
		count += 500
	
	for i in range(count,len(vocab),50):
		for j in range(i,i+50):
			if j<len(vocab):
				query += vocab[j] + ' '
		body.append({'text' : query.strip()})
		query=''
	response = requests.post(constructed_url, headers=headers, json=body)
	result = response.json()
	for j,i in enumerate(result):
			trans.update({body[j]['text']:i['text']})
	
	return trans

def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument("--subscription_key", default=None, type=str, required=True, help="Azure Subscription key for downloading transliterations")
	parser.add_argument("--input_file", default=None, type=str, required=True,
						help="The roman hindi words vocabulary ")
	
	args = parser.parse_args()
	input_file = args.input_file
	subscription_key = args.subscription_key

	req_token = get_token(subscription_key)
	headers = { 'Accept': 'application/json;text/xml',
			'Content-Type': 'application/json',
			'Ocp-Apim-Subscription-Key': subscription_key,
			'Authorization': req_token
			}

	vocab = []
	with open(input_file,'r+') as infile:
		con = infile.readlines()
	vocab = [x.strip('\n') for x in con]

	trans = get_transliteration(vocab, headers)
	with open('transliterations.txt','w+') as outfile:
		for i in trans.keys():
			words=i.split(' ')
			deva=trans.get(i).split(' ')
			for j,k in enumerate(words):
				outfile.write(k + "\t" + deva[j] +"\n")

if __name__ == "__main__":
	main()