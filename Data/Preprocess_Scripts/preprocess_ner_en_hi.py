# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

import csv
import os
import argparse

#reading original csv file
def make_temp_file(original_path):
	
	words=[]
	with open(original_path +'/annotatedData.csv','r',encoding='utf-8')as f:
		data = csv.reader(f, delimiter=',')
		for row in data:
			words.append(row)
	
	prev_id=words[1][0]
	with open('temp.txt','w') as outfile:
		for i in words[1:]:
			if i[0]==prev_id:
				if len(i)>2:
					outfile.write(i[0]+'\t'+i[1]+'\t'+i[2]+'\n')
			else:
				if len(i)>2:
					prev_id=i[0]
					outfile.write('\n'+i[0]+'\t'+i[1]+'\t'+i[2]+'\n')

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
				if j[0].split()[1] in ids:
					if mode=='test':
						if j[1]!='':
							outfile.write(j[1]+ '\t' + 'Other' + '\n')
					else:    
						outfile.write(j[1]+'\t'+j[2]+'\n')
					id_flag=True
			else:
				if id_flag:
					outfile.write('\n')
					id_flag=False

def make_devanagari(roman_path,deva_path,trans_pairs):
	with open(roman_path,'r') as infile:
		con=infile.readlines()
	sentences=[x.strip('\n') for x in con]

	with open(deva_path,'w') as outfile:
		for i in sentences:
			if i!='':
				j=i.split('\t')
				if j[0] in trans_pairs.keys():
					word=trans_pairs.get(j[0])
				else:
					word=j[0]

				outfile.write(word+'\t'+j[1]+'\n')
			else:
				outfile.write('\n')

def main():
	parser = argparse.ArgumentParser()

  # Required parameters
	parser.add_argument("--data_dir", default=None, type=str, required=True, help="Original data directory")
	parser.add_argument("--output_dir", default=None, type=str, required=True, help="Processed data directory")
	args = parser.parse_args()
	
	# setting paths
	original_path = args.data_dir+'/NER_EN_HI/temp/'
	new_path = args.output_dir +'/NER_EN_HI/'  
	id_dir = args.data_dir + '/NER_EN_HI/ID_Files'  

	if not os.path.exists(new_path):
		os.mkdir(new_path)
	if not os.path.exists(new_path+'Romanized'):
		os.mkdir(new_path+'Romanized')
	if not os.path.exists(new_path+'Devanagari'):
		os.mkdir(new_path+'Devanagari')

	# downloading transliterations
	trans_pairs={}
	if os.path.exists('transliterations.txt'):
		with open('transliterations.txt','r') as infile:
			con=infile.readlines()
		sent=[x.strip('\n') for x in con]

		for i in sent:
			if i!='':
				j=i.split('\t')
				trans_pairs.update({j[0]:j[1]})
	

	# make train, test and validation files
	make_temp_file(original_path)
	make_split_file(id_dir+'/train_ids.txt','temp.txt',new_path+'/Romanized/train.txt',mode='train')
	make_split_file(id_dir+'/test_ids.txt','temp.txt',new_path+'/Romanized/test.txt',mode='test')

	to_remove = [188]
	with open(new_path + '/Romanized/test.txt') as f:
		lines = f.read().strip().split('\n\n')
	lines_out = []
	for i in range(len(lines)):
		if i not in to_remove:
			lines_out.append(lines[i])
	with open(new_path + '/Romanized/test.txt', "w") as f:
		f.write("\n\n".join(lines_out) + "\n")

	make_split_file(id_dir+'/validation_ids.txt','temp.txt',new_path+'/Romanized/validation.txt',mode='valid')

	if len(trans_pairs.keys())>0:
		make_devanagari(new_path+'/Romanized/train.txt',new_path+'/Devanagari/train.txt',trans_pairs)
		make_devanagari(new_path+'/Romanized/test.txt',new_path+'/Devanagari/test.txt',trans_pairs)
		make_devanagari(new_path+'/Romanized/validation.txt',new_path+'/Devanagari/validation.txt',trans_pairs)

		# append all data in one file
		open(new_path+'Devanagari/all.txt', 'w+').writelines([l for l in open(new_path+'Devanagari/train.txt').readlines()])
		open(new_path+'Devanagari/all.txt', 'a').writelines([l for l in open(new_path+'Devanagari/test.txt').readlines()])
		open(new_path+'Devanagari/all.txt', 'a').writelines([l for l in open(new_path+'Devanagari/validation.txt').readlines()])
	
	
	# append all data in one file
	open(new_path+'Romanized/all.txt', 'w+').writelines([l for l in open(new_path+'Romanized/train.txt').readlines() ])
	open(new_path+'Romanized/all.txt', 'a').writelines([l for l in open(new_path+'Romanized/test.txt').readlines() ])
	open(new_path+'Romanized/all.txt', 'a').writelines([l for l in open(new_path+'Romanized/validation.txt').readlines() ])

	# delete temp files
	os.unlink('temp.txt')

if __name__=='__main__':
	main()