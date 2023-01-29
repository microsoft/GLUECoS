lines = []
file_path="all.txt"
with open(file_path, 'r') as infile:
    lines = infile.read().strip().split('\n\n')
#output = input("Enter file name")
for example in lines:
    example = example.split('\n')
    words=[]
    labels=[]
    for line in example:
        if(len(line.split('\t'))>=3):
                words.append(line.split('\t')[0])
                labels.append(line.split('\t')[1])
    #words = [if(len(line.split('\t')>=3)line.split('\t')[0] for line in example]
    #labels = [line.split('\t')[1] for line in example]
    for i,j in enumerate(labels):
        if j=='en':
            labels[i]='EN'
        elif j=='hi':
            labels[i]='HI'
        else:
            labels[i]='OTHER'
    num = len(words)
    for i in range(num):
        print(words[i]+'\t'+labels[i])
    print()



