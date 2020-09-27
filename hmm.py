from sklearn.model_selection import train_test_split
#preprocessing
file = open('CS563-NER-Dataset-10Types.txt','r')
outfile = open('input_file_hmm.txt','w')
lines = file.readlines()
for line in lines :
	word=line.split("\t")
	#print(word[0],word[1])
	if line.strip() !='':
		word[1]=word[1].replace("\n","")
		string=word[0]+"^"+word[1]+" "
		outfile.write(string)
	elif line.strip() =='':
		string="\n"
		outfile.write(string)

resultfile = open('output_file_hmm.txt','w')
train_data = open('input_file_hmm.txt','r')
train_data1 = train_data.readlines()
train, test = train_test_split(train_data1, train_size = 0.7,test_size = 0.3)

tag_pair_count = {}
tag_pair_prob = {}
transition = {}
emission = {}
start= {}
word_tag_pair= {}
tag_count = {}

def find_states(train):
	states = []
	for line in train:
		line = line.strip("\n")
		words = line.split(" ")
		for i in range(0,len(words)-1):
			tag = words[i].split("^")
			if tag[1] not in states:
				states.append(tag[1])
	return states

def transition_prob(data):
	for key, value in data.iteritems():
		if key[0] not in tag_count:
			tag_count[key[0]] = value 
		else:
			tag_count[key[0]] = tag_count[key[0]]+value
	for key,value in data.iteritems():
		if key not in tag_pair_prob:
			tag_pair_prob[key]=float(value)/float((tag_count[key[0]]))
	return tag_pair_prob

def transition_count(train):
	for line in train:
		list_ = []
		line = line.strip("\n")
		words = line.split(" ")
		for i in range(0,len(words)-1):
			i_split = words[i].split("^")
			list_.append(i_split[1])
		list_.append("^")
		for j in range(0,len(list_)-1):
			if(list_[j],list_[j+1]) not in tag_pair_count:
				tag_pair_count[list_[j],list_[j+1]]=1
			else:
				tag_pair_count[list_[j],list_[j+1]]=tag_pair_count[list_[j],list_[j+1]]+1
	return tag_pair_count

def prob_miss(data,list1,list2):
	for i in range(len(list1)):
		for j in range(len(list2)):
			if (list1[i],list2[j]) not in data:
				transition[list1[i],list2[j]]=float(0.000001)
			else:
				transition[list1[i],list2[j]]=float(data[list1[i],list2[j]])
	return transition

def calculate_transition():
	transition_count(train)
	transition_prob(tag_pair_count)
	states=find_states(train_data1)
	transition=prob_miss(tag_pair_prob,states,states)

def word_tag_count(train):
	for line in train:
		line = line.strip("\n")
		words = line.split(" ")
		for i in range(0,len(words)-1):
			i_split = words[i].split("^")
			if(i_split[0],i_split[1]) not in word_tag_pair:
				word_tag_pair[i_split[0],i_split[1]]=1
			else:
				word_tag_pair[i_split[0],i_split[1]]=word_tag_pair[i_split[0],i_split[1]]+1
	return word_tag_pair

def word_tag_prob(data):
	for key,value in data.iteritems():
		word_tag_pair[key]=float(value)/float(tag_count[key[1]])
	return word_tag_pair

def unique_words(data):
	list1=[]
	for key,value in data.iteritems():
		if key[0] not in list1:
			list1.append(key[0])
	return list1

def emission_table(word_tag_pair):
	list2=find_states(train_data1)
	list1=unique_words(word_tag_pair)
	for i in range(len(list1)):
		for j in range(len(list2)):
			if(list1[i],list2[j]) not in word_tag_pair:
				emission[list1[i],list2[j]]=float(0.00001)
			else:
				emission[list1[i],list2[j]]=word_tag_pair[list1[i],list2[j]]
	return emission

def calculate_emission():
	word_tag_pair=word_tag_count(train)
	word_tag_pair1=word_tag_prob(word_tag_pair)
	emission_table(word_tag_pair1)

def calculate_start_prob(train,states):
	count =0
	for line in train:
		count = count+1
		line = line.strip("\n")
		words = line.split(" ")
		i_split = words[0].split("^")
		if (i_split[1]) not in start:
			start[i_split[1]]=1
		else:
			start[i_split[1]]=start[i_split[1]]+1
	for key,value in start.iteritems():
		start[key]=float(value)/float(count)
	for i in range(len(states)):
		if (states[i]) not in start:
			start[states[i]]=0.000001
	return start

def roundoff_prob(transition,emission):
	for key,value in transition.iteritems():
		transition[key]=round(value,7)
	for key,value in emission.iteritems():
		emission[key] = round(value,7)
	return transition,emission

def file_result(words,tags,result):
	for i in range(len(words)):
		string=words[i]+"\t"+tags[i]+"\t"+result[i]
		resultfile.write(string)
		resultfile.write("\n")

def seperate_tag_word(line):
	wordlist=[]
	taglist=[]
	line = line.strip("\n")
	words = line.split(" ")
	for i in range(0,len(words)-1):
			i_split = words[i].split("^")
			wordlist.append(i_split[0])
			taglist.append(i_split[1])
	return wordlist,taglist

def emission_prob(emission,word,tag):
	# print emission
	if(word,tag) not in emission:
		return (float(0.00001))
	else:
		if(emission[word,tag])>float(0.0):
			return round(emission[word,tag],7)
		else:
			return float(0.00001)

def viterbi(test,states,emission,transition,start):
	# print transition
	viterbi = [{}]
	count =0
	for line in test:
		viterbi = [{}]
		count = count + 1
		words,tags=seperate_tag_word(line)
#		print("reached")
		for st in states:
			value =float(emission_prob(emission,words[0],st))
			viterbi[0][st] = {"prob": start[st] * value, "prev": None}
		for t in range(1, len(words)):
			viterbi.append({})
			for state in states:
				max_tr_prob = max(viterbi[t-1][prev_st]["prob"]*transition[prev_st,state] for prev_st in states)
				for prev_st in states:
					if viterbi[t-1][prev_st]["prob"] * transition[prev_st,state] == max_tr_prob:
						value =float(emission_prob(emission,words[0],state))
						max_prob = max_tr_prob * value
						viterbi[t][state] = {"prob": max_prob, "prev": prev_st}
						break

		result = []
		max_prob = max(value["prob"] for value in viterbi[-1].values())
		previous = None

		for state, data in viterbi[-1].items():
			if data["prob"] == max_prob:
				result.append(state)
				previous = state
				break
		for t in range(len(viterbi) - 2, -1, -1):
			result.insert(0, viterbi[t + 1][previous]["prev"])
			previous = viterbi[t + 1][previous]["prev"]
		file_result(words,tags,result)

if __name__ == '__main__':
	states=find_states(train_data1)
	calculate_transition()
	calculate_emission()
	transition,emission=roundoff_prob(transition,emission)
	start=calculate_start_prob(train,states)	
	viterbi(test,states,emission,transition,start)