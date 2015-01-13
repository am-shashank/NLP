#!/usr/bin/python
# -*- coding: latin-1 -*-
import os
import xml.etree.ElementTree as eTree
import re
from collections import Counter
import itertools
import math
import re
import fileinput
from os.path import basename
import subprocess
import shutil
import fileinput
from liblinearutil import *
import pickle
import operator
from nltk import sent_tokenize, word_tokenize
from random import shuffle
import re, subprocess, itertools
from nltk import FreqDist, sent_tokenize, word_tokenize
import numpy as np
import re, subprocess, itertools
from tree import TreeParser
from liblinearutil import train, problem, svm_read_problem, predict
from nltk.corpus import stopwords
import string
#function to serialize objects
def save_object(obj, filename):
     filehandler = open(filename, 'wb')
     pickle.dump(obj, filehandler, pickle.HIGHEST_PROTOCOL)

#function to deserialize objects
def load_object(filename):
        file_handler = open(filename,'rb')
        object_file = pickle.load(file_handler)
        file_handler.close()
        return object_file

#list files with absolute paths recursively within directory
def get_all_files(directory):
    file_list = []
    for root,dirnames, filenames in os.walk(directory):
        for filename in filenames:
            abs_path = os.path.join(root, filename)
            file_list.append(abs_path)
    return file_list


#preprocess list of files and output to directory. The list of files is specified in the raw_text_file
def preprocess(raw_text_file, corenlp_output):
    ret_code = os.system("java -cp /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-09.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-06-models.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/xom.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -filelist "+raw_text_file+" -outputDirectory "+ corenlp_output)
    if(ret_code!=0):
        raise Exception("Error creating XML files using StanfordCoreNLP")

def load_file_sentences(filename):
    fullstring = open(filename, 'r').read()        
    return [sen.lower() for sen in sent_tokenize(fullstring)] 


def load_file_tokens(filename):
    sentences = load_file_sentences(filename);
    toks = [];
    delimiter = re.compile("[^0-9A-Za-z]+");
    for s in sentences: 
        toks.extend(delimiter.split(s.strip()))
    return [token.lower() for token in toks if len(token) > 0 ]


def load_collection_tokens(directory):
        directory = directory + '/'
        files = get_all_files(directory)
        tokens = []
	ctr = 0
        for relative_path in files:
                path = directory + relative_path
#               print path
                tokens_in_current_file = load_file_tokens(path)
		if len(tokens_in_current_file) < 50:
			ctr += 1
              #  tokens.extend(tokens_in_current_file)
#       TODO: Same as above TODO        
        print ctr 


stops = set(stopwords.words("english"))
## LEXICAL FEATURES: 
def extract_single_file_words(xml_file):
    words = [];
    with open(xml_file) as f:
        all_lines = f.readlines();
        for line in all_lines:
            match_obj = WORD_PATTERN.search(line.strip());
            if match_obj:
                token = match_obj.group(1).strip().lower();
		#do tokenization, remove stop words
		#if token not in stops and token not in string.punctuation:
                words.append(token);
    return words;

def extract_single_file_NER(xml_file):
	ners = [];
	with open(xml_file) as f:
		all_lines = f.readlines();
		for line in all_lines:
			match_obj = NER_PATTERN.search(line.strip());
			if match_obj:
				token = match_obj.group(1).strip();
				if(token != 'O'):
					ners.append(token);
	f.close()
	return ners;

def extract_top_NER(file_list):
	all_ners = []
	for xml_file in file_list:
		file_ners = extract_single_file_NER(xml_file)
		all_ners = all_ners + file_ners
	top_ners = sorted( FreqDist(all_ners).items(), key=operator.itemgetter(1), reverse=True )[:100]
	return [ner for ner,score in top_ners ]


def map_ners(file_name, top_ners):
	ner_vector = []
	file_ners = extract_single_file_NER(file_name)
	ner_freq = Counter(file_ners)
	count = 0
	with open(file_name) as f:
        	all_lines = f.readlines();
		for line in all_lines:
			match_obj = WORD_PATTERN.search(line.strip());
			if match_obj:
				count = count + 1
	print count
	for ner in top_ners:
		if ner in ner_freq:
			ner_vector.append(float(ner_freq[ner])/float(count))
	
	return ner_vector

def extract_top_words(files):
    all_toks = [];
    count = 0;
    for f in files:
        toks = extract_single_file_words(f);
        all_toks.extend( toks );
        count = count +1
    top_words =  sorted( FreqDist(all_toks).items(), key=operator.itemgetter(1), reverse=True )[:10000];
    return [word for word,score in top_words];

def map_unigrams(filename, top_words):
    file_toks = extract_single_file_words(filename);
    freq = FreqDist(file_toks)
    return [int(freq[w]>0) for w in top_words ]

# DEPENDENCIES : 
#extract dependencies from an xml file within the <dep> tag

def extract_basic_dependencies(xml_file,dep_freq):
     tree = eTree.parse(xml_file)
     root = tree.getroot()
     for token in root.iter('dep'):
        type = token.attrib['type']
        gov = token.find('governor').text.lower()
        dep = token.findd('dependent').text.lower()
        if (type,gov,dep) in dep_freq:
                dep_freq[(type,gov,dep)] += 1
        else:
                dep_freq[(type,gov,dep)] = 1
     return dep_freq

#get the top freqeunt dependencies from the xml_directory
def extract_top_dependencies(xml_files):
     top_dep =  [];
     dep_freq = {}
     for xml_file in xml_files:
     	tree = eTree.parse(xml_file)
        root = tree.getroot()
   	for token in root.iter('dep'):
     	  	type = token.attrib['type']
        	gov = token.find('governor').text.lower()
        	dep = token.find('dependent').text.lower()
        	if (type,gov,dep) in dep_freq:
        	        dep_freq[(type,gov,dep)] += 1
        	else:
     			top_dep = sorted(dep_freq, key=dep_freq.get,reverse=True)[:10000]
     return top_dep;
 
def map_dependencies(xml_file, dep_list):
    vec = []
    dep_file = []
    tree = eTree.parse(xml_file)
    root = tree.getroot()
    #get the dependencies from the xml file
    for token in root.iter('dep'):
        gov = token.find('governor').text.lower()
        dep = token.find('dependent').text.lower()
	dep_file.append((type,gov,dep))
    for top_dep in dep_list:
        if top_dep in dep_file:
                vec.append(1)
        else:
                vec.append(0)
    return vec;

## SYNTACTIC PRODUCTION RULES 
def extract_single_file_prod_rules(xml_file):
    prod_rules = [];
    with open(xml_file) as f:
        all_lines = f.readlines();
        for line in all_lines:
            match_obj = PARSE_TREE_PATTERN.search(line.strip());
            if match_obj:
                parse_tree= match_obj.group(1).strip();
                tree = TreeParser(parse_tree).tree;
                prod_rules.extend(tree.getProdRules());
    return prod_rules;

def extract_prod_rules(files):
    token_list = [];
    for f in files:
        token_list.extend(extract_single_file_prod_rules(f));
    top_prod =  sorted( FreqDist(token_list).items(), key=operator.itemgetter(1), reverse=True );
    top_prod = top_prod[:10000];
    return [ rule for rule,score in top_prod];
    

def map_prod_rules(xml_file, rule_list):
    onefile_list = extract_single_file_prod_rules(xml_file);
    return [int(rule in onefile_list) for rule in rule_list];


def extract_mrc_db():
	f = open("/project/cis/nlp/tools/MRC/MRC_parsed/MRC_words",'r')
	mrc_words = [line.rstrip('\n') for line in f]
	return sorted(mrc_words)

def map_mrc_db(xml_file,mrc_words):
	lead_words = extract_single_file_words(xml_file)
	mrc_vec = []
	for word in mrc_words:
		mrc_vec.append(lead_words.count(word)/(len(lead_words)*1.0))
	return mrc_vec

#read the word and corresponding scores to a dictionary
def read_to_dict(filename):
	dic = {}
	lines = open(filename,"r").readlines()
        for line in lines:
                word = line.split()[0]
                value = line.split()[1]
                dic[word] = int(value)
	return dic

def vectorize_word_score(di,lead_words):
	vec = 230*[0]
	#calculate range
	min_word = min(di, key=di.get)
	min_score = di[min_word]
	max_word = max(di, key=di.get)
	max_score = di[max_word]
	r = max_score - min_score
	
	for lead_word in lead_words:
		if lead_word in di:
			score = di[lead_word]
			interval = int(math.ceil((score - min_score)/(r*1.0)))
			vec[interval] += 1
	l = len(lead_words)
	for i in range(0,len(vec)):
		vec[i] =float( vec[i])/l
	return vec
	
def map_word_score(xml_file):
	lead_words = extract_single_file_words(xml_file)
	imag = {}
        fam = {}
        conc = {}
        aoa = {}
        meanc={}
        imag = read_to_dict("/project/cis/nlp/tools/MRC/MRC_parsed/IMAG")
        fam=  read_to_dict("/project/cis/nlp/tools/MRC/MRC_parsed/FAM")
        conc =  read_to_dict("/project/cis/nlp/tools/MRC/MRC_parsed/CONC")
        aoa =  read_to_dict("/project/cis/nlp/tools/MRC/MRC_parsed/AOA")
        meanc =  read_to_dict("/project/cis/nlp/tools/MRC/MRC_parsed/MEANC")
 	imag_vec = vectorize_word_score(imag, lead_words)
	fam_vec = vectorize_word_score(fam, lead_words)
	conc_vec = vectorize_word_score(conc, lead_words) 
	aoa_vec = vectorize_word_score(aoa, lead_words) 
	meanc_vec = vectorize_word_score(meanc, lead_words) 			
	return imag_vec + fam_vec + conc_vec + aoa_vec + meanc_vec


def get_mi_weights(bg_corpus, topic_corpus): 
    bg_dict = FreqDist(bg_corpus)
    bg_item_ratio = dict([(w, (bg_dict[w]+0.0)/(len(bg_corpus) + 0.0)) for w in bg_dict])
    topic_dict = FreqDist(topic_corpus);
    topic_item_ratio = dict([(w, (topic_dict[w]+0.0)/(len(topic_corpus) + 0.0)) for w in topic_dict])
    keyitems = [w for w in bg_dict if ((bg_dict[w] >= 5)and(topic_dict.has_key(w)))]
    mi_weight = dict([(w, math.log(topic_item_ratio[w]/(bg_item_ratio[w]))) for w in keyitems])
    return sorted(mi_weight.iteritems(), key=operator.itemgetter(1), reverse=True)

def get_mi_top(bg_corpus, topic_corpus, K):
    ## bg_corpus and topic_corpus are lists of words
    sorted_mi = get_mi_weights(bg_corpus, topic_corpus)
    return [x for x,_ in sorted_mi[:K]]

def mi_feature():
	f=open("/home1/c/cis530/project/train_labels.txt","r")
        lines=f.readlines()
        input_data={}
        for line in lines:
                temp_list=line.split()
                input_data[temp_list[0]]=int(temp_list[1])
	files=get_all_files("/home1/c/cis530/project/train_data")
	path="/home1/c/cis530/project/train_data/"
	topic_words1=[]
	topic_words2=[]
	for f in files:
		f1 = os.path.basename(f)
		if input_data[f1]==1:
			topic_words1=topic_words1+load_file_tokens(f)
		elif input_data[f1]==-1:
			topic_words2=topic_words2+load_file_tokens(f)	
	mi=get_mi_top(topic_words1+topic_words2, topic_words1, 500)
	mi=mi+get_mi_top(topic_words1+topic_words2, topic_words2, 500)
	return mi



def map_mi(filename, mi, flag=False):
	if flag==False:
		path="/home1/c/cis530/project/train_data/"
	elif flag==True:
		path="/home1/c/cis530/project/test_data/"
	x=set(load_file_tokens(filename))
	feature=[]
	for item in mi:
		if item in x:
			feature.append(1)
		else:
			feature.append(0)
	return feature

#return a string representation index:non_zero_value for all values in input vector
def non_zero_string(vec):
        vec_str = ""
        for i in range(0,len(vec)):
                if vec[i]!=0:
                        vec_str += str(i+1)+":"+str(vec[i])+" "
        return vec_str



def process_corpus( xml_files, top_words, top_dependencies, syntactic_prod_rules, mrc_words, top_mi, file_label, flag="train" ) :
#1 binary lexical
#3 binary dependency relations
#4 binary syntactic production rules
#5 All features
#6 MRC Words
#7 MRC Concreteness
#8 MI
    #file descriptors to write results
    bin_lex_file = ""
    lex_expanded_file = ""
    bin_dep_file = ""
    bin_syn_file = ""
    all_file = ""
    if flag == "test":
        bin_lex_file = open("test_1.txt","w")
        bin_dep_file = open("test_3.txt","w")
        bin_syn_file = open("test_4.txt","w")
        all_file = open("test_5.txt","w")
	mrc_file = open("test_6.txt","w")
	mrc_prod = open("test_7.txt","w")
    else:
        bin_lex_file = open("train_1.txt","w")
        bin_dep_file = open("train_3.txt","w")
        bin_syn_file = open("train_4.txt","w")
        all_file = open("train_5.txt","w")
	mrc_file = open("train_6.txt","w")
	mrc_prod = open("train_7.txt","w")

    for xml_file in xml_files:
		file_name = os.path.basename(xml_file)
                bin_lex_vec = map_unigrams(xml_file,top_words)
                vec_str1 = file_label[file_name]+" "+non_zero_string(bin_lex_vec)
                bin_lex_file.write(vec_str1+"\n")

                bin_dep_vec = map_dependencies(xml_file,top_dependencies)
                vec_str3 = file_label[file_name]+" "+non_zero_string(bin_dep_vec)
                bin_dep_file.write(vec_str3+"\n")

                bin_syn_vec = map_prod_rules(xml_file,syntactic_prod_rules)
                vec_str4 = file_label[file_name]+" "+non_zero_string(bin_syn_vec)
                bin_syn_file.write(vec_str4+"\n")
		
		mrc_vec = map_mrc_db(xml_file,mrc_words)
                vec_str6 = file_label[file_name]+" "+non_zero_string(mrc_vec)
                mrc_file.write(vec_str6+"\n")

		word_score_vec = map_word_score(xml_file)
		vec_str7 = file_label[file_name]+" "+non_zero_string(word_score_vec) 
		mrc_prod.write(vec_str7+"\n")
		
		mi_vec = []
		if flag=="test":
			mi_vec = map_mi(xml_file, top_mi, True)
		else:
			mi_vec = map_mi(xml_file, top_mi)
					
                vec_str5 = file_label[file_name]+" "+non_zero_string(bin_lex_vec + bin_dep_vec + bin_syn_vec + mrc_vec + word_score_vec + mi_vec)
                all_file.write(vec_str5+"\n")


    return 0;


#TO DO make sure the test data is pre-processed
def process_corpus_test_data( xml_files, top_words, top_dependencies, syntactic_prod_rules, mrc_words, top_mi, file_label, flag="train" ) :
#1 binary lexical
#2 lexical with expansion
#3 binary dependency relations
#4 binary syntactic production rules
#5 All except expanded lexical features (1+3+4)
#6 MRC Words
#7 MRC Conc
#8 MI

    #file descriptors to write results
    bin_lex_file = ""
    lex_expanded_file = ""
    bin_dep_file = ""
    bin_syn_file = ""
    all_file = ""
    mrc_file = ""
    mrc_prod = ""
    if flag == "test":
	bin_lex_file = open("test_1.txt","w")
        bin_dep_file = open("test_3.txt","w")
        bin_syn_file = open("test_4.txt","w")
        all_file = open("test_5.txt","w")
	mrc_file = open("test_6.txt","w")
	mrc_prod = open("test_7.txt","w")
        for xml_file in xml_files:
		file_name = os.path.basename(xml_file)
                
		bin_lex_vec = map_unigrams(xml_file,top_words)
                vec_str1 = "1 "+non_zero_string(bin_lex_vec)
                bin_lex_file.write(vec_str1+"\n")

                bin_dep_vec = map_dependencies(xml_file,top_dependencies)
                vec_str3 = "1 "+non_zero_string(bin_dep_vec)
                bin_dep_file.write(vec_str3+"\n")

                bin_syn_vec = map_prod_rules(xml_file,syntactic_prod_rules)
                vec_str4 = "1 "+non_zero_string(bin_syn_vec)
                bin_syn_file.write(vec_str4+"\n")

                mrc_vec = map_mrc_db(xml_file,mrc_words)
                vec_str6 = "1 "+non_zero_string(mrc_vec)
                mrc_file.write(vec_str6+"\n")

                word_score_vec = map_word_score(xml_file)
                vec_str7 = "1 "+non_zero_string(word_score_vec)
                mrc_prod.write(vec_str7+"\n")

                mi_vec = []
                if flag=="test":
                        mi_vec = map_mi(xml_file, top_mi, True)
                else:
                        mi_vec = map_mi(xml_file, top_mi)

                vec_str5 = "1 "+non_zero_string(bin_lex_vec + bin_dep_vec + bin_syn_vec + mrc_vec + word_score_vec + mi_vec)
                all_file.write(vec_str5+"\n")

    else:
	bin_lex_file = open("train_1.txt","w")
        bin_dep_file = open("train_3.txt","w")
        bin_syn_file = open("train_4.txt","w")
        all_file = open("train_5.txt","w")
	mrc_file = open("train_6.txt","w")
	mrc_prod = open("train_7.txt","w")
	for xml_file in xml_files:
 		file_name = os.path.basename(xml_file)
                
		bin_lex_vec = map_unigrams(xml_file,top_words)
                vec_str1 = file_label[file_name]+" "+non_zero_string(bin_lex_vec)
                bin_lex_file.write(vec_str1+"\n")

                bin_dep_vec = map_dependencies(xml_file,top_dependencies)
                vec_str3 = file_label[file_name]+" "+non_zero_string(bin_dep_vec)
                bin_dep_file.write(vec_str3+"\n")

                bin_syn_vec = map_prod_rules(xml_file,syntactic_prod_rules)
                vec_str4 = file_label[file_name]+" "+non_zero_string(bin_syn_vec)
                bin_syn_file.write(vec_str4+"\n")

                mrc_vec = map_mrc_db(xml_file,mrc_words)
                vec_str6 = file_label[file_name]+" "+non_zero_string(mrc_vec)
                mrc_file.write(vec_str6+"\n")

                word_score_vec = map_word_score(xml_file)
                vec_str7 = file_label[file_name]+" "+non_zero_string(word_score_vec)
                mrc_prod.write(vec_str7+"\n")

                mi_vec = []
                if flag=="test":
                        mi_vec = map_mi(xml_file, top_mi, True)
                else:
                        mi_vec = map_mi(xml_file, top_mi)

                vec_str5 = file_label[file_name]+" "+non_zero_string(bin_lex_vec + bin_dep_vec + bin_syn_vec + mrc_vec + word_score_vec + mi_vec)
                all_file.write(vec_str5+"\n")
 
    return 0;

#read the labels and file names in dictionary 
def get_file_labels(filename):
	#dictionary containing file name and label mappings
	file_label = {}
	f = open(filename,'r')
	for line in f:
		name,label,overlap = line.split()
		file_label[name+".xml"]= label
	save_object(file_label,"file_label_obj")
	return file_label	
	
#train and predict the files files using liblinear
def run_classifier(train_file, test_file):
    output_tuple = ();
    y,x = svm_read_problem(train_file)
    #no of lines starting from 1 
    weight2 = y.count(1)/(len(y)*1.0)
    weight1 = 1.0 - weight2
    model = train(y,x,'-s 0 -w1 '+ str(weight1) +' -w-1 '+str(weight2))
    y, x = svm_read_problem(test_file)
    p_labs, p_acc, p_vals = predict(y,x, model,'-b 1')
    ctr = 0
    for i in range(len(p_labs)):
	if y[i] == p_labs[i]:
		ctr+=1
    return ctr/(len(y)*1.0)

#train and predict the files files using liblinear
def run_classifier_test_data(train_file, test_file):
    output_tuple = ();
    y,x = svm_read_problem(train_file)
    #no of lines starting from 1 
    weight2 = y.count(1)/(len(y)*1.0)
    weight1 = 1.0 - weight2
    model = train(y,x,'-s 0 -w1 '+ str(weight1) +' -w-1 '+str(weight2))
    y, x = svm_read_problem(test_file)
    p_labs, p_acc, p_vals = predict(y,x, model,'-b 1')
    return p_labs 

def write_test_results(train_set, test_set):
	f = open("test.txt","w")
	top_words = extract_top_words(train_set)
	top_prod = extract_prod_rules(train_set)
	top_dependencies = extract_top_dependencies(train_set)
	mrc_words = extract_mrc_db()
	top_mi = mi_feature()

	
	process_corpus_test_data( train_set, top_words, top_dependencies, top_prod, mrc_words, top_mi, file_labels, "train" )
	process_corpus_test_data( test_set, top_words, top_dependencies, top_prod, mrc_words, top_mi, file_labels, "test" )

#	Uncomment thes 2 lines and comment above 2 lines to run on known test data
#	process_corpus( train_set, top_words, top_dependencies, top_prod, mrc_words, top_mi, file_labels, "train" ) 
#	process_corpus( test_set, top_words, top_dependencies, top_prod, mrc_words, top_mi, file_labels, "test" ) 


	p_labs = run_classifier_test_data("train_5.txt","test_5.txt")
	i=0
	for test_file in test_set:
		fn_xml = os.path.basename(test_file)
		fn = fn_xml[:-4]		
		f.write(fn+" "+str(int(p_labs[i]))+"\n")
		i+=1
	f.close()

def get_shuffled_files(file_dir):
	file_list = get_all_files(file_dir)
	shuffle(file_list)
	i = 0
	num_files = len(file_list)
	num_files_per_group = num_files/10
	remaining_files = num_files % 10
	groups = []
	group_number = 0
	shuffle_index = 0
	while(group_number < 10):
		groups.append([])
		j = 0
		while(j < num_files_per_group):
			groups[group_number].append(file_list[shuffle_index])
			j = j + 1
			shuffle_index = shuffle_index + 1
		group_number = group_number + 1

	groups[9].append(file_list[shuffle_index])
	groups[9].append(file_list[shuffle_index + 1])
	return groups


#take k sets as train and and 10-k as test. Repeat this 10c2 times and take the average performance
def perform_kfold_validation(groups,file_labels, flag="train"):
	#select every fold except the ith fold
	sum_acc = 0
	top_words = []
	top_dependencies = []
	syntactic_prod_rules = []
	mrc_wordss = []
	for i in range(len(groups)):
		test_set = groups[i]
		train_set = []
		for j in range(len(groups)):
			if j != i:
				train_set += groups[j]
		top_words = extract_top_words(train_set)
		#print top_words
		top_dependencies = extract_top_dependencies(train_set)
		syntactic_prod_rules = extract_prod_rules(train_set)
		mrc_words = extract_mrc_db()
		top_mi = mi_feature()
		
		process_corpus( train_set, top_words, top_dependencies, syntactic_prod_rules, mrc_words, top_mi, file_labels, "train" ) 
		process_corpus( test_set, top_words, top_dependencies, syntactic_prod_rules, mrc_words, top_mi, file_labels, "test" ) 


		#process_corpus(test_set,top_prod,file_labels,"test")
		acc = run_classifier("train_5.txt","test_5.txt")		
		sum_acc += acc
	return sum_acc*100.0/len(groups)


def get_top_and_bottom_500(labels_file):
	file_names = open(labels_file, 'r')
	file_names= file_names.readlines()
	file_list = []
	for file_name in file_names:
		this_line = 'train_xml_data/' + file_name.split()[0] + '.xml'
		file_list.append(this_line)

	top_500 = file_list[0:500]
	bottom_500 = file_list[1782:2282]
	test_set = file_list[700:1200]
	return top_500 + bottom_500, test_set
		
CORENLP_PATH = '/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09';
PARSE_TREE_PATTERN = re.compile('^<parse>(.*?)</parse>$');
WORD_PATTERN = re.compile('^<word>(.*?)</word>$');
NER_PATTERN = re.compile('<NER>(.*?)</NER>$')
train_directory = '/home1/c/cis530/project/train_data'
train_xml = "train_xml_data/"
train_labels = "/home1/c/cis530/project/train_labels.txt"
test_xml = "test_xml_data/"
file_label={}
file_labels = get_file_labels(train_labels)
