#!/usr/bin/python
# -*- coding: latin-1 -*-
import os
import xml.etree.ElementTree as eTree
import re
from collections import Counter
import itertools
import math
import fileinput
from os.path import basename
import subprocess
import shutil


#NOTE - PLEASE REMOVE THE OS.REMOVE AND SHUTIL.RMTREE STATEMENTS IN THE FUNCTIONS IF YOU WANT TO LOOK AT THE INTERMEDIATE RESULTS AND VERIFY INFORMATION. I HAVE WRITTEN THESE STATEMENTS TO AUTOCLEAN THE TEMPORARY FILES, DIRECTORIES THAT THE PROGRAM CREATES



#preprocess list of files and output to directory. The list of files is specified in the raw_text_file
def preprocess(raw_text_file, corenlp_output):
    ret_code = os.system("java -cp /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-09.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-06-models.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/xom.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -filelist "+raw_text_file+" -outputDirectory "+ corenlp_output)
    if(ret_code!=0):
	raise Exception("Error creating XML files using StanfordCoreNLP")

#preprocess a single file and output to directory.
def preprocess1(raw_text_file, corenlp_output):
    ret_code = os.system("java -cp /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-09.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/stanford-corenlp-2012-07-06-models.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/xom.jar:/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner -file "+raw_text_file+" -outputDirectory "+ corenlp_output)
    if(ret_code!=0):
	raise Exception("Error creating XML files using StanfordCoreNLP")

#load tokens from a file
def load_file_tokens(filepath):
        f=open(filepath,"r")
        tokens=[]
        lines=f.readlines()
        for line in lines:
                words_in_line=line.split()
                for word in words_in_line:
                        tokens.append(word)
        return tokens

#process the file and replace NAMED ENTITY TAGS from XML file
def process_file(input_xml,output_file):
    out_file = open(output_file,'w')
    tree = eTree.parse(input_xml)
    root = tree.getroot()
    
    #add STOP at the beginning of the file
    out_file.write("STOP ")

    temp = "" #keep track of last word to check if it ends with "."
    prevNER ="" #keep track of previous NER to check consecutive NERs
    for token in root.iter('token'):
	    word = token.find('word').text
	    temp = word
	    NER = token.find('NER').text
	    if(word=="."):
		out_file.write("STOP ")
	    else:
		if NER!="O" and NER!=prevNER:
		    out_file.write(NER+" ")
		else:
		    #handle special characters
		    temp_string=word
                    counter=0
                    punctuation=0
                    while counter<len(temp_string):
                        if temp_string[counter].isalnum():
                            punctuation=0
                        else:
                            punctuation=punctuation+1
			counter=counter+1
		    #remove the token if it has only punctuations
                    if punctuation==len(temp_string):
			prevNER = NER
                        continue
                    else:
                        out_file.write(word.lower()+" ")	
            prevNER = NER
	    #print "Prev NER:" +prevNER
    if temp != ".":
	out_file.write("STOP")	
    out_file.close()
 
 
 #list files with absolute paths recursively within directory
def get_all_files(directory):
    file_list = []
    for root,dirnames, filenames in os.walk(directory):
	for filename in filenames:
	    abs_path = os.path.join(root, filename)
	    file_list.append(abs_path)
    return file_list

#get collection of tokens from directory
def get_tokens_directory(directory):
	file_list = get_all_files(directory)
	tokens = []
	for raw_trainfile in file_list:
	    tokens.append(open(raw_trainfile,"r").read().split())
	#convert 2d list to 1d list
	tokens =  [word for sublist in tokens for word in sublist]
	return tokens

 #return last 100 lines of file. Similar to tail in UNIX	
def tail(f, lines=100):
    	total_lines_wanted = lines
    	BLOCK_SIZE = 1024
    	f.seek(0, 2)
    	block_end_byte = f.tell()
    	lines_to_go = total_lines_wanted
    	block_number = -1
    	blocks = [] # blocks of size BLOCK_SIZE, in reverse order starting
    	            # from the end of the file
    	while lines_to_go > 0 and block_end_byte > 0:
        	if (block_end_byte - BLOCK_SIZE > 0):
        	    # read the last block we haven't yet read
        	    f.seek(block_number*BLOCK_SIZE, 2)
        	    blocks.append(f.read(BLOCK_SIZE))
        	else:
        	    # file too small, start from begining
        	    f.seek(0,0)
        	    # only read what was not read
        	    blocks.append(f.read(block_end_byte))
        	lines_found = blocks[-1].count('\n')
        	lines_to_go -= lines_found
       		block_end_byte -= BLOCK_SIZE
       		block_number -= 1
    	all_read_text = ''.join(reversed(blocks))
    	return '\n'.join(all_read_text.splitlines()[-total_lines_wanted:]) 
	
#create unigram, bigram and trigram language models
def gen_language_models(directory,out_file,uni_mod,bi_mod,tri_mod):
	out_fd = open(out_file,"w")
	file_list = get_all_files(directory)
	for fn in file_list:
	    tokens = open(fn,"r").read().split()
	    #IGNORE first STOP
	    tokens = tokens[1:]
	    for index,item in enumerate(tokens):
		if item=="STOP":
		    out_fd.write("\n")
		else:
		    out_fd.write(item+" ")
	out_fd.close()
	
	#create n-gram language models
	os.system("/home1/c/cis530/hw2/srilm/ngram-count -unk -text "+out_file+" -lm "+uni_mod+" -order 1")
	os.system("/home1/c/cis530/hw2/srilm/ngram-count -unk -text "+out_file+" -lm "+bi_mod+" -order 2 -cdiscount 0.75 -interpolate")
	os.system("/home1/c/cis530/hw2/srilm/ngram-count -unk -text "+out_file+" -lm "+tri_mod+" -cdiscount 0.75 -interpolate")
	#extract last 100 lines from the language model
	uni_fd = open("unigram.srilm","w")
	uni_fd.write(tail(open(uni_mod)))
	uni_fd.close()
	bi_fd = open("bigram.srilm","w")
	bi_fd.write(tail(open(bi_mod)))
	bi_fd.close()
	tri_fd = open("trigram.srilm","w")
	tri_fd.write(tail(open(tri_mod)))	
	tri_fd.close()

'''def clean():
    os.remove("temp_agshash")
    os.remove("get_all_ppl_test_agshash")
    shutil.rmtree("stanfordCoreNLPOutput_agshash")
    os.remove("fill_in_the_blanks_test")'''
    
#get perplexity of n-gram model on test directory
def get_srilm_ppl_for_file(lm_file, test_file):
        current_dir=os.getcwd()
        f=open(current_dir+"/concatenate.txt","w")
        tokens=load_file_tokens(test_file)
	counter=1
        content=""
        while counter<len(tokens):
                if (tokens[counter])=="STOP":
                        f.write(content)
                        content=""
                        f.write("\n")
                else:
                        content=content+tokens[counter]+" "
                counter=counter+1

        f.close()
        command="/home1/c/cis530/hw2/srilm/ngram -unk -lm "+lm_file+" -ppl "+current_dir+"/concatenate.txt"
        ppl=os.popen(command,'r').read()
        x=re.split(" |\n",ppl)
        counter=0
        while counter<len(x):
                if x[counter]=="ppl=":
                        break;
                counter=counter+1
        ppl=float(x[counter+1])
        os.remove("concatenate.txt")
        return ppl

#get perplexity of the constructed bigram model on test data in directory
def get_all_ppl(bigrammodel,directory):
	out_fd = open("get_all_ppl_test_agshash","w")
	file_list = get_all_files(directory)
	out_fd.write('STOP ')
	for fn in file_list:
	    tokens = open(fn,"r").read().split()
	    #IGNORE first STOP
	    tokens = tokens[1:]
	    out_fd.write(' '.join(tokens))
	    out_fd.write(' ')
	out_fd.close()
	return bigrammodel.getppl("get_all_ppl_test_agshash")

 
def get_all_ppl_srilm(lm_file,directory):
	out_fd = open("temp_agshash","w")
	file_list = get_all_files(directory)
	for fn in file_list:
	    tokens = open(fn,"r").read().split()
	    #IGNORE first STOP
	    tokens = tokens[1:]
	    for index,item in enumerate(tokens):
		if item=="STOP":
		    out_fd.write("\n")
		else:    
		     out_fd.write(item+" ")
	
	out_fd.close()
	ngram_output = os.popen("/home1/c/cis530/hw2/srilm/ngram -unk -lm "+ lm_file+ " -ppl temp_agshash").read().split()
	index = ngram_output.index("ppl=")
	os.remove("temp_agshash")
	return ngram_output[index+1]

#rankt the language models generated so far using the test data set
def write_ppl_results(bigrammodel,directory):
    lmid = []
    lmid.append(get_all_ppl(bigrammodel,directory))
    lmid.append(float(get_all_ppl_srilm("uni_gram",directory)))
    lmid.append(float(get_all_ppl_srilm("bi_gram",directory)))
    lmid.append(float(get_all_ppl_srilm("tri_gram",directory)))
    #print lmid
    sorted_indices =  [i[0] for i in sorted(enumerate(lmid), key=lambda x:x[1])]
    f = open("results.txt","a")
    f.write("\nLM ranking:" + str(sorted_indices[0])+" "+str(sorted_indices[1])+" "+str(sorted_indices[2])+" "+str(sorted_indices[3]))
    f.close()
    
#get perplexity of 2 files using a n-gram language model
def get_distinctive_measure(lm_file,mem_quote_file,nonmem_quote_file):
    mem_ppl = get_srilm_ppl_for_file(lm_file,mem_quote_file)
    nonmem_ppl = get_srilm_ppl_for_file(lm_file,nonmem_quote_file)
    return (mem_ppl,nonmem_ppl)
    
#get percentage of memorable quotes with higher perplexity in the specified directory
def distinctive_highppl_percentage(lm_file,directory):
    file_list = get_all_files(directory)
    fh = open("quotes_list_files_agshash","w")
    if not os.path.exists("stanfordCoreNLPOutput_agshash"):
	os.mkdir("stanfordCoreNLPOutput_agshash")
    if not os.path.exists("processedOutput_agshash"):
	os.mkdir("processedOutput_agshash")
    cnt = 0
    total = len(file_list)/2
    for file_name in file_list:
	fh.write(file_name+"\n")
    fh.close()
    preprocess("quotes_list_files_agshash", "stanfordCoreNLPOutput_agshash/")
    preprocessed_list = get_all_files("stanfordCoreNLPOutput_agshash/")
    for preprocessed_file in preprocessed_list:
	file_base_name = basename(preprocessed_file)
	#get file number of the memorable quote file
	num = file_base_name.split("_")[0]
	#check if the quote file is a memorable quote or not
	isMem = file_base_name.split("_")[1]
	if isMem[0]=="m":
	    memFile = num+"_mem.txt.xml"
	    not_memFile = num+"_not_mem.txt.xml"
	    process_file("stanfordCoreNLPOutput_agshash/"+memFile,"processedOutput_agshash/"+memFile+".output")
	    process_file("stanfordCoreNLPOutput_agshash/"+not_memFile,"processedOutput_agshash/"+not_memFile+".output")
	    mem_ppl = get_srilm_ppl_for_file(lm_file,"processedOutput_agshash/"+memFile+".output")
	    nonmem_ppl = get_srilm_ppl_for_file(lm_file,"processedOutput_agshash/"+not_memFile+".output")
	    if(mem_ppl>nonmem_ppl):
		cnt+=1
    res = (1.0*cnt/total)*100
    fd = open("results.txt","a")
    fd.write("\nPercentage of Memorable Quotes from LM 3 with higher perplexity:"+ str(res)+"%")
    #remove the temporary directories
    shutil.rmtree("stanfordCoreNLPOutput_agshash")
    shutil.rmtree("processedOutput_agshash")
    fd.close()
    return res

#Calculate perplexity of the entire sentence for all choices and compare the perplexities
def get_bestfit(sentence,wordlist,bigrammodel):
    min_ppl = float("inf")
    answer="" #store the best choice with least perplexity
    if not os.path.exists("FillInTheBlanks_agshash"):
	os.mkdir("FillInTheBlanks_agshash")
    for index,word in enumerate(wordlist):
	new_sentence = sentence.replace("<blank>",word)
	fd = open("fill_in_the_blanks_test_agshash","w")
	fd.write(new_sentence.lower())
	fd.close()
	preprocess1("fill_in_the_blanks_test_agshash","FillInTheBlanks_agshash/")
	process_file("FillInTheBlanks_agshash/fill_in_the_blanks_test_agshash.xml","FillInTheBlanks_agshash/fill_in_the_blanks_test_agshash.xml.output")
	ppl = bigrammodel.getppl("FillInTheBlanks_agshash/fill_in_the_blanks_test_agshash.xml.output")
	if(ppl<min_ppl):
	    min_ppl=ppl
	    answer=word
    os.remove("fill_in_the_blanks_test_agshash")
    shutil.rmtree("FillInTheBlanks_agshash")
    return answer

#calculate accuracy of bestfit using different sentences
def write_fill_in_the_blanks_results(bigrammodel):
    fd = open("results.txt","a")
    cnt=0.0
    bestfit = get_bestfit("Stocks <blank> this morning.",["plunged","walked","discovered","rise"],bigrammodel)
    print "plunged:"+bestfit
    if(bestfit=="plunged"):
	cnt+=1
    
    bestfit = get_bestfit("Stocks plunged this morning, despite a cut in interest <blank> by the Federal Reserve.",["rates","patients","researchers","levels"],bigrammodel)
    print "rates:"+bestfit
    if(bestfit=="rates"):
	cnt+=1
    
    bestfit = get_bestfit("Stocks plunged this morning, despite a cut in interest rates by the <blank> Reserve.",["Federal","university","bank","Internet"],bigrammodel)
    print "Federal:"+bestfit
    if(bestfit=="Federal"):
	cnt+=1
    
    bestfit = get_bestfit("Stocks plunged this morning, despite a cut in interest rates by the Federal Reserve, as Wall Street began <blank> for the first time.",["trading","wondering","recovering","hiring"],bigrammodel)
    print "trading:"+bestfit
    if(bestfit=="trading"):
	cnt+=1
    
    bestfit = get_bestfit("Stocks plunged this morning, despite a cut in interest rates by the Federal Reserve, as Wall Street began trading for the first time since last Tuesday’s <blank> attacks.",["terrorist","heart","doctor","alien"],bigrammodel)
    print "terrorist:"+bestfit
    if(bestfit=="terrorist"):
	cnt+=1
    
    #report the accuracy 
    fd.write("\nAccuracy: "+str(cnt/5.0*100.0))
    fd.close()
    
#predict the next word using the previous word
def fill_blank(sentence,bigrammodel):
    fd = open("fill_blank_test_agshash","w")
    fd.write(sentence)
    fd.close()
    if not os.path.exists("FillInTheBlanks_agshash"):
	os.mkdir("FillInTheBlanks_agshash")
    preprocess1("fill_blank_test_agshash","FillInTheBlanks_agshash/")
    process_file("FillInTheBlanks_agshash/fill_blank_test_agshash.xml","FillInTheBlanks_agshash/fill_blank_test_agshash.xml.output")
    
    #find bigram occuring with <blank>  in the processed file i.e context
    prevWord=''
    for word in open("FillInTheBlanks_agshash/fill_blank_test_agshash.xml.output").read().split():
	if(word=='xyzblankxyz'):
	    break;
	prevWord = word
    
    if bigrammodel.bigram_table.has_key(prevWord)==False:
	prevWord = "<UNK>"
	
    max = -9999	
    answer = ""
    if prevWord in bigrammodel.bigram_table:
	key1 = bigrammodel.bigram_table[prevWord]
    	for key2 in key1:
		if(bigrammodel.logprob(prevWord,key2)>max):
	    		max = bigrammodel.logprob(prevWord,key2)
	    		answer = key2
    os.remove("fill_blank_test_agshash")
    shutil.rmtree("FillInTheBlanks_agshash")
    return answer

#test fill_blank on sentences    
def write_fill_blank_results(bigrammodel):
    fd = open("results.txt","a")
    result = fill_blank("With great powers comes great xyzblankxyz",bigrammodel)
    fd.write("\nresponsibility:"+result)
    
    result = fill_blank("Say hello to my little xyzblankxyz",bigrammodel)
    fd.write("\nfriend:"+result)
    
    result = fill_blank("Hope is the quintessential human delusion, simultaneously the source of your greatest strength,and your greatest xyzblankxyz",bigrammodel)
    fd.write("\nweakness:"+result)
    
    result = fill_blank("You either die a hero or you live long enough to see yourself become the xyzblankxyz",bigrammodel)
    fd.write("\nvillian:"+result)
    
    result = fill_blank("May the Force be with xyzblankxyz",bigrammodel)
    fd.write("\nyou:"+result)
    
    result = fill_blank("Every gun makes its own xyzblankxyz",bigrammodel)
    fd.write("\ntune:"+result)
    
    fd.close()
    
    
class BigramModel:
    words = [] #tokens of all trainfiles
    bigram_table = {} #counts of bi-grams
    word_cnt = Counter()
    freq = {} #to store word frequencies
    size_vocabulory = 0
    
    #create a bi-gram language model
    def __init__(self,trainfiles):
	self.freq["STOP"] = 1
	self.freq["<UNK>"] = 0
    	       
	self.words.append("STOP")
        #get tokens from every file in training directory
	for file_path in trainfiles:
	    fh = open(file_path,"r")
	    for word in fh.read().split()[1:]:
		if self.freq.has_key(word):
		    self.freq[word] +=1
		else:
		    self.freq[word] =1
		self.words.append(word)
	
	#find words that occur only once in the list
	single_tokens = [word for word, count in self.freq.iteritems() if count == 1 ]
	
	#replace the words that occur only once with <UNK>	
	for index,item in enumerate(self.words):
	    if item in single_tokens:
		self.words[index] = "<UNK>"
		del self.freq[item]
		self.freq["<UNK>"]+=1
	
	self.size_vocabulory = len(self.freq)
	prevWord = ''
	for word in self.words:
	    #etup bigram if this isn't the first word
            if prevWord != '':
                if self.bigram_table.has_key(prevWord)==False:
        	    self.bigram_table[prevWord] = {}
        	    #add one smoothing
                    self.bigram_table[prevWord][word] = 2                
	        else:
                    if self.bigram_table[prevWord].has_key(word):
                       	self.bigram_table[prevWord][word] = self.bigram_table[prevWord][word] + 1
	            else:
        	        self.bigram_table[prevWord][word] = 2
	    prevWord = word

    #get the log probability from the bigram table	
    def logprob(self,context,event):
	if self.freq.has_key(context)==False:
		context = "<UNK>"
	if self.freq.has_key(event)==False:
		event = "<UNK>"
		
		
	#sum up all counts in the row
	sum_row=0
	#no of non-zero elements in the dictionary
	cnt_row = 0
	#flag to check if <context,event> occur in bigram_table
	found = 0
	
	if self.bigram_table.has_key(context)==True:
	    key1 = self.bigram_table[context]
	    for key2 in key1:
		sum_row += key1[key2]	
		cnt_row+=1
		if key2==event:
		    found = 1
	sum_row+= (self.size_vocabulory - cnt_row)
	#<context,event> not in bigram table. So, their bigram count is 1 after add-1 smoothing
	if found==0:
	    prob = 1.0 / sum_row
	else:
	    prob = (self.bigram_table[context][event]) / (1.0*sum_row)
	return math.log10(prob)
    
    #write the log-probabilities of all possible bigrams in the vocabulory
    def print_model(self,output_file):
	f = open(output_file,"w")
	for context in self.freq:
	    for event in self.freq:
		f.write(context+":"+event+" "+str(self.logprob(context,event))+" ")
	    f.write("\n")
	f.close()
    
    #get the perplexity of the test data using the bi-gram lanugage model
    def getppl(self,testfile):
	tokens = open(testfile,"r").read().split()
	#calculate perplexity
	sum=0.0
	for i in range(1,len(tokens)):
		sum+= self.logprob(tokens[i-1],tokens[i])
	l = sum/float(len(tokens))
	return math.pow(10,-l)
