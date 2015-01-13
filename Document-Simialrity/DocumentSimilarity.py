import os
import re
from collections import Counter
import math
from operator import itemgetter
from nltk import sent_tokenize, word_tokenize

#list files recursively within directory
def get_all_files(directory):
    file_list = []
    for root,dirnames, filenames in os.walk(directory):
        for filename in filenames:
            abs_path = os.path.join(root, filename)
            rel_path = abs_path[len(directory):]
            file_list.append(rel_path)
    return file_list

#list of all tokens (processed words) in a file
def load_file_tokens(filepath):
    file_tokens=[]
    with open(filepath) as f:
        content = f.read().split()
        for word in content:
                #replace all special characters with space
                tokenized_word = re.sub('[^A-Za-z0-9]+', ' ', word).lower()
                #check for NULL string
                if(tokenized_word):
                        #now split the word on newly inserted space and filter out all NULL
                        non_alphanum_words = filter(None,tokenized_word.split(' '))
                        file_tokens.extend(non_alphanum_words)
    return file_tokens

#list of all tokens (processed words) in a directory
def load_collection_tokens(directory):
    collection_tokens = []
    for root,dirnames, filenames in os.walk(directory):
        for filename in filenames:
            collection_tokens.append(load_file_tokens(os.path.join(root, filename)))
    return [word for sublist in collection_tokens for word in sublist]

#get Normalized term frequency of items in list
def get_tf(itemlist):
    tf = Counter(itemlist)
    max_term = max(tf, key=tf.get)
    max_count = tf[max_term]
for key in tf.keys():
        tf[key] = float (tf[key]) / float (max_count)
    return tf

#get Inverse Document frequency of items in list
def get_idf(itemlist):
    N = len(itemlist)
    unique_tokens = set([x for sublist in itemlist for x in sublist])
    idf = {}
    #count number of lists that contain token
    for token in unique_tokens:
        df_term = 0
        for sublist in itemlist:
            if token in sublist:
                df_term = df_term + 1
        idf[token] = math.log(N/float(df_term))
    #add artificial unknown word to dictionary
    idf['<U N K>'] =   math.log(N)
    return idf

#get tokens with top TF.IDF values
def get_tfidf_top(dict1, dict2, k):
    temp = Counter({})
    for tf_key in dict1.keys():
        if tf_key in dict2.keys():
            temp[tf_key] = float(dict1[tf_key]) * float (dict2[tf_key])
    popular_words = sorted(temp, key = temp.get, reverse = True)
    return popular_words[:k]

#helper function to return a dictionary of MI values
def get_mi(bg_terms,topic_terms):
    mi_dict = Counter({})
    for term in topic_terms:
        if term not in mi_dict.keys() and topic_terms.count(term) > 5:
            #calculate numerator
            p_w_topic = topic_terms.count(term) / float (len(topic_terms))
            #calculate denominator
            p_w = bg_terms.count(term) / float (len(bg_terms))
            mi_dict[term] = math.log(p_w_topic/p_w)
            if mi_dict[term] < 0:
                mi_dict[term] = 0
    return mi_dict
#get tokens with top MI values
def get_mi_top(bg_terms, topic_terms,k):
    mi_dict = get_mi(bg_terms,topic_terms)
    return [k for k,v in mi_dict.most_common(k)]

#record the MI values and corresponding tokens in a specified file
def write_mi_weights(directory, outfilename):
    topic_terms = load_collection_tokens(directory)
    bg_terms = load_collection_tokens('/home1/c/cis530/hw1/data/corpus')
    mi_dict = get_mi(bg_terms,topic_terms)
    f=open(outfilename,'a')
    for key,value in mi_dict.iteritems():
        f.write(key+'\t'+ str(value) +'\n' )
    f.close()
    return

#calculate precision
def get_precision(L_1,L_2):
    L1 = set(L_1)
    L2 = set(L_2)
    intersection_set  = L1.intersection(L2)
    return float (len(intersection_set))/len(L1)

#calculate recall
def get_recall(L_1,L_2):
    L1 = set(L_1)
    L2 = set(L_2)
    intersection_set  = L1.intersection(L2)
    return float (len(intersection_set))/len(L2)

#calculate fmeasure
def get_fmeasure(L_1,L_2):
    return ((2*get_precision(L_1,L_2)*get_recall(L_1,L_2)) + (get_precision(L_1,L_2)+get_recall(L_1,L_2)))

#read Brown clusters file to dictionary      
def read_brown_cluster():
    file_path = '/home1/c/cis530/hw1/data/brownwc.txt'
    bc_dict = {}
    content = open(file_path).readlines()
    for line in content:
      li = line.split('\t')
      bc_dict[li[1]] = li[0]
    return bc_dict
  #Get clusters in file 
def load_file_clusters(filepath,bc_dict):
    file_tokens = load_file_tokens(filepath)
    word_cluster_id = []
    for token in file_tokens:
      if token in bc_dict:
        word_cluster_id.append(bc_dict[token])
    return word_cluster_id

#Get clusters in directory 
def load_collection_clusters(directory,bc_dict):
    collection_tokens = load_collection_tokens(directory)
    word_cluster_id = []
    for token in collection_tokens:
      if token in bc_dict:
        word_cluster_id.append(bc_dict[token])
    return word_cluster_id

#get Inverse documnet frequency for the clusters
def get_idf_clusters(bc_dict):
    document_cluster_list = []
    dir_list = [name for name in os.listdir('/home1/c/cis530/hw1/data/all_data/') if os.path.isdir(os.path.join('/home1/c/cis530/hw1/data/all_data/', name))]
    for dir_name in dir_list:
        document_cluster_list.append(load_collection_clusters('/home1/c/cis530/hw1/data/all_data/'+dir_name,bc_dict))
    return get_idf(document_cluster_list)

#record TF.IDF values of the clusters in the specified file
def write_tfidf_weights(directory,outfilename,bc_dict):
    idf_dict = get_idf_clusters(bc_dict)
    tf_dict = get_tf(load_collection_clusters(directory,bc_dict))
    tfidf_dict = {}
    f=open(outfilename,'a')
    for tf_key in tf_dict:
      if tf_key in idf_dict:
        tfidf_dict[tf_key] = float(tf_dict[tf_key]) * float (idf_dict[tf_key])
    for key,value in tfidf_dict.iteritems():
        f.write(key+'\t'+ str(value) +'\n' )
    f.close()
    return
#map clusters to unique integers
def create_feature_space(list):
    feature_space_dict = {}
    ctr=0
    for item in set(list):
        feature_space_dict[item] = ctr
        ctr+=1
    return feature_space_dict

#create a vector of clusters
def vectorize(feature_space, lst):
    vector = []
    for feature in feature_space:
        if feature in lst:
                vector.insert(feature_space[feature],1)
        else:
                vector.insert(feature_space[feature],0)
    return  vector

#caluclate similarity of 2 vectors
def cosine_similarity(X, Y):
    if(all(v == 0 for v in X) or all(v == 0 for v in Y)):
        return 0
    mag_x = 0
    mag_y = 0
    prod = 0
    for  i in range(0,len(X)):
        prod+= float(X[i])*float(Y[i])
        mag_x+= float(X[i])*float(X[i])
        mag_y+= float(Y[i])*float(Y[i])
    return float(prod)/(math.sqrt(mag_x)*math.sqrt(mag_y))

#rank the specified test documents with respect to the repersentative file
def rank_doc_sim(rep_file,method,test_path,bc_dict):
    list_rep = []
    vector_rep = []
    list_test = []
    vector_test = []
    result = []
    f = open(rep_file,"r")
    cluster_value = {}
    for line in f.readlines():
        temp = line.strip('\n').split("\t")
        #store the key values of the file in a dict
cluster_value[temp[0]]=temp[1]
        list_rep.append(temp[0])
    feature_space = create_feature_space(list_rep)
    for feature in feature_space:
        vector_rep.insert(feature_space[feature],cluster_value[feature])
    for root,dirnames, filenames in os.walk(test_path):
            for test_file in filenames:
                if(method=="tfidf"):
                    list_test = load_file_clusters(test_path+"/"+test_file,bc_dict)
                else:
                    list_test = load_file_tokens(test_path+"/"+test_file)
                vector_test = vectorize(feature_space,list_test)
                tup = (test_file,cosine_similarity(vector_rep,vector_test))
                result.append(tup)
    result = sorted(result,key=lambda x: x[1],reverse=True)
    return result[:100]

#record precision and recall values in a file
def write_precision_recall(itemlist, outfilename="/home1/a/agshash/CL/hw1/results.txt"):
    mi_list = get_mi_top(load_collection_tokens('/home1/c/cis530/hw1/data/all_data/'),load_collection_tokens('/home1/c/cis530/hw1/data/corpus/starbucks/'),100)
    tf_dict = get_tf(load_collection_tokens('/home1/c/cis530/hw1/data/corpus/starbucks/'))
    tf_top = sorted(tf_dict.keys(),key=lambda x: tf_dict[x],reverse=True)[:100]
    idf_dict = get_idf(itemlist)
    tfidf_list = get_tfidf_top(tf_dict,idf_dict, 50)
    #calculate precision/recall   
    p1 = get_precision(mi_list,tfidf_list)
    r1 = get_recall(mi_list,tfidf_list)
    p2 = get_precision(tf_top,tfidf_list)
    r2 = get_recall(tf_top,tfidf_list)
    #obtain a ranking of documents based on cosine similarity
    result1 = rank_doc_sim("starbucks_tfidf_weights.txt","tfidf",'/home1/c/cis530/hw1/data/mixed/',read_brown_cluster())
    #calculate precision for ranked documents
    no_relevant = 0
    for tup in result1:
        #count no of relevant documents retrieved
        if tup[0].startswith('starbucks'):
                no_relevant+=1
    files_test_path = get_all_files('/home1/c/cis530/hw1/data/mixed/')
    no_docs_database = 0
    for f in files_test_path:
        #count no of actually relevant documents
        if f.startswith('starbucks'):
                no_docs_database=no_docs_database + 1
    p3 = float (no_relevant) / no_docs_database
    result2 = rank_doc_sim("starbucks_mi_weights.txt","mi",'/home1/c/cis530/hw1/data/mixed/',read_brown_cluster())
    no_relevant = 0
    for tup in result2:
        if tup[0].startswith('starbucks'):
                no_relevant+=1
    p4 = float(no_relevant) / no_docs_database
    #record the calculated precision/recall values in a file
    f = open(outfilename,"a")
    f.write(str(p1) + ","+str(r1)+"\n")
    f.write(str(p2)+","+str(r2)+"\n")
    f.write(str(p4)+"\n")
    f.write(str(p3))
    f.close()
    return
               
