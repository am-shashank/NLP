Document Classifier
===================

Focus was on developing a supervised classifier for identifying information-dense news texts, which report important factual information in direct, succinct manner. develop a supervised classifier for the information-dense/non-informative distinction. The main focus is on feature engineering. The challenge was to come up with features that distinguish between the
two types of text.

-------------------

We used a combination of several lexical and syntactic features
- Frequent Words
- MRC Dataset : This database allows us to capture different properties of the words. So, a stimuli can be created to differ on word frequency and control on imagibility and we can try lots of such different combinations. For this feature, we used the bag of 4920 MRC words as our feature vector. The value of every element in the vector is the number of times it appeared in the lead divided by the total number of words in the lead file.
- MRC Correctness: This is similar to MRC database. Except that, here we cluster the MRC words based on their scores of age of acquisition, imagibility,concreteness, familiarity and mean correctness. We divide the scores of AOA, IMAG, CONC, FAM, MEANC into intervals and create 4 vectors. The value of every element in the feature vector is the fraction of words in the lead filecorresponding to that interval. We then used the extended list of all these 4 vectors as our feature vector. 
- Mutual Information: Here, we captured the mutual information between the terms and the information dense texts. Similarly, we computed MI for non-information dense texts. We then used these both lists of 500 words each, with high MI values as our binary feature vector.
- Dependency Relations: Dependency parsers essentially capture the relationship between words and so by knowing this relationship, we can evaluate events more accurately. So they can reflect some syntactic information specific to information dense and non-information dense documents.
- Production Rules: Constituency parse tree breaks a text into sub-phrases. Non-terminals in the tree are types of phrases, the terminals are the words in the sentence. These are represented in the form of rules. Such rules capture important syntactic information which is crucial in document classification. We observed that many non-information dense documents contain catch phrases and peculiar sentences which could be captured using production rules.

Other Improvements:

Training the model on a subset of train data with very high word overlap and very low word overlap turns out to be way better instead of training the model on the entire set of training data (with high, low and average word overlap). We used 10 fold cross validation to tweak parameters like the no of top words, dependency relations and production rules and conducted many other experiments with it. It turned out that 10000 was a good number. We also tried using the entire set of words, dependency relations and production rules as our feature vector but that performed badly.

Resources and tools:

- StanfordCoreNLP: This tool is useful for tokenization and extraction of features like dependencies, production rules, NER etc. The xml file returned by this tool can be easily parsed using a readily available parser and from our experience with this tool, we felt that the tool was very accurate and easy to use.
- LibSVM: This tool is used for building a document classification model. 
- MRC Psychological Database: It becomes very difficult to experiment when you want to control several properties of words (like frequency, familiarity, concreteness, imagibility etc) or if one is to avoid confounding them with
experimental interest. MRC database takes care of such problems by scoring words on different properties.
- Word2vec: This tool gives us the distributed vector representations of the words. The major benefit of using such representations is that similar words are closer in the vector space making our model very robust. We used this to
implement a similarity matrix in conjunction with a unigram model.


Algorithm:

Input: Training data with class labels, testing data
Output: 
 - Predicted labels of test data
 - Accuracy (No of labels predicted correctly / actual labels)

Steps

- Pre-process the training data using StanfordCoreNLP to generate xml files.
- Select 500 (approximately 20% of the total train set) files with highest human summary word overlap and 500 files with lowest human summary word overlap as the train data. Among the remaining files, randomly choose a few as the validation set.
- Feature Extraction:
  -- Top 10000 words
  -- Top 10000 production rules
  -- Top 10000 dependency relations
  -- Top 500 MI words from Information Dense texts and top 500 MI words from Non-information dense texts
  -- Bag of 4923 words from MRC database
  -- Bag of 230 intervals for every word property (Imagibility, Familiarity, Concreteness, Age of Acquisition, Mean correctness) representing the fraction of words with scores in that corresponding interval of that
property 
- Map every lead file to the above mentioned feature spaces. Use the combination of these feature vectors to represent the lead file 
- Train the LibSVM model on feature vectors of the lead files in the selected train set.
- Using this model, predict the labels of the test set.
