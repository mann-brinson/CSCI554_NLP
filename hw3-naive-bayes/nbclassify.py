import os
import string
from string import digits
import json
import math
import sys

class NbClassify():
    def __init__(self, model_file, stopwords):
        self.stop_set = self.extract_stopwords(stopwords)
        self.trans = self.punc_remove() #For removing punctuation
        self.remove_digits = str.maketrans('', '', digits) #For removing digits
        self.prior_probs, self.cond_probs = self.read_model(model_file)
        self.predictions = [] 
         
    def extract_stopwords(self, stopwords): # __init__()
        '''Make a set of stopwords from the stopwords file'''
        s_fd = open(stopwords, 'r') 
        stop_raw = s_fd.readlines()
        stop_set = set([i.strip() for i in stop_raw])
        return stop_set

    def punc_remove(self): # __init__()
        '''Make a dict that will remove special characters'''
        sp_chars = [i for i in string.punctuation]
        del_d = {sp_char: '' for sp_char in sp_chars} 
        del_d[' '] = ''
        trans = str.maketrans(del_d) 
        return trans
    
    def read_model(self, model_file): # __init__()
        '''Read and store the model file into memory.'''
        with open(model_file) as json_file:
            model_dict = json.load(json_file)
        prior_probs = model_dict['prior_probs']
        cond_probs = model_dict['cond_probs']
        return prior_probs, cond_probs
    
    def classify(self, test_path):
        '''Read in test files one-by-one, extract features, and classify doc using model.'''
        for root, directories, files in os.walk(test_path, topdown=False):
            for name in directories:
                dir_split = os.path.join(root, name).split('/')
                if len(dir_split) == 15: #15 in vocareum
                    doc_files = [f for f in os.listdir(os.path.join(root, name))]
                    for doc_name in doc_files:
                        doc_file = os.path.join(root, name) + '/' + doc_name
                        self.extract_features_classify(doc_file)
                        
    def extract_features_classify(self, doc_file): # called by classify()
        '''Extract features from document/review and populate the class_feat_counts.
        Return the selected label for both classes.
        doc_file - full path to the review text'''

        #initialize bayes scores with prior probs
        bayes_scores = {'trust': {'truthful': math.log(self.prior_probs['trust']['truthful'], 10), 
                                  'deceptive': math.log(self.prior_probs['trust']['deceptive'], 10)},
                        'sent': {'positive': math.log(self.prior_probs['sent']['positive'], 10),
                                 'negative': math.log(self.prior_probs['sent']['negative'], 10)}}

        r_fd = open(doc_file, 'r') 
        raw_text = r_fd.readlines()[0]
        word_list = raw_text.split(' ')
        for word in word_list:
            word1 = word.lower().strip()
            #Remove digits and punctuation
            word2 = word1.translate(self.remove_digits)
            word_list2 = word2.split(',')
            for word_i in word_list2:
                word_ii = word_i.translate(self.trans)

                #Edge case: check for inter-word commas
                if word_ii == '': continue

                #Remove stopwords
                if word_ii in self.stop_set: continue

                #For each class/label, check if the word exists within the class/label cond_probs
                #If yes, then select the cond_prob and multiply it to the bayes_scorees[c][l]
                for c in bayes_scores.keys():
                    labels = list(bayes_scores[c].keys())
                    for l in labels:
                        if word_ii in self.cond_probs[c][l]:
                            current_prob = self.cond_probs[c][l][word_ii]
                            prev_probs = bayes_scores[c][l]
                            bayes_scores[c][l] = prev_probs + math.log(current_prob, 10)

        #Select max class label for each class
        preds = []
        for c in bayes_scores.keys():
            label = max(bayes_scores[c], key=lambda key: bayes_scores[c][key])
            preds.append(label)
        preds.append(doc_file)
        self.predictions.append(preds)
        
    def output_results(self, output_file):
        fd = open(output_file, "w")
        for pred in test.predictions:
            fd.write(' '.join(pred) + '\n')
        fd.close()

#PARAMETERS
model_file = 'nbmodel.txt'
# test_path = 'op_spam_training_data'
test_path = sys.argv[1]
stopwords = 'stopwords.txt'
output_file = 'nboutput.txt'

#DRIVER
test = NbClassify(model_file, stopwords)
test.classify(test_path)
test.output_results(output_file)



