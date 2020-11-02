import os
import string
from string import digits
import json
import sys

class PercepClassify():
    def __init__(self, model_path, stopwords):
        self.stop_set = self.extract_stopwords(stopwords)
        self.trans = self.punc_remove() #For removing punctuation
        self.remove_digits = str.maketrans('', '', digits) #For removing digits
        
        #Bias: Looks like self.bias = {'trust': 2, 'sent': 4}
        self.bias = self.get_weights("bias", model_path)
        
        #Weights: Looks like self.weights = {'trust': {w1: 3, w2: -1, ...}, 'sent': {w1: 2, w2: -3, ...}}
        self.weights = self.get_weights("weights", model_path)
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

    def get_weights(self, w_type, model_path): #__init__()
        '''Get bias from model file.
        w_type - either "weights" or "bias".
        model_path - path to model file'''
        with open(model_path) as json_file:
            model_dict = json.load(json_file)
        return model_dict[w_type]  

    def classify(self, test_path):
        '''Read in test files one-by-one, extract features, and classify doc using model.'''
        for root, directories, files in os.walk(test_path, topdown=False):
            for name in directories:
                dir_split = os.path.join(root, name).split('/')
                if len(dir_split) == 4 and dir_split[-1] in ['fold1']: #15 in vocareum
                    doc_files = [f for f in os.listdir(os.path.join(root, name))]
                    for doc_name in doc_files:
                        doc_file = os.path.join(root, name) + '/' + doc_name
                        self.extract_features_classify(doc_file)
                        
    def extract_features_classify(self, doc_file): # called by classify()
        '''Extract features from document/review and multiply against associated weights.
        Return the selected label for both classes.
        doc_file - full path to the review text'''

        #initialize activation score
        act_scores = {"trust": self.bias["trust"], "sent": self.bias["sent"]}
        
        #Build up a feature vector for the record
        feat_vec = dict()
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
                    
                #Add to feature vector
                if word_ii not in feat_vec: feat_vec[word_ii] = 1
                else: feat_vec[word_ii] += 1
        
        #Compute activation score from feat_vec * weights         
        for feat in feat_vec.keys():
            for c in act_scores.keys():
                if feat in self.weights[c]:
                    score = self.weights[c][feat] * feat_vec[feat]
                    prev_score = act_scores[c]
                    act_scores[c] = prev_score + score
        
        #Convert act scores into labels
        preds = []
        for c in act_scores.keys():
            if c == 'trust':
                if act_scores[c] > 0: preds.append('truthful')
                else: preds.append('deceptive')
            elif c == 'sent':
                if act_scores[c] > 0: preds.append('positive')
                else: preds.append('negative')
        preds.append(doc_file)
        self.predictions.append(preds)
        
    def output_results(self, output_file):
        fd = open(output_file, "w")
        for pred in self.predictions:
            fd.write(' '.join(pred) + '\n')
        fd.close()

#PARAMETERS
# model_path = 'averagedmodel.txt'
# model_path = 'vanillamodel.txt'
# model_path = '/path/to/vanillamodel.txt'
model_path = sys.argv[1]


# test_path = 'op_spam_training_data'
test_path = sys.argv[2]
stopwords = 'stopwords.txt'
output_file = 'percepoutput.txt'

#DRIVER
test = PercepClassify(model_path, stopwords)
test.classify(test_path)
test.output_results(output_file)


