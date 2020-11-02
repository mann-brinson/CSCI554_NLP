import os
import string
from string import digits
import json
import sys

class NbTrain():
    def __init__(self, input_path, stopwords):
        self.input_path = input_path
        self.stop_set = self.extract_stopwords(stopwords)
        self.priors = {'trust': {'truthful': 0, 'deceptive': 0},
                       'sent': {'positive': 0, 'negative': 0}}
        self.prior_probs = {'trust': {'truthful': 0, 'deceptive': 0},
                            'sent': {'positive': 0, 'negative': 0}}
        
        self.trans = self.punc_remove() #For removing punctuation
        self.remove_digits = str.maketrans('', '', digits) #For removing digits

        #Keep track of class_type>class>feature>count
        self.class_feat_counts = {'trust': {'truthful': dict(), 'deceptive': dict()},
                                    'sent': {'positive': dict(), 'negative': dict()}} 
        
        self.feat_count_full = 0 #For a given class, length of union between the class's two label sets
    
    def punc_remove(self): # called by __init__
        '''Make a dict that will remove special characters'''
        sp_chars = [i for i in string.punctuation]
        del_d = {sp_char: '' for sp_char in sp_chars} 
        del_d[' '] = ''
        trans = str.maketrans(del_d) 
        return trans
    
    def extract_stopwords(self, stopwords): # called by __init__()
        '''Make a set of stopwords from the stopwords file'''
        s_fd = open(stopwords, 'r') 
        stop_raw = s_fd.readlines()
        stop_set = set([i.strip() for i in stop_raw])
        return stop_set
    
    def class_feat_count(self): #STEP 1
        ''' Read each review file and extract the file's labels: class_trust and class_sent.
        For each review, tokenize and add each token to the class_feat_counts'''
        for root, directories, files in os.walk(self.input_path, topdown=False):
            for name in directories:
                dir_split = os.path.join(root, name).split('/')
                if len(dir_split) == 4:
                    #Get class labels
                    class_sent = dir_split[1].split('_')[0]
                    class_trust = dir_split[2].split('_')[0]
                    doc_files = [f for f in os.listdir(os.path.join(root, name))]
                    for doc_name in doc_files:
                        doc_file = os.path.join(root, name) + '/' + doc_name
                        self.extract_features(doc_file, class_trust, class_sent)
        #Compute prior_probs
        for c in self.priors.keys():
            c_token_count = sum(self.priors[c].values())
            labels = list(self.priors[c].keys())
            for l in labels:
                c_l_prob = self.priors[c][l] / c_token_count
                self.prior_probs[c][l] = c_l_prob
                        
    def extract_features(self, doc_file, class_trust, class_sent): # called by class_feat_count()
        '''Extract features from document/review and populate the class_feat_counts.
        doc_file - full path to the review text'''
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
                    
                #Update priors
                self.priors['trust'][class_trust] += 1
                self.priors['sent'][class_sent] += 1             
                    
                #Add trust feature
                if word_ii not in self.class_feat_counts['trust'][class_trust]:
                    self.class_feat_counts['trust'][class_trust][word_ii] = 1 #Create feature count
                else:
                    self.class_feat_counts['trust'][class_trust][word_ii] += 1 #Update feature count
                
                #Add sentiment feature
                if word_ii not in self.class_feat_counts['sent'][class_sent]:
                    self.class_feat_counts['sent'][class_sent][word_ii] = 1 #Create feature count
                else:
                    self.class_feat_counts['sent'][class_sent][word_ii] += 1 #Update feature count      
                
    def count_feats(self): #STEP 2
        '''Count all features for a given classe's two binary labels.
        Store the result in self.feat_count_full
        class_type - the type of class you want to count features for either trust or sent.'''
        c = list(test.priors.keys())[0]
        labels = list(test.priors[c].keys())
        class_feats = set()
        for l in labels:
            feats = test.class_feat_counts[c][l].keys()
            [class_feats.add(feat) for feat in feats]
        n_class_feats = len(class_feats)
        self.feat_count_full = n_class_feats
        
    def plus_one_smooth(self): #STEP 3
        '''Add one to each existing feature count. 
        Add non-occurring features to each feature vector. '''
        #STEP 1: For each class, for each label, +1 to each feature
        for c in self.class_feat_counts.keys():
            labels = list(self.class_feat_counts[c].keys())
            for l in labels:
                for key in self.class_feat_counts[c][l].keys():
                    self.class_feat_counts[c][l][key] += 1
                    
        #STEP 2: For each class, find the diffs between label1 and label2, and add features
        for c in self.class_feat_counts.keys():
            labels = list(self.class_feat_counts[c].keys())
            set_a = set(self.class_feat_counts[c][labels[0]].keys())
            set_b = set(self.class_feat_counts[c][labels[1]].keys())

            not_in_a = set_b.difference(set_a)
            not_in_b = set_a.difference(set_b)

            #Add the null instances to each count
            for feat in not_in_a: 
                self.class_feat_counts[c][labels[0]][feat] = 1
            for feat in not_in_b:
                self.class_feat_counts[c][labels[1]][feat] = 1
                
        #STEP 3: For each class/label/feature_set, divide by: (class/label prior + feat_count_full)
        for c in self.class_feat_counts.keys():
            labels = list(self.class_feat_counts[c].keys())
            for l in labels:
                for key in self.class_feat_counts[c][l].keys():
                    feat_count = self.class_feat_counts[c][l][key]
                    self.class_feat_counts[c][l][key] = feat_count / (self.priors[c][l] + self.feat_count_full)
                    
    def output_model(self, output_file): #STEP 4
        '''Serialize the priors and conditional probabilities into a model .txt file.
        output_file - the name of the file to output'''
        m_dict = {'prior_probs': self.prior_probs, 'cond_probs': self.class_feat_counts}
        model = json.dumps(m_dict, indent=1)
        fd = open(output_file, "w")
        fd.write(model)
        fd.close()

#PARAMETERS
# input_path = 'op_spam_training_data'
input_path = sys.argv[1]
stopwords = 'stopwords.txt'
output_file = 'nbmodel.txt'

#DRIVER: Fit the model
test = NbTrain(input_path, stopwords)
test.class_feat_count()
test.count_feats()
test.plus_one_smooth()
test.output_model(output_file)

