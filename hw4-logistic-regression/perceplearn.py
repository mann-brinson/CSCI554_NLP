import os
import string
from string import digits
import json
import sys
import random
random.seed(42)

class PercepTrain():
    def __init__(self, input_path, stopwords):
        self.input_path = input_path
        self.stop_set = self.extract_stopwords(stopwords)
        
        self.trans = self.punc_remove() #For removing punctuation
        self.remove_digits = str.maketrans('', '', digits) #For removing digits

        #Keep track of class_type>class>feature>count
        self.class_label_map = {'trust': {'truthful': 1, 'deceptive': -1},
                                'sent': {'positive': 1, 'negative': -1}}
    
        self.class_weights_vanilla = {'trust': dict(), 'sent': dict()}
        self.bias_vanilla = {'trust': 0, 'sent': 0}
    
        self.class_weights_cache = {'trust': dict(), 'sent': dict()}
        self.bias_cache = {'trust': 0, 'sent': 0}

        self.total_instances_seen = 1
    
    def punc_remove(self): # called by __init__()
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
    
    def learn_weights(self, n_epochs): #STEP 1
        ''' Read each review file and extract the file's labels: class_trust and class_sent.
        For each review, tokenize and add each token to the class_feat_counts'''
        #Read each training file, and add to a list
        training_list = []
        for root, directories, files in os.walk(self.input_path, topdown=False):
            for name in directories:
                dir_split = os.path.join(root, name).split('/')
                if len(dir_split) == 4 and dir_split[-1] in ['fold2', 'fold3', 'fold4']: #15 in vocareum
                    #Get class labels
                    class_sent = dir_split[1].split('_')[0]
                    class_trust = dir_split[2].split('_')[0]
                    doc_files = [f for f in os.listdir(os.path.join(root, name))]
                    for doc_name in doc_files:
                        doc_file = os.path.join(root, name) + '/' + doc_name
                        training_list.append(doc_file)
                            
        #Randomly shuffle the list of training instances
        random.shuffle(training_list)
        
        #Run through the shuffled list of training instances n_epochs times
        for epoch in range(n_epochs):
            for doc_file in training_list:
                class_sent = doc_file.split('/')[-4].split('_')[0]
                class_trust = doc_file.split('/')[-3].split('_')[0]
                self.extract_features(doc_file, trust=class_trust, sent=class_sent)
                        
    def extract_features(self, doc_file, **kwargs): # called by learn_weights()
        '''Extract features from document/review and populate the class_feat_counts.
        doc_file - full path to the review text'''
        r_fd = open(doc_file, 'r') 
        raw_text = r_fd.readlines()[0]
        word_list = raw_text.split(' ')
        
        feat_vector = dict()
        
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
                if word_ii not in feat_vector: feat_vector[word_ii] = 1
                else: feat_vector[word_ii] += 1

        #Calculate activation score for each class  
        class_scores = self.compute_activation(feat_vector)

        #For each classifier, check if the prediction is incorrect, if so, update weights
        for c in kwargs.keys():
            label_c = self.class_label_map[c][kwargs[c]] #Gets a +1 or -1 label for class>label
            if label_c * class_scores[c] <= 0:
                #Update weights
                for feat in feat_vector.keys():
                    i_feat_val = feat_vector[feat]
                    count = self.total_instances_seen

                    #update weights
                    if feat not in self.class_weights_vanilla[c]:
                        self.class_weights_vanilla[c][feat] = label_c * i_feat_val
                    else:
                        self.class_weights_vanilla[c][feat] += label_c * i_feat_val

                    #update cached weights
                    if feat not in self.class_weights_cache[c]:
                        self.class_weights_cache[c][feat] = label_c * count * i_feat_val
                    else:
                        self.class_weights_cache[c][feat] += label_c * count * i_feat_val

                self.bias_vanilla[c] += label_c #update bias
                self.bias_cache[c] += label_c * count #update cached bias
        self.total_instances_seen += 1 #increment counter regardless of update
                    
    def compute_activation(self, feat_vector): #called by extract_features()
        '''For a given instance, calculate activation score for each class'''
        class_scores = {'trust': 0, 'sent': 0}
        #add sum(w_i * x_i) to score, for all instance features
        for feat in feat_vector.keys():
            for c in class_scores.keys():
                if feat in self.class_weights_vanilla[c]:
                    feat_weight = self.class_weights_vanilla[c][feat]
                    class_scores[c] += feat_vector[feat] * feat_weight  
        #add biases to score
        for c in self.bias_vanilla:
            class_scores[c] += self.bias_vanilla[c]
        return class_scores
    
    #def output_models(self, output_vanilla, output_avg): #STEP 3
    def output_models(self, **kwargs):
        '''Output the model files.'''
        models = dict() 
        #Get vanilla model
        v_dict = {"bias": self.bias_vanilla, "weights": self.class_weights_vanilla}
        
        #Create averaged model
        count = self.total_instances_seen
        avg_dict = {"bias": {"trust": 0, "sent": 0}, 
                    "weights": {"trust": dict(), "sent": dict()}}
        
        for c in self.class_weights_vanilla: #weights
            for w_key in self.class_weights_vanilla[c].keys():
                w_val = self.class_weights_vanilla[c][w_key]
                w_val_avg = w_val - (self.class_weights_cache[c][w_key] / count)
                avg_dict["weights"][c][w_key] = w_val_avg
                
        for c in self.bias_vanilla: #bias
            b_val = self.bias_vanilla[c]
            b_val_avg = b_val - (self.bias_cache[c] / count)
            avg_dict["bias"][c] = b_val_avg
            
        models["output_vanilla"] = v_dict
        models["output_avg"] = avg_dict
        
        #Write the results to to the provided output files
        for c in kwargs.keys():
            output_file = kwargs[c]
            model = json.dumps(models[c], indent=1)
            fd = open(output_file, "w")
            fd.write(model)
            fd.close()

#PARAMETERS
#input_path = 'op_spam_training_data'
input_path = sys.argv[1]
stopwords = 'stopwords.txt'
n_epochs = 10

output_vanilla = 'vanillamodel.txt'
output_avg = 'averagedmodel.txt'

#DRIVER
test = PercepTrain(input_path, stopwords)
test.learn_weights(n_epochs)
test.output_models(output_vanilla=output_vanilla, output_avg=output_avg)

