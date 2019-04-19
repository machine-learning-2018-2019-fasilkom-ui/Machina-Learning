import math

class TF_IDF:
    def __init__(self):
        self.sms = []
        self.label = []

        self.n_iklan = 0
        self.dict_iklan = {}
        self.dict_non_iklan = {}

        self.n_non_iklan = 0
        self.all_iklan = {}
        self.all_non_iklan = {}

        self.all_words = {}

    def set_train_and_label(self, sms, label):
        self.sms = sms
        self.label = label

    def compute_freq(self):
        for doc_number in range(len(self.sms)):
            line = self.sms[doc_number].split(" ")
            label = self.label[doc_number]
            for word in line:
                if word in self.all_words:
                    self.all_words[word] += 1
                else:
                    self.all_words[word] = 1
            if label == 1:
                self.n_iklan += 1
                for word in line:
                    if word in self.all_iklan:
                        self.all_iklan[word] += 1
                    else:
                        self.all_iklan[word] = 1
                    if word in self.dict_iklan:
                        if doc_number in self.dict_iklan[word]:
                            self.dict_iklan[word][doc_number] += 1
                        else:
                            self.dict_iklan[word][doc_number] = 1
                    else:
                        self.dict_iklan[word] = {}
                        self.dict_iklan[word][doc_number] = 1  
            else:
                self.n_non_iklan += 1
                for word in line:
                    if word in self.all_non_iklan:
                        self.all_non_iklan[word] += 1
                    else:
                        self.all_non_iklan[word] = 1
                    if word in self.dict_non_iklan:
                        if doc_number in self.dict_non_iklan[word]:
                            self.dict_non_iklan[word][doc_number] += 1
                        else:
                            self.dict_non_iklan[word][doc_number] = 1
                    else:
                        self.dict_non_iklan[word] = {}
                        self.dict_non_iklan[word][doc_number] = 1

    def get_tf_idf_scores(self):
        scores = {}
        for key, value in self.all_words.items():
            frequency_iklan = 0
            frequency_non_iklan = 0
            df_iklan = 0
            df_non_iklan = 0
            idf_iklan = 0
            idf_non_iklan = 0
            # score(word) = frequency_word * idf

            if key in self.all_iklan:
                frequency_iklan = self.all_iklan[key]
            
            if key in self.all_non_iklan:
                frequency_non_iklan = self.all_non_iklan[key]

            # compute df iklan = unique document
            if key in self.dict_iklan:
                df_iklan = len(self.dict_iklan[key])
            
            if key in self.dict_non_iklan:
                df_non_iklan = len(self.dict_non_iklan[key])
            
            scores[key] = {}
            scores[key][0] = 0
            scores[key][1] = 0

            if df_iklan != 0:
                scores[key][1] = frequency_iklan * math.log(self.n_iklan / df_iklan)

            if df_non_iklan != 0:
                scores[key][0] = frequency_non_iklan * math.log(self.n_non_iklan / df_non_iklan)
        
        return scores
            

# sms = ['a b c g', 'd e f', 'a a a a a a a b c', 'a b c h', 'd e f', 'a a a a a a a b c']
# label = [0, 0, 0, 1, 1, 1]

# tf = TF_IDF()
# tf.set_train_and_label(sms, label)
# tf.compute_freq()
# scores = tf.get_tf_idf_score()
# print(scores)
