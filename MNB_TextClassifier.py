import numpy as np
from tf_idf import TF_IDF

class MNB_TextClassifier:
    def __init__(self):
        pass
    
    def proba_y_given_x(self, y, x):
        proba = self.proba_y(y)
        for term in x:
            proba *= self.condprob[term][y] if term in self.terms else self.condprob[''][y]
        return proba
    
    def proba_y(self, y):
        return self.prior[y]

    def fit(self, X, y, tfidf=False):
        X_input = X.copy()
        y_input = y.copy()

        if not type(X) == np.ndarray:
            X = X.str.split(' ')
            X = np.array(X)
        if not type(y) == np.ndarray:
            y = np.array(y)
        if not X.shape[0] == y.shape[0]:
            raise Exception('number of rows in X isn\'t match with number of rows in y')

        self.prior = {}
        self.condprob = {}
        self.is_tfidf = tfidf

        self.terms = {term for row in X for term in row}
        self.classes = {cls for cls in y}

        m = X.shape[0]
        for cls in self.classes:
            self.prior[cls] = sum([1 if y[i] == cls else 0 for i in range(m)]) / m

        if self.is_tfidf:
            # Scoring TF-IDF
            tf_idf = TF_IDF()
            tf_idf.set_train_and_label(X_input.tolist(), y_input.tolist())
            tf_idf.compute_freq()
            self.scores = tf_idf.get_tf_idf_scores()

            denominator_smooth = sum([self.scores[term][cls] for term in self.terms for cls in self.classes])
            word_scores_sum = {}
            for cls in self.classes:
                word_scores_sum[cls] = sum([self.scores[term][cls] for term in self.terms])
            
            # Compute conditional probability (self.condprob)
            for term in self.terms:
                if not term in self.condprob:
                    self.condprob[term] = {}

                for cls in self.classes:                    
                    nominator = self.scores[term][cls] + 1
                    denominator = word_scores_sum[cls] + len(self.terms)
                    self.condprob[term][cls] = nominator / denominator
            # Unknown term
            self.condprob[''] = {}
            for cls in self.classes:
                self.condprob[''][cls] = 1 / (word_scores_sum[cls] + len(self.terms))
        else:
            # Word count dictionary
            word_sum = {}
            for i in range(m):
                for j in range(len(X[i])):
                    if not X[i][j] in word_sum:
                        word_sum[X[i][j]] = {}
                    if not y[i] in word_sum[X[i][j]]:
                        word_sum[X[i][j]][y[i]] = 0 
                    if not 1 - y[i] in word_sum[X[i][j]]:
                        word_sum[X[i][j]][1 - y[i]] = 0 
                    word_sum[X[i][j]][y[i]] += 1

            words_count_in_cls = {}
            for i in range(m):
                if not y[i] in words_count_in_cls:
                    words_count_in_cls[y[i]] = 0
                words_count_in_cls[y[i]] += len(X[i])

            # Compute conditional probability (self.condprob)
            for term in self.terms:
                if not term in self.condprob:
                    self.condprob[term] = {}
                for cls in self.classes:
                    nominator = word_sum[term][cls] + 1
                    denominator = words_count_in_cls[cls] + len(self.terms)
                    self.condprob[term][cls] = nominator / denominator
            # Unknown term
            self.condprob[''] = {}
            for cls in self.classes:
                self.condprob[''][cls] = 1 / (words_count_in_cls[cls] + len(self.terms))

    def predict_single(self, x):
        return 1 if self.proba_y_given_x(1, x) > self.proba_y_given_x(0, x) else 0
        
    def predict(self, X):
        X = X.str.split(' ')
        return [self.predict_single(x) for x in X]
