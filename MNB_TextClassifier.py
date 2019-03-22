import numpy as np

class MNB_TextClassifier:
    def __init__(self):
        pass

    def __scoring(self, X, y):
        pass
    
    def proba_y_given_x(self, y, x):
        proba = self.proba_y(y)
        for term in x:
            proba *= self.condprob[term][y] if term in self.terms else self.condprob[''][y]
        return proba
    
    def proba_y(self, y):
        return self.prior[y]

    def fit(self, X, y, tfidf=False):
        if not type(X) == np.ndarray:
            X = np.array(X)
        if not type(y) == np.ndarray:
            y = np.array(y)
        if not X.shape[0] == y.shape[0]:
            raise Exception

        self.prior = {}
        self.condprob = {}
        self.score = {}
        self.is_tfidf = tfidf

        self.terms = {term for row in X for term in row}
        self.classes = {cls for cls in y}

        m = X.shape[0]
        for cls in self.classes:
            self.prior[cls] = sum([1 if y[i] == cls else 0 for i in range(m)]) / m

        if self.is_tfidf:
            # Scoring TF-IDF
            pass
            # Compute conditional probability (self.condprob)
            pass
        else:
            for term in self.terms:
                if not term in self.condprob:
                    self.condprob[term] = {}
                for cls in self.classes:
                    nominator = sum([1 if X[i][j] == term and y[i] == cls else 0 for i in range(m) for j in range(len(X[i]))]) + 1
                    denominator = sum([len(X[i]) if y[i] == cls else 0 for i in range(m)]) + len(self.terms)
                    self.condprob[term][cls] = nominator / denominator
            # Unknown term
            self.condprob[''] = {}
            for cls in self.classes:
                self.condprob[''][cls] = 1 / (sum([len(X[i]) if y[i] == cls else 0 for i in range(m)]) + len(self.terms))

    def predict_single(self, x):
        return 1 if self.proba_y_given_x(1, x) > self.proba_y_given_x(0, x) else 0
        
    def predict(self, X):
        return [self.predict_single(x) for x in X]
