import time
import pandas as pd
from sklearn.metrics import accuracy_score

from MNB_TextClassifier import MNB_TextClassifier
from utils import fair_train_test_split
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('clean_dataset.csv')
    X = df['Teks']
    y = df['label']
    X_train, X_test, y_train, y_test = fair_train_test_split(X, y, test_size=0.3)

    t1 = time.time()
    clf_without_tf_idf = MNB_TextClassifier()
    clf_without_tf_idf.fit(X_train, y_train, tfidf=False)
    y_pred = clf_without_tf_idf.predict(X_test)
    t2 = time.time()
    print('Accuracy of MNB_TextClassifier without TF-IDF : ', accuracy_score(y_pred, y_test))
    print('Time elapsed :', t2 - t1)
    
    print()

    t1 = time.time()
    clf_with_tf_idf = MNB_TextClassifier()
    clf_with_tf_idf.fit(X_train, y_train, tfidf=True)
    y_pred = clf_with_tf_idf.predict(X_test)
    t2 = time.time()
    print('Accuracy of MNB_TextClassifier with TF-IDF : ', accuracy_score(y_pred, y_test))
    print('Time elapsed :', t2 - t1)
