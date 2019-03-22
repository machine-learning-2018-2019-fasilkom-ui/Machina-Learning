import pandas as pd
from sklearn.metrics import accuracy_score

from MNB_TextClassifier import MNB_TextClassifier
from utils import fair_train_test_split

if __name__ == '__main__':
    df = pd.read_csv('clean_dataset.csv')
    X = df['Teks']
    y = df['label']
    X_train, X_test, y_train, y_test = fair_train_test_split(X, y, test_size=0.1)

    clf_without_tf_idf = MNB_TextClassifier()
    clf_without_tf_idf.fit(X_train, y_train, tfidf=False)
    y_pred = clf_without_tf_idf.predict(X_test)
    print('Accuracy of MNB_TextClassifier without TF-IDF : ', accuracy_score(y_pred, y_test))
    
    clf_with_tf_idf = MNB_TextClassifier()
    clf_with_tf_idf.fit(X_train, y_train, tfidf=True)
    y_pred = clf_with_tf_idf.predict(X_test)
    print('Accuracy of MNB_TextClassifier with TF-IDF : ', accuracy_score(y_pred, y_test))
