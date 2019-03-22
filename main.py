import pandas as pd
from sklearn.metrics import accuracy_score

from MNB_TextClassifier import MNB_TextClassifier
from utils import fair_train_test_split

if __name__ == '__main__':
    df = pd.read_csv('clean_dataset.csv')
    X = df['Teks']
    y = df['label']
    X_train, X_test, y_train, y_test = fair_train_test_split(X, y, test_size=0.1)

    clf = MNB_TextClassifier()
    clf.fit(X_train, y_train, tfidf=True)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_pred, y_test))
