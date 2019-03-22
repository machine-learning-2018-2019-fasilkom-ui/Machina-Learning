import pandas as pd
from sklearn.model_selection import train_test_split

def fair_train_test_split(X, y, test_size=0.5):
    classes = {cls for cls in y}
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []

    for cls in classes:
        X_train, X_test, y_train, y_test = train_test_split(X[y == cls], y[y == cls], test_size=test_size)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    return pd.concat(X_trains, axis=0), pd.concat(X_tests, axis=0), pd.concat(y_trains, axis=0), pd.concat(y_tests, axis=0)