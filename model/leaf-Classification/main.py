import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn import preprocessing

from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),

]

def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)
    classes = list(le.classes_)
    test_ids = test.id

    train = train.drop(['species', 'id'], axis=1)
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes


def get_data():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    train, labels, test, test_ids, classes = encode(train, test)
    return train, labels, test, test_ids, classes

def model_selection():
    train, labels, test, test_ids, classes = get_data()
    n_samples = len(train)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, val_index in sss.split(np.zeros(n_samples), labels):
        X_train, X_val = train.values[train_index], train.values[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        X_train = preprocessing.scale(X_train)
        X_val = preprocessing.scale(X_val)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__

        print("="*30)
        print(name)

        print('****Results****')
        train_predictions = clf.predict(X_val)
        acc = accuracy_score(y_val, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        prob_predictions = clf.predict_proba(X_val)
        l1 = log_loss(y_val, prob_predictions)
        print("Log Loss: {}".format(l1))


    print("="*30)

def submission():
    favorite_clf = KNeighborsClassifier()
    train, labels, test, test_ids, classes = get_data()
    X_train = train.values
    y_train = labels
    favorite_clf.fit(X_train, y_train)

    test_predictions = favorite_clf.predict_proba(test)
    submission = pd.DataFrame(test_predictions, columns=classes)
    submission.insert(0, 'id', test_ids)
    submission.reset_index()

    submission.to_csv('submission.csv', index = False)




if __name__ == '__main__':
    submission()
