from test_svm import read_data
from sklearn.model_selection import train_test_split
import numpy as np
import setting
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from random import shuffle
from statistics import mean
from sklearn.model_selection import StratifiedKFold
# [0.79166667 0.79166667 0.73913043 0.73913043 0.69565217]
# ['1' '2' '11' '9' '2' '11' '1' '3' '2' '20' '12' '21' '19' '22']
# ['1', '2', '5', '9', '2', '10', '1', '3', '4', '7', '12', '18', '19', '22']
if __name__ == '__main__':
    tranform = read_data.Tranform_data()

    X, y = tranform.get_train_data(setting.TRAIN_PATH)
    kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=100)
    cvscores = []
    acc = []
    c = list(zip(X, y))

    shuffle(c)

    X, y = zip(*c)
    for train, test in kfold.split(X, y):
        X_train, y_train = [X[train[int(idx)]] for idx in range(len(train))], [y[train[int(idx)]] for idx in
                                                                               range(len(train))]
        X_test, y_test = [X[test[int(idx)]] for idx in range(len(test))], [y[test[int(idx)]] for idx in
                                                                           range(len(test))]
        X_train, Y_train, vectornizer = tranform.conv_train_data(X_train, y_train)
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_test = vectornizer.transform(X_test)
        clf = LinearSVC()
        clf.fit(X_train, Y_train)
        t = clf.decision_function(X_test)
        t = t.argsort()
        print(clf.classes_)

        Y_pred = clf.predict(X_test)
        print(Y_pred)
        print(y_test)
        print(accuracy_score(y_test, Y_pred))
        acc.append(accuracy_score(y_test, Y_pred))
    print(mean(acc))
    # c = list(zip(X_train, Y_train))
    #
    # shuffle(c)
    #
    # X_train, Y_train = zip(*c)
    # clf = LinearSVC()
    # scores = cross_val_score(clf, X_train, Y_train, cv=7)
    # print(scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #
    # X_test_raw, Y_test = tranform.get_train_data(setting.TEST_PATH)
    # X_test = vectornizer.transform(X_test_raw)
    #
    # clf = LinearSVC()
    # clf.fit(X_train, Y_train)
    # Y_pred = clf.predict(X_test)
    # print(Y_pred)
    # print(Y_test)
    # with open('result.csv', 'w') as f:
    #     f.write('label,id\n')
    #     for i in range(len(X_test_raw)):
    #         id = 'test_' + ''.join(['0'] * (6 - len(str(i)))) + str(i)
    #         class_id = str(Y_pred[i])
    #         f.write(id + ',' + class_id + '\n')

