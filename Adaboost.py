import numpy as np
import re

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def question1():
    f1 = open('DogsVsCats.test', 'r')
    train = []
    for line in f1:
        string = re.sub(' .*?:', ' ', line)
        str1 = re.sub(' ', ',', string)
        image_list = str1.strip(',').split(',')
        image_list = list(map(float, image_list))
        train.append(image_list)
    f1.close()

    f2 = open('DogsVsCats.train', 'r')
    test = []
    for line in f2:
        string = re.sub(' .*?:', ' ', line)
        str2 = re.sub(' ', ',', string)
        image_list = str2.strip(',').split(',')
        image_list = list(map(float, image_list))
        test.append(image_list)
    f2.close()

    trainy = np.array(train)[:, 0]
    trainx = np.array(train)[:, 1:]
    testy = np.array(test)[:, 0]
    testx = np.array(test)[:, 1:]

    Kf = KFold(n_splits=10, random_state=2, shuffle=True)

    svm_lin = []
    svm_pol = []
    for _, (train_index, test_index) in enumerate(Kf.split(trainy, trainx)):
        X_train_val = trainx[train_index]
        y_train_val = trainy[train_index]
        X_test_val = testx[test_index]
        y_test_val = testy[test_index]

        svm = SVC(kernel='linear')
        svm.fit(X_train_val, y_train_val)
        y_pre = svm.predict(X_test_val)
        svm_lin.append(accuracy_score(y_test_val, y_pre))  # 模型评估

        svm = SVC(kernel='poly', degree=5)
        svm.fit(X_train_val, y_train_val)
        y_pre = svm.predict(X_test_val)
        svm_pol.append(accuracy_score(y_test_val, y_pre))

    Val_acc_linear = np.mean(np.array(svm_lin))
    Val_acc_poly = np.mean(np.array(svm_pol))

    svm = SVC(kernel='linear')
    svm.fit(trainx, trainy)
    y_pre = svm.predict(trainx)
    print("linear model:\nthe accuracy of training,validation and test ")
    print(accuracy_score(trainy, y_pre))
    print(Val_acc_linear)
    y_pre = svm.predict(testx)
    print(accuracy_score(testy, y_pre))

    svm = SVC(kernel='poly', degree=5)
    svm.fit(trainx, trainy)
    y_pre = svm.predict(trainx)
    print("poly model:\nthe accuracy of training,validation and test ")
    print(accuracy_score(trainy, y_pre))
    print(Val_acc_poly)
    y_pre = svm.predict(testx)
    print(accuracy_score(testy, y_pre))


def question2and3():
    f1 = open('DogsVsCats.test', 'r')
    train = []
    for line in f1:
        string = re.sub(' .*?:', ' ', line)
        str1 = re.sub(' ', ',', string)
        image_list = str1.strip(',').split(',')
        image_list = list(map(float, image_list))
        train.append(image_list)
    f1.close()

    f2 = open('DogsVsCats.train', 'r')
    test = []
    for line in f2:
        string = re.sub(' .*?:', ' ', line)
        str2 = re.sub(' ', ',', string)
        image_list = str2.strip(',').split(',')
        image_list = list(map(float, image_list))
        test.append(image_list)
    f2.close()

    trainy = np.array(train)[:, 0]
    trainx = np.array(train)[:, 1:]
    testy = np.array(test)[:, 0]
    testx = np.array(test)[:, 1:]

    ada = AdaBoostClassifier(n_estimators=10,
                             base_estimator=SVC(kernel='poly', degree=5, probability=True))  # probability=True))
    ada.fit(trainx, trainy)
    y_pre = ada.predict(testx)
    print("k = 10,Accuracy:", accuracy_score(testy, y_pre))

    ada = AdaBoostClassifier(n_estimators=20,
                             base_estimator=SVC(kernel='poly', degree=5, probability=True))  # probability=True))
    ada.fit(trainx, trainy)
    y_pre = ada.predict(testx)
    print("k = 20,Accuracy:", accuracy_score(testy, y_pre))