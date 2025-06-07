"""
Filename: AlgorythmRunner.py
Author: Tomasz Zalewski, Antoni Kowalski
Description: Uruchamianie wielokrotnie wybranego algorytmu oraz prezentacja wyników.
"""

from DataUtils import divide_data, prepare_data
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from RouletteForest import RandomForest
import numpy as np
import time
import os


def run_made_algorythm(
    data_name, re_runs, tree_count, seeds, type, use_train_data=False
):

    filePath = os.path.expanduser(f"./data/{data_name}.data")
    scores = []

    t = time.time()

    for i in range(re_runs):
        train_data, test_data = divide_data(filePath, seeds[i], 0.6, randomise=True)
        forest = RandomForest(tree_count, train_data, type)
        if use_train_data is True:
            # użycie zbioru treningowego dla testu przeuczenia
            scores.append(forest.predict(train_data))
        else:
            scores.append(forest.predict(test_data))

    elapsed = time.time() - t

    print("Our RF:\n")
    print(f"Min: {min(scores)}\n")
    print(f"Max: {max(scores)}\n")
    print(f"Mean: {np.mean(scores)}\n")
    print(f"Std: {np.std(scores)}\n")
    print(f"Time: {elapsed}")


def run_sklearn_algorythms(data_name, re_runs, tree_count, seeds, type):

    filePath = os.path.expanduser(f"./data/{data_name}_fixed.data")

    scores_rf = []
    scores_svm = []

    t_rf = time.time()

    for i in range(re_runs):
        train_data, test_data = divide_data(filePath, seeds[i], 0.6, randomise=True)
        train_answers, train_attributes = prepare_data(train_data)
        test_answers, test_attributes = prepare_data(test_data)

        forest = RandomForestClassifier(n_estimators=tree_count, criterion=type)
        forest.fit(train_attributes, train_answers)
        forest_pred = forest.predict(test_attributes)

        correct_rf = 0
        for i in range(len(forest_pred)):
            if forest_pred[i] == test_answers[i]:
                correct_rf += 1
        scores_rf.append(correct_rf / len(forest_pred))

    elapsed_rf = time.time() - t_rf

    print("Sklearn RF:\n")
    print(f"Min: {min(scores_rf)}\n")
    print(f"Max: {max(scores_rf)}\n")
    print(f"Mean: {np.mean(scores_rf)}\n")
    print(f"Std: {np.std(scores_rf)}\n")
    print(f"Time: {elapsed_rf}")

    t_svm = time.time()

    for i in range(re_runs):
        train_data, test_data = divide_data(filePath, seeds[i], 0.6, randomise=True)
        train_answers, train_attributes = prepare_data(train_data)
        test_answers, test_attributes = prepare_data(test_data)

        clf = svm.SVC()
        clf.fit(train_attributes, train_answers)
        svm_pred = clf.predict(test_attributes)
        correct_svm = 0
        for i in range(len(forest_pred)):
            if svm_pred[i] == test_answers[i]:
                correct_svm += 1
        scores_svm.append(correct_svm / len(svm_pred))

    elapsed_svm = time.time() - t_svm

    print("Sklearn SVM:\n")
    print(f"Min: {min(scores_svm)}\n")
    print(f"Max: {max(scores_svm)}\n")
    print(f"Mean: {np.mean(scores_svm)}\n")
    print(f"Std: {np.std(scores_svm)}\n")
    print(f"Time: {elapsed_svm}")
