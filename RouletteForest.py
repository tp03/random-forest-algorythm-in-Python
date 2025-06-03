from ID3 import ID3
import numpy as np
from DataUtils import dataToList, remove_ans
from collections import defaultdict

class RandomForest:

    def __init__(self, tree_count, data):

        self.count = tree_count
        self.data = data
        self.tree_list = []
        self.bootstrap_bagging()

    def bootstrap_bagging(self):
        n = len(self.data)
        for i in range(self.count):
            chosen_data = np.random.choice(self.data, n, True)
            self.tree_list.append(ID3(chosen_data))

    def vote(self, data_sample):
        votes = defaultdict(int)
        for tree in self.tree_list:
            result = tree.classify(data_sample, tree.root)
            votes[result] += 1
        return max(votes, key=votes.get)    

    def predict(self, test_data):
        test_data = dataToList(test_data)
        test_answers, test_attributes = remove_ans(test_data, attributes_columns=False, ansLast = True)
        correct = 0
        for i in range(len(test_attributes)):
            result = self.vote(test_attributes[i])
            print(f"Expected: {test_answers[i]}, got: {result}")
            if result == test_answers[i]:
                correct += 1
        print(f"Accuracy: {correct/len(test_attributes)}")