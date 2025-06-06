"""
Filename: ID3.py
Author: Tomasz Zalewski, Antoni Kowalski
Description: Implementacja algorytmu ID3 do budowy drzewa decyzyjnego.
"""
from math import log
from collections import defaultdict, Counter
from TreeNode import TreeNode
import numpy as np
from DataUtils import remove_ans, dataToList

class ID3:

    def __init__(self, input_data, classification_type):
        self.classify_method = classification_type   
        if len(input_data)>0:
            self.data_list = dataToList(input_data)
            self.answers, self.attributes = remove_ans(self.data_list, attributes_columns=True)
            self.most_common_answer = Counter(self.answers).most_common(1)[0][0]
            self.root = self.induce(list(range(len(self.attributes))), self.attributes, self.answers, list(range(len(self.answers)))) 
             
    def entropy(self, input):
        diffClasses = defaultdict(int)
        for a in input:
            diffClasses[a] += 1
        sum = 0
        for c in diffClasses.values():
            sum += (c/len(input))*log(c/len(input))
        return -sum
    
    def gini(self, input):
        counts = Counter(input)
        total = len(input)
        impurity = 1
        for count in counts.values():
            prob = count / total
            impurity -= prob ** 2
        return impurity
    
    def giniGain(self, answers, attribute_values_column):
        diffAttr = defaultdict(list)
        for i, val in enumerate(attribute_values_column):
            diffAttr[val].append(answers[i])

        weighted_gini = 0
        total_len = len(answers)
        for group in diffAttr.values():
            weighted_gini += (len(group) / total_len) * self.gini(group)

        return weighted_gini

    def branchesSetEntropy(self, attribute_values_column, answers): 
        diffAttrAnsSum = defaultdict(int)
        diffAttr = {}
        sum = 0
        for i, attr_val in enumerate(attribute_values_column):
            if attr_val not in diffAttr:
                diffAttr[attr_val] = []
            diffAttr[attr_val].append(answers[i])
            diffAttrAnsSum[attr_val] += 1
        for attr_val in diffAttr:
            sum += (diffAttrAnsSum[attr_val]/len(answers))*self.entropy(diffAttr[attr_val])
        return sum

    def infGain(self, answers, attribute_values_column): 
        return self.entropy(answers) - self.branchesSetEntropy(attribute_values_column, answers)

    def getUsedAttrAndAnswers(self, answers_ids, answers_values, attribute_values):
        attribute_vals = []
        answers_vals = []
        for id in answers_ids:
            answers_vals.append(answers_values[id])

        for col in attribute_values:
            filtered_col = [col[id] for id in answers_ids]
            attribute_vals.append(filtered_col)

        return attribute_vals, answers_vals

    def induce(self, attribute_ids,full_attribute_values, full_answers_values, answers_ids): 
        attribute_values, answers_values = self.getUsedAttrAndAnswers(answers_ids, full_answers_values, full_attribute_values)
        if len(set(answers_values)) == 1:
            return TreeNode(isLeaf=True, answer=answers_values[0])
        if not attribute_ids:
            most_common_answer = max(set(answers_values), key=answers_values.count)
            return TreeNode(isLeaf=True, answer=most_common_answer)
        
        gains = defaultdict(float)
        for id in attribute_ids:
            if self.classify_method == "entropy":
                gains[id] = self.infGain(answers_values, attribute_values[id])
            if self.classify_method == "gini":
                gains[id] = self.giniGain(answers_values, attribute_values[id])    

        gains = self.randomize(gains)    
        chosen_id = self.roulette(gains)
        root = TreeNode(attribute_index=chosen_id)
        
        new_attribute_ids = attribute_ids.copy()
        if attribute_ids and chosen_id in new_attribute_ids:
            new_attribute_ids.remove(chosen_id)

        new_branches = defaultdict(list)
        for idx in answers_ids:
            attr_val = full_attribute_values[chosen_id][idx]
            new_branches[attr_val].append(idx)

        for attr_val, new_answers_ids in new_branches.items():
            child_node = self.induce(new_attribute_ids, full_attribute_values, full_answers_values, new_answers_ids)
            root.add_child(attr_val, child_node)
        
        return root
    
    def randomize(self, infGains):
        arg_count = max(1, round(np.sqrt(len(infGains))))
        chosen_idx = np.random.choice(list(infGains.keys()), arg_count, False)
        chosen_args = defaultdict(float)
        for index in chosen_idx:
            chosen_args[index] = infGains[index]
        return chosen_args    

    def roulette(self, gains):
        sum_of_gains = sum(gains.values())
        if sum_of_gains == 0:
            return np.random.choice(list(gains.keys()), 1)[0]
        probabilities = []
        for gain in gains.values():
            if gain > 0 and self.classify_method == "entropy":
                probabilities.append(gain/sum_of_gains)
            elif gain > 0 and self.classify_method == "gini":
                probabilities.append(sum_of_gains/gain)   
            else:
                probabilities.append(0.0)
        probabilities = [p/sum(probabilities) for p in probabilities]
        return np.random.choice(list(gains.keys()), 1, False,
              p=probabilities)[0]                                                       

    def classify(self, attributes, tree_root):
        if tree_root.isLeaf:
            return tree_root.answer
        attribute_value = attributes[tree_root.attribute_index]
        if attribute_value not in tree_root.children:
            return self.most_common_answer
        return self.classify(attributes, tree_root.children[attribute_value])

