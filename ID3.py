from math import log
from collections import defaultdict
from TreeNode import TreeNode
import numpy as np
from DataUtils import remove_ans, dataToList

class ID3:

    def __init__(self, input_data):
        self.data_list = dataToList(input_data)
        self.answers, self.attributes = remove_ans(self.data_list, attributes_columns=True, ansLast = False)
        self.root = self.induce(list(range(len(self.attributes))), self.attributes, self.answers, list(range(len(self.answers))))
             
    def entropy(self, input):
        diffClasses = defaultdict(int)
        for a in input:
            diffClasses[a] += 1
        sum = 0
        for c in diffClasses.values():
            sum += (c/len(input))*log(c/len(input))
        return -sum

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
        
        infGains = defaultdict(float)
        for id in attribute_ids:
            infGains[id] = self.infGain(answers_values, attribute_values[id])

        infGains = self.randomize(infGains)    
        chosen_id = self.roulette(infGains)
        chosen_id = chosen_id[0]
        root = TreeNode(attribute_index=chosen_id)
        
        if attribute_ids and chosen_id in attribute_ids:
            attribute_ids.remove(chosen_id)

        new_branches = defaultdict(list)
        for idx in answers_ids:
            attr_val = full_attribute_values[chosen_id][idx]
            new_branches[attr_val].append(idx)

        for attr_val, new_answers_ids in new_branches.items():
            child_node = self.induce(attribute_ids, full_attribute_values, full_answers_values, new_answers_ids)
            root.add_child(attr_val, child_node)
        
        return root
    
    def randomize(self, infGains):

        arg_count = round(np.sqrt(len(list(infGains.keys()))))
        chosen_idx = np.random.choice(list(infGains.keys()), arg_count, False)
        chosen_args = {}
        for index in chosen_idx:
            chosen_args[index] = infGains[index]
        return chosen_args    

    def roulette(self, infGains):

        sum_of_gains = sum(infGains.values())
        if sum_of_gains == 0:
            return [max(infGains, key=infGains.get)]
        probabilities = []
        for gain in infGains.values():
            if gain > 0:
                probabilities.append(gain/sum_of_gains)
            else:
                probabilities.append(0.0)

        return np.random.choice(list(infGains.keys()), 1, False,
              p=probabilities)                                                       

    def classify(self, attributes, tree_root, default_answer=None):
        if tree_root.isLeaf:
            return tree_root.answer
        attribute_value = attributes[tree_root.attribute_index]
        if attribute_value not in tree_root.children:
            return default_answer
        return self.classify(attributes, tree_root.children[attribute_value], default_answer)

