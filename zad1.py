
from math import log
from collections import defaultdict
import random

def divide_data(data_file_name, train_ratio=0.6, randomise=False):
    with open(data_file_name, 'r') as f:
        data = f.readlines()
    
    if randomise:
        random.shuffle(data)

    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    return train_data, test_data

# train_data, test_data = divide_data('agaricus-lepiota.data')
# split_data('agaricus-lepiota.data')

class TreeNode:
    def __init__(self, attribute_index=None, isLeaf=False, answer=None):
        self.attribute_index = attribute_index
        self.isLeaf = isLeaf
        self.answer = answer
        self.children = {}

    def add_child(self, attribute_value, child_node):
        self.children[attribute_value] = child_node

def entropy(input):
    diffClasses = defaultdict(int)
    for a in input:
        diffClasses[a] += 1
    sum = 0
    for c in diffClasses.values():
        sum += (c/len(input))*log(c/len(input))
    return -sum

def branchesSetEntropy(attribute_values_column, answers): # attributes are a list of chars just like answers
    diffAttrAnsSum= defaultdict(int)
    diffAttr = {}
    sum = 0
    for i, attr_val in enumerate(attribute_values_column):
        if attr_val not in diffAttr:
            diffAttr[attr_val] = []
        diffAttr[attr_val].append(answers[i])
        diffAttrAnsSum[attr_val] += 1
    for attr_val in diffAttr:
        sum += (diffAttrAnsSum[attr_val]/len(answers))*entropy(diffAttr[attr_val])
    return sum

def infGain(answers, attribute_values_column): # attributes are one column of attributes
    return entropy(answers) - branchesSetEntropy(attribute_values_column, answers)

def remove_ans(data,attributes_columns=True, ansLast=True):
    answers = []
    attributes = []
    for line in data:
        if ansLast:
            answers.append(line[-1])
            attributes.append(line[:-1])
        else:
            answers.append(line[0])
            attributes.append(line[1:])
    if not attributes_columns:
        return answers, attributes
    transposed_attributes = [list(col) for col in zip(*attributes)]
    return answers, transposed_attributes

def dataToList(data):
    data_list = []
    for line in data:
        line = line.split(',')
        data_list.append(line)
    return data_list

def getUsedAttrAndAnswers(answers_ids, answers_values, attribute_values):
    attribute_vals = []
    answers_vals = []
    for id in answers_ids:
        answers_vals.append(answers_values[id])

    for col in attribute_values:
        filtered_col = [col[id] for id in answers_ids]
        attribute_vals.append(filtered_col)

    return attribute_vals, answers_vals

def ID3(attribute_ids,full_attribute_values, full_answers_values, answers_ids): # only answers in answers_ids count
    attribute_values, answers_values = getUsedAttrAndAnswers(answers_ids, full_answers_values, full_attribute_values)
    if len(set(answers_values)) == 1:
        return TreeNode(isLeaf=True, answer=answers_values[0])
    if not attribute_ids:
        most_common_answer = max(set(answers_values), key=answers_values.count)
        return TreeNode(isLeaf=True, answer=most_common_answer)
    
    infGains = defaultdict(float)
    for id in attribute_ids:
        infGains[id] = infGain(answers_values, attribute_values[id])
        
    max_id = max(infGains, key=infGains.get)
    root = TreeNode(attribute_index=max_id)
    
    if attribute_ids and max_id in attribute_ids:
        attribute_ids.remove(max_id)

    new_branches = defaultdict(list)
    for idx in answers_ids:
        attr_val = full_attribute_values[max_id][idx]
        new_branches[attr_val].append(idx)

    for attr_val, new_answers_ids in new_branches.items():
        child_node = ID3(attribute_ids, full_attribute_values, full_answers_values, new_answers_ids)
        root.add_child(attr_val, child_node)

    return root

def ID3_init(data):
    data_list = dataToList(data)
    answers, attributes = remove_ans(data_list,attributes_columns=True, ansLast = True)
    return ID3(list(range(len(attributes))), attributes, answers, list(range(len(answers))))

def classify(attributes, tree, default_answer=None):
    if tree.isLeaf:
        return tree.answer
    attribute_value = attributes[tree.attribute_index]
    if attribute_value not in tree.children:
        return default_answer
    return classify(attributes, tree.children[attribute_value], default_answer)

filePath = 'nursery.data'
train_data, test_data = divide_data(filePath, 0.75, randomise=True)
tree = ID3_init(train_data)

test_data = dataToList(test_data)
test_answers, test_attributes = remove_ans(test_data, attributes_columns=False, ansLast = True)

correct = 0
for i in range(len(test_attributes)):
    result = classify(test_attributes[i], tree, ' <=50K')
    print(f"Expected: {test_answers[i]}, got: {result}")
    if result == test_answers[i]:
        correct += 1
print(f"Accuracy: {correct/len(test_attributes)}")
