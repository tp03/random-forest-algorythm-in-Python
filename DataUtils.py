import random

def remove_ans(data,attributes_columns=True, ansLast=False):
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

def divide_data(data_file_name, train_ratio=0.6, randomise=False):
    with open(data_file_name, 'r') as f:
        data = f.readlines()
    
    if randomise:
        random.shuffle(data)

    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    return train_data, test_data