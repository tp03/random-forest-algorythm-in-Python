import random

nursery_categories = {
    0: ['usual', 'pretentious', 'great_pret'],
    1: ['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
    2: ['complete', 'completed', 'incomplete', 'foster'],
    3: ['1', '2', '3', 'more'],
    4: ['convenient', 'less_conv', 'critical'],
    5: ['convenient', 'inconv'],
    6: ['nonprob', 'slightly_prob', 'problematic'],
    7: ['recommended', 'priority', 'not_recom'],
    8: ['recommend\n', 'priority\n', 'not_recom\n', 'very_recom\n', 'spec_prior\n']
}

cars_categories = {
    0: ['low', 'med', 'high', 'vhigh'],
    1: ['low', 'med', 'high', 'vhigh'],
    2: ['2', '3', '4', '5more'],
    3: ['2', '4', 'more'],
    4: ['small', 'med', 'big'],
    5: ['low', 'med', 'high'],
    6: ['unacc\n', 'acc\n', 'good\n', 'vgood\n']
}

cancer_categories = {
    0: ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
    1: ['lt40', 'ge40', 'premeno'],
    2: ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'],
    3: ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'],
    4: ['yes', 'no', '?'],
    5: ['1', '2', '3'],
    6: ['left', 'right'],
    7: ['left_up', 'left_low', 'right_up', 'right_low', 'central', '?'],
    8: ['yes', 'no'],
    9: ['no-recurrence-events\n', 'recurrence-events\n']
}


def data_number_fix(names_dict, file_name):
    with open(f"{file_name}.data", 'r') as file_handle:
        with open(f"{file_name}_fixed.data", 'w') as write_handle:
            for line in file_handle:
                new_line = ""
                args = line.split(',')
                for index, arg in enumerate(args):
                    for idx, value in enumerate(names_dict[index]):
                        if arg == value:
                            new_line += str(idx)
                            if index < len(args) - 1:
                                new_line += ","
                            else:
                                new_line += "\n"
                            break        
                write_handle.write(new_line)                                 

def remove_ans(data,attributes_columns=True):
    answers = []
    attributes = []
    for line in data:
        answers.append(line[-1])
        attributes.append(line[:-1])
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

def divide_data(data_file_name, seed, train_ratio=0.6, randomise=False):
    with open(data_file_name, 'r') as f:
        data = f.readlines()
    
    if randomise:
        random.seed(seed)
        random.shuffle(data)

    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    return train_data, test_data

def prepare_data(data):
    data = dataToList(data)
    return remove_ans(data, attributes_columns=False)