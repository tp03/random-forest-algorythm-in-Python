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


def nursery_fix():
    with open("nursery.data", 'r') as file_handle:
        with open("nursery_fixed.data", 'w') as write_handle:
            for line in file_handle:
                new_line = ""
                args = line.split(',')
                for index, arg in enumerate(args):
                    for idx, value in enumerate(nursery_categories[index]):
                        if arg == value:
                            new_line += str(idx)
                            if index < len(args) - 1:
                                new_line += ","
                            else:
                                new_line += "\n"
                            break        
                write_handle.write(new_line)                                 

nursery_fix()

def data_file_maker(ucirepo, name):
    X = ucirepo.data.features 
    y = ucirepo.data.targets

    with open(f"{name}.data", 'w') as file_handle:
        for i in range(len(X)):
            x = X.iloc[i].to_list()        
            for argument in x:
                file_handle.write(f"{argument},")
            file_handle.write(f"{y.iloc[i].to_list()[0]}")
            file_handle.write('\n')
    file_handle.close()    

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