from DataUtils import divide_data, dataToList, remove_ans
from RouletteForest import RandomForest
from ucimlrepo import fetch_ucirepo 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# create banknote authentication data file
# banknote_authentication = fetch_ucirepo(id=267) 
# data_file_maker(banknote_authentication, "banknote") 

# create banknote authentication data file
# banknote_authentication = fetch_ucirepo(id=267) 
# data_file_maker(banknote_authentication, "banknote") 

filePath = 'nursery.data'
train_data, test_data = divide_data(filePath, 0.75, randomise=True)
forst = RandomForest(30, train_data)

forst.predict(test_data)

filePath = 'nursery_fixed.data'
train_data, test_data = divide_data(filePath, 0.75, randomise=True)  

train_data = dataToList(train_data)
train_answers, train_attributes = remove_ans(train_data, attributes_columns=False, ansLast = True)

test_data = dataToList(test_data)
test_answers, test_attributes = remove_ans(test_data, attributes_columns=False, ansLast = True)

forest = RandomForestClassifier(n_estimators=30)
forest.fit(train_attributes, train_answers)
pred = forest.predict(test_attributes)
correct = 0
for i in range(len(pred)):
    if pred[i] == test_answers[i]:
        correct += 1      
print(f"sklearn RF accuracy: {correct/len(pred)}")

clf = svm.SVC()
clf.fit(train_attributes, train_answers)
svm_clf = clf.predict(test_attributes)
correct = 0
for i in range(len(svm_clf)):
    if svm_clf[i] == test_answers[i]:
        correct += 1      
print(f"sklearn SVM accuracy: {correct/len(svm_clf)}")