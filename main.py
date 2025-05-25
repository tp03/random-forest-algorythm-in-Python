from DataUtils import divide_data
from RouletteForest import RandomForest


filePath = 'agaricus-lepiota.data'
# filePath = 'breast-cancer.data'
train_data, test_data = divide_data(filePath, 0.6, randomise=False)
forst = RandomForest(100, train_data)
forst.predict(test_data)