# from sklearn import svm
# from sklearn import neighbors
# from sklearn import grid_search
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.externals import joblib
import numpy as np
import csv
import sys
import util
import random

# read from Train.csv
trainFile = open(sys.argv[1], 'r')
reader = csv.reader(trainFile)
csvHeader = reader.next()
trainData = []
for row in reader:
    row[:] = [int(x) for x in row]
    trainData.append(row)
trainFile.close()

rowSize = len(trainData[0])
totalSize = len(trainData)

spID = random.sample(range(20000), 5000)
aaa = [csvHeader]
bbb = [csvHeader]
for i in range(20000):
    if i in spID:
        aaa.append(trainData[i])
    else:
        bbb.append(trainData[i])
xfile = open("toTest.csv", 'w')
csvCursor = csv.writer(xfile)
for i in range(len(aaa)):
    csvCursor.writerow(aaa[i])
xfile.close()
xfile = open("toTrain.csv", 'w')
csvCursor = csv.writer(xfile)
for i in range(len(bbb)):
    csvCursor.writerow(bbb[i])
xfile.close()


data2train_x, data2train_y = util.transfer(trainData[5001:], True)
data2train_x = np.array(data2train_x)
data2train_y = np.array(data2train_y)

# scaler = preprocessing.StandardScaler().fit(data2train_x)
# data2train_x = scaler.transform(data2train_x)

# model = ensemble.RandomForestClassifier(n_estimators = 200, max_features = None, oob_score = True, random_state = 10)
model = ensemble.RandomForestClassifier(n_estimators = 1000, oob_score = True, random_state = 10)
model.fit(data2train_x, data2train_y)
joblib.dump(model, sys.argv[2])

# param_test1 = {'n_estimators':range(250, 321, 10)}
# gsearch1 = grid_search.GridSearchCV(estimator = ensemble.RandomForestClassifier(random_state = 0), param_grid = param_test1, cv=5)
# gsearch1.fit(data2train_x, data2train_y)
# print gsearch1.grid_scores_ 
# print gsearch1.best_params_, gsearch1.best_score_


# data2test_x, data2test_y = util.transfer(trainData[:5000], True)
# data2test_x= np.array(data2test_x)
# data2test_y= np.array(data2test_y)
# data2test_x = scaler.transform(data2test_x)
# print model.score(data2test_x, data2test_y)


# temp = sorted(trainData, key = lambda x: x[5])
# temp.reverse()
# print avpre(temp, 235)
# print partial(temp, 235)

# xfile = open(sys.argv[2], 'w')
# csvCursor = csv.writer(xfile)
# csvCursor.writerow(csvHeader)
# temp = sorted(trainData, key = lambda x: x[3])
# temp.reverse()
# for index in range(5000):
#     csvCursor.writerow(temp[index])
# xfile.close()
