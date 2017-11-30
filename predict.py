from sklearn.externals import joblib
import sys
import csv
import numpy as np
import util

model = joblib.load(sys.argv[1])

# readfile
testfile = open(sys.argv[2], 'r')
reader = csv.reader(testfile)
csvHeader = reader.next()
testData = []
for row in reader:
    row[:] = [int(x) for x in row]
    testData.append(row)
testfile.close()

data2test_x, data2test_y = util.transfer(testData[:5000], True)
data2test_x= np.array(data2test_x)
data2test_y= np.array(data2test_y)

print model.feature_importances_
print model.score(data2test_x, data2test_y)
test_prob = model.predict_proba(data2test_x)
rankTest = []
for index in range(len(data2test_x)):
    rankTest.append([index + 1, test_prob[index][1], data2test_y[index]])
rankTest = sorted(rankTest, key = lambda x : x[1])
rankTest.reverse()
print util.avpre(rankTest, 500)
