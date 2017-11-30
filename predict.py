from sklearn.externals import joblib
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import sys
import util

model = joblib.load(sys.argv[1])


testData= pd.read_csv(sys.argv[2])

# testData = testData.as_matrix()
# testX = testData[:, 1:-2]
# testY = testData[:, -1]
testX, testY = util.transfer(testData, hasY=True)

print model.feature_importances_
print model.score(testX, testY)
test_prob = model.predict_proba(testX)
rankTest = []
for index in range(len(testX)):
    rankTest.append([index + 1, test_prob[index][1], testY[index]])
rankTest = sorted(rankTest, key = lambda x : x[1])
rankTest.reverse()
print util.avpre(rankTest, 500)
