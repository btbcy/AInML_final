from sklearn.externals import joblib
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import sys
import util

model = joblib.load(sys.argv[1])

testData = pd.read_csv(sys.argv[2])
testX, testY = util.transfer(testData, hasY=False)

test_prob = model.predict_proba(testX)
rankTest = []
for index in range(len(testX)):
    rankTest.append([index + 1, test_prob[index][1]])
rankTest = sorted(rankTest, key = lambda x : -x[1])
rankTest = np.asarray(rankTest)
public_test = pd.DataFrame({'Rank_ID': rankTest[:, 0]})
public_test.Rank_ID = public_test.Rank_ID.astype('int')
public_test.to_csv('public1.csv', index=False)

#--------------------------
testData = pd.read_csv(sys.argv[3])
testX, testY = util.transfer(testData, hasY=False)

test_prob = model.predict_proba(testX)
rankTest = []
for index in range(len(testX)):
    rankTest.append([index + 5001, test_prob[index][1]])
rankTest = sorted(rankTest, key = lambda x : -x[1])
rankTest = np.asarray(rankTest)
public_test = pd.DataFrame({'Rank_ID': rankTest[:, 0]})
public_test.Rank_ID = public_test.Rank_ID.astype('int')
public_test.to_csv('private.csv', index=False)
