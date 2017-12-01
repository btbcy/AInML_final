from sklearn.externals import joblib
from sklearn import cross_validation, metrics
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import sys
import util

model = joblib.load(sys.argv[1])

testData= pd.read_csv(sys.argv[2])
testX, testY = util.transfer(testData, hasY=True)

print model.feature_importances_
print "score\t", model.score(testX, testY)
test_prob = model.predict_proba(testX)
print "auc\t", metrics.roc_auc_score(testY, test_prob[:, 1])

rankTest = []
for index in range(len(testX)):
    rankTest.append([index + 1, test_prob[index][1], testY[index]])
rankTest = sorted(rankTest, key = lambda x : x[1])
rankTest.reverse()
print "kaggle\t", util.avpre(rankTest, 500)
