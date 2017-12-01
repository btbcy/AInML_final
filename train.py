from sklearn import ensemble
from sklearn import linear_model
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import xgboost as xgb
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import numpy as np
import sys
import util

trainData = pd.read_csv(sys.argv[1])
# print trainData.info()
# print trainData.describe()

# fig = plt.figure()
# fig.set(alpha = 0.2)

# plt.subplot2grid((1, 4), (0, 0))
# plt.scatter(trainData.Y, trainData.AGE)
# plt.ylabel(u"AGE")
# plt.grid(b = True, which = 'major', axis = 'y')
# plt.subplot2grid((1, 4), (0, 1))
# trainData.Y.value_counts().plot(kind = 'bar')
# plt.title('Y')
# plt.subplot2grid((1, 4), (0, 2), colspan = 2)
# trainData.AGE[trainData.EDUCATION == 3].plot(kind='kde')
# trainData.AGE[trainData.EDUCATION == 4].plot(kind='kde')
# trainData.AGE[trainData.EDUCATION == 5].plot(kind='kde')
# trainData.AGE[trainData.EDUCATION == 6].plot(kind='kde')
# plt.xlabel(u"AGE")
# plt.legend((u'edu3', u'edu4', u'edu5', u'edu6'), loc='best')
# plt.show()

# Y_0 = trainData.MARRIAGE[trainData.Y == 0].value_counts()
# Y_1 = trainData.MARRIAGE[trainData.Y == 1].value_counts()
# df = pd.DataFrame({u'Y=1':Y_1, u'Y=0':Y_0})
# df.plot(kind = 'bar', stacked=True)
# plt.show()

# trainData = trainData.as_matrix()
# trainX = trainData[:, 1:-2]
# trainY = trainData[:, -1]
trainX, trainY = util.transfer(trainData, hasY=True)

# param_test1 = {'max_depth':range(3, 14, 2), 'min_samples_split':range(50, 201, 20)}
# gsearch1 = GridSearchCV(estimator=ensemble.RandomForestClassifier(n_estimators=130, criterion='entropy', oob_score=True, random_state=10), param_grid=param_test1, iid=False, scoring='roc_auc', cv=5)
# gsearch1.fit(trainX, trainY)
# print gsearch1.grid_scores_
# print gsearch1.best_params_, gsearch1.best_score_

# model = ensemble.RandomForestClassifier(criterion = 'entropy', n_estimators = 130, min_samples_split = 130, max_depth = 13, max_features = 'auto', oob_score = True, random_state = 10)
# model = ensemble.GradientBoostingClassifier(n_estimators = 100, random_state = 10)
# model = ensemble.ExtraTreesClassifier(criterion = 'entropy', n_estimators = 130, bootstrap=True, oob_score = True, random_state = 10)
model = xgb.XGBClassifier(n_estimators = 150)
# model = ensemble.AdaBoostClassifier(n_estimators = 50, random_state = 10)
model.fit(trainX, trainY)
# joblib.dump(model, sys.argv[2])

#-------------------------------------------------
testData= pd.read_csv("toTest.csv")
testX, testY = util.transfer(testData, hasY=True)

# numRounds = 300
# params = {'booster':'gbtree'}
# xgbtrain = xgb.DMatrix(trainX, trainY)
# xgbtest = xgb.DMatrix(testX, testY)
# watchlist = [(xgbtrain, 'train'), (xgbtest, 'val')]
# model = xgb.train(params, dtrain=xgbtrain, num_boost_round=numRounds, evals=watchlist, early_stopping_rounds = 100)
# print model.get_score()

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
