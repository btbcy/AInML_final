from sklearn import ensemble
from sklearn.externals import joblib
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

model = ensemble.RandomForestClassifier(n_estimators = 100, oob_score = True, random_state = 10)
# model = ensemble.GradientBoostingClassifier(n_estimators = 100, random_state = 10)
model.fit(trainX, trainY)
joblib.dump(model, sys.argv[2])
