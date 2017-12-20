from sklearn import ensemble
from sklearn.externals import joblib
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import sys
import util

trainData = pd.read_csv(sys.argv[1])
dataX, dataY= util.transfer(trainData, hasY=True)

model = list()
# model.append(ensemble.RandomForestClassifier(criterion='entropy', n_estimators=130, max_depth=12, oob_score=True, random_state=10))
model.append(ensemble.ExtraTreesClassifier(criterion='entropy', n_estimators=98, max_depth=16, bootstrap=True, oob_score=True, random_state=10))
model.append(ensemble.GradientBoostingClassifier(n_estimators=126, max_depth=4, max_leaf_nodes=30, random_state=10))
model.append(xgb.XGBClassifier(n_estimators=158, max_depth=4))
model.append(lgb.LGBMClassifier(n_estimators=125, max_depth=5, num_leaves=100))
model.append(ensemble.VotingClassifier(estimators=[('et', model[0]), ('gdb', model[1]), ('xgb', model[2]), ('lgb', model[3])], voting='soft', weights=[1, 1, 1, 1]))

for i in range(len(model)):
    model[i].fit(dataX, dataY)

joblib.dump(model[len(model)-1], 'model.m')

