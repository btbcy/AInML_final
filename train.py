from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
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
import warnings
warnings.filterwarnings('ignore')

def get_oof(clf, x_train, y_train, x_test):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=10)
    train_iter = np.zeros((x_train.shape[0], )) # 1 * 15000
    test_iter = np.zeros((x_test.shape[0], ))   # 1 * 5000
    test_skf = np.empty((5, x_test.shape[0]))   # 5 * 5000

    for idx, (train_idx, test_idx) in enumerate(kf.split(x_train)):
        kf_x_train = x_train[train_idx] # 12000 * feature
        kf_y_train = y_train[train_idx] # 12000 * 1
        kf_x_test = x_train[test_idx]   # 3000 * feature

        clf.fit(kf_x_train, kf_y_train)
        train_iter[test_idx] = clf.predict_proba(kf_x_test)[:, 1] # 1* 3000
        test_skf[idx, :] = clf.predict_proba(x_test)[:, 1] # 1 * 5000

    test_iter[:] = test_skf.mean(axis=0)   # 1 * 5000
    train_iter = train_iter.reshape(-1, 1) # 15000 * 1
    test_iter = test_iter.reshape(-1, 1)   # 5000 * 1
    return train_iter, test_iter

trainData = pd.read_csv(sys.argv[1])

trainX, trainY = util.transfer(trainData, hasY=True)

# param_test1 = {'max_depth':range(3, 14, 2), 'min_samples_split':range(50, 201, 20)}
# gsearch1 = GridSearchCV(estimator=ensemble.RandomForestClassifier(n_estimators=130, criterion='entropy', oob_score=True, random_state=10), param_grid=param_test1, iid=False, scoring='roc_auc', cv=5)
# gsearch1.fit(trainX, trainY)
# print gsearch1.grid_scores_
# print gsearch1.best_params_, gsearch1.best_score_

# model = ensemble.RandomForestClassifier(criterion = 'entropy', n_estimators = 130, min_samples_split = 130, max_depth = 13, max_features = 'auto', oob_score = True, random_state = 10)
# gbc = ensemble.GradientBoostingClassifier(n_estimators = 100, random_state = 10)
# etc = ensemble.ExtraTreesClassifier(criterion = 'entropy', n_estimators = 130, bootstrap=True, oob_score = True, random_state = 10)
model = xgb.XGBClassifier(n_estimators = 150)
# adc = ensemble.AdaBoostClassifier(n_estimators = 50, random_state = 10)
model.fit(trainX, trainY)
# joblib.dump(model, sys.argv[2])

#-------------------------------------------------
# testData = pd.read_csv("toTest.csv")
# testX, testY = util.transfer(testData, hasY=True)
# testX, testY = util.transfer(trainData[:5000], hasY=True)

testData = pd.read_csv("Test_Public.csv")
testX, testY = util.transfer(testData, hasY=False)

# rf_oof_train, rf_oof_test = get_oof(rfc, trainX, trainY, testX)
# gb_oof_train, gb_oof_test = get_oof(gbc, trainX, trainY, testX)
# et_oof_train, et_oof_test = get_oof(etc, trainX, trainY, testX)
# xg_oof_train, xg_oof_test = get_oof(xgc, trainX, trainY, testX)
# ad_oof_train, ad_oof_test = get_oof(adc, trainX, trainY, testX)
# base_predictions_train = pd.DataFrame({
#     'RandomForest': rf_oof_train.ravel(),
#     'GradientBoost': gb_oof_train.ravel(),
#     'ExtraTrees': et_oof_train.ravel(),
#     'XGBoost': xg_oof_train.ravel(),
#     'AdaBoost': ad_oof_train.ravel()
# })
# trainX_2nd = np.concatenate((rf_oof_train, gb_oof_train, et_oof_train, xg_oof_train, ad_oof_train), axis=1)
# testX_2nd = np.concatenate((rf_oof_test, gb_oof_test, et_oof_test, xg_oof_test, ad_oof_test), axis=1)
# model = xgb.XGBClassifier(n_estimators = 100)
# model.fit(trainX, trainY)

# print model.feature_importances_
# print "score\t", model.score(testX, testY)
test_prob = model.predict_proba(testX)
test_pre = model.predict(testX)
# print "auc\t", metrics.roc_auc_score(testY, test_prob[:, 1])

rankTest = []
for index in range(len(testX)):
    if testY is not None:
        rankTest.append([index + 1, test_prob[index][1], testY[index]])
    else:
        rankTest.append([index + 1, test_prob[index][1], test_pre[index]])
rankTest = sorted(rankTest, key = lambda x : x[1])
rankTest.reverse()
# print "kaggle\t", util.avpre(rankTest, 500)
rankTest = np.asarray(rankTest)
public_test = pd.DataFrame({'Rank_ID': rankTest[:, 0]})
public_test.Rank_ID = public_test.Rank_ID.astype('int')
public_test.to_csv("public3.csv", index=False)

