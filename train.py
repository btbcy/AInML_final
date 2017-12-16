from sklearn import ensemble
# from sklearn import model_selection
# from sklearn import metrics
# from sklearn.model_selection import GridSearchCV
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
warnings.simplefilter(action='ignore', category=(UserWarning, RuntimeWarning))

# def get_oof(clf, x_train, y_train, x_test):
#     kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=10)
#     train_iter = np.zeros((x_train.shape[0], )) # 1 * 15000
#     test_iter = np.zeros((x_test.shape[0], ))   # 1 * 5000
#     test_skf = np.empty((5, x_test.shape[0]))   # 5 * 5000
# 
#     for idx, (train_idx, test_idx) in enumerate(kf.split(x_train)):
#         kf_x_train = x_train[train_idx] # 12000 * feature
#         kf_y_train = y_train[train_idx] # 12000 * 1
#         kf_x_test = x_train[test_idx]   # 3000 * feature
# 
#         clf.fit(kf_x_train, kf_y_train)
#         train_iter[test_idx] = clf.predict_proba(kf_x_test)[:, 1] # 1* 3000
#         test_skf[idx, :] = clf.predict_proba(x_test)[:, 1] # 1 * 5000
# 
#     test_iter[:] = test_skf.mean(axis=0)   # 1 * 5000
#     train_iter = train_iter.reshape(-1, 1) # 15000 * 1
#     test_iter = test_iter.reshape(-1, 1)   # 5000 * 1
#     return train_iter, test_iter

trainData = pd.read_csv(sys.argv[1])

dataX, dataY= util.transfer(trainData, hasY=True)

model = list()
# model.append(ensemble.RandomForestClassifier(criterion='entropy', n_estimators=130, max_depth=12, oob_score=True, random_state=10))
model.append(ensemble.ExtraTreesClassifier(criterion='entropy', n_estimators=100, max_depth=16, bootstrap=True, oob_score=True, random_state=10))
model.append(ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=10))
model.append(xgb.XGBClassifier(n_estimators=160, max_depth=4))
# joblib.dump(model, sys.argv[2])

#-------------------------------------------------
isTestPublic = False

if not isTestPublic:

    testData = pd.read_csv("Test_PublicA.csv")
    testX, testY = util.transfer(testData, hasY=True)

    #---------------
    # param_test1 = {'max_depth':range(3, 14, 2), 'min_samples_split':range(100, 801, 200)}
    # gsearch1 = GridSearchCV(estimator=ensemble.GradientBoostingClassifier(learning_rate=0.1, n_estimators=95, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10), param_grid=param_test1, iid=False, scoring='roc_auc', cv=5)
    # gsearch1.fit(trainX, trainY)
    # print gsearch1.grid_scores_
    # print ""
    # print gsearch1.best_params_, gsearch1.best_score_
    #----------------

    test_prob = list()
    for i in range(len(model)):
        model[i].fit(dataX, dataY)
        test_prob.append(model[i].predict_proba(testX))
    rankTest = []
    testY = testY.as_matrix()
    for index in range(len(testX)):
        avsum = 0.0
        for nm in range(len(model)):
            avsum += test_prob[nm][index][1]
        # avsum /= len(model)
        rankTest.append([avsum, testY[index]])
    rankTest = sorted(rankTest, key = lambda x : x[0])
    rankTest.reverse()
    print "ekaggle  ", util.avpre(rankTest, 500)

    for nm in range(len(model)):
        rankTest = []
        for index in range(len(testX)):
            rankTest.append([test_prob[nm][index][1], testY[index]])
        rankTest = sorted(rankTest, key = lambda x : x[0])
        rankTest.reverse()
        print "model", nm, " ", util.avpre(rankTest, 500)
        print model[nm].feature_importances_

    # model[0].fit(trainX, trainY)
    # # print adc.feature_importances_
    # # print "score\t", adc.score(testX, testY)
    # test_prob1 = model[0].predict_proba(testX)
    # # print "auc\t", metrics.roc_auc_score(testY, test_prob1[:, 1])
    # rankTest = []
    # for index in range(len(testX)):
    #     rankTest.append([test_prob1[index][1], testY[index]])
    # rankTest = sorted(rankTest, key = lambda x : x[0])
    # rankTest.reverse()
    # print "kaggle\t", util.avpre(rankTest, 500)


    testData = pd.read_csv("Test_PrivateA.csv")
    testX, testY = util.transfer(testData, hasY=True)
    print ""
    test_prob = list()
    for i in range(len(model)):
        model[i].fit(dataX, dataY)
        test_prob.append(model[i].predict_proba(testX))
    rankTest = []
    testY = testY.as_matrix()
    for index in range(len(testX)):
        avsum = 0.0
        for nm in range(len(model)):
            avsum += test_prob[nm][index][1]
        # avsum /= len(model)
        rankTest.append([avsum, testY[index]])
    rankTest = sorted(rankTest, key = lambda x : x[0])
    rankTest.reverse()
    print "ekaggle  ", util.avpre(rankTest, 500)

    for nm in range(len(model)):
        rankTest = []
        for index in range(len(testX)):
            rankTest.append([test_prob[nm][index][1], testY[index]])
        rankTest = sorted(rankTest, key = lambda x : x[0])
        rankTest.reverse()
        print "model", nm, " ", util.avpre(rankTest, 500)

else:
    testData = pd.read_csv("Test_Public.csv")
    testX, testY = util.transfer(testData, hasY=False)

    test_prob = list()
    for i in range(len(model)):
        model[i].fit(dataX, dataY)
        test_prob.append(model[i].predict_proba(testX))
    rankTest = []
    for index in range(len(testX)):
        avsum = 0.0
        for nm in range(len(model)):
            avsum += test_prob[nm][index][1]
        avsum /= len(model)
        rankTest.append([index + 1, avsum])
    rankTest = sorted(rankTest, key = lambda x : x[1])
    rankTest.reverse()
    rankTest = np.asarray(rankTest)
    public_test = pd.DataFrame({'Rank_ID': rankTest[:, 0]})
    public_test.Rank_ID = public_test.Rank_ID.astype('int')
    public_test.to_csv("public.csv", index=False)

    # model.fit(dataX, dataY)
    # test_prob = model.predict_proba(testX)

    # rankTest = []
    # for index in range(len(testX)):
    #     rankTest.append([index + 1, test_prob[index][1]])
    # rankTest = sorted(rankTest, key = lambda x : x[1])
    # rankTest.reverse()
    # rankTest = np.asarray(rankTest)
    # public_test = pd.DataFrame({'Rank_ID': rankTest[:, 0]})
    # public_test.Rank_ID = public_test.Rank_ID.astype('int')
    # public_test.to_csv("public2.csv", index=False)

