from sklearn import ensemble
from sklearn import model_selection
# from sklearn import metrics
# from sklearn.model_selection import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.externals import joblib
import xgboost as xgb
import lightgbm as lgb
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
# model.append(ensemble.ExtraTreesClassifier(criterion='entropy', n_estimators=98, max_depth=16, bootstrap=True, oob_score=True, random_state=10))
# model.append(ensemble.GradientBoostingClassifier(n_estimators=126, max_depth=4, max_leaf_nodes=30, random_state=10))
# model.append(xgb.XGBClassifier(n_estimators=158, max_depth=4))
model.append(lgb.LGBMClassifier(n_estimators=125, max_depth=5, num_leaves=100))
# model.append(ensemble.RandomForestClassifier(criterion='entropy', n_estimators=135, max_depth=12, oob_score=True, random_state=10))
# model.append(ensemble.GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=10))
# model.append(ensemble.VotingClassifier(estimators=[('et', model[0]), ('gdb', model[1]), ('xgb', model[2])], voting='soft', weights=[1, 1, 1.4]))
# model.append(ensemble.VotingClassifier(estimators=[('et', model[0]), ('gdb', model[1]), ('xgb', model[2]), ('lgb', model[3])], voting='soft', weights=[1, 1, 1, 1]))
# joblib.dump(model, sys.argv[2])

#-------------------------------------------------
isTestPublic = False

if not isTestPublic:

    testData = pd.read_csv("Test_PublicA.csv")
    testX, testY = util.transfer(testData, hasY=True)

    lr = lgb.LGBMClassifier(n_estimators=1000, max_depth=6, num_leaves=100)
    data_leavesX = []
    test_leavesX = []
    for i in range(len(model)):
        model[i].fit(dataX, dataY)
        train_leaves = model[i].apply(dataX)
        test_leaves = model[i].apply(testX)
        leaves = np.concatenate((train_leaves, test_leaves), axis=0)

        if len(leaves.shape) > 2:
            data_leavesX.append(leaves[:, :, 0])
        else:
            data_leavesX.append(leaves)
        data_leavesX[i] = data_leavesX[i].astype(np.int32)
        rows, cols = data_leavesX[i].shape
        cum_count = np.zeros((1, cols), dtype=np.int32)
        cum_count[0][0] = len(np.unique(data_leavesX[i][:, 0]))
        for idx in range(1, cols):
            cum_count[0][idx] = len(np.unique(data_leavesX[i][:, idx])) + cum_count[0][idx-1]
        transXdict = {}
        for bbb in range(cols):
            initial_index = cum_count[0][bbb-1] + 1 if bbb != 0 else 1
            for aaa in range(rows):
                if data_leavesX[i][aaa, bbb] not in transXdict:
                    transXdict[data_leavesX[i][aaa, bbb]] = initial_index
                    data_leavesX[i][aaa, bbb] = initial_index
                    initial_index += 1
                else:
                    data_leavesX[i][aaa, bbb] = transXdict[data_leavesX[i][aaa, bbb]]
        test_leavesX.append(data_leavesX[i][len(dataX):, :])
        data_leavesX[i] = data_leavesX[i][:len(dataX), :]

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
        test_prob.append(model[i].predict_proba(testX)[:, 1])

    for nm in range(len(model)):
        rankTest = []
        for index in range(len(testX)):
            rankTest.append([test_prob[nm][index], testY[index]])
        rankTest = sorted(rankTest, key = lambda x : -x[0])
        print "model", nm, " ", util.avpre(rankTest, 500)
        # print model[nm].feature_importances_

    for i in range(len(model)):
        lr.fit(data_leavesX[i], dataY)
        prob2nd = lr.predict_proba(test_leavesX[i])[:, 1]
        rankTest = []
        for index in range(len(testX)):
            rankTest.append([prob2nd[index], testY[index]])
        rankTest = sorted(rankTest, key = lambda x : -x[0])
        print "model", nm, " ", util.avpre(rankTest, 500)

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
        test_prob.append(model[i].predict_proba(testX))

    for nm in range(len(model)):
        rankTest = []
        for index in range(len(testX)):
            rankTest.append([test_prob[nm][index][1], testY[index]])
        rankTest = sorted(rankTest, key = lambda x : -x[0])
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

