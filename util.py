import pandas as pd
from pandas import Series, DataFrame
import numpy as np

def avpre(data, k):
    sumY = 0.0
    sumAvY = 0.0
    for i in range(k):
        sumY += data[i][-1]
        sumAvY += sumY / (i+1)
    return sumAvY / k

def rank(dataX, dataY, data_prob):
    rank = []
    for index in range(len(dataX)):
        rank.append([index + 1, data_prob[index][1], dataY[index]])
    rank = sorted(rank, key = lambda x : x[1])
    rank.reverse()
    return avpre(rank, 500)

def transfer(data, hasY = False):
    dataY = None
    if hasY:
        dataY = data['Y']
        data = data.drop(['Y'], axis=1)
    data['MARcorssAGE'] = data['MARRIAGE'] * data['AGE']
    data['USE1'] = data['LIMIT_BAL'] - data['BILL_AMT1'] - data['PAY_AMT1']
    data['PAY_DI'] = data['PAY_1'] - data['PAY_2']
    data['BILL_DI1'] = data['BILL_AMT1'] - data['BILL_AMT2']
    data['BILL_DI12_PAY'] = data['BILL_AMT1'] - data['BILL_AMT2'] - data['PAY_AMT1'] # ?
    # data['PAY_DI'] = data['PAY_2'] - data['PAY_3']
    # data['PAY_DI2'] = (data['PAY_1'] - data['PAY_2']) / (data['PAY_1'] + 3)
    # data['PdivB'] = data['PAY_AMT1'] / (data['BILL_AMT1'] + 0.001)
    # data['USE_RATIO'] = data['USE'] / data['LIMIT_BAL']
    # data['BILL_DI2'] = data['BILL_AMT2'] - data['BILL_AMT3']
    # data['BILL_LIMIT_RATIO_1'] = data['BILL_AMT1'] / data['LIMIT_BAL']

    data['AGE'] = (data['AGE'] / 5).astype(int)
    # data['LIMIT_BAL'] = (data['LIMIT_BAL'] / 5000).astype(int)
    # data['PAY_AMT1'] = data['PAY_AMT1'].apply(lambda x: log(x+1))

    data = data.drop(['SEX'], axis=1)
    data = data.drop(['MARRIAGE'], axis=1)
    # print data.head(0)
    dataX = data.as_matrix()
    dataX = dataX[:, 1:]
    return dataX, dataY
