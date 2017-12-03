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

def partial(data, num):
    sumY = 0.0
    for i in range(num):
        sumY += data[i][-1]
    return sumY / num

def rank(dataX, dataY, data_prob):
    rank = []
    for index in range(len(dataX)):
        rank.append([index + 1, data_prob[index][1], dataY[index]])
    rank = sorted(rank, key = lambda x : x[1])
    rank.reverse()
    return avpre(rank, 500)

def discre(data, index,  maximum, minimum, number):
    return data

def transfer(data, hasY = False):
    dataY = None
    if hasY:
        dataY = data['Y']
        data = data.drop(['Y'], axis=1)
    data['MARcorssAGE'] = data['MARRIAGE'] * data['AGE']
    data['USE'] = data['LIMIT_BAL'] - data['BILL_AMT1'] - data['PAY_AMT1']
    data['LIMIT_AGE'] = data['LIMIT_BAL'] / data['AGE']

    data = data.drop(['SEX'], axis=1)
    data = data.drop(['MARRIAGE'], axis=1)
    # print data.describe(include='all')
    dataX = data.as_matrix()
    dataX = dataX[:, 1:]
    return dataX, dataY
