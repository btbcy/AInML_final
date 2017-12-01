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
    dataY = []
    if hasY:
        dataY = data['Y']
        data = data.drop(['Y'], axis=1)
    data['BILLaddPAY_1'] = data['BILL_AMT1'] + data['PAY_AMT1']
    data['BILLaddPAY_2'] = data['BILL_AMT2'] + data['PAY_AMT2']
    data['BILLaddPAY_3'] = data['BILL_AMT3'] + data['PAY_AMT3']
    data['BILLaddPAY_4'] = data['BILL_AMT4'] + data['PAY_AMT4']
    data['BILLaddPAY_5'] = data['BILL_AMT5'] + data['PAY_AMT5']
    data['BILLaddPAY_6'] = data['BILL_AMT6'] + data['PAY_AMT6']
    data['MARcorssAGE'] = data['MARRIAGE'] * data['AGE']
    data = data.drop(['Train_ID'], axis=1)
    data = data.drop(['SEX'], axis=1)
    # data = data.drop(['MARRIAGE'], axis=1)
    # data['PAYadd'] = data['PAY_1'] + data['PAY_2'] * 0.8 + data['PAY_3'] * 0.6 + data['PAY_4'] * 0.4 + data['PAY_5'] * 0.2 + data['PAY_6'] * 0.1
    dataX = data.as_matrix()
    return dataX, dataY
