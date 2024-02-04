# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 08:16:45 2020

@author: ASUS

"""
import numpy as np
import copy
from math import sqrt
from scipy import stats
from sklearn import preprocessing,metrics
from lifelines.utils import concordance_index

def mse(y_true, y_pred):
    """
    计算均方误差（Mean Squared Error）

    参数:
        y_true: 实际观测值，一维NumPy数组或列表
        y_pred: 预测值，一维NumPy数组或列表

    返回值:
        均方误差（MSE）
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算误差平方和
    mse = np.mean((y_true - y_pred) ** 2)

    return mse


def ci(y_true, y_pred):
    """
    计算Concordance Index (CI)

    参数:
        y_true: 实际观测值，一维NumPy数组或列表
        y_pred: 预测值，一维NumPy数组或列表

    返回值:
        Concordance Index (CI)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ci = concordance_index(y_true, y_pred)

    return ci

def rmse(y,predict):
    y = np.array(y)
    predict = np.array(predict)
    rmse = np.sqrt(np.mean((y - predict)**2))
    return rmse

def pearson(y,predict):    
    y = np.array(y)
    predict = np.array(predict)
    rp = np.corrcoef(y.T, predict.T)[0,1]
    return rp

def getROCE(predList,targetList,roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index,x] for index,x in enumerate(predList)]
    predList = sorted(predList,key = lambda x:x[1],reverse = True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce


