#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:35:50 2017

@author: liuzhen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:32:39 2017

@author: liuzhen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:24:23 2017

@author: liuzhen
"""

import numpy
from numpy import*

#-------------------------------------------------------------------
"""
class LinearRegression:
    def getAverageTheta():
        
"""
#-------------------------------------------------------------------
class Dataset:
    
    def __init__(self, dataset):
        self.dataset = dataset
        
#-------------------separate the dataset into k fold---------------------------
    def kFoldCrossingValidation(self, teSetNum, NumOfFold):
        self.teSetNum = teSetNum
        self.sizeOfFold = (int)( (self.dataset.shape[0] - teSetNum) / NumOfFold )
        count = 0
        fold = 0
        self.foldsList = numpy.zeros( (self.dataset.shape[1], NumOfFold, self.sizeOfFold) )
        self.testDatasetList = numpy.zeros( (self.dataset.shape[1], teSetNum ) )
        
        for i in range(dataset.shape[1]):
            for j in range(teSetNum):
                self.testDatasetList[i][j] = dataset[j + (NumOfFold) * self.sizeOfFold, i]

        for i in range(dataset.shape[1]):
            fold = 0
            count = 0
            for j in range( (NumOfFold) * self.sizeOfFold ):
                if self.sizeOfFold <= count:
                    fold = fold + 1
                    count = 0
                self.foldsList[i][fold][count] = dataset[j, i]                    
                count = count + 1

#-------------------get a single fold of a dataset column----------------------
    def getFoldingSubdataset(self, datasetColNum, foldNum):
        return self.foldsList[datasetColNum, foldNum, 0:]
    
#-------------------get whole testing dataset----------------------------------
    def getTestingSubdataset(self, datasetColNum):
        return self.testDatasetList[datasetColNum, 0:]
    
    def getSizeOfFold(self):
        return self.sizeOfFold

#------------------------------------------------------------------------------
import csv
ifile = open('Desktop/kc_house_data.csv', "r")
reader = csv.reader(ifile)
# 0 = price, 
# 1 = bedrooms,
# 2 = bathrooms
# 3 = sqft_living
# 4 = floors
# 5 = waterfront
# 6 = view
# 7 = grade
# 8 = sqft_above
# 9 = sqft_basement
# 10 = yr_renovated
# 11 = lat
# 12 = sqft_living15
dataset = numpy.zeros( (21613, 11) )
rownum = 0

#-------------------get all data from kc_house_data.cvs file and put into a 2-d array---
for row in reader:
    colnum = 0
    for col in row:
        if rownum != 0:
            if colnum == 2:
                # price
                dataset[rownum - 1][0] = col
            elif colnum == 3:
                # bedrooms
                dataset[rownum - 1][1] = col
            elif colnum == 4:
                # bathrooms
                dataset[rownum - 1][2] = col
            
            elif colnum == 5:
                # sqft_living
                dataset[rownum - 1][3] = col
            elif colnum == 7:
                # floors
                dataset[rownum - 1][4] = col
            elif colnum == 8:
                # waterfront
                dataset[rownum - 1][5] = col
            elif colnum == 9:
                # view
                dataset[rownum - 1][6] = col
            elif colnum == 11:
                # grade
                dataset[rownum - 1][7] = col
            elif colnum == 12:
                # sqft_above
                dataset[rownum - 1][8] = col
            elif colnum == 17:
                # lat
                dataset[rownum - 1][9] = col
            elif colnum == 19:
                # sqft_living15
                dataset[rownum - 1][10] = col
            
        colnum = colnum + 1
    rownum = rownum + 1

numOfTestingData = 2000
numOfFold = 5
# initialize class Dataset
foldingData = Dataset(dataset)
# start to separate all columns from dataset into k fold (k = 5)
foldingData.kFoldCrossingValidation(numOfTestingData, numOfFold)

# alpha is coefficient of L2 norm
# calculate average theta vector in OLS Regression and Ridge Regression
# use for loop to get 5-fold validation's training set and test set
# calculate R-square in 5-fold validation(both OSL and Ridge)
alpha = 0.5
olsAveR = 0
ridgeAveR = 0
olsAveTheta = mat([0])
ridgeAveTheta = mat([0])
for i in range(numOfFold):    
    trSet = numpy.zeros( (dataset.shape[1], foldingData.getSizeOfFold() * 4) )
    vSet = numpy.zeros( (dataset.shape[1], foldingData.getSizeOfFold()) )
    alphaSet = numpy.eye(dataset.shape[1])
    for j in range(dataset.shape[1]):
        count = 0
        for k in range(numOfFold):
            for l in range(foldingData.getSizeOfFold()):
                if k == i:
                    vSet[j][l] = foldingData.getFoldingSubdataset(j, k)[l]
                else:
                    trSet[j][count] = foldingData.getFoldingSubdataset(j, k)[l]
                    count = count + 1
    x0 = numpy.ones(foldingData.getSizeOfFold() * 4)
    V_x0 = numpy.ones(foldingData.getSizeOfFold())
    X = ( mat( [ x0, trSet[1, 0:foldingData.getSizeOfFold() * 4],
                     trSet[2, 0:foldingData.getSizeOfFold() * 4],
                     trSet[3, 0:foldingData.getSizeOfFold() * 4],
                     trSet[4, 0:foldingData.getSizeOfFold() * 4],
                     trSet[5, 0:foldingData.getSizeOfFold() * 4],
                     trSet[6, 0:foldingData.getSizeOfFold() * 4],
                     trSet[7, 0:foldingData.getSizeOfFold() * 4],
                     trSet[8, 0:foldingData.getSizeOfFold() * 4],
                     trSet[9, 0:foldingData.getSizeOfFold() * 4],
                     trSet[10, 0:foldingData.getSizeOfFold() * 4]]) ).T
    Y = ( mat( [trSet[0][0:foldingData.getSizeOfFold() * 4]] ) ).T
    validationX = ( mat( [ V_x0, vSet[1, 0:foldingData.getSizeOfFold()],
                         vSet[2, 0:foldingData.getSizeOfFold()],
                         vSet[3, 0:foldingData.getSizeOfFold()],
                         vSet[4, 0:foldingData.getSizeOfFold()],
                         vSet[5, 0:foldingData.getSizeOfFold()],
                         vSet[6, 0:foldingData.getSizeOfFold()],
                         vSet[7, 0:foldingData.getSizeOfFold()],
                         vSet[8, 0:foldingData.getSizeOfFold()],
                         vSet[9, 0:foldingData.getSizeOfFold()],
                         vSet[10, 0:foldingData.getSizeOfFold()]] ) ).T
    validationY = ( mat( [vSet[0][0:foldingData.getSizeOfFold()]] ) ).T

    theta_ols = ((X.T) * X ).I * (X.T) * Y
    theta_ridge = ((X.T) * X + alpha * alphaSet).I * (X.T) * Y
    olsAveTheta = olsAveTheta + theta_ols
    ridgeAveTheta = ridgeAveTheta + theta_ridge
    y_olsHat = validationX * theta_ols
    y_ridgeHat = validationX * theta_ridge
    vy_mean = sum(validationY[:]) / validationY.shape[0]
    y_commonSST = (validationY) - vy_mean
    SST = sum(multiply(y_commonSST,y_commonSST)[:])
    y_olsSSR = y_olsHat - vy_mean
    y_ridgeSSR = y_ridgeHat - vy_mean
    olsSSR = sum(multiply(y_olsSSR,y_olsSSR)[:])
    ridgeSSR = sum(multiply(y_ridgeSSR,y_ridgeSSR)[:])
    olsR = (olsSSR)/(SST)
    ridgeR = (ridgeSSR) / (SST)
    olsAveR = olsAveR + olsR
    ridgeAveR = ridgeAveR + ridgeR 

olsAveR = olsAveR / 5
ridgeAveR = ridgeAveR / 5
olsAveTheta = olsAveTheta / 5
ridgeAveTheta = ridgeAveTheta / 5

#-----------------------------put theta matrix back to testing set---------------------------------
yHat = numpy.zeros(numOfTestingData)
yRidgeHat = numpy.zeros(numOfTestingData)
testingSetX0 = numpy.ones(numOfTestingData)

# input testing set into our models
# output R-square of OLS and Ridge
for i in range(dataset.shape[1]):
    for j in range(numOfTestingData):
        if i == 0:
            yHat[j] = yHat[j] + (testingSetX0[j] * olsAveTheta[i]) 
            yRidgeHat[j] = yRidgeHat[j] + (testingSetX0[j] * ridgeAveTheta[i]) 
        else:
            yHat[j] = yHat[j] + (foldingData.getTestingSubdataset(i)[j] * olsAveTheta[i])
            yRidgeHat[j] = yRidgeHat[j] + (foldingData.getTestingSubdataset(i)[j] * ridgeAveTheta[i]) 

olsTestMean = sum(foldingData.getTestingSubdataset(0)[:]) / foldingData.getTestingSubdataset(0)[:].shape[0]
olsTestSST_a = foldingData.getTestingSubdataset(0) - olsTestMean
olsTestSST = sum(multiply(olsTestSST_a,olsTestSST_a)[:])
olsTestSSR_a = yHat - olsTestMean
olsTestSSR = sum(multiply(olsTestSSR_a,olsTestSSR_a)[:])
olsFinalRSquare = olsTestSSR / olsTestSST

ridgeTestSSR_a = yRidgeHat - olsTestMean
ridgeTestSSR = sum(multiply(ridgeTestSSR_a,ridgeTestSSR_a)[:])
ridgeFinalRSquare = ridgeTestSSR / olsTestSST

print("")
print("OLS R-Square is " + str(olsFinalRSquare))
print("Ridge R-Square is " + str(ridgeFinalRSquare))

ifile.close()
#-------------------------------------------------------------------