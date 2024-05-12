###
# saveModel.py
#
# Joshua Mehlman
# ENGR 859 Spring 2024
# Term Project
#
# Squirrl Or Bird Detector
#
###
#
# Read in tagged log files 
# confusion matrix
# Calculate Sensitivity aka recall
# Calculate Specificity
# 
###

import csv
import matplotlib.pyplot as plt
import torch
from torchmetrics import  ConfusionMatrix
from torchmetrics.classification import MulticlassSpecificity, MulticlassRecall
#from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics 
import numpy as np


showBadImage = False

# Which runs do we want
#runDir = "sucess"
#logFileNameList = ["240509_1-13_log", "240509_1-17_log", "240509_1-19_log", "240509_2-3_log", "240510-1_log"]
runDir = "retrained_LeNetV5"
logFileNameList = ["2_log"]

# Where our log files
baseDir = "../../images"
logFileDir = baseDir+"/"+runDir+'/' 

classes = ['Bird', 'Empty', 'Squirrel'] # Its loaded alphabeticly
if showBadImage:
    classes = ['Bird', 'Empty', 'Squirrel', 'Bad Image'] # Its loaded alphabeticly

preds =[] # for confusion matrix
lables =[] # for confusion matrix

np.set_printoptions(precision=2)


for logFileName in logFileNameList:
    wrongGuesFile = logFileDir + logFileName + '_wrong.csv'
    csvColNames = ['imageNum', 'birdConf', 'noneConf', 'squiConf', 'predict', 'manuaTag']
    with open(wrongGuesFile, 'w', newline='') as badFile: # make a new file
        csvWriter = csv.DictWriter(badFile, fieldnames=csvColNames)
        csvWriter.writeheader()

    logFileDirName = logFileDir + logFileName + '.csv'
    print(f"Log file: {logFileDirName}")
    predsThis =[] # for confusion matrix
    lablesThis =[] # for confusion matrix

    birdCount = 0
    squiCount = 0
    noneCount = 0

    birdMax = 0
    squiMax = 0
    noneMax = 0
    birdMaxImage = 0
    noneMaxImage = 0
    squiMaxImage = 0


    with open(logFileDirName) as logFile:
        logReader = csv.reader(logFile, delimiter=",")
        headder = next(logReader) # skip the headder

        for row in logReader:
            #FileNum, Bird, None, squirrel,manual tag
            imageNum = int(row[0])
            birdConf = float(row[1])
            noneConf = float(row[2])
            squiConf = float(row[3])
            manuaTag = int(row[4])
            #print(f"imageNum: {imageNum}, birdConf: {birdConf}, noneConf: {noneConf}, squirrlConf: {squiConf}, tag: {manuaTag}")
            maxConf = max(birdConf, noneConf, squiConf)
            predict = 1
            if maxConf == birdConf:
                predict = 0
            if maxConf == squiConf:
                predict = 2

            if manuaTag == predict:
                if predict == 0 and birdConf > birdMax: 
                    birdMax = birdConf
                    birdMaxImage = imageNum
                    #print(f"updating bird max: {birdMax}, {birdMaxImage}")
                if predict == 1 and noneConf > noneMax: 
                    noneMax = noneConf
                    noneMaxImage = imageNum
                if predict == 2 and squiConf > squiMax: 
                    squiMax = squiConf
                    squiMaxImage = imageNum
            else: 
                with open(wrongGuesFile, 'a', newline='') as badFile: # make a new file
                    csvWriter = csv.DictWriter(badFile, fieldnames=csvColNames)
                    csvWriter.writerow({'imageNum': imageNum, 
                                        'birdConf': birdConf, 'noneConf':noneConf, 'squiConf':squiConf, 
                                        'predict':predict, 'manuaTag':manuaTag})


            if(manuaTag == 0): birdCount += 1
            if(manuaTag == 1): noneCount += 1
            if(manuaTag == 2): squiCount += 1

            if showBadImage or (manuaTag < 3):
                predsThis.append(predict) #for confusion matrix
                lablesThis.append(manuaTag)

                # For the combined
                preds.append(predict)
                lables.append(manuaTag)


    #this log
    print(f"Label Counts, bird: {birdCount}, none: {noneCount}, squirll: {squiCount}")
    print(f"Max Confidences bird: {birdMax:.2f}, {birdMaxImage} none: {noneMax:.2f}, {noneMaxImage} squirll: {squiMax:.2f}, {squiMaxImage}")
    confMatrixData = metrics.confusion_matrix(lablesThis, predsThis)
    confMatrix = metrics.ConfusionMatrixDisplay(confusion_matrix=confMatrixData, display_labels=classes)

    specificity = MulticlassSpecificity(num_classes=len(classes), average=None)
    spec = specificity(torch.Tensor(predsThis), torch.Tensor(lablesThis))
    sensitivity = MulticlassRecall(num_classes=len(classes), average=None)
    recal = sensitivity(torch.Tensor(predsThis), torch.Tensor(lablesThis))

    confMatrix.plot()
    #plt.title(f"{logFileName} \nSpecificity: {spec:.2f}\nRecal: {recal:.2f}")
    plt.title(f"{logFileName} \nSpecificity: {spec.numpy()}\nRecal: {recal.numpy()}")
    cmFileDirName = logFileDir + 'CF_' + logFileName + '.png'
    plt.savefig(cmFileDirName)
    #plt.show()

#Conf matrix for the sum of all files
#print(f"lab: {lables}")
#print(f"pred: {preds}")
confMatrixData = metrics.confusion_matrix(lables, preds)
confMatrix = metrics.ConfusionMatrixDisplay(confusion_matrix=confMatrixData, display_labels=classes)

specificity = MulticlassSpecificity(num_classes=len(classes), average='none')
spec = specificity(torch.Tensor(preds), torch.Tensor(lables))
sensitivity = MulticlassRecall(num_classes=len(classes), average='none')
recal = sensitivity(torch.Tensor(preds), torch.Tensor(lables))

confMatrix.plot()
#plt.title(f"Combined\nSpecificity: {spec:.2f}\nRecal: {recal:.2f}")
plt.title(f"Combined\nSpecificity: {spec.numpy()}\nSensitivity: {recal.numpy()}")
cmFileDirName = logFileDir + 'CF_combined.png'
print(f"combined file name: {cmFileDirName}")
plt.savefig(cmFileDirName)