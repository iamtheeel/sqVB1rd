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
# File operations to help with tagging
###



# Rename files to include the run number
import os, fnmatch
from pathlib import Path
from random import random

moveFromCam = False
testTrainSplit = True\

baseDir = "../../images/"
#sqb = "Bird"
sqb = "Empty"
cull = True
#sqb = "Squirrel"


    # Where our existing images are
    #runDir = "240406_0_shed"
    #runDir = "240406_1_shed2"
    #runDir = "240406_2"
    #runDir = "240407_1"
    #runDir = "240407_2"
    #runDir = "240407_3"
    #runDir = "240407_4"
    #runDir = "240408_1"
    #runDir = "240408_2"
    #runDir = "240409_1"
    #runDir = "240409_2"
    #runDir = "240504_1"
    #runDir = "240505_2"
    #runDir = "240505_3"
    #runDir = "240506"
    #runDir = "240507_2"
    #runDir = "240509_1"
    #runDir = "240509_1"
    #runDir = "240509_2"
runDir = "240510"

if moveFromCam:
    inDir = baseDir+'/sucess/'+runDir +'/' + sqb
    #move data from photo folder to data dir
    outDir = baseDir+'/sucess/combImages/' +sqb +'/'
    #baseDir = "../../images/"
    #inDir = baseDir+'/'+runDir +'/' + sqb
    #outDir = baseDir+'/'+ runDir +'/reNamed/' +sqb +'/'

if testTrainSplit:
    inDir = baseDir+'/TrainingTestSet/Combined/' +sqb
    outDir = baseDir+'/TrainingTestSet/'

# Make a new dir for Where we put the new images
dirPath = Path(outDir)
dirPath.mkdir(exist_ok=True)


teCount = 0
trCount = 0
cuCount = 0

listing = os.scandir(inDir)
for file in listing:
    if fnmatch.fnmatch(file, '*.JPG'):
        imageFile_str =inDir+'/'+file.name

        if moveFromCam:
            newName_str = outDir+'/'+runDir+'-'+file.name

        ## Todo, cut the number of Emptys

        if testTrainSplit:
            randNum = round(random()*100)
            # 10% to test
            testPerc = 10

            if cull:
                # 99% to test
                cullPerc = 90

                if(randNum < cullPerc):
                    testTrain = 'cull'
                    cuCount += 1
                elif(randNum < 92):
                    testTrain = 'test'
                    teCount += 1
                else:
                    trCount += 1
                    testTrain = 'train'
            else:

                if(randNum < testPerc):
                    testTrain = 'test'
                    teCount += 1
                elif(randNum < 70):
                    cuCount += 1
                    testTrain = 'cull'
                else:
                    trCount += 1
                    testTrain = 'train'
            newName_str = outDir+testTrain +'/' + sqb

        #shutil.copy2(imageFile_str, newName_str)
        #print(f"Copy file: {imageFile_str} to: {newName_str}: {randNum}")
print(f"train: {trCount}, test: {teCount}, cull: {cuCount}")