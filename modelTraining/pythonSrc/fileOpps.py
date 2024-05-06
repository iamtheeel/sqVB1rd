###
# saveModel.py
#
# Joshua Mehlman
# ENGR 859 Spring 2024
# Term Project
#
# Squirrl Or Bird Detector
#
# File operations to help with tagging
#
# Rename files to include the run number
import os, fnmatch, shutil
from pathlib import Path
from random import random

moveFromCam = True
testTrainSplit = False

if moveFromCam:
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
    runDir = "240505_3"

    #move data from photo folder to data dir
    baseDir = "../../images"
    inDir = baseDir+"/"+runDir+'/'+"DCIM"
    outDir = baseDir+'/'+runDir+"/reNamedImages"

if testTrainSplit:
    splitPerc = 10
    #sqb = "Bird"
    #sqb = "Empty"
    sqb = "Squirrel"
    baseDir = "../data"
    inDir = baseDir+'/all/' + sqb
    outDir = baseDir+'/'

# Make a new dir for Where we put the new images
dirPath = Path(outDir)
dirPath.mkdir(exist_ok=True)


listing = os.scandir(inDir)
for file in listing:
    if fnmatch.fnmatch(file, '*.JPG'):
        imageFile_str =inDir+'/'+file.name

        if moveFromCam:
            newName_str = outDir+'/'+runDir+'-'+file.name

        if testTrainSplit:
            randNum = round(random()*100)
            if(randNum <splitPerc):
                testTrain = 'test'
            else:
                testTrain = 'train'
            newName_str = outDir+testTrain +'/' + sqb

        shutil.copy2(imageFile_str, newName_str)
        print(f"Copoy file: {imageFile_str} to: {newName_str}")