#
# Joshua Mehlman
# ENGR 859 Spring 2024
# Term Project
#
# Squirrl Or Bird Detector
#
# Data augmentation/reducing
#
# Rename files to include the run number
import os, fnmatch, shutil
#from pathlib import Path
#from random import random
import torchvision.transforms as trans
from PIL import Image

runDir = "240504_1"
#sqb = "Squirrel"
sqb = "Bird"

#classes = ['Bird', 'Empty', 'Squirrel'] # Its loaded alphabeticlyh


baseDir = "../../images" 
train_dir = baseDir + '/tagged' + '/train'


rotTrans = trans.Compose([ 
    trans.RandomRotation(degrees=(-10, 10)),
    trans.ToTensor()
])

perspTrans = trans.Compose([ 
    trans.RandomPerspective(distortion_scale=0.25, p=1), #p=1, means every image
    trans.ToTensor()
])
#clasDir = train_dir + '/' + sqb
dataDir = baseDir + '/' + runDir +'/' + 'tagged' + '/train/' + sqb

print(f"{dataDir}")
listing = os.scandir(dataDir)
for file in listing:
    if fnmatch.fnmatch(file, '*.JPG'):
        imageFile_str =dataDir+'/'+file.name
        image = Image.open(imageFile_str)  

        newImg = rotTrans(image)
        newImg = trans.ToPILImage()(newImg)
        newImgName = 'rot_' + file.name
        newImageFile_str =dataDir + '/'+ newImgName
        newImg.save(newImageFile_str)

        newImg = perspTrans(image)
        newImg = trans.ToPILImage()(newImg)
        newImgName = 'per_' + file.name
        newImageFile_str =dataDir + '/'+ newImgName
        newImg.save(newImageFile_str)

        #print(f"{dataDir}, {file.name}, {newImgName}")